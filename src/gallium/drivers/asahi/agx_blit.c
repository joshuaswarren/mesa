/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020-2021 Collabora, Ltd.
 * Copyright 2019 Sonny Jiang <sonnyj608@gmail.com>
 * Copyright 2019 Advanced Micro Devices, Inc.
 * Copyright 2014 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include "asahi/compiler/agx_debug.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/layout/layout.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "gallium/auxiliary/util/u_blitter.h"
#include "gallium/auxiliary/util/u_dump.h"
#include "nir/pipe_nir.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/box.h"
#include "util/u_sampler.h"
#include "util/u_surface.h"
#include "agx_state.h"
#include "glsl_types.h"
#include "nir.h"
#include "nir_builder_opcodes.h"
#include "shader_enums.h"

/* For block based blit kernels, we hardcode the maximum tile size which we can
 * always achieve. This simplifies our life.
 */
#define TILE_WIDTH  32
#define TILE_HEIGHT 32

static enum pipe_format
effective_format(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_Z32_FLOAT:
   case PIPE_FORMAT_Z24X8_UNORM:
      return PIPE_FORMAT_R32_FLOAT;
   case PIPE_FORMAT_Z16_UNORM:
      return PIPE_FORMAT_R16_UNORM;
   case PIPE_FORMAT_S8_UINT:
      return PIPE_FORMAT_R8_UINT;
   default:
      return format;
   }
}

static void *
asahi_blit_compute_shader(struct pipe_context *ctx, struct asahi_blit_key *key)
{
   const nir_shader_compiler_options *options =
      ctx->screen->nir_options[MESA_SHADER_COMPUTE];

   nir_builder b_ =
      nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, options, "blit_cs");
   nir_builder *b = &b_;
   b->shader->info.workgroup_size[0] = TILE_WIDTH;
   b->shader->info.workgroup_size[1] = TILE_HEIGHT;
   b->shader->info.num_ubos = 1;

   BITSET_SET(b->shader->info.textures_used, 0);
   BITSET_SET(b->shader->info.samplers_used, 0);
   BITSET_SET(b->shader->info.images_used, 0);

   nir_def *zero = nir_imm_int(b, 0);

   nir_def *params[4];
   b->shader->num_uniforms = ARRAY_SIZE(params);
   for (unsigned i = 0; i < b->shader->num_uniforms; ++i) {
      params[i] = nir_load_ubo(b, 2, 32, zero, nir_imm_int(b, i * 8),
                               .align_mul = 4, .range = ~0);
   }

   nir_def *trans_offs = params[0];
   nir_def *trans_scale = params[1];
   nir_def *dst_offs_2d = params[2];
   nir_def *dimensions_el_2d = params[3];

   nir_def *phys_id_el_nd = nir_trim_vector(
      b, nir_load_global_invocation_id(b, 32), key->array ? 3 : 2);
   nir_def *phys_id_el_2d = nir_trim_vector(b, phys_id_el_nd, 2);
   nir_def *layer = key->array ? nir_channel(b, phys_id_el_nd, 2) : NULL;

   /* Offset within the tile. We're dispatched for the entire tile but the
    * beginning might be out-of-bounds, so fix up.
    */
   nir_def *offs_in_tile_el_2d = nir_iand_imm(b, dst_offs_2d, 31);
   nir_def *logical_id_el_2d = nir_isub(b, phys_id_el_2d, offs_in_tile_el_2d);

   nir_def *image_pos_2d = nir_iadd(b, logical_id_el_2d, dst_offs_2d);
   nir_def *image_pos_nd = image_pos_2d;
   if (layer) {
      image_pos_nd =
         nir_vector_insert_imm(b, nir_pad_vector(b, image_pos_nd, 3), layer, 2);
   }

   nir_def *in_bounds;
   if (key->aligned) {
      in_bounds = nir_imm_true(b);
   } else {
      in_bounds = nir_ige(b, logical_id_el_2d, nir_imm_ivec2(b, 0, 0));
      in_bounds =
         nir_iand(b, in_bounds, nir_ilt(b, logical_id_el_2d, dimensions_el_2d));
   }

   nir_def *colour0, *colour1;
   nir_push_if(b, nir_ball(b, in_bounds));
   {
      /* For pixels within the copy area, texture from the source */
      nir_def *coords_el_2d =
         nir_ffma(b, nir_u2f32(b, logical_id_el_2d), trans_scale, trans_offs);

      nir_def *coords_el_nd = coords_el_2d;
      if (layer) {
         coords_el_nd = nir_vector_insert_imm(
            b, nir_pad_vector(b, coords_el_nd, 3), nir_u2f32(b, layer), 2);
      }

      colour0 = nir_tex(b, coords_el_nd, .texture_index = 0, .sampler_index = 0,
                        .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP,
                        .dim = GLSL_SAMPLER_DIM_2D, .is_array = key->array,
                        .dest_type = nir_type_uint32);
   }
   nir_push_else(b, NULL);
   {
      /* For out-of-bounds pixels, copy in the destination */
      colour1 = nir_image_load(
         b, 4, 32, nir_imm_int(b, 0), nir_pad_vec4(b, image_pos_nd), zero, zero,
         .image_array = key->array, .image_dim = GLSL_SAMPLER_DIM_2D,
         .access = ACCESS_IN_BOUNDS, .dest_type = nir_type_uint32);
   }
   nir_pop_if(b, NULL);
   nir_def *color = nir_if_phi(b, colour0, colour1);

   enum asahi_blit_clamp clamp = ASAHI_BLIT_CLAMP_NONE;
   bool src_sint = util_format_is_pure_sint(key->src_format);
   bool dst_sint = util_format_is_pure_sint(key->dst_format);
   if (util_format_is_pure_integer(key->src_format) &&
       util_format_is_pure_integer(key->dst_format)) {

      if (src_sint && !dst_sint)
         clamp = ASAHI_BLIT_CLAMP_SINT_TO_UINT;
      else if (!src_sint && dst_sint)
         clamp = ASAHI_BLIT_CLAMP_UINT_TO_SINT;
   }

   if (clamp == ASAHI_BLIT_CLAMP_SINT_TO_UINT)
      color = nir_imax(b, color, nir_imm_int(b, 0));
   else if (clamp == ASAHI_BLIT_CLAMP_UINT_TO_SINT)
      color = nir_umin(b, color, nir_imm_int(b, INT32_MAX));

   nir_def *local_offset = nir_imm_int(b, 0);
   nir_def *lid = nir_trim_vector(b, nir_load_local_invocation_id(b), 2);
   lid = nir_u2u16(b, lid);

   /* Pure integer formatss need to be clamped in software, at least in some
    * cases. We do so on store. Piglit gl-3.0-render-integer checks this, as
    * does KHR-GL33.packed_pixels.*.
    *
    * TODO: Make this common code somehow.
    */
   const struct util_format_description *desc =
      util_format_description(key->dst_format);
   unsigned c = util_format_get_first_non_void_channel(key->dst_format);

   if (desc->channel[c].size <= 16 &&
       util_format_is_pure_integer(key->dst_format)) {

      unsigned bits[4] = {
         desc->channel[0].size ?: desc->channel[0].size,
         desc->channel[1].size ?: desc->channel[0].size,
         desc->channel[2].size ?: desc->channel[0].size,
         desc->channel[3].size ?: desc->channel[0].size,
      };

      if (util_format_is_pure_sint(key->dst_format))
         color = nir_format_clamp_sint(b, color, bits);
      else
         color = nir_format_clamp_uint(b, color, bits);

      color = nir_u2u16(b, color);
   }

   /* The source texel has been converted into a 32-bit value. We need to
    * convert it to a tilebuffer format that can then be converted to the
    * destination format in the PBE hardware. That's the renderable format for
    * the destination format, which must exist along this path. This mirrors the
    * flow of fragment and end-of-tile shaders.
    */
   enum pipe_format tib_format =
      ail_pixel_format[effective_format(key->dst_format)].renderable;

   nir_store_local_pixel_agx(b, color, nir_imm_intN_t(b, 1, 16), lid, .base = 0,
                             .write_mask = 0xf, .format = tib_format,
                             .explicit_coord = true);

   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP);

   nir_push_if(b, nir_ball(b, nir_ieq_imm(b, lid, 0)));
   {
      nir_def *pbe_index = nir_imm_intN_t(b, 2, 16);
      nir_image_store_block_agx(
         b, pbe_index, local_offset, image_pos_nd, .format = tib_format,
         .image_dim = GLSL_SAMPLER_DIM_2D, .image_array = key->array,
         .explicit_coord = true);
   }
   nir_pop_if(b, NULL);
   b->shader->info.cs.image_block_size_per_thread_agx =
      util_format_get_blocksize(key->dst_format);

   return pipe_shader_from_nir(ctx, b->shader);
}

static bool
asahi_compute_blit_supported(enum agx_chip chip,
                             const struct pipe_blit_info *info)
{
   /* Gallium's compute blitter reaches agx_launch_grid() and therefore
    * agx_cdm_launch(), whose packet is the AGX2 compute command format, and
    * agx_upload_textures()'s legacy storage PBE descriptor. AGX3-encoding
    * parts need the native Ioto program/launch path Honeykrisp already uses;
    * on HAL200 the internal block-image kernel aborts in
    * agx_native_descriptor_rejected("storage PBE") before it is ever
    * submitted. Take the fully supported render blitter until native Gallium
    * compute is wired. */
   if (agx_chip_command_stream_generation(chip) != AGX_CS_GENERATION_AGX2)
      return false;

   return (info->src.box.depth == info->dst.box.depth) && !info->alpha_blend &&
          !info->num_window_rectangles && !info->sample0_only &&
          !info->scissor_enable && !info->window_rectangle_include &&
          !info->swizzle_enable && info->src.resource->nr_samples <= 1 &&
          info->dst.resource->nr_samples <= 1 &&
          !util_format_is_depth_and_stencil(info->src.format) &&
          !util_format_is_depth_and_stencil(info->dst.format) &&
          info->src.box.depth >= 0 &&
          info->mask == util_format_get_mask(info->src.format) &&
          /* XXX: texsubimage pbo failing otherwise, needs investigation */
          info->dst.format != PIPE_FORMAT_B5G6R5_UNORM &&
          info->dst.format != PIPE_FORMAT_B5G5R5A1_UNORM &&
          info->dst.format != PIPE_FORMAT_B5G5R5X1_UNORM &&
          info->dst.format != PIPE_FORMAT_R5G6B5_UNORM &&
          info->dst.format != PIPE_FORMAT_R5G5B5A1_UNORM &&
          info->dst.format != PIPE_FORMAT_R5G5B5X1_UNORM;
}

static void
asahi_compute_save(struct agx_context *ctx)
{
   struct asahi_blitter *blitter = &ctx->compute_blitter;
   struct agx_stage *stage = &ctx->stage[MESA_SHADER_COMPUTE];

   assert(!blitter->active && "recursion detected, driver bug");

   pipe_resource_reference(&blitter->saved_cb.buffer, stage->cb[0].buffer);
   memcpy(&blitter->saved_cb, &stage->cb[0],
          sizeof(struct pipe_constant_buffer));

   blitter->has_saved_image = stage->image_mask & BITFIELD_BIT(0);
   if (blitter->has_saved_image) {
      pipe_resource_reference(&blitter->saved_image.resource,
                              stage->images[0].resource);
      memcpy(&blitter->saved_image, &stage->images[0],
             sizeof(struct pipe_image_view));
   }

   pipe_sampler_view_reference(&blitter->saved_sampler_view,
                               &stage->textures[0]->base);

   blitter->saved_num_sampler_states = stage->sampler_count;
   memcpy(blitter->saved_sampler_states, stage->samplers,
          stage->sampler_count * sizeof(void *));

   blitter->saved_cs = stage->shader;
   blitter->active = true;
}

static void
asahi_compute_restore(struct agx_context *ctx)
{
   struct pipe_context *pctx = &ctx->base;
   struct asahi_blitter *blitter = &ctx->compute_blitter;

   if (blitter->has_saved_image) {
      pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0,
                              &blitter->saved_image);
      pipe_resource_reference(&blitter->saved_image.resource, NULL);
   }

   /* take_ownership=true so do not unreference */
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &blitter->saved_cb);
   blitter->saved_cb.buffer = NULL;

   if (blitter->saved_sampler_view) {
      pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 1, 0,
                              &blitter->saved_sampler_view);

      blitter->saved_sampler_view = NULL;
   }

   if (blitter->saved_num_sampler_states) {
      pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0,
                                blitter->saved_num_sampler_states,
                                blitter->saved_sampler_states);
   }

   pctx->bind_compute_state(pctx, blitter->saved_cs);
   blitter->saved_cs = NULL;
   blitter->active = false;
}

static void
asahi_compute_blit(struct pipe_context *ctx, const struct pipe_blit_info *info,
                   struct asahi_blitter *blitter)
{
   if (info->src.box.width == 0 || info->src.box.height == 0 ||
       info->dst.box.width == 0 || info->dst.box.height == 0)
      return;

   assert(asahi_compute_blit_supported(agx_device(ctx->screen)->chip, info));
   asahi_compute_save(agx_context(ctx));

   unsigned depth = info->dst.box.depth;
   bool array = depth > 1;

   struct pipe_resource *src = info->src.resource;
   struct pipe_resource *dst = info->dst.resource;
   struct pipe_sampler_view src_templ = {0}, *src_view;

   float src_width = (float)u_minify(src->width0, info->src.level);
   float src_height = (float)u_minify(src->height0, info->src.level);

   float x_scale =
      (info->src.box.width / (float)info->dst.box.width) / src_width;

   float y_scale =
      (info->src.box.height / (float)info->dst.box.height) / src_height;

   /* Expand the grid so destinations are in tiles */
   unsigned expanded_x0 = info->dst.box.x & ~(TILE_WIDTH - 1);
   unsigned expanded_y0 = info->dst.box.y & ~(TILE_HEIGHT - 1);
   unsigned expanded_x1 =
      align(info->dst.box.x + info->dst.box.width, TILE_WIDTH);
   unsigned expanded_y1 =
      align(info->dst.box.y + info->dst.box.height, TILE_HEIGHT);

   /* But clamp to the destination size to save some redundant threads */
   expanded_x1 =
      MIN2(expanded_x1, u_minify(info->dst.resource->width0, info->dst.level));
   expanded_y1 =
      MIN2(expanded_y1, u_minify(info->dst.resource->height0, info->dst.level));

   /* Calculate the width/height based on the expanded grid */
   unsigned width = expanded_x1 - expanded_x0;
   unsigned height = expanded_y1 - expanded_y0;

   unsigned data[] = {
      fui(0.5f * x_scale + (float)info->src.box.x / src_width),
      fui(0.5f * y_scale + (float)info->src.box.y / src_height),
      fui(x_scale),
      fui(y_scale),
      info->dst.box.x,
      info->dst.box.y,
      info->dst.box.width,
      info->dst.box.height,
   };

   struct pipe_constant_buffer cb = {
      .buffer_size = sizeof(data),
      .user_buffer = data,
   };
   ctx->set_constant_buffer(ctx, MESA_SHADER_COMPUTE, 0, &cb);

   struct pipe_image_view image = {
      .resource = dst,
      .access = PIPE_IMAGE_ACCESS_WRITE | PIPE_IMAGE_ACCESS_DRIVER_INTERNAL,
      .shader_access = PIPE_IMAGE_ACCESS_WRITE,
      .format = info->dst.format,
      .u.tex.level = info->dst.level,
      .u.tex.first_layer = info->dst.box.z,
      .u.tex.last_layer = info->dst.box.z + depth - 1,
      .u.tex.single_layer_view = !array,
   };
   ctx->set_shader_images(ctx, MESA_SHADER_COMPUTE, 0, 1, 0, &image);

   if (!blitter->sampler[info->filter]) {
      struct pipe_sampler_state sampler_state = {
         .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
         .min_img_filter = info->filter,
         .mag_img_filter = info->filter,
         .compare_func = PIPE_FUNC_ALWAYS,
         .seamless_cube_map = true,
         .max_lod = 31.0f,
      };

      blitter->sampler[info->filter] =
         ctx->create_sampler_state(ctx, &sampler_state);
   }

   ctx->bind_sampler_states(ctx, MESA_SHADER_COMPUTE, 0, 1,
                            &blitter->sampler[info->filter]);

   /* Initialize the sampler view. */
   u_sampler_view_default_template(&src_templ, src, src->format);
   src_templ.format = info->src.format;
   src_templ.target = array ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D;
   src_templ.swizzle_r = PIPE_SWIZZLE_X;
   src_templ.swizzle_g = PIPE_SWIZZLE_Y;
   src_templ.swizzle_b = PIPE_SWIZZLE_Z;
   src_templ.swizzle_a = PIPE_SWIZZLE_W;
   src_templ.u.tex.first_layer = info->src.box.z;
   src_templ.u.tex.last_layer = info->src.box.z + depth - 1;
   src_templ.u.tex.first_level = info->src.level;
   src_templ.u.tex.last_level = info->src.level;
   src_view = ctx->create_sampler_view(ctx, src, &src_templ);
   ctx->set_sampler_views(ctx, MESA_SHADER_COMPUTE, 0, 1, 0, &src_view);
   ctx->sampler_view_release(ctx, src_view);

   struct asahi_blit_key key = {
      .src_format = info->src.format,
      .dst_format = info->dst.format,
      .array = array,
      .aligned = info->dst.box.width == width && info->dst.box.height == height,
   };
   struct hash_entry *ent = _mesa_hash_table_search(blitter->blit_cs, &key);
   void *cs = NULL;

   if (ent) {
      cs = ent->data;
   } else {
      cs = asahi_blit_compute_shader(ctx, &key);
      _mesa_hash_table_insert(
         blitter->blit_cs, ralloc_memdup(blitter->blit_cs, &key, sizeof(key)),
         cs);
   }

   assert(cs != NULL);
   ctx->bind_compute_state(ctx, cs);

   struct pipe_grid_info grid_info = {
      .block = {TILE_WIDTH, TILE_HEIGHT, 1},
      .last_block = {width % TILE_WIDTH, height % TILE_HEIGHT, 1},
      .grid =
         {
            DIV_ROUND_UP(width, TILE_WIDTH),
            DIV_ROUND_UP(height, TILE_HEIGHT),
            depth,
         },
   };
   ctx->launch_grid(ctx, &grid_info);
   ctx->set_shader_images(ctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
   ctx->set_constant_buffer(ctx, MESA_SHADER_COMPUTE, 0, NULL);
   ctx->set_sampler_views(ctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);

   asahi_compute_restore(agx_context(ctx));
}

void
agx_blitter_save(struct agx_context *ctx, struct blitter_context *blitter,
                 enum asahi_blitter_op op)
{
   util_blitter_save_vertex_shader(blitter,
                                   ctx->stage[MESA_SHADER_VERTEX].shader);
   util_blitter_save_tessctrl_shader(blitter,
                                     ctx->stage[MESA_SHADER_TESS_CTRL].shader);
   util_blitter_save_tesseval_shader(blitter,
                                     ctx->stage[MESA_SHADER_TESS_EVAL].shader);
   util_blitter_save_geometry_shader(blitter,
                                     ctx->stage[MESA_SHADER_GEOMETRY].shader);
   util_blitter_save_so_targets(blitter, ctx->streamout.num_targets,
                                ctx->streamout.targets, MESA_PRIM_UNKNOWN);
   util_blitter_save_vertex_buffers(blitter, ctx->vertex_buffers,
                                    util_last_bit(ctx->vb_mask));
   util_blitter_save_vertex_elements(blitter, ctx->attributes);
   util_blitter_save_rasterizer(blitter, ctx->rast);
   util_blitter_save_viewport(blitter, &ctx->viewport[0]);

   if (op & ASAHI_SAVE_FRAGMENT_STATE) {
      util_blitter_save_blend(blitter, ctx->blend);
      util_blitter_save_depth_stencil_alpha(blitter, ctx->zs);
      util_blitter_save_stencil_ref(blitter, &ctx->stencil_ref);
      util_blitter_save_sample_mask(blitter, ctx->sample_mask, 0);

      util_blitter_save_scissor(blitter, &ctx->scissor[0]);
      util_blitter_save_fragment_shader(
         blitter, ctx->stage[MESA_SHADER_FRAGMENT].shader);

      if (op & ASAHI_SAVE_FRAGMENT_CONSTANT) {
         util_blitter_save_fragment_constant_buffer_slot(
            blitter, ctx->stage[MESA_SHADER_FRAGMENT].cb);
      }
   }

   if (op & ASAHI_SAVE_FRAMEBUFFER) {
      util_blitter_save_framebuffer(blitter, &ctx->framebuffer);
   }

   if (op & ASAHI_SAVE_TEXTURES) {
      util_blitter_save_fragment_sampler_states(
         blitter, ctx->stage[MESA_SHADER_FRAGMENT].sampler_count,
         (void **)(ctx->stage[MESA_SHADER_FRAGMENT].samplers));

      util_blitter_save_fragment_sampler_views(
         blitter, ctx->stage[MESA_SHADER_FRAGMENT].texture_count,
         (struct pipe_sampler_view **)ctx->stage[MESA_SHADER_FRAGMENT].textures);
   }

   /* Saving the condition is what lets u_blitter turn it off for the
    * operation and restore it afterwards. */
   if (op & ASAHI_DISABLE_RENDER_COND) {
      util_blitter_save_render_condition(blitter,
                                         (struct pipe_query *)ctx->cond_query,
                                         ctx->cond_cond, ctx->cond_mode);
   }
}

/* G17P stores a multisample colour tile by resolving it directly into a
 * matching single-sample PBE target at EOT. Attach exact full-resource API
 * resolves to the still-active source batch instead of first requesting the
 * unwitnessed multisample PBE store needed by a second render pass.
 *
 * DRI additionally seeds its private multisample window drawable from the WSI
 * image. Keep that reverse copy restricted to the original WSI binding shape;
 * its later swap is the operation that supplies the direct resolve target.
 */
static bool
agx_g17p_rgba8_resolve_format(enum pipe_format format)
{
   return format == PIPE_FORMAT_R8G8B8A8_UNORM ||
          format == PIPE_FORMAT_B8G8R8A8_UNORM ||
          format == PIPE_FORMAT_R8G8B8X8_UNORM ||
          format == PIPE_FORMAT_B8G8R8X8_UNORM;
}

bool
agx_g17p_direct_resolve_blit_shape(const struct pipe_blit_info *info,
                                   bool source_multisampled)
{
   struct pipe_resource *src = info->src.resource;
   struct pipe_resource *dst = info->dst.resource;
   if (!src || !dst)
      return false;

   struct pipe_resource *msaa = source_multisampled ? src : dst;
   struct pipe_resource *single = source_multisampled ? dst : src;
   const unsigned wsi_bind =
      PIPE_BIND_DISPLAY_TARGET | PIPE_BIND_SCANOUT | PIPE_BIND_SHARED;
   const bool wsi_seed = !source_multisampled &&
                         !(msaa->bind & wsi_bind) &&
                         (single->bind & wsi_bind);

   return (util_res_sample_count(msaa) == 2 ||
           util_res_sample_count(msaa) == 4) &&
          util_res_sample_count(single) == 1 && src->format == dst->format &&
          agx_g17p_rgba8_resolve_format(src->format) &&
          (source_multisampled || wsi_seed) &&
          (single->target == PIPE_TEXTURE_2D ||
           single->target == PIPE_TEXTURE_RECT) &&
          single->last_level == 0 && single->array_size == 1 &&
          single->depth0 == 1 &&
          !info->src.level && !info->dst.level && !info->src.box.x &&
          !info->src.box.y && !info->src.box.z && !info->dst.box.x &&
          !info->dst.box.y && !info->dst.box.z &&
          info->src.box.depth == 1 && info->dst.box.depth == 1 &&
          info->src.box.width == (int)src->width0 &&
          info->src.box.height == (int)src->height0 &&
          info->dst.box.width == (int)dst->width0 &&
          info->dst.box.height == (int)dst->height0 &&
          src->width0 == dst->width0 && src->height0 == dst->height0 &&
          info->src.format == info->dst.format &&
          info->src.format == src->format && info->mask == PIPE_MASK_RGBA &&
          info->filter == PIPE_TEX_FILTER_NEAREST && !info->scissor_enable &&
          !info->alpha_blend && !info->sample0_only &&
          !info->swizzle_enable && !info->num_window_rectangles;
}

static bool
agx_g17p_attach_direct_resolve(struct agx_context *ctx,
                               struct pipe_resource *msaa,
                               struct pipe_resource *single,
                               bool preserve_source)
{
   struct agx_batch *batch =
      agx_writer_get(ctx, agx_resource(msaa)->bo->handle);

   if (!batch || !agx_batch_is_active(batch) || !batch->initialized ||
       (batch->key.resolve && batch->key.resolve != single) ||
       batch->key.nr_cbufs != 1 ||
       batch->key.cbufs[0].texture != msaa ||
       batch->key.cbufs[0].level != 0 ||
       batch->key.cbufs[0].first_layer != 0 ||
       batch->key.cbufs[0].last_layer != 0)
      return false;

   batch->g17p_resolve_preserve |= preserve_source;
   if (!batch->key.resolve) {
      pipe_resource_reference(&batch->key.resolve, single);
      if (ctx->batch == batch)
         ctx->batch = NULL;
      agx_batch_writes(batch, agx_resource(single), 0);
   }
   return true;
}

bool
agx_g17p_staged_resolve_blit_shape(const struct pipe_blit_info *info)
{
   if (!info || !info->src.resource || !info->dst.resource)
      return false;

   const struct pipe_resource *src = info->src.resource;
   const struct pipe_resource *dst = info->dst.resource;
   return (util_res_sample_count(src) == 2 ||
           util_res_sample_count(src) == 4) &&
          util_res_sample_count(dst) == 1 &&
          (src->target == PIPE_TEXTURE_2D ||
           src->target == PIPE_TEXTURE_RECT) &&
          src->last_level == 0 && src->array_size == 1 && src->depth0 == 1 &&
          agx_g17p_rgba8_resolve_format(src->format) &&
          info->src.level == 0 && info->src.box.z == 0 &&
          info->src.box.depth == 1 && info->src.format == src->format &&
          !(info->mask & ~PIPE_MASK_RGBA) && (info->mask & PIPE_MASK_RGBA) &&
          !info->sample0_only;
}

/* A partial or converting resolve cannot write its final destination directly
 * at EOT, while storing the multisample image for a later shader pass has no
 * grounded G17P PBE encoding. Resolve the complete live tile into a temporary
 * single-sample image first; the ordinary render blitter can then apply the
 * caller's rectangle, mask, scale, swizzle, and destination format exactly. */
static bool
agx_g17p_staged_resolve_blit(struct agx_context *ctx,
                             const struct pipe_blit_info *info)
{
   struct pipe_resource *src = info->src.resource;
   struct pipe_resource templ = {
      .target = PIPE_TEXTURE_2D,
      .format = src->format,
      .width0 = src->width0,
      .height0 = src->height0,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 0,
      .nr_samples = 1,
      .nr_storage_samples = 1,
      .usage = PIPE_USAGE_DEFAULT,
      .bind = PIPE_BIND_RENDER_TARGET | PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *resolved =
      ctx->base.screen->resource_create(ctx->base.screen, &templ);
   if (!resolved)
      return false;

   if (!agx_g17p_attach_direct_resolve(ctx, src, resolved, true)) {
      pipe_resource_reference(&resolved, NULL);
      return false;
   }

   struct pipe_blit_info staged = *info;
   staged.src.resource = resolved;
   staged.src.level = 0;
   staged.src.format = src->format;
   staged.src.box.z = 0;
   staged.src.box.depth = 1;
   agx_blit(&ctx->base, &staged);
   pipe_resource_reference(&resolved, NULL);
   return true;
}

bool
agx_g17p_stencil_plane_blit_shape(const struct pipe_blit_info *info)
{
   if (!info || !info->src.resource || !info->dst.resource ||
       !(info->mask & PIPE_MASK_S) || (info->mask & ~PIPE_MASK_ZS))
      return false;

   const struct util_format_description *src =
      util_format_description(info->src.format);
   const struct util_format_description *dst =
      util_format_description(info->dst.format);

   return util_format_has_stencil(src) && util_format_has_stencil(dst) &&
          (!(info->mask & PIPE_MASK_Z) ||
           (util_format_has_depth(src) && util_format_has_depth(dst)));
}

void
agx_blit(struct pipe_context *pipe, const struct pipe_blit_info *info)
{
   struct agx_context *ctx = agx_context(pipe);

   /* G17P does not have a grounded fragment ABI for exporting stencil. That
    * is not a hardware-resource limitation: standalone stencil is native S8,
    * and u_transfer_helper represents every packed API Z/S image as a native
    * depth image plus a separate native S8 image.
    *
    * Lower the stencil aspect at that representation boundary. Depth keeps
    * using the depth blitter, whose G17P depthf sequence is Apple-witnessed.
    * Reinterpret each physical S8 image as R8_UINT and use the ordinary
    * integer-colour blitter, avoiding an unwitnessed stencilf contract.
    * R8_UINT and S8_UINT have the same one-byte AIL layout and integer
    * conversion, so this is a lossless plane copy, including scaling and
    * array-layer selection performed by u_blitter.
    */
   if (agx_chip_command_stream_generation(agx_device(pipe->screen)->chip) !=
          AGX_CS_GENERATION_AGX2 &&
       agx_g17p_stencil_plane_blit_shape(info)) {
      struct agx_resource *src = agx_resource(info->src.resource);
      struct agx_resource *dst = agx_resource(info->dst.resource);

      const bool src_has_depth = util_format_has_depth(
         util_format_description(info->src.format));
      const bool dst_has_depth = util_format_has_depth(
         util_format_description(info->dst.format));
      if ((!src_has_depth || src->separate_stencil) &&
          (!dst_has_depth || dst->separate_stencil)) {
         if (info->mask & PIPE_MASK_Z) {
            struct pipe_blit_info depth = *info;
            depth.src.format = src->layout.format;
            depth.dst.format = dst->layout.format;
            depth.mask = PIPE_MASK_Z;
            agx_blit(pipe, &depth);
         }

         struct pipe_blit_info stencil = *info;
         stencil.src.resource = src->separate_stencil
                                   ? &src->separate_stencil->base
                                   : info->src.resource;
         stencil.src.format = PIPE_FORMAT_R8_UINT;
         stencil.dst.resource = dst->separate_stencil
                                   ? &dst->separate_stencil->base
                                   : info->dst.resource;
         stencil.dst.format = PIPE_FORMAT_R8_UINT;
         stencil.mask = PIPE_MASK_R;
         /* Resolve stencil by selecting sample zero, but preserve every
          * sample when copying between multisample images. G17P sample
          * shading handles the latter through the shared static lowering.
          * Equal-sample-count G18 copies must preserve each source sample;
          * its render blitter supports the per-sample shading path.
          */
         if (agx_device(pipe->screen)->chip == AGX_CHIP_G18G) {
            const bool preserve_samples =
               util_res_sample_count(stencil.src.resource) ==
                  util_res_sample_count(stencil.dst.resource);
            stencil.sample0_only =
               info->src.resource->nr_samples > 1 && !preserve_samples;
         } else {
            stencil.sample0_only = info->src.resource->nr_samples > 1 &&
                                   info->dst.resource->nr_samples <= 1;
         }
         agx_blit(pipe, &stencil);
         return;
      }
   }

   if (agx_device(pipe->screen)->debug & AGX_DBG_FAULTTRACE) {
      fprintf(stderr,
              "M5_BLIT %s %ux%u L%u -> %s %ux%u L%u mask=%#x filter=%u "
              "scissor=%u samples=%u->%u\n",
              util_format_name(info->src.format), info->src.box.width,
              info->src.box.height, info->src.level,
              util_format_name(info->dst.format), info->dst.box.width,
              info->dst.box.height, info->dst.level, info->mask, info->filter,
              info->scissor_enable, info->src.resource->nr_samples,
              info->dst.resource->nr_samples);
   }


   if (info->render_condition_enable && !agx_render_condition_check(ctx))
      return;

   /* G17G: a blit that only copies bits (same view format, no scaling,
    * filtering, masking, scissor or swizzle) goes through the CPU copy that
    * resource_copy_region uses. This covers glCopyImageSubData between
    * different formats of one size, which the state tracker turns into a
    * reinterpreting blit; the draw path refuses most of those formats. */
   if (agx_chip_g17_m5(agx_device(pipe->screen)->chip) &&
       (info->src.format != info->src.resource->format ||
        info->dst.format != info->dst.resource->format) &&
       util_can_blit_via_copy_region(info, true, false)) {
      enum pipe_format sf = info->src.resource->format;
      enum pipe_format df = info->dst.resource->format;
      if (!util_format_is_compressed(sf) && !util_format_is_compressed(df) &&
          !util_format_is_depth_or_stencil(sf) &&
          !util_format_is_depth_or_stencil(df) &&
          info->src.resource->nr_samples <= 1 &&
          info->dst.resource->nr_samples <= 1 &&
          util_format_get_blocksize(sf) == util_format_get_blocksize(df) &&
          util_format_get_blocksize(sf) ==
             util_format_get_blocksize(info->src.format)) {
         util_resource_copy_region(pipe, info->dst.resource, info->dst.level,
                                   info->dst.box.x, info->dst.box.y,
                                   info->dst.box.z, info->src.resource,
                                   info->src.level, &info->src.box);
         return;
      }
   }

   /* Legalize compression /before/ calling into u_blitter to avoid recursion.
    * u_blitter bans recursive usage.
    */
   agx_legalize_compression(ctx, agx_resource(info->dst.resource),
                            info->dst.format);

   agx_legalize_compression(ctx, agx_resource(info->src.resource),
                            info->src.format);

   /* The current G18 fused EOT resolve replaces the multisample store. GL
    * requires the source samples to survive a resolve, including subsequent
    * texture fetches and resolves. Store them first and use the render blitter
    * to fetch and average every sample until EOT can preserve both outputs. */
   const bool g18_msaa4 =
      agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G &&
      agx_g18_msaa4_enabled();

   /* G17G resolves through u_blitter's sample fetches instead. Attaching
    * the resolve to the batch that drew the multisample image makes its EOT
    * store only the resolved target, so the samples themselves were never
    * written and the next pass reloaded stale memory (dEQP-GLES3
    * multisample.fbo_4_samples.stencil: samples a stencil test rejected
    * came back zero; gpu/msstencil.c in m5air). */
   if (agx_native_graphics(agx_device(ctx->base.screen)) &&
       agx_g17p_direct_resolve_blit_shape(info, true) &&
       agx_g17p_attach_direct_resolve(ctx, info->src.resource,
                                      info->dst.resource, true))
      return;

   if (agx_native_graphics(agx_device(ctx->base.screen)) &&
       agx_g17p_staged_resolve_blit_shape(info) &&
       agx_g17p_staged_resolve_blit(ctx, info))
      return;

   /* The render blitter fetches every stored sample and performs the format's
    * resolve in the shader. A non-layered framebuffer can select any valid
    * layer of the supported array storage; u_blitter carries that layer into
    * the source coordinates. Keep the single-layer extent and storage checks.
    */
   if (g18_msaa4 && util_res_sample_count(info->src.resource) > 1 &&
       (!agx_g18_msaa4_storage(agx_device(ctx->base.screen),
                                agx_resource(info->src.resource),
                                info->src.format) ||
        info->src.level != 0 || info->src.box.z < 0 ||
        (unsigned)info->src.box.z >= info->src.resource->array_size ||
        info->src.box.depth != 1)) {
      mesa_loge("asahi: G18G multisample blit source %s %ux%u has "
                "unsupported sample count, format or storage layout "
                "(level=%u layer=%d depth=%d samples=%u target=%u)",
                util_format_name(info->src.format), info->src.box.width,
                info->src.box.height, info->src.level, info->src.box.z,
                info->src.box.depth, util_res_sample_count(info->src.resource),
                info->src.resource->target);
      return;
   }

   if (asahi_compute_blit_supported(agx_device(ctx->base.screen)->chip,
                                    info)) {
      asahi_compute_blit(pipe, info, &ctx->compute_blitter);
      return;
   }

   if (!util_blitter_is_blit_supported(ctx->blitter, info)) {
      fprintf(stderr, "\n");
      util_dump_blit_info(stderr, info);
      fprintf(stderr, "\n\n");
      UNREACHABLE("Unsupported blit");
   }

   /* Handle self-blits */
   agx_flush_writer(ctx, agx_resource(info->dst.resource), "Blit");

   agx_blitter_save(
      ctx, ctx->blitter,
      ASAHI_BLIT |
         (info->render_condition_enable ? 0 : ASAHI_DISABLE_RENDER_COND));
   if (agx_device(pipe->screen)->debug & AGX_DBG_FAULTTRACE)
      fprintf(stderr, "M5_BLIT entering u_blitter\n");
   util_blitter_blit(ctx->blitter, info, NULL);

   /* The first DRI drawable validation seeds its private MSAA colourbuffer
    * from the WSI image. Keep that copy in the same tile batch, and publish
    * the tile directly back to the WSI image when the swap resolve arrives.
    */
   if (agx_native_graphics(agx_device(ctx->base.screen)) &&
       agx_g17p_direct_resolve_blit_shape(info, false)) {
      if (agx_g17p_attach_direct_resolve(ctx, info->dst.resource,
                                       info->src.resource, false)) {
         struct agx_batch *batch =
            agx_writer_get(ctx, agx_resource(info->dst.resource)->bo->handle);
         batch->g17p_resolve_reload = true;
      }
   }
}

static bool
try_copy_via_blit(struct pipe_context *pctx, struct pipe_resource *dst,
                  unsigned dst_level, unsigned dstx, unsigned dsty,
                  unsigned dstz, struct pipe_resource *src, unsigned src_level,
                  const struct pipe_box *src_box)
{
   struct agx_context *ctx = agx_context(pctx);

   if (dst->target == PIPE_BUFFER)
      return false;

   /* TODO: Handle these for rusticl copies */
   if (dst->target != src->target)
      return false;

   struct pipe_blit_info info = {
      .dst =
         {
            .resource = dst,
            .level = dst_level,
            .box.x = dstx,
            .box.y = dsty,
            .box.z = dstz,
            .box.width = src_box->width,
            .box.height = src_box->height,
            .box.depth = src_box->depth,
            .format = dst->format,
         },
      .src =
         {
            .resource = src,
            .level = src_level,
            .box = *src_box,
            .format = src->format,
         },
      .mask = util_format_get_mask(src->format),
      .filter = PIPE_TEX_FILTER_NEAREST,
      .scissor_enable = 0,
      .swizzle_enable = 0,
   };

   /* snorm formats don't round trip, so don't use them for copies */
   if (util_format_is_snorm(info.dst.format))
      info.dst.format = util_format_snorm_to_sint(info.dst.format);

   if (util_format_is_snorm(info.src.format))
      info.src.format = util_format_snorm_to_sint(info.src.format);

   if (util_blitter_is_blit_supported(ctx->blitter, &info) &&
       info.dst.format == info.src.format) {

      agx_blit(pctx, &info);
      return true;
   } else {
      return false;
   }
}

/* Copy between buffers on the GPU, in the context's submission order. */
void
agx_copy_buffer_range(struct pipe_context *pctx, struct pipe_resource *dst,
                      unsigned dst_offset, struct pipe_resource *src,
                      unsigned src_offset, unsigned size)
{
   struct agx_batch *batch = agx_get_compute_batch(agx_context(pctx));
   agx_batch_init_state(batch);

   uint64_t dst_addr = agx_map_gpu(agx_resource(dst)) + dst_offset;
   uint64_t src_addr = agx_map_gpu(agx_resource(src)) + src_offset;

   agx_batch_reads(batch, agx_resource(src));
   agx_batch_writes_range(batch, agx_resource(dst), dst_offset, size);

   /* Bring dst to 16 bytes so the bulk can use vectorized copies, which need
    * dst, src and size aligned. Failing to check for alignment causes subtle
    * and hard-to-debug issues!
    */
   unsigned head = MIN2((16 - (dst_addr & 0xf)) & 0xf, size);
   if (head && ((src_addr + head) & 0xf) == 0) {
      libagx_copy_uchar(batch, agx_1d(head), AGX_BARRIER_ALL, dst_addr,
                        src_addr);
      dst_addr += head;
      src_addr += head;
      size -= head;
   }

   if (size >= 16 && (dst_addr & 0xf) == 0 && (src_addr & 0xf) == 0) {
      unsigned uint4s = size / 16;
      unsigned bytes = uint4s * 16;

      libagx_copy_uint4(batch, agx_1d(uint4s), AGX_BARRIER_ALL, dst_addr,
                        src_addr);

      dst_addr += bytes;
      src_addr += bytes;
      size -= bytes;
   }

   if (size) {
      libagx_copy_uchar(batch, agx_1d(size), AGX_BARRIER_ALL, dst_addr,
                        src_addr);
   }
}

void
agx_resource_copy_region(struct pipe_context *pctx, struct pipe_resource *dst,
                         unsigned dst_level, unsigned dstx, unsigned dsty,
                         unsigned dstz, struct pipe_resource *src,
                         unsigned src_level, const struct pipe_box *src_box)
{
   if (agx_chip_g17_m5(agx_device(pctx->screen)->chip)) {
      /* Copy through transfer maps, which synchronize with the GPU: G17G
       * has no precompiled copy kernels (agx_launch_precomp refuses the
       * batch), and a copy blit's render target is mostly outside the
       * native PBE profile (mip levels, layers, most formats), so its batch
       * would be dropped while the blit reported success. */
      util_resource_copy_region(pctx, dst, dst_level, dstx, dsty, dstz, src,
                                src_level, src_box);
      return;
   }

   if (dst->target == PIPE_BUFFER && src->target == PIPE_BUFFER) {
      assert(dst->format == src->format);
      unsigned bs = util_format_get_blocksize(dst->format);

      agx_copy_buffer_range(pctx, dst, dstx * bs, src, src_box->x * bs,
                            bs * src_box->width);
      return;
   }

   if (try_copy_via_blit(pctx, dst, dst_level, dstx, dsty, dstz, src, src_level,
                         src_box))
      return;

   /* CPU fallback */
   util_resource_copy_region(pctx, dst, dst_level, dstx, dsty, dstz, src,
                             src_level, src_box);
}

/* G17G: generate mipmaps on the CPU where the GPU cannot. The blitter
 * renders into the levels above the base, and a render whose target the
 * native PBE profile refuses is dropped while the generation reports
 * success. Box-filter 2x2 (2x2x2 for 3D) in float through texture maps;
 * sRGB unpacks to linear and packs back. Returning false leaves compressed,
 * integer and depth formats to the generic path.
 *
 * The profile does take a level of a 2D texture (a one-level image at the
 * level's offset, agx_batch_upload_pbe_internal), and then the blitter is
 * the right tool: the CPU path waits for the GPU, reads the base level back
 * and uploads every level, 74 ms for a 3456x2160 RGBA8 texture on an M5 Pro
 * against 1 ms rendered. Such a texture goes to the generic path when every
 * level it would render into passes a dry run of that PBE.
 * ASAHI_M5_CPU_MIPMAP=1 keeps the CPU path, for comparison.
 */
bool
agx_generate_mipmap_g17g(struct pipe_context *pctx, struct pipe_resource *prsc,
                         enum pipe_format format, unsigned base_level,
                         unsigned last_level, unsigned first_layer,
                         unsigned last_layer)
{
   const struct util_format_unpack_description *unpack =
      util_format_unpack_description(format);
   const struct util_format_pack_description *pack =
      util_format_pack_description(format);
   if (util_format_is_compressed(format) ||
       util_format_is_pure_integer(format) ||
       util_format_is_depth_or_stencil(format) || prsc->nr_samples > 1 ||
       !unpack || !unpack->unpack_rgba || !pack || !pack->pack_rgba_float)
      return false;

   if (prsc->target == PIPE_TEXTURE_2D &&
       !debug_get_bool_option("ASAHI_M5_CPU_MIPMAP", false)) {
      bool renderable = true;
      for (unsigned level = base_level + 1;
           level <= last_level && renderable; ++level) {
         renderable = agx_g17g_level_renderable(agx_context(pctx), prsc,
                                                format, level, 0);
      }
      if (renderable)
         return false;
   }

   const bool is_3d = prsc->target == PIPE_TEXTURE_3D;
   const unsigned bs = util_format_get_blocksize(format);

   for (unsigned level = base_level + 1; level <= last_level; ++level) {
      const unsigned sw = u_minify(prsc->width0, level - 1);
      const unsigned sh = u_minify(prsc->height0, level - 1);
      const unsigned sd = is_3d ? u_minify(prsc->depth0, level - 1) : 1;
      const unsigned dw = u_minify(prsc->width0, level);
      const unsigned dh = u_minify(prsc->height0, level);
      const unsigned dd = is_3d ? u_minify(prsc->depth0, level) : 1;
      const unsigned layers = is_3d ? 1 : last_layer - first_layer + 1;

      for (unsigned l = 0; l < layers; ++l) {
         const unsigned z0 = is_3d ? 0 : first_layer + l;
         struct pipe_box sbox, dbox;
         u_box_3d(0, 0, z0, sw, sh, sd, &sbox);
         u_box_3d(0, 0, z0, dw, dh, dd, &dbox);
         struct pipe_transfer *src_xfer, *dst_xfer;
         const uint8_t *src = pctx->texture_map(
            pctx, prsc, level - 1, PIPE_MAP_READ, &sbox, &src_xfer);
         if (!src)
            return false;
         uint8_t *dst = pctx->texture_map(
            pctx, prsc, level, PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE,
            &dbox, &dst_xfer);
         if (!dst) {
            pctx->texture_unmap(pctx, src_xfer);
            return false;
         }

         for (unsigned z = 0; z < dd; ++z) {
            for (unsigned y = 0; y < dh; ++y) {
               for (unsigned x = 0; x < dw; ++x) {
                  float acc[4] = {0};
                  unsigned n = 0;
                  for (unsigned oz = 0; oz < (sd > 1 ? 2 : 1); ++oz) {
                     for (unsigned oy = 0; oy < (sh > 1 ? 2 : 1); ++oy) {
                        for (unsigned ox = 0; ox < (sw > 1 ? 2 : 1); ++ox) {
                           const unsigned sx = MIN2(2 * x + ox, sw - 1);
                           const unsigned sy = MIN2(2 * y + oy, sh - 1);
                           const unsigned sz = MIN2(2 * z + oz, sd - 1);
                           float px[4];
                           util_format_unpack_rgba(
                              format, px,
                              src + sz * src_xfer->layer_stride +
                                 sy * src_xfer->stride + sx * bs,
                              1);
                           for (unsigned c = 0; c < 4; ++c)
                              acc[c] += px[c];
                           n++;
                        }
                     }
                  }
                  for (unsigned c = 0; c < 4; ++c)
                     acc[c] /= n;
                  util_format_pack_rgba(format,
                                        dst + z * dst_xfer->layer_stride +
                                           y * dst_xfer->stride + x * bs,
                                        acc, 1);
               }
            }
         }

         pctx->texture_unmap(pctx, dst_xfer);
         pctx->texture_unmap(pctx, src_xfer);
      }
   }
   return true;
}
