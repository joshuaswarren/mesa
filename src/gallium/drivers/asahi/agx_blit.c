/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2020-2021 Collabora, Ltd.
 * Copyright 2019 Sonny Jiang <sonnyj608@gmail.com>
 * Copyright 2019 Advanced Micro Devices, Inc.
 * Copyright 2014 Broadcom
 * SPDX-License-Identifier: MIT
 */

#include <stdint.h>
#include "asahi/compiler/agx_nir_texture.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx3_compute_program_core.h"
#include "asahi/lib/agx_bg_eot_key.h"
#include "asahi/lib/agx_tile_mip.h"
#include "asahi/vulkan/hk_g17p_meta_copy_core.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "gallium/auxiliary/util/u_blitter.h"
#include "gallium/auxiliary/util/u_dump.h"
#include "gallium/auxiliary/util/u_gen_mipmap.h"
#include "nir/pipe_nir.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/hash_table.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_math.h"
#include "util/u_inlines.h"
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
      /* Restore what the app bound, not the batch a colourless pass was
       * merged into (agx_try_zs_merge), or later draws would write colour. */
      util_blitter_save_framebuffer(
         blitter, ctx->zs_merge ? &ctx->zs_merge_real : &ctx->framebuffer);
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
/* AGX3 end-of-tile IMGWBLK averages samples in hardware for every native
 * normalized and 16-bit floating-point colour selector (0x75, 0x95, 0x15),
 * matching Borealis's hk_agx3_format_needs_shader_resolve(). Keep integer
 * formats (sample-zero semantics), sRGB (linear averaging), 32-bit floats and
 * packed R11G11B10F/R9G9B9E5F words (raw-bit IMGWBLK) on the shader resolve
 * path. */
static bool
agx_g17p_direct_resolve_format(enum pipe_format format)
{
   if (util_format_is_depth_or_stencil(format) ||
       util_format_is_pure_integer(format) ||
       util_format_is_srgb(format) ||
       format == PIPE_FORMAT_R11G11B10_FLOAT ||
       format == PIPE_FORMAT_R9G9B9E5_FLOAT)
      return false;

   if (util_format_is_float(format) &&
       util_format_get_component_bits(format, UTIL_FORMAT_COLORSPACE_RGB, 0) == 32)
      return false;

   return agx_g17p_mrt_eot_selector(format) != 0;
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
          agx_g17p_direct_resolve_format(src->format) &&
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
       batch->key.cbufs[0].format != single->format ||
       batch->key.cbufs[0].level != 0 ||
       batch->key.cbufs[0].first_layer != 0 ||
       batch->key.cbufs[0].last_layer != 0 ||
       batch->key.width != msaa->width0 ||
       batch->key.height != msaa->height0 ||
       !agx_g17p_rt0_tile_layout_supported(&batch->tilebuffer_layout))
      return false;

   /* Do not fuse a resolve into a batch that already references the resolve
    * target (e.g. sampled during the render pass), because the fused EOT
    * store would race with reads in other tiles of the same pass. The WSI
    * seed path (preserve_source == false) intentionally reads the WSI image
    * once to seed the private MSAA buffer and resolves back into it. */
   if (!batch->key.resolve && preserve_source &&
       agx_batch_uses_bo(batch, agx_resource(single)->bo))
      return false;

   batch->g17p_resolve_preserve |= preserve_source;
   if (!batch->key.resolve) {
      pipe_resource_reference(&batch->key.resolve, single);
      batch->g17p_resolve_blit_attached = true;
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
          agx_g17p_direct_resolve_format(src->format) &&
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
                                   ? &src->separate_stencil->base.b
                                   : info->src.resource;
         stencil.src.format = PIPE_FORMAT_R8_UINT;
         stencil.dst.resource = dst->separate_stencil
                                   ? &dst->separate_stencil->base.b
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
      agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G;

   /* G17G resolves through u_blitter's sample fetches instead. Attaching
    * the resolve to the batch that drew the multisample image makes its EOT
    * store only the resolved target, so the samples themselves were never
    * written and the next pass reloaded stale memory (dEQP-GLES3
    * multisample.fbo_4_samples.stencil: samples a stencil test rejected
    * came back zero; gpu/msstencil.c in m5air). */
   const struct agx_device *dev = agx_device(ctx->base.screen);

   if (agx_chip_native_agx3(dev->chip) &&
       agx_g17p_direct_resolve_blit_shape(info, true) &&
       agx_g17p_attach_direct_resolve(ctx, info->src.resource,
                                      info->dst.resource, true))
      return;

   if (agx_chip_native_agx3(dev->chip) &&
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
   if (agx_chip_native_agx3(dev->chip) &&
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
 */
static bool
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

   if (prsc->target == PIPE_TEXTURE_2D) {
      bool renderable = true;
      for (unsigned level = base_level + 1;
           level <= last_level && renderable; ++level) {
         renderable = agx_level_renderable(agx_context(pctx), prsc,
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

static bool
agx_agx3_mipmap_format_supported(enum pipe_format format)
{
   switch (format) {
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
      return true;
   default:
      return false;
   }
}

bool
agx_generate_mipmap_agx3_supported(const struct agx_device *dev,
                                   const struct agx_resource *rsrc,
                                   enum pipe_format format,
                                   unsigned base_level, unsigned last_level,
                                   unsigned first_layer, unsigned last_layer)
{
   if (!dev || !rsrc || !agx_chip_native_agx3(dev->chip))
      return false;

   const struct pipe_resource *prsc = &rsrc->base.b;
   switch (prsc->target) {
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      break;
   default:
      return false;
   }

   if (prsc->nr_samples > 1 || rsrc->layout.sample_count_sa != 1 ||
       rsrc->layout.tiling != AIL_TILING_GPU || rsrc->layout.compressed ||
       rsrc->layout.format != format ||
       !agx_agx3_mipmap_format_supported(format))
      return false;

   if (base_level >= last_level || last_level > prsc->last_level ||
       last_level >= rsrc->layout.levels || first_layer > last_layer ||
       last_layer >= prsc->array_size || prsc->width0 > UINT16_MAX + 1u ||
       prsc->height0 > UINT16_MAX + 1u)
      return false;

   for (unsigned level = base_level; level <= last_level; ++level) {
      const struct ail_tile tile = rsrc->layout.tilesize_el[level];
      struct agx_tile_address_config cfg;
      if (!agx_tile_address_config_init(rsrc->layout.stride_el[level],
                                        tile.width_el, tile.height_el, &cfg))
         return false;
   }

   return true;
}

/* Ensure the current compute batch has room for one native AGX3 direct
 * compute dispatch record plus the stream terminate emitted by
 * agx_flush_compute(). Returns true if the batch was flushed and replaced. */
static bool
agx_agx3_mipmap_ensure_cdm_space(struct agx_context *ctx,
                                 struct agx_batch **batchp,
                                 struct agx_resource *rsrc, unsigned level,
                                 unsigned last_level)
{
   const size_t min_bytes =
      AGX3_G17P_NATIVE_COMPUTE_DISPATCH_SIZE + AGX3_CDM_STREAM_TERMINATE_SIZE;
   struct agx_batch *batch = *batchp;

   if ((size_t)(batch->cdm.end - batch->cdm.current) >= min_bytes)
      return false;

   agx_flush_batch(ctx, batch);
   batch = agx_get_compute_batch(ctx);
   agx_batch_init_state(batch);
   agx_batch_reads(batch, rsrc);
   for (unsigned l = level; l <= last_level; ++l)
      agx_batch_writes(batch, rsrc, l);
   *batchp = batch;
   return true;
}

/*
 * Compressed UNORM8x4 mip chains on HAL200 (G16G, G17P), the scheme Borealis
 * uses (hk_g17p_compressed_blit(), build_g17p_mip_shader()).
 *
 * util_gen_mipmap() would store every level of a compressed chain in its own
 * render pass. On classic-submission G16G a mip render pass in a chain costs
 * 0.13-0.24 ms, against compute at 0.13-0.24 ns per texel
 * (HK_CLASSIC_MIP_PBE_MIN_TEXELS), so a 2560x1440 chain would take about ten
 * times as long as it does uncompressed. Instead, one compute dispatch per
 * level reads the source level through the texture unit, which decodes
 * compressed tiles, writes the level's texels raw at their GPU-tiled
 * addresses, and marks each 16x16 metadata tile of the level uncompressed.
 * Levels smaller than a metadata tile have no metadata.
 *
 * Every dispatch of the compute blitter ends in the strong post-dispatch
 * flush (agx_g17p_append_compute()), which orders the next level's texture
 * reads after this level's stores, as each Borealis dispatch does.
 *
 * The filter is libagx_mipmap_unorm8x4{,_odd}'s: weights in 1/256 per axis,
 * rounded to four fractional bits, then to the nearest code with exact ties
 * toward 128.
 */
#define AGX_G17P_MIP_WG_SIZE 16

enum agx_g17p_mip_key {
   AGX_G17P_MIP_ODD = BITFIELD_BIT(0),
   AGX_G17P_MIP_EXACT_2X2 = BITFIELD_BIT(1),
   AGX_G17P_MIP_METADATA = BITFIELD_BIT(2),
   AGX_G17P_MIP_ALIGNED_16 = BITFIELD_BIT(3),
   AGX_G17P_MIP_TILE_64X64 = BITFIELD_BIT(4),
};

struct agx_g17p_mip_params {
   uint32_t dst_lo, dst_hi;
   uint32_t metadata_lo, metadata_hi;
   uint32_t axis_lo, axis_hi;
   uint32_t width;
   uint32_t height;
   uint32_t scale_x;
   uint32_t scale_y;
   uint32_t tiles_per_row;
   uint32_t tile_width_log2;
   uint32_t tile_height_log2;
   uint32_t metadata_interleave_bits;
   uint32_t metadata_major_is_x;
   uint32_t uncompressed_mode;
};
static_assert(sizeof(struct agx_g17p_mip_params) == 64,
              "G17P mip parameters are tightly packed");

#define mip_param(b, field)                                                    \
   nir_load_ubo(b, 1, 32, nir_imm_int(b, 0),                                   \
                nir_imm_int(b, offsetof(struct agx_g17p_mip_params, field)),   \
                .align_mul = 4, .range = ~0)

#define mip_param64(b, field)                                                  \
   nir_pack_64_2x32_split(b, mip_param(b, field##_lo), mip_param(b, field##_hi))

/* x with a zero bit inserted above each of its low 16 bits */
static nir_def *
agx_nir_spread_bits(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 8)), 0x00ff00ff);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 4)), 0x0f0f0f0f);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 2)), 0x33333333);
   return nir_iand_imm(b, nir_ior(b, v, nir_ishl_imm(b, v, 1)), 0x55555555);
}

/* The even bits of the low 8 bits of v, packed: inverse of the spread */
static nir_def *
agx_nir_compact_bits_8(nir_builder *b, nir_def *v)
{
   v = nir_iand_imm(b, v, 0x55);
   v = nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 1)), 0x33);
   return nir_iand_imm(b, nir_ior(b, v, nir_ushr_imm(b, v, 2)), 0x0f);
}

static void *
agx_g17p_mip_compute_shader(struct pipe_context *pctx, unsigned key)
{
   const bool odd = key & AGX_G17P_MIP_ODD;
   const bool exact_2x2 = key & AGX_G17P_MIP_EXACT_2X2;
   const bool metadata = key & AGX_G17P_MIP_METADATA;
   const bool aligned_16 = key & AGX_G17P_MIP_ALIGNED_16;
   const bool tile_64x64 = key & AGX_G17P_MIP_TILE_64X64;
   assert(!(odd && exact_2x2));

   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, pctx->screen->nir_options[MESA_SHADER_COMPUTE],
      "agx-g17p-mip");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = AGX_G17P_MIP_WG_SIZE;
   b->shader->info.workgroup_size[1] = AGX_G17P_MIP_WG_SIZE;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.num_ubos = 1;
   b->shader->info.num_textures = 1;
   BITSET_SET(b->shader->info.textures_used, 0);
   BITSET_SET(b->shader->info.textures_used_by_txf, 0);

   /* Lay the 16x16 invocations of a workgroup out in Morton order, matching
    * both one 16x16 compression metadata tile and the GPU-tiled texel layout:
    * each 32-invocation subgroup covers an 8x4 block (128 contiguous bytes).
    */
   nir_def *lid = nir_load_local_invocation_id(b);
   nir_def *index = nir_ior(b, nir_channel(b, lid, 0),
                            nir_ishl_imm(b, nir_channel(b, lid, 1), 4));
   nir_def *wg = nir_load_workgroup_id(b);
   nir_def *wg_x = nir_channel(b, wg, 0);
   nir_def *wg_y = nir_channel(b, wg, 1);
   nir_def *x =
      nir_ior(b, nir_ishl_imm(b, wg_x, 4), agx_nir_compact_bits_8(b, index));
   nir_def *y = nir_ior(b, nir_ishl_imm(b, wg_y, 4),
                        agx_nir_compact_bits_8(b, nir_ushr_imm(b, index, 1)));
   nir_def *width = (!aligned_16 || odd) ? mip_param(b, width) : NULL;
   nir_def *height = !aligned_16 ? mip_param(b, height) : NULL;

   nir_if *x_if = NULL, *y_if = NULL;
   if (!aligned_16) {
      x_if = nir_push_if(b, nir_ult(b, x, width));
      x_if->control = nir_selection_control_dont_flatten;
      y_if = nir_push_if(b, nir_ult(b, y, height));
      y_if->control = nir_selection_control_dont_flatten;
   }

   {
      nir_def *x0, *y0, *x1, *y1;
      nir_def *weights[4] = {NULL};

      if (exact_2x2) {
         x0 = nir_ishl_imm(b, x, 1);
         y0 = nir_ishl_imm(b, y, 1);
         x1 = nir_ior_imm(b, x0, 1);
         y1 = nir_ior_imm(b, y0, 1);
      } else {
         nir_def *wx, *wy;
         if (odd) {
            /* agx_build_mip_axis() entries: lower | weight << 16 */
            nir_def *axis = mip_param64(b, axis);
            nir_def *x_entry = nir_load_global(
               b, 1, 32, nir_iadd(b, axis, nir_u2u64(b, nir_imul_imm(b, x, 4))),
               .align_mul = 4);
            nir_def *y_entry = nir_load_global(
               b, 1, 32,
               nir_iadd(b, axis,
                        nir_u2u64(b, nir_imul_imm(b, nir_iadd(b, width, y), 4))),
               .align_mul = 4);
            x0 = nir_iand_imm(b, x_entry, 0xffff);
            y0 = nir_iand_imm(b, y_entry, 0xffff);
            wx = nir_iand_imm(b, nir_ushr_imm(b, x_entry, 16), 0xff);
            wy = nir_iand_imm(b, nir_ushr_imm(b, y_entry, 16), 0xff);
         } else {
            nir_def *scale_x = mip_param(b, scale_x);
            nir_def *scale_y = mip_param(b, scale_y);
            x0 = nir_imul(b, x, scale_x);
            y0 = nir_imul(b, y, scale_y);
            wx = nir_ishl_imm(b, nir_iadd_imm(b, scale_x, -1), 7);
            wy = nir_ishl_imm(b, nir_iadd_imm(b, scale_y, -1), 7);
         }

         x1 = nir_iadd(b, x0, nir_b2i32(b, nir_ine_imm(b, wx, 0)));
         y1 = nir_iadd(b, y0, nir_b2i32(b, nir_ine_imm(b, wy, 0)));
         nir_def *ix = nir_isub(b, nir_imm_int(b, 256), wx);
         nir_def *iy = nir_isub(b, nir_imm_int(b, 256), wy);

         weights[0] = nir_imul(b, ix, iy);
         weights[1] = nir_imul(b, wx, iy);
         weights[2] = nir_imul(b, ix, wy);
         weights[3] = nir_imul(b, wx, wy);
      }

      nir_def *coords[4] = {
         nir_vec2(b, x0, y0),
         nir_vec2(b, x1, y0),
         nir_vec2(b, x0, y1),
         nir_vec2(b, x1, y1),
      };
      nir_def *texels[4];
      for (unsigned i = 0; i < 4; ++i) {
         texels[i] =
            nir_txf(b, coords[i], .texture_index = 0,
                    .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_uint32,
                    .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
      }

      nir_def *packed = NULL;
      for (unsigned c = 0; c < 4; ++c) {
         nir_def *out;
         if (exact_2x2) {
            nir_def *sum = nir_iadd(
               b,
               nir_iadd(b, nir_channel(b, texels[0], c),
                        nir_channel(b, texels[1], c)),
               nir_iadd(b, nir_channel(b, texels[2], c),
                        nir_channel(b, texels[3], c)));
            /* The general rounding below when all four weights are 16384
             * (sum in 0..1020): sum <= 511 rounds ties up, sum >= 512 rounds
             * ties down, both toward 128. */
            out = nir_ushr_imm(
               b, nir_isub(b, nir_iadd_imm(b, sum, 2), nir_ushr_imm(b, sum, 9)),
               2);
         } else {
            nir_def *value = NULL;
            for (unsigned i = 0; i < 4; ++i) {
               nir_def *term =
                  nir_imul(b, nir_channel(b, texels[i], c), weights[i]);
               value = value ? nir_iadd(b, value, term) : term;
            }

            nir_def *fixed4 = nir_ushr_imm(b, nir_iadd_imm(b, value, 2048), 12);
            out = nir_ushr_imm(b, fixed4, 4);
            nir_def *rem = nir_iand_imm(b, fixed4, 15);
            nir_def *up = nir_ior(b, nir_ugt_imm(b, rem, 8),
                                  nir_iand(b, nir_ieq_imm(b, rem, 8),
                                           nir_ult_imm(b, out, 128)));
            out = nir_iadd(b, out, nir_b2i32(b, up));
         }

         nir_def *byte = nir_ishl_imm(b, out, 8 * c);
         packed = packed ? nir_ior(b, packed, byte) : byte;
      }

      nir_def *element;
      if (tile_64x64) {
         /* A 16x16 workgroup is already Morton-ordered in index (bits 0..7);
          * its position inside a 64x64 tile supplies Morton bits 8..11. */
         nir_def *tile = nir_iadd(
            b, nir_imul(b, nir_ushr_imm(b, wg_y, 2), mip_param(b, tiles_per_row)),
            nir_ushr_imm(b, wg_x, 2));
         nir_def *morton_hi = nir_ior(
            b,
            nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, wg_x, 1), 8),
                    nir_ishl_imm(b, nir_iand_imm(b, wg_y, 1), 9)),
            nir_ior(b, nir_ishl_imm(b, nir_iand_imm(b, wg_x, 2), 9),
                    nir_ishl_imm(b, nir_iand_imm(b, wg_y, 2), 10)));
         element = nir_ior(b, nir_ior(b, nir_ishl_imm(b, tile, 12), morton_hi),
                           index);
      } else {
         nir_def *tile_width_log2 = mip_param(b, tile_width_log2);
         nir_def *tile_height_log2 = mip_param(b, tile_height_log2);
         nir_def *tile_x_mask =
            nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), tile_width_log2), -1);
         nir_def *tile_y_mask =
            nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), tile_height_log2), -1);
         nir_def *tile = nir_iadd(
            b,
            nir_imul(b, nir_ushr(b, y, tile_height_log2),
                     mip_param(b, tiles_per_row)),
            nir_ushr(b, x, tile_width_log2));
         nir_def *morton = nir_ior(
            b, agx_nir_spread_bits(b, nir_iand(b, x, tile_x_mask)),
            nir_ishl_imm(b, agx_nir_spread_bits(b, nir_iand(b, y, tile_y_mask)),
                         1));
         element = nir_ior(
            b, nir_ishl(b, tile, nir_iadd(b, tile_width_log2, tile_height_log2)),
            morton);
      }

      nir_store_global(b, packed,
                       nir_iadd(b, mip_param64(b, dst),
                                nir_u2u64(b, nir_imul_imm(b, element, 4))),
                       .align_mul = 4);
   }

   if (!aligned_16) {
      nir_pop_if(b, y_if);
      nir_pop_if(b, x_if);
   }

   if (metadata) {
      /* Each 16x16 workgroup is one metadata tile (wg_x, wg_y), indexed as
       * libagx's index_metadata() does; one invocation stores it. */
      nir_if *meta_if = nir_push_if(b, nir_ieq_imm(b, index, 0));
      meta_if->control = nir_selection_control_dont_flatten;
      {
         nir_def *bits = mip_param(b, metadata_interleave_bits);
         nir_def *mask =
            nir_iadd_imm(b, nir_ishl(b, nir_imm_int(b, 1), bits), -1);
         nir_def *major = nir_bcsel(
            b, nir_ine_imm(b, mip_param(b, metadata_major_is_x), 0), wg_x, wg_y);
         nir_def *meta_idx = nir_iadd(
            b,
            nir_ior(b, agx_nir_spread_bits(b, nir_iand(b, wg_x, mask)),
                    nir_ishl_imm(
                       b, agx_nir_spread_bits(b, nir_iand(b, wg_y, mask)), 1)),
            nir_ishl(b, nir_iand(b, major, nir_inot(b, mask)), bits));
         nir_def *mode = mip_param(b, uncompressed_mode);

         nir_store_global(b, nir_vec2(b, mode, mode),
                          nir_iadd(b, mip_param64(b, metadata),
                                   nir_u2u64(b, nir_imul_imm(b, meta_idx, 8))),
                          .align_mul = 8);
      }
      nir_pop_if(b, meta_if);
   }

   return pipe_shader_from_nir(pctx, b->shader);
}

#define AGX_G17P_BLOCK_MIP_TILE 32

/* Exact 2x downsample into a compressed level of a HAL200/AGX3B image through
 * the compressed PBE, matching Borealis's build_g17p_block_mip_shader(). Each
 * invocation samples the centre of its 2x2 source block at unnormalized pixel
 * coordinates (2x + 1, 2y + 1), writes the pixel to the workgroup's
 * imageblock, and one invocation per 32x32 workgroup stores the compressed
 * tile through the level's compressed storage PBE. */
static void *
agx_g17p_block_mip_compute_shader(struct pipe_context *pctx)
{
   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, pctx->screen->nir_options[MESA_SHADER_COMPUTE],
      "agx-g17p-block-mip");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = AGX_G17P_BLOCK_MIP_TILE;
   b->shader->info.workgroup_size[1] = AGX_G17P_BLOCK_MIP_TILE;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.num_textures = 1;
   b->shader->info.num_images = 1;
   BITSET_SET(b->shader->info.textures_used, 0);
   BITSET_SET(b->shader->info.samplers_used, 0);
   BITSET_SET(b->shader->info.images_used, 0);

   nir_def *gid = nir_trim_vector(b, nir_load_global_invocation_id(b, 32), 2);
   nir_def *lid = nir_trim_vector(b, nir_load_local_invocation_id(b), 2);
   nir_def *u = nir_u2f32(
      b, nir_ior_imm(b, nir_ishl_imm(b, nir_channel(b, gid, 0), 1), 1));
   nir_def *v = nir_u2f32(
      b, nir_ior_imm(b, nir_ishl_imm(b, nir_channel(b, gid, 1), 1), 1));
   nir_def *texel =
      nir_tex(b, nir_vec2(b, u, v), .lod = nir_imm_float(b, 0.0),
              .texture_index = 0, .sampler_index = 0,
              .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP,
              .dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_float32);

   const enum pipe_format isa_format =
      ail_pixel_format[PIPE_FORMAT_R8G8B8A8_UNORM].renderable;
   nir_store_local_pixel_agx(b, texel, nir_imm_intN_t(b, 1, 16),
                             nir_u2u16(b, lid), .base = 0, .write_mask = 0xf,
                             .format = isa_format, .explicit_coord = true);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP);

   nir_push_if(b, nir_ball(b, nir_ieq_imm(b, lid, 0)));
   {
      nir_image_store_block_agx(b, nir_imm_int(b, 0), nir_imm_int(b, 0), gid,
                                .format = isa_format,
                                .image_dim = GLSL_SAMPLER_DIM_2D,
                                .image_array = false, .explicit_coord = true);
   }
   nir_pop_if(b, NULL);
   b->shader->info.cs.image_block_size_per_thread_agx =
      util_format_get_blocksize(PIPE_FORMAT_R8G8B8A8_UNORM);

   return pipe_shader_from_nir(pctx, b->shader);
}

/* Private 2D UNORM8x4 mip chains that the compute path below generates. Like
 * Borealis on images that are not attachments, keep chains whose top level
 * is below the size where a compressed level pays for its render-pass upload
 * plain (hk_mip_level_through_pbe()); GL cannot tell which textures will be
 * rendered to. */
bool
agx_g17p_compressed_mip_chain_allowed(const struct agx_device *dev,
                                      const struct pipe_resource *templ)
{
   return agx_chip_gen9_color_compression(dev->chip) &&
          templ->target == PIPE_TEXTURE_2D && templ->depth0 == 1 &&
          templ->array_size <= 1 && templ->nr_samples <= 1 &&
          templ->last_level > 0 &&
          agx_agx3_mipmap_format_supported(templ->format) &&
          templ->width0 <= UINT16_MAX + 1u &&
          templ->height0 <= UINT16_MAX + 1u &&
          hk_mip_level_through_pbe(dev->chip, templ->width0, templ->height0);
}

struct agx_g17p_mip_level_plan {
   struct agx_g17p_mip_params params;
   unsigned key;
   uint32_t src_w, src_h;
   uint32_t groups_x, groups_y;
};

static void
agx_generate_mipmap_agx3_levels(struct agx_context *ctx,
                                struct agx_batch **batchp,
                                struct agx_resource *rsrc,
                                unsigned start_level, unsigned last_level,
                                unsigned first_layer, unsigned last_layer,
                                bool has_2level);

static bool
agx_g17p_can_precompiled_tail(const struct agx_device *dev,
                              struct agx_context *ctx,
                              const struct agx_resource *rsrc,
                              unsigned tail_start, unsigned last_level,
                              uint64_t *out_meta_start, uint64_t *out_meta_end)
{
   *out_meta_start = 0;
   *out_meta_end = 0;

   if (!agx_chip_native_agx3(dev->chip))
      return false;

   if (!agx_get_precompiled(&ctx->bg_eot, LIBAGX_MIPMAP_UNORM8X4) ||
       !agx_get_precompiled(&ctx->bg_eot, LIBAGX_MIPMAP_UNORM8X4_ODD))
      return false;

   const struct ail_layout *layout = &rsrc->layout;
   for (unsigned level = tail_start - 1; level <= last_level; ++level) {
      const struct ail_tile tile = layout->tilesize_el[level];
      struct agx_tile_address_config cfg;
      if (!agx_tile_address_config_init(layout->stride_el[level],
                                        tile.width_el, tile.height_el, &cfg))
         return false;
   }

   unsigned last_comp = 0;
   bool has_meta = false;
   for (unsigned level = tail_start; level <= last_level; ++level) {
      if (ail_is_level_logically_compressed(layout, level)) {
         if (!ail_is_level_allocated_compressed(layout, level))
            return false;
         has_meta = true;
         last_comp = level;
      }
   }

   if (has_meta) {
      if (!agx_get_precompiled(&ctx->bg_eot, LIBAGX_FILL_UINT4) ||
          !ail_is_level_allocated_compressed(layout, tail_start))
         return false;

      const uint64_t meta_start = layout->level_offsets_compressed_B[tail_start];
      const uint64_t meta_end =
         (last_comp + 1 < layout->levels &&
          ail_is_level_allocated_compressed(layout, last_comp + 1))
            ? layout->level_offsets_compressed_B[last_comp + 1]
            : layout->compression_layer_stride_B;
      if ((meta_start & 15u) || (meta_end & 15u) || meta_end <= meta_start)
         return false;

      *out_meta_start = meta_start;
      *out_meta_end = meta_end;
   }

   return true;
}

static bool
agx_g17p_plan_mip_level(const struct agx_resource *rsrc, unsigned level,
                        struct agx_g17p_mip_level_plan *plan)
{
   const struct pipe_resource *prsc = &rsrc->base.b;
   const struct ail_layout *layout = &rsrc->layout;
   const uint32_t src_w = u_minify(prsc->width0, level - 1);
   const uint32_t src_h = u_minify(prsc->height0, level - 1);
   const uint32_t width = u_minify(prsc->width0, level);
   const uint32_t height = u_minify(prsc->height0, level);

   for (unsigned axis = 0; axis < 2; ++axis) {
      const uint64_t src = axis ? src_h : src_w;
      const uint64_t dst = axis ? height : width;
      if (!(src == dst || src == dst * 2 || src == dst * 2 + 1))
         return false;
   }

   const struct ail_tile tile = layout->tilesize_el[level];
   struct agx_tile_address_config tiles;
   if (!agx_tile_address_config_init(layout->stride_el[level],
                                          tile.width_el, tile.height_el,
                                          &tiles))
      return false;

   const bool odd = src_w == (uint64_t)width * 2 + 1 ||
                    src_h == (uint64_t)height * 2 + 1;
   const bool exact_2x2 = !odd && src_w == (uint64_t)width * 2 &&
                          src_h == (uint64_t)height * 2;
   const bool metadata = ail_is_level_logically_compressed(layout, level);
   const uint32_t groups_x = DIV_ROUND_UP(width, AGX_G17P_MIP_WG_SIZE);
   const uint32_t groups_y = DIV_ROUND_UP(height, AGX_G17P_MIP_WG_SIZE);

   *plan = (struct agx_g17p_mip_level_plan){
      .key = (odd ? AGX_G17P_MIP_ODD : 0) |
             (exact_2x2 ? AGX_G17P_MIP_EXACT_2X2 : 0) |
             (metadata ? AGX_G17P_MIP_METADATA : 0) |
             (groups_x * AGX_G17P_MIP_WG_SIZE == width &&
                    groups_y * AGX_G17P_MIP_WG_SIZE == height
                 ? AGX_G17P_MIP_ALIGNED_16
                 : 0) |
             (tiles.tile_width_log2 == 6 && tiles.tile_height_log2 == 6
                 ? AGX_G17P_MIP_TILE_64X64
                 : 0),
      .src_w = src_w,
      .src_h = src_h,
      .groups_x = groups_x,
      .groups_y = groups_y,
   };

   const uint64_t base = agx_map_gpu((struct agx_resource *)rsrc);
   const uint64_t dst = base + ail_get_layer_level_B(layout, 0, level);
   plan->params = (struct agx_g17p_mip_params){
      .dst_lo = (uint32_t)dst,
      .dst_hi = (uint32_t)(dst >> 32),
      .width = width,
      .height = height,
      .scale_x = src_w == width ? 1 : 2,
      .scale_y = src_h == height ? 1 : 2,
      .tiles_per_row = tiles.tiles_per_row,
      .tile_width_log2 = tiles.tile_width_log2,
      .tile_height_log2 = tiles.tile_height_log2,
   };

   if (metadata) {
      /* One workgroup per metadata tile; the metadata grid must be the
       * level's 16x16-texel grid. */
      const uint32_t width_tl =
         ail_metadata_width_tl((struct ail_layout *)layout, level);
      const uint32_t height_tl =
         ail_metadata_height_tl((struct ail_layout *)layout, level);
      if (width_tl != groups_x || height_tl != groups_y)
         return false;

      const uint64_t meta = base + layout->metadata_offset_B +
                            layout->level_offsets_compressed_B[level];
      plan->params.metadata_lo = (uint32_t)meta;
      plan->params.metadata_hi = (uint32_t)(meta >> 32);
      plan->params.metadata_interleave_bits =
         util_logbase2_ceil(MIN2(width_tl, height_tl));
      plan->params.metadata_major_is_x = width_tl > height_tl;
      plan->params.uncompressed_mode =
         (uint32_t)ail_tile_mode_uncompressed(layout->format);
   }

   return true;
}

static bool
agx_generate_mipmap_sampled(struct pipe_context *pctx,
                            struct agx_resource *rsrc,
                            enum pipe_format format, unsigned base_level,
                            unsigned last_level, unsigned first_layer,
                            unsigned last_layer)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   const struct pipe_resource *prsc = &rsrc->base.b;
   const struct ail_layout *layout = &rsrc->layout;

   /* The shader samples the source and stores Morton-ordered texels directly.
    * Only the Gen9 compressed layout has a metadata writer here. Gen10 plain
    * tiles need no metadata and use the same address calculation.
    */
   const bool supported_layout =
      layout->compressed ? agx_chip_gen9_color_compression(dev->chip) &&
                           layout->policy != AIL_LAYOUT_POLICY_M5
                         : agx_chip_descriptor_generation(dev->chip) ==
                              AGX_DESCRIPTOR_GENERATION_10;
   if (!supported_layout || prsc->width0 > UINT16_MAX + 1u ||
       prsc->height0 > UINT16_MAX + 1u ||
       prsc->target != PIPE_TEXTURE_2D || prsc->nr_samples > 1 ||
       layout->sample_count_sa != 1 || layout->tiling != AIL_TILING_GPU ||
       layout->format != format || !agx_agx3_mipmap_format_supported(format) ||
       first_layer != 0 || last_layer != 0 || base_level >= last_level ||
       last_level > prsc->last_level || last_level >= layout->levels ||
       last_level - base_level > PIPE_MAX_TEXTURE_LEVELS)
      return false;

   /* Large levels amortize a render pass and benefit from the PBE storing
    * compressed tiles for the next level. On HAL200/AGX3B, exact 2x2
    * compressed levels use the compressed compute block-mip path instead,
    * keeping all levels compressed in one compute batch without render-pass
    * or cross-queue handoff overhead. */
   const unsigned first_src = base_level;
   const bool block_mip_supported =
      layout->compressed && hk_g17p_block_mip_supported(dev->chip);
   while ((prsc->bind & PIPE_BIND_RENDER_TARGET) && base_level < last_level) {
      const unsigned w = u_minify(prsc->width0, base_level);
      const unsigned h = u_minify(prsc->height0, base_level);
      if ((w | h) & 1 ||
          !(block_mip_supported
               ? hk_compressed_mip_level_through_pbe(dev->chip, w, h, w / 2,
                                                     h / 2)
               : hk_mip_level_through_pbe(dev->chip, w / 2, h / 2)) ||
          !agx_level_renderable(ctx, &rsrc->base.b, format, base_level + 1, 0))
         break;
      ++base_level;
   }

   /* Leading exact 2x2 compressed levels stay compressed via the compute
    * block-mip shader; the first remaining level (if any) decodes the last
    * compressed level into raw GPU-tiled storage for the precompiled tail. */
   unsigned block_last = base_level;
   if (block_mip_supported) {
      while (block_last < last_level) {
         const unsigned level = block_last + 1;
         const unsigned sw = u_minify(prsc->width0, level - 1);
         const unsigned sh = u_minify(prsc->height0, level - 1);
         const unsigned dw = u_minify(prsc->width0, level);
         const unsigned dh = u_minify(prsc->height0, level);
         if (sw != dw * 2u || sh != dh * 2u ||
             !ail_is_level_logically_compressed(layout, level))
            break;
         block_last = level;
      }
   }

   if (block_last > base_level) {
      if (!ctx->compute_blitter.g17p_block_mip_cs) {
         ctx->compute_blitter.g17p_block_mip_cs =
            agx_g17p_block_mip_compute_shader(pctx);
         if (!ctx->compute_blitter.g17p_block_mip_cs)
            return false;
      }
      if (!ctx->compute_blitter.g17p_block_mip_sampler) {
         struct pipe_sampler_state sampler_state = {
            .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
            .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
            .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
            .min_img_filter = PIPE_TEX_FILTER_LINEAR,
            .mag_img_filter = PIPE_TEX_FILTER_LINEAR,
            .min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST,
            .compare_func = PIPE_FUNC_ALWAYS,
            .seamless_cube_map = true,
            .unnormalized_coords = true,
         };
         ctx->compute_blitter.g17p_block_mip_sampler =
            pctx->create_sampler_state(pctx, &sampler_state);
         if (!ctx->compute_blitter.g17p_block_mip_sampler)
            return false;
      }
   }

   const unsigned sampled_start = block_last + 1;

   /* Level sampled_start - 1 may hold compressed tiles and is decoded through
    * the sampled compute shader into raw GPU-tiled level sampled_start. Every
    * subsequent level reads that uncompressed level and can run directly via
    * precompiled libagx_mipmap_unorm8x4{,_2level,_odd} (plus one contiguous
    * libagx_fill_uint4 to mark remaining compressed metadata tiles
    * uncompressed), avoiding per-level sampler-view and launch_grid overhead. */
   uint64_t meta_start = 0, meta_end = 0;
   const bool precompiled_tail =
      sampled_start < last_level &&
      agx_g17p_can_precompiled_tail(dev, ctx, rsrc, sampled_start + 1,
                                    last_level, &meta_start, &meta_end);
   const unsigned sampled_last =
      precompiled_tail ? sampled_start : last_level;

   /* Plan every sampled level before recording anything: an unsupported level
    * makes the whole request fall back to util_gen_mipmap(). */
   struct agx_g17p_mip_level_plan plans[PIPE_MAX_TEXTURE_LEVELS];
   for (unsigned level = sampled_start; level <= sampled_last; ++level) {
      struct agx_g17p_mip_level_plan *plan = &plans[level - sampled_start];
      if (!agx_g17p_plan_mip_level(rsrc, level, plan))
         return false;

      if (!ctx->compute_blitter.g17p_mip_cs[plan->key]) {
         ctx->compute_blitter.g17p_mip_cs[plan->key] =
            agx_g17p_mip_compute_shader(pctx, plan->key);
         if (!ctx->compute_blitter.g17p_mip_cs[plan->key])
            return false;
      }
   }

   if (base_level > first_src) {
      if (!util_gen_mipmap(pctx, &rsrc->base.b, format, first_src, base_level,
                           first_layer, last_layer, PIPE_TEX_FILTER_LINEAR))
         return false;
      if (base_level == last_level)
         return true;
   }

   asahi_compute_save(ctx);

   /* glGenerateMipmap runs no application compute shader. */
   struct agx_query *cs_invocations =
      ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS];
   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = NULL;

   struct agx_batch *batch = agx_get_compute_batch(ctx);
   agx_batch_init_state(batch);
   for (unsigned level = base_level + 1; level <= last_level; ++level)
      agx_batch_writes(batch, rsrc, level);

   void *bound_cs = NULL;
   if (block_last > base_level) {
      pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0, 1,
                                &ctx->compute_blitter.g17p_block_mip_sampler);
      pctx->bind_compute_state(pctx, ctx->compute_blitter.g17p_block_mip_cs);
      bound_cs = ctx->compute_blitter.g17p_block_mip_cs;

      for (unsigned level = base_level + 1; level <= block_last; ++level) {
         if (unlikely(ctx->batch != batch)) {
            batch = agx_get_compute_batch(ctx);
            agx_batch_init_state(batch);
            for (unsigned l = level; l <= last_level; ++l)
               agx_batch_writes(batch, rsrc, l);
         }

         struct pipe_sampler_view templ;
         u_sampler_view_default_template(&templ, &rsrc->base.b,
                                         PIPE_FORMAT_R8G8B8A8_UNORM);
         templ.target = PIPE_TEXTURE_2D;
         templ.u.tex.first_level = level - 1;
         templ.u.tex.last_level = level - 1;
         templ.u.tex.first_layer = 0;
         templ.u.tex.last_layer = 0;
         struct pipe_sampler_view *view =
            pctx->create_sampler_view(pctx, &rsrc->base.b, &templ);
         pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &view);
         pctx->sampler_view_release(pctx, view);

         struct pipe_image_view image = {
            .resource = &rsrc->base.b,
            .access = PIPE_IMAGE_ACCESS_WRITE | PIPE_IMAGE_ACCESS_DRIVER_INTERNAL,
            .shader_access = PIPE_IMAGE_ACCESS_WRITE,
            .format = PIPE_FORMAT_R8G8B8A8_UNORM,
            .u.tex.level = level,
            .u.tex.first_layer = 0,
            .u.tex.last_layer = 0,
            .u.tex.single_layer_view = true,
         };
         pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &image);

         const uint32_t width = u_minify(prsc->width0, level);
         const uint32_t height = u_minify(prsc->height0, level);
         const struct pipe_grid_info grid = {
            .block = {AGX_G17P_BLOCK_MIP_TILE, AGX_G17P_BLOCK_MIP_TILE, 1},
            .grid = {DIV_ROUND_UP(width, AGX_G17P_BLOCK_MIP_TILE),
                     DIV_ROUND_UP(height, AGX_G17P_BLOCK_MIP_TILE), 1},
         };
         pctx->launch_grid(pctx, &grid);
      }

      pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
   }

   for (unsigned level = sampled_start; level <= sampled_last; ++level) {
      struct agx_g17p_mip_level_plan *plan = &plans[level - sampled_start];

      /* Re-acquire only if a full encoder replaced the compute batch. */
      if (unlikely(ctx->batch != batch)) {
         batch = agx_get_compute_batch(ctx);
         agx_batch_init_state(batch);
         for (unsigned l = level; l <= last_level; ++l)
            agx_batch_writes(batch, rsrc, l);
      }

      if (plan->key & AGX_G17P_MIP_ODD) {
         const uint32_t width = plan->params.width;
         const uint32_t height = plan->params.height;
         struct agx_ptr axis = agx_pool_alloc_aligned(
            &batch->pool, (size_t)(width + height) * sizeof(uint32_t), 4);
         uint32_t *entries = axis.cpu;
         ASSERTED const bool x_ok =
            agx_build_mip_axis(plan->src_w, width, entries);
         ASSERTED const bool y_ok =
            agx_build_mip_axis(plan->src_h, height, entries + width);
         assert(x_ok && y_ok);
         plan->params.axis_lo = (uint32_t)axis.gpu;
         plan->params.axis_hi = (uint32_t)(axis.gpu >> 32);
      }

      struct pipe_constant_buffer cb = {
         .buffer_size = sizeof(plan->params),
         .user_buffer = &plan->params,
      };
      pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &cb);

      /* An integer view of the same channel class reads the source's raw
       * bytes and keeps its compression (ail_is_view_compatible()). */
      struct pipe_sampler_view templ;
      u_sampler_view_default_template(&templ, &rsrc->base.b,
                                      PIPE_FORMAT_R8G8B8A8_UINT);
      templ.target = PIPE_TEXTURE_2D;
      templ.u.tex.first_level = level - 1;
      templ.u.tex.last_level = level - 1;
      templ.u.tex.first_layer = 0;
      templ.u.tex.last_layer = 0;
      struct pipe_sampler_view *view =
         pctx->create_sampler_view(pctx, &rsrc->base.b, &templ);
      pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &view);
      pctx->sampler_view_release(pctx, view);

      void *cs = ctx->compute_blitter.g17p_mip_cs[plan->key];
      if (cs != bound_cs) {
         pctx->bind_compute_state(pctx, cs);
         bound_cs = cs;
      }

      const struct pipe_grid_info grid = {
         .block = {AGX_G17P_MIP_WG_SIZE, AGX_G17P_MIP_WG_SIZE, 1},
         .grid = {plan->groups_x, plan->groups_y, 1},
      };
      pctx->launch_grid(pctx, &grid);
   }

   if (precompiled_tail) {
      if (unlikely(ctx->batch != batch)) {
         batch = agx_get_compute_batch(ctx);
         agx_batch_init_state(batch);
         agx_batch_reads(batch, rsrc);
         for (unsigned l = sampled_start + 1; l <= last_level; ++l)
            agx_batch_writes(batch, rsrc, l);
      }

      if (meta_end > meta_start) {
         agx_agx3_mipmap_ensure_cdm_space(ctx, &batch, rsrc, sampled_start + 1,
                                          last_level);
         const uint32_t mode =
            (uint32_t)ail_tile_mode_uncompressed(layout->format);
         const uint64_t meta_addr =
            agx_map_gpu(rsrc) + layout->metadata_offset_B + meta_start;
         libagx_fill_uint4(batch, agx_2d((meta_end - meta_start) / 16u, 1),
                           AGX_BARRIER_NONE, meta_addr, 0, mode, mode, mode,
                           mode);
      }

      const bool has_2level =
         agx_get_precompiled(&ctx->bg_eot, LIBAGX_MIPMAP_UNORM8X4_2LEVEL) !=
         NULL;
      agx_generate_mipmap_agx3_levels(ctx, &batch, rsrc, sampled_start + 1,
                                      last_level, 0, 0, has_2level);
   }

   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = cs_invocations;
   pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, NULL);
   pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
   if (!ctx->compute_blitter.saved_num_sampler_states &&
       block_last > base_level)
      pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0, 0, NULL);
   asahi_compute_restore(ctx);
   return true;
}

static void
agx_generate_mipmap_agx3_levels(struct agx_context *ctx,
                                struct agx_batch **batchp,
                                struct agx_resource *rsrc,
                                unsigned start_level, unsigned last_level,
                                unsigned first_layer, unsigned last_layer,
                                bool has_2level)
{
   struct agx_batch *batch = *batchp;
   const struct pipe_resource *prsc = &rsrc->base.b;
   const struct ail_layout *layout = &rsrc->layout;
   const uint64_t base_addr = agx_map_gpu(rsrc);

   for (unsigned level = start_level; level <= last_level;) {
      const uint32_t src_w = u_minify(prsc->width0, level - 1);
      const uint32_t src_h = u_minify(prsc->height0, level - 1);
      const uint32_t dst_w = u_minify(prsc->width0, level);
      const uint32_t dst_h = u_minify(prsc->height0, level);

      const struct ail_tile src_tile = layout->tilesize_el[level - 1];
      const struct ail_tile dst_tile = layout->tilesize_el[level];
      struct agx_tile_address_config src_cfg, dst_cfg;
      const bool ok_src = agx_tile_address_config_init(
         layout->stride_el[level - 1], src_tile.width_el, src_tile.height_el,
         &src_cfg);
      const bool ok_dst = agx_tile_address_config_init(
         layout->stride_el[level], dst_tile.width_el, dst_tile.height_el,
         &dst_cfg);
      assert(ok_src && ok_dst);
      (void)ok_src;
      (void)ok_dst;

      bool fused2 = false;
      uint32_t dst2_w = 0, dst2_h = 0;
      struct agx_tile_address_config dst2_cfg = {0};
      if (has_2level && level + 1 <= last_level &&
          src_w == dst_w * 2u && src_h == dst_h * 2u &&
          src_cfg.tile_width_log2 >= 2u && src_cfg.tile_height_log2 >= 2u &&
          dst_cfg.tile_width_log2 >= 1u && dst_cfg.tile_height_log2 >= 1u) {
         dst2_w = u_minify(prsc->width0, level + 1);
         dst2_h = u_minify(prsc->height0, level + 1);
         if (dst_w == dst2_w * 2u && dst_h == dst2_h * 2u) {
            const struct ail_tile dst2_tile = layout->tilesize_el[level + 1];
            const bool ok_dst2 = agx_tile_address_config_init(
               layout->stride_el[level + 1], dst2_tile.width_el,
               dst2_tile.height_el, &dst2_cfg);
            assert(ok_dst2);
            (void)ok_dst2;
            fused2 = true;
            for (unsigned layer = first_layer; layer <= last_layer; ++layer) {
               const uint64_t s_addr =
                  base_addr + ail_get_layer_level_B(layout, layer, level - 1);
               const uint64_t d1_addr =
                  base_addr + ail_get_layer_level_B(layout, layer, level);
               if ((s_addr & 0xfu) || (d1_addr & 0xfu)) {
                  fused2 = false;
                  break;
               }
            }
         }
      }

      if (fused2) {
         const uint32_t tile_log2_packed =
            dst2_cfg.tile_width_log2 | (dst2_cfg.tile_height_log2 << 4) |
            (dst_cfg.tile_width_log2 << 8) | (dst_cfg.tile_height_log2 << 12) |
            (src_cfg.tile_width_log2 << 16) | (src_cfg.tile_height_log2 << 20);
         const struct agx_grid grid =
            agx_2d(ALIGN_POT(dst2_w, 16), ALIGN_POT(dst2_h, 8));

         for (unsigned layer = first_layer; layer <= last_layer; ++layer) {
            agx_agx3_mipmap_ensure_cdm_space(ctx, &batch, rsrc, level,
                                             last_level);

            const uint64_t dst2_addr =
               base_addr + ail_get_layer_level_B(layout, layer, level + 1);
            const uint64_t dst1_addr =
               base_addr + ail_get_layer_level_B(layout, layer, level);
            const uint64_t src_addr =
               base_addr + ail_get_layer_level_B(layout, layer, level - 1);
            const enum agx_barrier barrier =
               (layer == last_layer) ? AGX_BARRIER_ALL : AGX_BARRIER_NONE;

            libagx_mipmap_unorm8x4_2level(
               batch, grid, barrier, dst2_addr, dst1_addr, src_addr, dst2_w,
               dst2_h, dst2_cfg.tiles_per_row, dst_cfg.tiles_per_row,
               src_cfg.tiles_per_row, tile_log2_packed);
         }

         level += 2;
         continue;
      }

      const bool odd = (src_w == (uint64_t)dst_w * 2u + 1u) ||
                       (src_h == (uint64_t)dst_h * 2u + 1u);
      struct agx_ptr axis = {0};
      if (odd) {
         axis = agx_pool_alloc_aligned(
            &batch->pool, (size_t)(dst_w + dst_h) * sizeof(uint32_t), 4);
         uint32_t *entries = axis.cpu;
         const bool x_ok = agx_build_mip_axis(src_w, dst_w, entries);
         const bool y_ok = agx_build_mip_axis(src_h, dst_h, entries + dst_w);
         assert(x_ok && y_ok);
         (void)x_ok;
         (void)y_ok;
      }

      for (unsigned layer = first_layer; layer <= last_layer; ++layer) {
         if (agx_agx3_mipmap_ensure_cdm_space(ctx, &batch, rsrc, level,
                                              last_level) &&
             odd) {
            axis = agx_pool_alloc_aligned(
               &batch->pool, (size_t)(dst_w + dst_h) * sizeof(uint32_t), 4);
            uint32_t *entries = axis.cpu;
            const bool x_ok = agx_build_mip_axis(src_w, dst_w, entries);
            const bool y_ok =
               agx_build_mip_axis(src_h, dst_h, entries + dst_w);
            assert(x_ok && y_ok);
            (void)x_ok;
            (void)y_ok;
         }

         const uint64_t dst_addr =
            base_addr + ail_get_layer_level_B(layout, layer, level);
         const uint64_t src_addr =
            base_addr + ail_get_layer_level_B(layout, layer, level - 1);
         const enum agx_barrier barrier =
            (layer == last_layer) ? AGX_BARRIER_ALL : AGX_BARRIER_NONE;

         if (odd) {
            libagx_mipmap_unorm8x4_odd(
               batch, agx_2d(dst_w, dst_h), barrier, dst_addr, src_addr,
               axis.gpu, dst_w, dst_h, dst_cfg.tiles_per_row,
               src_cfg.tiles_per_row, dst_cfg.tile_width_log2,
               dst_cfg.tile_height_log2, src_cfg.tile_width_log2,
               src_cfg.tile_height_log2);
         } else {
            libagx_mipmap_unorm8x4(
               batch, agx_2d(dst_w, dst_h), barrier, dst_addr, src_addr, dst_w,
               dst_h, src_w == dst_w ? 1u : 2u, src_h == dst_h ? 1u : 2u,
               dst_cfg.tiles_per_row, src_cfg.tiles_per_row,
               dst_cfg.tile_width_log2, dst_cfg.tile_height_log2,
               src_cfg.tile_width_log2, src_cfg.tile_height_log2);
         }
      }

      ++level;
   }

   *batchp = batch;
}

/* Native AGX3 compute mipchain generation for uncompressed GPU-tiled UNORM8x4
 * 2D / 2D_ARRAY / CUBE textures. Runs all levels in one CDM compute batch via
 * libagx_mipmap_unorm8x4 / libagx_mipmap_unorm8x4_2level /
 * libagx_mipmap_unorm8x4_odd (matching Borealis's hk_g17p_compute_blit()),
 * avoiding one 3D render pass per mip level. Unsupported formats or layouts
 * return false and fall back to util_gen_mipmap(). */
static bool
agx_generate_mipmap_agx3(struct pipe_context *pctx, struct pipe_resource *prsc,
                         enum pipe_format format, unsigned base_level,
                         unsigned last_level, unsigned first_layer,
                         unsigned last_layer)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   struct agx_resource *rsrc = agx_resource(prsc);

   if (!agx_generate_mipmap_agx3_supported(dev, rsrc, format, base_level,
                                           last_level, first_layer, last_layer))
      return false;

   if (!agx_get_precompiled(&ctx->bg_eot, LIBAGX_MIPMAP_UNORM8X4) ||
       !agx_get_precompiled(&ctx->bg_eot, LIBAGX_MIPMAP_UNORM8X4_ODD))
      return false;

   const bool has_2level =
      agx_get_precompiled(&ctx->bg_eot, LIBAGX_MIPMAP_UNORM8X4_2LEVEL) != NULL;

   struct agx_batch *batch = agx_get_compute_batch(ctx);
   agx_batch_init_state(batch);
   agx_batch_reads(batch, rsrc);
   for (unsigned level = base_level + 1; level <= last_level; ++level)
      agx_batch_writes(batch, rsrc, level);

   agx_generate_mipmap_agx3_levels(ctx, &batch, rsrc, base_level + 1,
                                   last_level, first_layer, last_layer,
                                   has_2level);
   return true;
}

/* pipe_context::generate_mipmap. A native path declines what it does not
 * handle, and the blitter then generates the levels here rather than leaving
 * them to the state tracker: under u_threaded_context, tc_generate_mipmap()
 * has already told the state tracker the driver generates them, and
 * tc_call_generate_mipmap() asserts that it did. tc_generate_mipmap() only
 * accepts a format the driver renders to, which util_gen_mipmap() needs. */
bool
agx_generate_mipmap(struct pipe_context *pctx, struct pipe_resource *prsc,
                    enum pipe_format format, unsigned base_level,
                    unsigned last_level, unsigned first_layer,
                    unsigned last_layer)
{
   const enum agx_chip chip = agx_device(pctx->screen)->chip;
   if (agx_generate_mipmap_sampled(pctx, agx_resource(prsc), format, base_level,
                                   last_level, first_layer, last_layer))
      return true;

   const bool native =
      agx_chip_g17_m5(chip)
         ? agx_generate_mipmap_g17g(pctx, prsc, format, base_level,
                                    last_level, first_layer, last_layer)
         : agx_generate_mipmap_agx3(pctx, prsc, format, base_level,
                                    last_level, first_layer, last_layer);

   return native ||
          util_gen_mipmap(pctx, prsc, format, base_level, last_level,
                          first_layer, last_layer, PIPE_TEX_FILTER_LINEAR);
}

#define AGX_G17P_BUF2IMG_TILE 32
#define AGX_G17P_TILED_UPLOAD_PER_THREAD 4u

struct agx_g17p_block_buf2img_params {
   uint32_t src_lo;
   uint32_t src_hi;
   uint32_t src_stride_B;
   uint32_t tile_x0;
   uint32_t tile_y0;
   uint32_t box_x;
   uint32_t box_y;
   uint32_t box_w;
   uint32_t box_h;
   uint32_t mip_w;
   uint32_t mip_h;
   uint32_t pad;
};
static_assert(sizeof(struct agx_g17p_block_buf2img_params) == 48,
              "G17P block buf2img parameters are tightly packed");

#define block_b2i_param(b, field)                                              \
   nir_load_ubo(                                                               \
      b, 1, 32, nir_imm_int(b, 0),                                             \
      nir_imm_int(b, offsetof(struct agx_g17p_block_buf2img_params, field)),   \
      .align_mul = 4, .range = ~0)

#define block_b2i_param64(b, field)                                            \
   nir_pack_64_2x32_split(b, block_b2i_param(b, field##_lo),                   \
                          block_b2i_param(b, field##_hi))

struct agx_g17p_tiled_buf2img_params {
   uint32_t src_lo;
   uint32_t src_hi;
   uint32_t dst_lo;
   uint32_t dst_hi;
   uint32_t src_stride_B;
   uint32_t tiles_per_row;
   uint32_t tile_w_log2;
   uint32_t tile_h_log2;
   uint32_t tile_x0;
   uint32_t tile_y0;
   uint32_t x0;
   uint32_t y0;
   uint32_t width;
   uint32_t height;
   uint32_t pad[2];
};
static_assert(sizeof(struct agx_g17p_tiled_buf2img_params) == 64,
              "G17P tiled buf2img parameters are tightly packed");

#define tiled_b2i_param(b, field)                                              \
   nir_load_ubo(                                                               \
      b, 1, 32, nir_imm_int(b, 0),                                             \
      nir_imm_int(b, offsetof(struct agx_g17p_tiled_buf2img_params, field)),   \
      .align_mul = 4, .range = ~0)

#define tiled_b2i_param64(b, field)                                            \
   nir_pack_64_2x32_split(b, tiled_b2i_param(b, field##_lo),                   \
                          tiled_b2i_param(b, field##_hi))

static int
agx_g17p_block_buf2img_slot(enum pipe_format format,
                            enum pipe_format *pbe_out,
                            enum pipe_format *tex_out)
{
   if (util_format_is_depth_or_stencil(format) ||
       util_format_is_compressed(format))
      return -1;

   /* agx3_tile_store_explicit() admits 8-bit 1/4-channel stores (imgwblk
    * lane_class 3 -> UNORM8 PBE) and 32-bit 1-channel stores (lane_class 2
    * -> R32_UINT PBE). The shader packs raw source bytes directly into the
    * imageblock with I8/I32 and partial tiles read untouched texels via UINT. */
   int slot = -1;
   enum pipe_format pbe = PIPE_FORMAT_NONE;
   enum pipe_format tex = PIPE_FORMAT_NONE;
   switch (ail_pixel_format[format].channels) {
   case AGX_CHANNELS_R8:
      slot = 0;
      pbe = PIPE_FORMAT_R8_UNORM;
      tex = PIPE_FORMAT_R8_UINT;
      break;
   case AGX_CHANNELS_R8G8B8A8:
      slot = 1;
      pbe = PIPE_FORMAT_R8G8B8A8_UNORM;
      tex = PIPE_FORMAT_R8G8B8A8_UINT;
      break;
   case AGX_CHANNELS_R32:
      slot = 2;
      pbe = PIPE_FORMAT_R32_UINT;
      tex = PIPE_FORMAT_R32_UINT;
      break;
   default:
      return -1;
   }

   if (util_format_get_blocksize(format) != util_format_get_blocksize(pbe) ||
       !ail_formats_compatible(format, pbe) ||
       !ail_formats_compatible(format, tex) ||
       ail_pixel_format[pbe].renderable == PIPE_FORMAT_NONE ||
       ail_pixel_format[tex].renderable == PIPE_FORMAT_NONE)
      return -1;

   if (pbe_out)
      *pbe_out = pbe;
   if (tex_out)
      *tex_out = tex;
   return slot;
}

enum pipe_format
agx_g17p_block_buf2img_format(enum pipe_format format)
{
   enum pipe_format pbe = PIPE_FORMAT_NONE;
   return agx_g17p_block_buf2img_slot(format, &pbe, NULL) >= 0 ? pbe
                                                               : PIPE_FORMAT_NONE;
}

bool
agx_g17p_block_buf2img_aligned(const struct pipe_box *box, uint32_t mip_w,
                               uint32_t mip_h)
{
   return (box->x & (AGX_G17P_BUF2IMG_TILE - 1)) == 0 &&
          (box->y & (AGX_G17P_BUF2IMG_TILE - 1)) == 0 &&
          ((box->width & (AGX_G17P_BUF2IMG_TILE - 1)) == 0 ||
           (uint32_t)(box->x + box->width) == mip_w) &&
          ((box->height & (AGX_G17P_BUF2IMG_TILE - 1)) == 0 ||
           (uint32_t)(box->y + box->height) == mip_h);
}

static nir_def *
agx_g17p_block_buf2img_unpack(nir_builder *b, enum pipe_format view_format,
                              nir_def *raw)
{
   switch (view_format) {
   case PIPE_FORMAT_R8_UINT:
      return nir_u2u16(b, raw);
   case PIPE_FORMAT_R8G8B8A8_UINT: {
      nir_def *b0 = nir_u2u16(b, nir_iand_imm(b, raw, 0xff));
      nir_def *b1 = nir_u2u16(b, nir_iand_imm(b, nir_ushr_imm(b, raw, 8), 0xff));
      nir_def *b2 = nir_u2u16(b, nir_iand_imm(b, nir_ushr_imm(b, raw, 16), 0xff));
      nir_def *b3 = nir_u2u16(b, nir_ushr_imm(b, raw, 24));
      return nir_vec4(b, b0, b1, b2, b3);
   }
   default:
      return raw;
   }
}

/* HAL200/AGX3B compressed buffer-to-image upload through the compressed PBE.
 * Each 32x32 workgroup loads one tile of raw bytes from the source buffer
 * (preserving untouched pixels via txf when partial), writes them to its
 * imageblock, and thread (0,0) commits the compressed tile with imgwblk. */
static void *
agx_g17p_block_buf2img_compute_shader(struct pipe_context *pctx,
                                      enum pipe_format view_format,
                                      bool partial)
{
   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, pctx->screen->nir_options[MESA_SHADER_COMPUTE],
      partial ? "agx-g17p-block-b2i-partial" : "agx-g17p-block-b2i");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = AGX_G17P_BUF2IMG_TILE;
   b->shader->info.workgroup_size[1] = AGX_G17P_BUF2IMG_TILE;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.num_ubos = 1;
   b->shader->info.num_images = 1;
   BITSET_SET(b->shader->info.images_used, 0);
   if (partial) {
      b->shader->info.num_textures = 1;
      BITSET_SET(b->shader->info.textures_used, 0);
      BITSET_SET(b->shader->info.textures_used_by_txf, 0);
   }

   const unsigned block_B = util_format_get_blocksize(view_format);
   const struct util_format_description *fdesc =
      util_format_description(view_format);
   const unsigned comps = fdesc->nr_channels;
   const unsigned ch_bits = fdesc->channel[0].size;
   const unsigned tib_bits = MAX2(ch_bits, 16u);
   const unsigned load_comps = ch_bits == 8 ? 1u : comps;
   const unsigned load_bits = ch_bits == 8 ? block_B * 8u : ch_bits;

   nir_def *wg = nir_load_workgroup_id(b);
   nir_def *lid = nir_trim_vector(b, nir_load_local_invocation_id(b), 2);
   nir_def *tile_x =
      nir_iadd(b, nir_channel(b, wg, 0), block_b2i_param(b, tile_x0));
   nir_def *tile_y =
      nir_iadd(b, nir_channel(b, wg, 1), block_b2i_param(b, tile_y0));
   nir_def *px = nir_ior(b, nir_ishl_imm(b, tile_x, 5), nir_channel(b, lid, 0));
   nir_def *py = nir_ior(b, nir_ishl_imm(b, tile_y, 5), nir_channel(b, lid, 1));

   nir_def *rx = nir_isub(b, px, block_b2i_param(b, box_x));
   nir_def *ry = nir_isub(b, py, block_b2i_param(b, box_y));
   nir_def *in_box = nir_iand(b, nir_ult(b, rx, block_b2i_param(b, box_w)),
                              nir_ult(b, ry, block_b2i_param(b, box_h)));

   nir_def *val;
   if (!partial) {
      nir_def *raw_in;
      nir_def *raw_zero = nir_imm_zero(b, load_comps, load_bits);
      nir_push_if(b, in_box);
      {
         nir_def *off =
            nir_iadd(b, nir_imul(b, ry, block_b2i_param(b, src_stride_B)),
                     nir_imul_imm(b, rx, block_B));
         raw_in = nir_load_global(
            b, load_comps, load_bits,
            nir_iadd(b, block_b2i_param64(b, src), nir_u2u64(b, off)),
            .align_mul = block_B,
            .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
      }
      nir_pop_if(b, NULL);
      nir_def *raw = nir_if_phi(b, raw_in, raw_zero);
      val = agx_g17p_block_buf2img_unpack(b, view_format, raw);
   } else {
      nir_def *val_in, *val_out;
      nir_push_if(b, in_box);
      {
         nir_def *off =
            nir_iadd(b, nir_imul(b, ry, block_b2i_param(b, src_stride_B)),
                     nir_imul_imm(b, rx, block_B));
         nir_def *raw = nir_load_global(
            b, load_comps, load_bits,
            nir_iadd(b, block_b2i_param64(b, src), nir_u2u64(b, off)),
            .align_mul = block_B,
            .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
         val_in = agx_g17p_block_buf2img_unpack(b, view_format, raw);
      }
      nir_push_else(b, NULL);
      {
         nir_def *in_mip =
            nir_iand(b, nir_ult(b, px, block_b2i_param(b, mip_w)),
                     nir_ult(b, py, block_b2i_param(b, mip_h)));
         nir_def *tex_val;
         nir_def *zero = nir_imm_zero(b, comps, tib_bits);
         nir_push_if(b, in_mip);
         {
            nir_def *texel = nir_txf(
               b, nir_vec2(b, px, py), .lod = nir_imm_int(b, 0),
               .texture_index = 0, .dim = GLSL_SAMPLER_DIM_2D,
               .dest_type = nir_type_uint32,
               .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
            texel = nir_trim_vector(b, texel, comps);
            tex_val = tib_bits == 16 ? nir_u2u16(b, texel) : texel;
         }
         nir_pop_if(b, NULL);
         val_out = nir_if_phi(b, tex_val, zero);
      }
      nir_pop_if(b, NULL);
      val = nir_if_phi(b, val_in, val_out);
   }

   const enum pipe_format isa_format = ail_pixel_format[view_format].renderable;
   nir_store_local_pixel_agx(b, val, nir_imm_intN_t(b, 1, 16),
                             nir_u2u16(b, lid), .base = 0,
                             .write_mask = BITFIELD_MASK(comps),
                             .format = isa_format, .explicit_coord = true);
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP);

   nir_push_if(b, nir_ball(b, nir_ieq_imm(b, lid, 0)));
   {
      nir_image_store_block_agx(b, nir_imm_int(b, 0), nir_imm_int(b, 0),
                                nir_vec2(b, px, py), .format = isa_format,
                                .image_dim = GLSL_SAMPLER_DIM_2D,
                                .image_array = false, .explicit_coord = true);
   }
   nir_pop_if(b, NULL);
   b->shader->info.cs.image_block_size_per_thread_agx = block_B;

   return pipe_shader_from_nir(pctx, b->shader);
}

/* HAL200/AGX3 uncompressed GPU-tiled buffer-to-image upload in Morton order,
 * matching Borealis's hk_g17p_build_tiled_upload(). Writes 64 consecutive
 * Morton elements per run (coalesced 128B+ stores) directly into tiled memory. */
static void *
agx_g17p_tiled_buf2img_compute_shader(struct pipe_context *pctx,
                                      unsigned block_B, unsigned per_thread)
{
   nir_builder build = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, pctx->screen->nir_options[MESA_SHADER_COMPUTE],
      "agx-g17p-tiled-b2i");
   nir_builder *b = &build;
   b->shader->info.workgroup_size[0] = 64;
   b->shader->info.workgroup_size[1] = 1;
   b->shader->info.workgroup_size[2] = 1;
   b->shader->info.num_ubos = 1;

   const unsigned bits = MIN2(block_B, 4u) * 8u;
   const unsigned comps = MAX2(block_B / 4u, 1u);
   const unsigned align_B = MIN2(block_B, 4u);

   nir_def *lid = nir_channel(b, nir_load_local_invocation_id(b), 0);
   nir_def *wg = nir_load_workgroup_id(b);
   nir_def *twl = tiled_b2i_param(b, tile_w_log2);
   nir_def *thl = tiled_b2i_param(b, tile_h_log2);
   nir_def *tile_x =
      nir_iadd(b, tiled_b2i_param(b, tile_x0), nir_channel(b, wg, 1));
   nir_def *tile_y =
      nir_iadd(b, tiled_b2i_param(b, tile_y0), nir_channel(b, wg, 2));
   nir_def *tile_el = nir_ishl(
      b,
      nir_iadd(b, nir_imul(b, tile_y, tiled_b2i_param(b, tiles_per_row)),
               tile_x),
      nir_iadd(b, twl, thl));
   nir_def *origin_x = nir_ishl(b, tile_x, twl);
   nir_def *origin_y = nir_ishl(b, tile_y, thl);
   nir_def *lane_x = agx_nir_compact_bits_8(b, lid);
   nir_def *lane_y = agx_nir_compact_bits_8(b, nir_ushr_imm(b, lid, 1));
   nir_def *x0 = tiled_b2i_param(b, x0);
   nir_def *y0 = tiled_b2i_param(b, y0);
   nir_def *width = tiled_b2i_param(b, width);
   nir_def *height = tiled_b2i_param(b, height);
   nir_def *pitch = tiled_b2i_param(b, src_stride_B);
   nir_def *src = tiled_b2i_param64(b, src);
   nir_def *dst = tiled_b2i_param64(b, dst);

   for (unsigned j = 0; j < per_thread; ++j) {
      nir_def *run = nir_iadd_imm(
         b, nir_imul_imm(b, nir_channel(b, wg, 0), per_thread), j);
      nir_def *m = nir_ior(b, nir_ishl_imm(b, run, 6), lid);
      nir_def *x = nir_ior(
         b, origin_x,
         nir_ior(b, nir_ishl_imm(b, agx_nir_compact_bits_8(b, run), 3),
                 lane_x));
      nir_def *y = nir_ior(
         b, origin_y,
         nir_ior(b,
                 nir_ishl_imm(
                    b, agx_nir_compact_bits_8(b, nir_ushr_imm(b, run, 1)), 3),
                 lane_y));
      nir_def *rx = nir_isub(b, x, x0);
      nir_def *ry = nir_isub(b, y, y0);
      nir_push_if(b, nir_iand(b, nir_ult(b, rx, width), nir_ult(b, ry, height)));
      {
         nir_def *src_off =
            nir_iadd(b, nir_imul(b, ry, pitch), nir_imul_imm(b, rx, block_B));
         nir_def *data = nir_load_global(
            b, comps, bits, nir_iadd(b, src, nir_u2u64(b, src_off)),
            .align_mul = align_B,
            .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
         nir_def *dst_off =
            nir_imul_imm(b, nir_iadd(b, tile_el, m), block_B);
         nir_store_global(b, data, nir_iadd(b, dst, nir_u2u64(b, dst_off)),
                          .align_mul = align_B);
      }
      nir_pop_if(b, NULL);
   }

   return pipe_shader_from_nir(pctx, b->shader);
}

bool
agx_g17p_tiled_buf2img_plan(const struct ail_layout *layout, unsigned level,
                            const struct pipe_box *box, unsigned buffer_offset,
                            unsigned row_stride, unsigned layer_stride,
                            uint32_t groups[3], uint32_t *per_thread_out)
{
   if (layout->tiling != AIL_TILING_GPU || level >= layout->levels ||
       ail_is_level_logically_compressed(layout, level) ||
       util_format_is_depth_or_stencil(layout->format))
      return false;

   const unsigned block_B = util_format_get_blocksize(layout->format);
   if (!util_is_power_of_two_nonzero(block_B) || block_B > 16u)
      return false;

   const unsigned align_B = MIN2(block_B, 4u);
   if (((buffer_offset | row_stride | layer_stride) & (align_B - 1u)) != 0)
      return false;

   const unsigned bw = util_format_get_blockwidth(layout->format);
   const unsigned bh = util_format_get_blockheight(layout->format);
   if (box->x < 0 || box->y < 0 || box->z < 0 || box->width <= 0 ||
       box->height <= 0 || box->depth <= 0 || (box->x % bw) != 0 ||
       (box->y % bh) != 0)
      return false;

   const uint32_t x0_el = (uint32_t)box->x / bw;
   const uint32_t y0_el = (uint32_t)box->y / bh;
   const uint32_t w_el = DIV_ROUND_UP((uint32_t)box->width, bw);
   const uint32_t h_el = DIV_ROUND_UP((uint32_t)box->height, bh);
   if (!w_el || !h_el)
      return false;

   const struct ail_tile tile = layout->tilesize_el[level];
   struct agx_tile_address_config cfg;
   if (!agx_tile_address_config_init(layout->stride_el[level], tile.width_el,
                                     tile.height_el, &cfg))
      return false;

   const uint32_t twl = cfg.tile_width_log2;
   const uint32_t thl = cfg.tile_height_log2;
   const uint32_t area_log2 = twl + thl;
   if ((twl != thl && twl != thl + 1u) || area_log2 < 6u || twl > 7u ||
       thl > 7u)
      return false;

   const uint32_t per_thread =
      MIN2(AGX_G17P_TILED_UPLOAD_PER_THREAD, 1u << (area_log2 - 6u));
   const uint32_t tx0 = x0_el >> twl;
   const uint32_t ty0 = y0_el >> thl;
   const uint64_t x1 = (uint64_t)x0_el + w_el - 1u;
   const uint64_t y1 = (uint64_t)y0_el + h_el - 1u;
   groups[0] = 1u << (area_log2 - 6u - util_logbase2(per_thread));
   groups[1] = (uint32_t)((x1 >> twl) - tx0 + 1u);
   groups[2] = (uint32_t)((y1 >> thl) - ty0 + 1u);
   if (groups[0] > 65535u || groups[1] > 65535u || groups[2] > 65535u)
      return false;

   const uint64_t src_span =
      (uint64_t)(h_el - 1u) * row_stride + (uint64_t)w_el * block_B;
   const uint64_t dst_end =
      (((uint64_t)(y1 >> thl) * cfg.tiles_per_row + (x1 >> twl) + 1u)
       << area_log2) *
      block_B;
   if (src_span > UINT32_MAX || dst_end > UINT32_MAX)
      return false;

   *per_thread_out = per_thread;
   return true;
}

static bool
agx_image_copy_buffer_block(struct pipe_context *pctx,
                            struct agx_resource *dst_rsrc,
                            struct agx_resource *src_rsrc,
                            unsigned buffer_offset, unsigned row_stride,
                            unsigned layer_stride, unsigned level,
                            const struct pipe_box *box)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   struct pipe_resource *dst = &dst_rsrc->base.b;
   const struct ail_layout *layout = &dst_rsrc->layout;

   if (!hk_g17p_block_mip_supported(dev->chip) || dst->nr_samples > 1 ||
       dst_rsrc->separate_stencil || layout->tiling != AIL_TILING_GPU ||
       !layout->compressed || level >= layout->levels ||
       !ail_is_level_logically_compressed(layout, level))
      return false;

   switch (dst->target) {
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_CUBE_ARRAY:
      break;
   default:
      return false;
   }

   enum pipe_format pbe_format = PIPE_FORMAT_NONE;
   enum pipe_format tex_format = PIPE_FORMAT_NONE;
   const int slot =
      agx_g17p_block_buf2img_slot(dst->format, &pbe_format, &tex_format);
   if (slot < 0)
      return false;

   const uint32_t mip_w = u_minify(dst->width0, level);
   const uint32_t mip_h = u_minify(dst->height0, level);
   if (box->x < 0 || box->y < 0 || box->z < 0 || box->width <= 0 ||
       box->height <= 0 || box->depth <= 0 ||
       (uint32_t)box->x + (uint32_t)box->width > mip_w ||
       (uint32_t)box->y + (uint32_t)box->height > mip_h ||
       (unsigned)box->z + (unsigned)box->depth > dst->array_size)
      return false;

   const unsigned block_B = util_format_get_blocksize(pbe_format);
   if (((buffer_offset | row_stride | layer_stride) & (block_B - 1u)) != 0)
      return false;

   const uint64_t slice_span = (uint64_t)(box->height - 1) * row_stride +
                               (uint64_t)box->width * block_B;
   if (slice_span > UINT32_MAX)
      return false;

   const bool aligned = agx_g17p_block_buf2img_aligned(box, mip_w, mip_h);
   const bool partial = !aligned && agx_resource_valid(dst_rsrc, level);

   void **cs_slot = &ctx->compute_blitter.g17p_block_buf2img_cs[slot][partial];
   if (!*cs_slot) {
      *cs_slot =
         agx_g17p_block_buf2img_compute_shader(pctx, tex_format, partial);
      if (!*cs_slot)
         return false;
   }

   const uint32_t tx0 = (uint32_t)box->x / AGX_G17P_BUF2IMG_TILE;
   const uint32_t ty0 = (uint32_t)box->y / AGX_G17P_BUF2IMG_TILE;
   const uint32_t tx1 =
      ((uint32_t)box->x + (uint32_t)box->width - 1u) / AGX_G17P_BUF2IMG_TILE;
   const uint32_t ty1 =
      ((uint32_t)box->y + (uint32_t)box->height - 1u) / AGX_G17P_BUF2IMG_TILE;
   const struct pipe_grid_info grid = {
      .block = {AGX_G17P_BUF2IMG_TILE, AGX_G17P_BUF2IMG_TILE, 1},
      .grid = {tx1 - tx0 + 1u, ty1 - ty0 + 1u, 1},
   };

   asahi_compute_save(ctx);
   struct agx_query *cs_invocations =
      ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS];
   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = NULL;

   struct agx_batch *batch = agx_get_compute_batch(ctx);
   agx_batch_init_state(batch);
   agx_batch_reads(batch, src_rsrc);
   agx_batch_writes(batch, dst_rsrc, level);

   pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0, 1,
                             &ctx->compute_blitter.sampler[0]);
   pctx->bind_compute_state(pctx, *cs_slot);

   const uint64_t base_src = agx_map_gpu(src_rsrc) + buffer_offset;
   for (unsigned z = 0; z < (unsigned)box->depth; ++z) {
      if (unlikely(ctx->batch != batch)) {
         batch = agx_get_compute_batch(ctx);
         agx_batch_init_state(batch);
         agx_batch_reads(batch, src_rsrc);
         agx_batch_writes(batch, dst_rsrc, level);
      }

      const unsigned layer = (unsigned)box->z + z;
      const uint64_t slice_src = base_src + (uint64_t)z * layer_stride;
      const struct agx_g17p_block_buf2img_params params = {
         .src_lo = (uint32_t)slice_src,
         .src_hi = (uint32_t)(slice_src >> 32),
         .src_stride_B = row_stride,
         .tile_x0 = tx0,
         .tile_y0 = ty0,
         .box_x = (uint32_t)box->x,
         .box_y = (uint32_t)box->y,
         .box_w = (uint32_t)box->width,
         .box_h = (uint32_t)box->height,
         .mip_w = mip_w,
         .mip_h = mip_h,
      };
      const struct pipe_constant_buffer cb = {
         .buffer_size = sizeof(params),
         .user_buffer = &params,
      };
      pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &cb);

      if (partial) {
         struct pipe_sampler_view templ;
         u_sampler_view_default_template(&templ, dst, tex_format);
         templ.target = PIPE_TEXTURE_2D;
         templ.u.tex.first_level = level;
         templ.u.tex.last_level = level;
         templ.u.tex.first_layer = layer;
         templ.u.tex.last_layer = layer;
         struct pipe_sampler_view *view =
            pctx->create_sampler_view(pctx, dst, &templ);
         pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &view);
         pctx->sampler_view_release(pctx, view);
      }

      const struct pipe_image_view image = {
         .resource = dst,
         .access = PIPE_IMAGE_ACCESS_WRITE | PIPE_IMAGE_ACCESS_DRIVER_INTERNAL,
         .shader_access = PIPE_IMAGE_ACCESS_WRITE,
         .format = pbe_format,
         .u.tex.level = level,
         .u.tex.first_layer = layer,
         .u.tex.last_layer = layer,
         .u.tex.single_layer_view = true,
      };
      pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 1, 0, &image);
      pctx->launch_grid(pctx, &grid);
   }

   pctx->set_shader_images(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
   if (partial && !ctx->compute_blitter.saved_sampler_view)
      pctx->set_sampler_views(pctx, MESA_SHADER_COMPUTE, 0, 0, 1, NULL);
   if (!ctx->compute_blitter.saved_num_sampler_states)
      pctx->bind_sampler_states(pctx, MESA_SHADER_COMPUTE, 0, 0, NULL);

   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = cs_invocations;
   asahi_compute_restore(ctx);
   return true;
}

static bool
agx_image_copy_buffer_tiled(struct pipe_context *pctx,
                            struct agx_resource *dst_rsrc,
                            struct agx_resource *src_rsrc,
                            unsigned buffer_offset, unsigned row_stride,
                            unsigned layer_stride, unsigned level,
                            const struct pipe_box *box)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   struct pipe_resource *dst = &dst_rsrc->base.b;
   const struct ail_layout *layout = &dst_rsrc->layout;

   if (!agx_chip_native_agx3(dev->chip) || dst->nr_samples > 1 ||
       dst_rsrc->separate_stencil)
      return false;

   uint32_t groups[3] = {0};
   uint32_t per_thread = 0;
   if (!agx_g17p_tiled_buf2img_plan(layout, level, box, buffer_offset,
                                    row_stride, layer_stride, groups,
                                    &per_thread))
      return false;

   const unsigned block_B = util_format_get_blocksize(dst->format);
   const unsigned block_slot = util_logbase2(block_B);
   const unsigned pt_slot = util_logbase2(per_thread);
   void **cs_slot =
      &ctx->compute_blitter.g17p_tiled_buf2img_cs[block_slot][pt_slot];
   if (!*cs_slot) {
      *cs_slot =
         agx_g17p_tiled_buf2img_compute_shader(pctx, block_B, per_thread);
      if (!*cs_slot)
         return false;
   }

   const unsigned bw = util_format_get_blockwidth(dst->format);
   const unsigned bh = util_format_get_blockheight(dst->format);
   const uint32_t x0_el = (uint32_t)box->x / bw;
   const uint32_t y0_el = (uint32_t)box->y / bh;
   const uint32_t w_el = DIV_ROUND_UP((uint32_t)box->width, bw);
   const uint32_t h_el = DIV_ROUND_UP((uint32_t)box->height, bh);

   const struct ail_tile tile = layout->tilesize_el[level];
   struct agx_tile_address_config cfg;
   ASSERTED const bool cfg_ok = agx_tile_address_config_init(
      layout->stride_el[level], tile.width_el, tile.height_el, &cfg);
   assert(cfg_ok);

   const struct pipe_grid_info grid = {
      .block = {64, 1, 1},
      .grid = {groups[0], groups[1], groups[2]},
   };

   asahi_compute_save(ctx);
   struct agx_query *cs_invocations =
      ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS];
   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = NULL;

   struct agx_batch *batch = agx_get_compute_batch(ctx);
   agx_batch_init_state(batch);
   agx_batch_reads(batch, src_rsrc);
   agx_batch_writes(batch, dst_rsrc, level);

   pctx->bind_compute_state(pctx, *cs_slot);

   const uint64_t base_src = agx_map_gpu(src_rsrc) + buffer_offset;
   const uint64_t base_dst = dst_rsrc->bo->va->addr;
   for (unsigned z = 0; z < (unsigned)box->depth; ++z) {
      if (unlikely(ctx->batch != batch)) {
         batch = agx_get_compute_batch(ctx);
         agx_batch_init_state(batch);
         agx_batch_reads(batch, src_rsrc);
         agx_batch_writes(batch, dst_rsrc, level);
      }

      const uint64_t slice_src = base_src + (uint64_t)z * layer_stride;
      const uint64_t slice_dst =
         base_dst + ail_get_layer_level_B(layout, (unsigned)box->z + z, level);
      const struct agx_g17p_tiled_buf2img_params params = {
         .src_lo = (uint32_t)slice_src,
         .src_hi = (uint32_t)(slice_src >> 32),
         .dst_lo = (uint32_t)slice_dst,
         .dst_hi = (uint32_t)(slice_dst >> 32),
         .src_stride_B = row_stride,
         .tiles_per_row = cfg.tiles_per_row,
         .tile_w_log2 = cfg.tile_width_log2,
         .tile_h_log2 = cfg.tile_height_log2,
         .tile_x0 = x0_el >> cfg.tile_width_log2,
         .tile_y0 = y0_el >> cfg.tile_height_log2,
         .x0 = x0_el,
         .y0 = y0_el,
         .width = w_el,
         .height = h_el,
      };
      const struct pipe_constant_buffer cb = {
         .buffer_size = sizeof(params),
         .user_buffer = &params,
      };
      pctx->set_constant_buffer(pctx, MESA_SHADER_COMPUTE, 0, &cb);
      pctx->launch_grid(pctx, &grid);
   }

   ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS] = cs_invocations;
   asahi_compute_restore(ctx);
   return true;
}

void
agx_image_copy_buffer(struct pipe_context *pctx, struct pipe_resource *dst,
                      struct pipe_resource *src, unsigned buffer_offset,
                      unsigned buffer_stride, unsigned buffer_layer_stride,
                      unsigned level, const struct pipe_box *box)
{
   if (box->width <= 0 || box->height <= 0 || box->depth <= 0)
      return;

   const bool to_image = dst->target != PIPE_BUFFER;
   struct pipe_resource *img = to_image ? dst : src;
   struct pipe_resource *buf = to_image ? src : dst;
   const enum pipe_format format = img->format;
   const unsigned blocksize = util_format_get_blocksize(format);
   const unsigned nblocksx = util_format_get_nblocksx(format, box->width);
   const unsigned nblocksy = util_format_get_nblocksy(format, box->height);
   const unsigned row_stride =
      buffer_stride ? buffer_stride : nblocksx * blocksize;
   const unsigned layer_stride =
      buffer_layer_stride ? buffer_layer_stride : nblocksy * row_stride;

   const uint64_t copy_bytes = (uint64_t)(box->depth - 1) * layer_stride +
                               (uint64_t)(nblocksy - 1) * row_stride +
                               (uint64_t)nblocksx * blocksize;
   if (buffer_offset >= buf->width0 ||
       copy_bytes > (uint64_t)buf->width0 - buffer_offset)
      return;

   if (to_image) {
      struct agx_resource *dst_rsrc = agx_resource(dst);
      struct agx_resource *src_rsrc = agx_resource(src);
      if (agx_image_copy_buffer_block(pctx, dst_rsrc, src_rsrc, buffer_offset,
                                      row_stride, layer_stride, level, box))
         return;

      if (agx_image_copy_buffer_tiled(pctx, dst_rsrc, src_rsrc, buffer_offset,
                                      row_stride, layer_stride, level, box))
         return;

      struct pipe_transfer *xfer = NULL;
      const void *map = pipe_buffer_map_range(
         pctx, src, buffer_offset, (unsigned)copy_bytes, PIPE_MAP_READ, &xfer);
      if (!map)
         return;

      pctx->texture_subdata(pctx, dst, level, 0, box, map, row_stride,
                            layer_stride);
      pipe_buffer_unmap(pctx, xfer);
      return;
   }

   struct pipe_transfer *src_xfer = NULL;
   struct pipe_transfer *dst_xfer = NULL;
   const void *src_map =
      pctx->texture_map(pctx, src, level, PIPE_MAP_READ, box, &src_xfer);
   if (!src_map)
      return;

   void *dst_map = pipe_buffer_map_range(
      pctx, dst, buffer_offset, (unsigned)copy_bytes,
      PIPE_MAP_WRITE | PIPE_MAP_DISCARD_RANGE, &dst_xfer);
   if (dst_map) {
      util_copy_box(dst_map, format, row_stride, layer_stride, 0, 0, 0,
                    box->width, box->height, box->depth, src_map,
                    src_xfer->stride, src_xfer->layer_stride, 0, 0, 0);
      pipe_buffer_unmap(pctx, dst_xfer);
   }
   pctx->texture_unmap(pctx, src_xfer);
}
