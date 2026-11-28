/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019-2020 Collabora, Ltd.
 * Copyright 2014-2017 Broadcom
 * Copyright 2010 Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "agx_state.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_debug.h"
#include "asahi/compiler/agx3_compute_program.h"
#include "asahi/compiler/agx3_graphics_program.h"
#include "asahi/vulkan/hk_g17p_patch.h"
#include "asahi/compiler/agx3_reencode_cache.h"
#include "asahi/compiler/agx3_transcode.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/genxml/agx_pack.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/lib/agx3_code_container_core.h"
#include "asahi/lib/agx3_g17p_grid_helper_core.h"
#include "asahi/lib/agx_chip_caps.h"
#include "asahi/lib/agx_g17p_draw_state_upload.h"
#include "asahi/lib/agx_helpers.h"
#include "asahi/lib/agx_m5_va.h"
#include "asahi/lib/agx_ppp.h"
#include "asahi/lib/agx_ppp_hal200_graphics_core.h"
#include "asahi/lib/agx_ppp_hal200_raster_core.h"
#include "asahi/lib/agx_ppp_hal300_graphics_core.h"
#include "asahi/lib/agx_ppp_hal300_depth_bias_scissor_core.h"
#include "asahi/lib/agx_ppp_hal300_depth_stencil_core.h"
#include "asahi/lib/agx_ppp_viewport_payload_core.h"
#include "asahi/lib/agx_ppp_w_clamp_cull2_core.h"
#include "asahi/lib/agx_usc.h"
#include "asahi/lib/agx_vdm_barrier_core.h"
#include "asahi/lib/agx_vdm_index_list_core.h"
#include "asahi/lib/agx_vdm_state_core.h"
#include "asahi/lib/agx_vdm_state_update_core.h"
#include "asahi/lib/agx_vdm_ppp_state_core.h"
#include "asahi/lib/agx_vdm_stream_link_core.h"
#include "asahi/lib/agx_va_runtime_hole.h"
#include "asahi/lib/agx_vdm_hal200_direct_draw_core.h"
#include "asahi/libagx/compression.h"
#include "asahi/libagx/query.h"
#include "asahi/libagx/tessellator.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builtin_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
#include "compiler/nir/nir_serialize.h"
#include "compiler/shader_enums.h"
#include "gallium/auxiliary/nir/pipe_nir.h"
#include "gallium/auxiliary/draw/draw_context.h"
#include "gallium/auxiliary/nir/tgsi_to_nir.h"
#include "gallium/auxiliary/tgsi/tgsi_from_mesa.h"
#include "gallium/auxiliary/util/u_draw.h"
#include "gallium/auxiliary/util/u_framebuffer.h"
#include "gallium/auxiliary/util/u_helpers.h"
#include "gallium/auxiliary/util/u_prim_restart.h"
#include "gallium/auxiliary/util/u_viewport.h"
#include "pipe/p_context.h"
#include "pipe/p_defines.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "poly/geometry.h"
#include "asahi/genxml/agx3_cmdstream.h"
#include "util/u_atomic.h"
#include "asahi/lib/agx_g18g_template_policy.h"
#include "util/bitscan.h"
#include "util/bitset.h"
#include "util/blend.h"
#include "util/blob.h"
#include "util/compiler.h"
#include "util/format/u_format.h"
#include "util/format/u_formats.h"
#include "util/half_float.h"
#include "util/hash_table.h"
#include "util/mesa-blake3.h"
#include "util/macros.h"
#include "util/ralloc.h"
#include "util/u_inlines.h"
#include "util/u_index_modify.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "util/u_transfer.h"
#include "util/u_upload_mgr.h"
#include "agx_bg_eot.h"
#include "agx_bo.h"

/* EXP-G1b's native access::write argument is a resolved PBE pointer followed
 * by this 32-byte control record. This is the same captured record used by
 * Honeykrisp's G17P compute storage path. */
static const uint8_t agx_g17p_native_write_control[32] = {
   0x08, 0x0e, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
};

/* G18G's second state-0 pointer. Native run112 kick 19 launch 1 (Apple's
 * texture2d<uint, access::write> kernel) binds u0:u1 -> the image's PBE and
 * u2:u3 -> this 32-byte record, which sits right after the PBE in the
 * driver's heap. Its first quadword, 00000700080f0000, decodes as a Gen5
 * sampler with clamp-to-edge addressing (J873 run 164 notes, section 2); it is not G17P's write control. Native's second quadword,
 * 0000001000005800, is a native-process value (shifted left by four it is
 * the page holding kick 19's image) and is left zero here, as Mesa's G18G
 * sampler slots already are past their first quadword; if it is a
 * border-colour pointer, clamp-to-edge never reads it. */
static const uint8_t agx_g18g_native_write_state1[32] =
   AGX_G18G_WRITE_STATE1_INIT;

#include "agx_device.h"
#include "agx_disk_cache.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_lower_vbo.h"
#include "agx_tilebuffer.h"
#include "libagx.h"
#include "libagx_dgc.h"
#include "libagx_shaders.h"
#include "nir_builder.h"
#include "nir_builder_opcodes.h"
#include "nir_intrinsics.h"
#include "nir_intrinsics_indices.h"
#include "nir_xfb_info.h"
#include "pool.h"

/* G17P adds the render-context base when a VDM record follows a nested PPP
 * pointer. Gallium's batch pool already returns the full context VA, so make
 * only that nested pointer relative. The root VDM stream remains absolute. */
static bool
agx_gallium_vdm_compact(struct agx_batch *batch, uint64_t gpu,
                        uint64_t *compact)
{
   if (!batch || !compact)
      return false;

   struct agx_bo *payload_bo = NULL;
   util_dynarray_foreach(&batch->pool.bos, struct agx_bo *, bo) {
      if (gpu >= (*bo)->va->addr && gpu - (*bo)->va->addr < (*bo)->size) {
         payload_bo = *bo;
         break;
      }
   }
   if (!payload_bo)
      return false;

   const uint64_t alias =
      agx_batch_get_g17p_vdm_alias(batch, payload_bo, gpu);
   if (!agx_g17p_vdm_payload_compact(alias, compact))
      return false;

   return true;
}

static bool
agx_gallium_ppp_record(enum agx_chip chip, uint64_t address, uint32_t words,
                        struct agx_vdm_ppp_state_core *out)
{
   if (agx_chip_hal300_path(chip) && !agx_m5_vdm_compact(address, &address))
      return false;
   return agx_vdm_ppp_state_core_pack(chip, address, words, out);
}

static bool
agx_gallium_ppp_fini(struct agx_batch *batch, uint8_t **out,
                     struct agx_ppp_update *ppp)
{
   if (!batch || !ppp)
      return false;

   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   if (agx_chip_hal300_path(dev->chip)) {
      struct agx_ppp_update compact = *ppp;
      if (!agx_m5_vdm_compact(ppp->gpu_base, &compact.gpu_base))
         return false;
      return agx_ppp_fini(dev->chip, out, &compact);
   }
   if (!agx_chip_native_agx3(dev->chip))
      return agx_ppp_fini(dev->chip, out, ppp);

   struct agx_ppp_update compact_ppp = *ppp;
   if (!agx_gallium_vdm_compact(batch, ppp->gpu_base,
                                &compact_ppp.gpu_base))
      return false;

   return agx_ppp_fini(dev->chip, out, &compact_ppp);
}
void
agx_legalize_compression(struct agx_context *ctx, struct agx_resource *rsrc,
                         enum pipe_format format)
{
   if (rsrc && !ail_is_view_compatible(&rsrc->layout, format)) {
      agx_decompress(ctx, rsrc, "Incompatible formats");
   }
}

static void
agx_set_shader_images(struct pipe_context *pctx, mesa_shader_stage shader,
                      unsigned start_slot, unsigned count,
                      unsigned unbind_num_trailing_slots,
                      const struct pipe_image_view *iviews)
{
   struct agx_context *ctx = agx_context(pctx);
   ctx->stage[shader].dirty |= AGX_STAGE_DIRTY_IMAGE;

   /* Unbind start_slot...start_slot+count */
   if (!iviews) {
      for (int i = start_slot;
           i < start_slot + count + unbind_num_trailing_slots; i++) {
         pipe_resource_reference(&ctx->stage[shader].images[i].resource, NULL);
      }

      ctx->stage[shader].image_mask &=
         ~(BITFIELD64_MASK(count + unbind_num_trailing_slots) << start_slot);
      return;
   }

   /* Images writeable with pixel granularity are incompatible with
    * compression. Decompress if necessary.
    *
    * Driver-internal images are used by the compute blitter and are exempt
    * from these transitions, as it only uses compressed images when safe.
    *
    * We do this upfront because agx_decompress and agx_legalize_compression can
    * call set_shader_images internall.
    */
   for (int i = 0; i < count; i++) {
      const struct pipe_image_view *image = &iviews[i];
      struct agx_resource *rsrc = agx_resource(image->resource);

      if (rsrc && !(image->access & PIPE_IMAGE_ACCESS_DRIVER_INTERNAL)) {
         if (!rsrc->layout.writeable_image &&
             (image->shader_access & PIPE_IMAGE_ACCESS_WRITE)) {

            agx_decompress(ctx, rsrc, "Shader image");
         }

         /* Readable images may be compressed but are still subject to format
          * reinterpretation rules.
          */
         agx_legalize_compression(ctx, rsrc, image->format);

         if (image->shader_access & PIPE_IMAGE_ACCESS_WRITE)
            assert(rsrc->layout.writeable_image ||
                   agx_resource_is_shared_interchange(rsrc));
      }
   }

   /* Bind start_slot...start_slot+count */
   for (int i = 0; i < count; i++) {
      const struct pipe_image_view *image = &iviews[i];
      struct pipe_image_view *dst = &ctx->stage[shader].images[start_slot + i];

      if (!image->resource) {
         util_copy_image_view(dst, NULL);
         ctx->stage[shader].image_mask &= ~BITFIELD64_BIT(start_slot + i);
      } else if (image->resource->target == PIPE_BUFFER) {
         unsigned clamped_size = agx_buffer_range_in_resource(
            image->resource, image->u.buf.offset, image->u.buf.size);

         /* Unbind a buffer image view with no whole elements inside the BO so
          * neither descriptor packing nor batch tracking addresses past it. */
         if (!clamped_size ||
             !agx_texture_buffer_size_el(image->format, clamped_size)) {
            util_copy_image_view(dst, NULL);
            ctx->stage[shader].image_mask &= ~BITFIELD64_BIT(start_slot + i);
         } else {
            util_copy_image_view(dst, image);
            dst->u.buf.size = clamped_size;
            ctx->stage[shader].image_mask |= BITFIELD64_BIT(start_slot + i);
         }
      } else {
         util_copy_image_view(dst, image);
         ctx->stage[shader].image_mask |= BITFIELD64_BIT(start_slot + i);
      }
   }

   /* Unbind start_slot+count...start_slot+count+unbind_num_trailing_slots */
   for (int i = 0; i < unbind_num_trailing_slots; i++) {
      ctx->stage[shader].image_mask &= ~BITFIELD64_BIT(start_slot + count + i);
      util_copy_image_view(&ctx->stage[shader].images[start_slot + count + i],
                           NULL);
   }
}

static void
agx_set_shader_buffers(struct pipe_context *pctx, mesa_shader_stage shader,
                       unsigned start, unsigned count,
                       const struct pipe_shader_buffer *buffers,
                       unsigned writable_bitmask)
{
   struct agx_context *ctx = agx_context(pctx);

   util_set_shader_buffers_mask(ctx->stage[shader].ssbo,
                                &ctx->stage[shader].ssbo_mask, buffers, start,
                                count);

   /* Every reader of the binding (the shader's bounds, the batch's resource
    * tracking, the CPU vertex stages) sees only the part inside the buffer. */
   for (unsigned i = start; i < start + count; ++i) {
      struct pipe_shader_buffer *sb = &ctx->stage[shader].ssbo[i];

      if (sb->buffer) {
         sb->buffer_size = agx_buffer_range_in_resource(
            sb->buffer, sb->buffer_offset, sb->buffer_size);
      }
   }

   ctx->stage[shader].dirty |= AGX_STAGE_DIRTY_SSBO;
   ctx->stage[shader].ssbo_writable_mask &= ~(BITFIELD_MASK(count) << start);
   ctx->stage[shader].ssbo_writable_mask |= writable_bitmask << start;
}

static bool
agx_front_face_ccw(const struct agx_context *ctx)
{
   bool ccw = ctx->rast->base.front_ccw;
   if (ctx->in_tess && !ctx->gs)
      ccw ^= !ctx->stage[MESA_SHADER_TESS_EVAL].shader->tess.ccw;
   return ccw;
}

static void
agx_set_blend_color(struct pipe_context *pctx,
                    const struct pipe_blend_color *state)
{
   struct agx_context *ctx = agx_context(pctx);

   if (state)
      memcpy(&ctx->blend_color, state, sizeof(*state));

   ctx->dirty |= AGX_DIRTY_BLEND_COLOR;
}

static void
agx_set_patch_vertices(struct pipe_context *pctx, unsigned char n)
{
   struct agx_context *ctx = agx_context(pctx);
   ctx->patch_vertices = n;
}

static void
agx_set_tess_state(struct pipe_context *pctx,
                   const float default_outer_level[4],
                   const float default_inner_level[2])
{
   struct agx_context *ctx = agx_context(pctx);

   memcpy(ctx->default_outer_level, default_outer_level, 4 * sizeof(float));
   memcpy(ctx->default_inner_level, default_inner_level, 2 * sizeof(float));
}

static void *
agx_create_blend_state(struct pipe_context *ctx,
                       const struct pipe_blend_state *state)
{
   struct agx_blend *so = CALLOC_STRUCT(agx_blend);
   struct agx_blend_key *key = &so->key;

   key->alpha_to_coverage = state->alpha_to_coverage;
   key->alpha_to_one = state->alpha_to_one;

   key->logicop_enable = state->logicop_enable;
   key->logicop_func = state->logicop_func;

   for (unsigned i = 0; i < PIPE_MAX_COLOR_BUFS; ++i) {
      unsigned rti = state->independent_blend_enable ? i : 0;
      struct pipe_rt_blend_state rt = state->rt[rti];

      if (state->logicop_enable || !rt.blend_enable) {
         /* No blending, but we get the colour mask below */
         key->rt[i].mode = agx_pack_blend_standard(
            PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO,
            PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, PIPE_BLENDFACTOR_ZERO);
      } else {
         key->rt[i].mode = agx_pack_blend_standard(
            rt.rgb_func, rt.rgb_src_factor, rt.rgb_dst_factor, rt.alpha_func,
            rt.alpha_src_factor, rt.alpha_dst_factor);
      }

      key->rt[i].colormask = rt.colormask;

      if (rt.colormask)
         so->store |= (PIPE_CLEAR_COLOR0 << i);

      if (!state->logicop_enable && !state->alpha_to_coverage &&
          !rt.blend_enable)
         so->opaque_overwrite_mask[i] = rt.colormask & PIPE_MASK_RGBA;
   }

   return so;
}

static void
agx_bind_blend_state(struct pipe_context *pctx, void *cso)
{
   struct agx_context *ctx = agx_context(pctx);
   ctx->blend = cso;
   ctx->dirty |= AGX_DIRTY_BLEND;
}

static bool agx_keep_one_fs_output(nir_builder *b, nir_intrinsic_instr *intr,
                                   void *data);
static bool agx_drop_unread_varying(nir_builder *b, nir_intrinsic_instr *intr,
                                    void *data);
static uint64_t agx_g17g_mrt_inputs(struct agx_uncompiled_shader *so,
                                    unsigned rt);

static const enum agx_stencil_op agx_stencil_ops[PIPE_STENCIL_OP_INVERT + 1] = {
   [PIPE_STENCIL_OP_KEEP] = AGX_STENCIL_OP_KEEP,
   [PIPE_STENCIL_OP_ZERO] = AGX_STENCIL_OP_ZERO,
   [PIPE_STENCIL_OP_REPLACE] = AGX_STENCIL_OP_REPLACE,
   [PIPE_STENCIL_OP_INCR] = AGX_STENCIL_OP_INCR_SAT,
   [PIPE_STENCIL_OP_DECR] = AGX_STENCIL_OP_DECR_SAT,
   [PIPE_STENCIL_OP_INCR_WRAP] = AGX_STENCIL_OP_INCR_WRAP,
   [PIPE_STENCIL_OP_DECR_WRAP] = AGX_STENCIL_OP_DECR_WRAP,
   [PIPE_STENCIL_OP_INVERT] = AGX_STENCIL_OP_INVERT,
};

static void
agx_pack_stencil(struct agx_fragment_stencil_packed *out,
                 struct pipe_stencil_state st)
{
   if (st.enabled) {
      agx_pack(out, FRAGMENT_STENCIL, cfg) {
         cfg.compare = (enum agx_zs_func)st.func;
         cfg.write_mask = st.writemask;
         cfg.read_mask = st.valuemask;

         cfg.depth_pass = agx_stencil_ops[st.zpass_op];
         cfg.depth_fail = agx_stencil_ops[st.zfail_op];
         cfg.stencil_fail = agx_stencil_ops[st.fail_op];
      }
   } else {
      agx_pack(out, FRAGMENT_STENCIL, cfg) {
         cfg.compare = AGX_ZS_FUNC_ALWAYS;
         cfg.write_mask = 0xFF;
         cfg.read_mask = 0xFF;

         cfg.depth_pass = AGX_STENCIL_OP_KEEP;
         cfg.depth_fail = AGX_STENCIL_OP_KEEP;
         cfg.stencil_fail = AGX_STENCIL_OP_KEEP;
      }
   }
}

static void *
agx_create_zsa_state(struct pipe_context *ctx,
                     const struct pipe_depth_stencil_alpha_state *state)
{
   struct agx_zsa *so = CALLOC_STRUCT(agx_zsa);
   assert(!state->depth_bounds_test && "todo");

   so->base = *state;

   /* Handle the enable flag */
   enum pipe_compare_func depth_func =
      state->depth_enabled ? state->depth_func : PIPE_FUNC_ALWAYS;

   /* Z func can otherwise be used as-is */
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_NEVER == AGX_ZS_FUNC_NEVER);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_LESS == AGX_ZS_FUNC_LESS);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_EQUAL == AGX_ZS_FUNC_EQUAL);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_LEQUAL == AGX_ZS_FUNC_LEQUAL);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_GREATER == AGX_ZS_FUNC_GREATER);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_NOTEQUAL == AGX_ZS_FUNC_NOT_EQUAL);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_GEQUAL == AGX_ZS_FUNC_GEQUAL);
   STATIC_ASSERT((enum agx_zs_func)PIPE_FUNC_ALWAYS == AGX_ZS_FUNC_ALWAYS);

   agx_pack(&so->depth, FRAGMENT_FACE, cfg) {
      cfg.depth_function = (enum agx_zs_func)depth_func;
      cfg.disable_depth_write = !state->depth_writemask;
   }

   agx_pack_stencil(&so->front_stencil, state->stencil[0]);

   if (state->stencil[1].enabled) {
      agx_pack_stencil(&so->back_stencil, state->stencil[1]);
   } else {
      /* One sided stencil */
      so->back_stencil = so->front_stencil;
   }

   if (depth_func != PIPE_FUNC_NEVER && depth_func != PIPE_FUNC_ALWAYS)
      so->load |= PIPE_CLEAR_DEPTH;

   if (state->depth_writemask) {
      so->load |= PIPE_CLEAR_DEPTH;
      so->store |= PIPE_CLEAR_DEPTH;
   }

   if (state->stencil[0].enabled) {
      so->load |= PIPE_CLEAR_STENCIL; /* TODO: Optimize */
      so->store |= PIPE_CLEAR_STENCIL;
   }

   return so;
}

static void
agx_bind_zsa_state(struct pipe_context *pctx, void *cso)
{
   struct agx_context *ctx = agx_context(pctx);
   ctx->zs = cso;
   ctx->dirty |= AGX_DIRTY_ZS;
}

static enum agx_polygon_mode
agx_translate_polygon_mode(unsigned mode)
{
   switch (mode) {
   case PIPE_POLYGON_MODE_FILL:
      return AGX_POLYGON_MODE_FILL;
   case PIPE_POLYGON_MODE_POINT:
      return AGX_POLYGON_MODE_POINT;
   case PIPE_POLYGON_MODE_LINE:
      return AGX_POLYGON_MODE_LINE;
   default:
      UNREACHABLE("Unsupported polygon mode");
   }
}

static void *
agx_create_rs_state(struct pipe_context *ctx,
                    const struct pipe_rasterizer_state *cso)
{
   struct agx_rasterizer *so = CALLOC_STRUCT(agx_rasterizer);
   so->base = *cso;

   agx_pack(so->cull, CULL, cfg) {
      cfg.cull_front = cso->cull_face & PIPE_FACE_FRONT;
      cfg.cull_back = cso->cull_face & PIPE_FACE_BACK;
      cfg.depth_clip = cso->depth_clip_near;
      cfg.depth_clamp = !cso->depth_clip_near;
      cfg.flat_shading_vertex =
         cso->flatshade_first ? AGX_PPP_VERTEX_0 : AGX_PPP_VERTEX_2;
      cfg.rasterizer_discard = cso->rasterizer_discard;
   };

   /* Two-sided polygon mode doesn't seem to work on G13. Apple's OpenGL
    * implementation lowers to multiple draws with culling. Warn.
    */
   if (unlikely(cso->fill_front != cso->fill_back)) {
      agx_msg("Warning: Two-sided fill modes are unsupported, "
              "rendering may be incorrect.\n");
   }

   so->polygon_mode = agx_translate_polygon_mode(cso->fill_front);
   so->line_width = agx_pack_line_width(cso->line_width);
   so->depth_bias = util_get_offset(cso, cso->fill_front);

   return so;
}

static void
agx_bind_rasterizer_state(struct pipe_context *pctx, void *cso)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_rasterizer *so = cso;

   bool base_cso_changed = (cso == NULL) || (ctx->rast == NULL);

   /* Check if scissor or depth bias state has changed, since scissor/depth bias
    * enable is part of the rasterizer state but everything else needed for
    * scissors and depth bias is part of the scissor/depth bias arrays */
   bool scissor_zbias_changed = base_cso_changed ||
                                (ctx->rast->base.scissor != so->base.scissor) ||
                                (ctx->rast->depth_bias != so->depth_bias);

   ctx->dirty |= AGX_DIRTY_RS;

   if (scissor_zbias_changed)
      ctx->dirty |= AGX_DIRTY_SCISSOR_ZBIAS;

   if (base_cso_changed ||
       (ctx->rast->base.sprite_coord_mode != so->base.sprite_coord_mode))
      ctx->dirty |= AGX_DIRTY_SPRITE_COORD_MODE;

   ctx->rast = so;
}

static bool
has_edgeflags(struct agx_context *ctx, enum mesa_prim mode)
{
   return ctx->stage[MESA_SHADER_VERTEX].shader->info.has_edgeflags &&
          mode == MESA_PRIM_TRIANGLES &&
          (ctx->rast->base.fill_front != PIPE_POLYGON_MODE_FILL);
}

static enum agx_wrap
agx_wrap_from_pipe(enum pipe_tex_wrap in)
{
   switch (in) {
   case PIPE_TEX_WRAP_REPEAT:
      return AGX_WRAP_REPEAT;
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      return AGX_WRAP_CLAMP_TO_EDGE;
   case PIPE_TEX_WRAP_MIRROR_REPEAT:
      return AGX_WRAP_MIRRORED_REPEAT;
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      return AGX_WRAP_CLAMP_TO_BORDER;
   case PIPE_TEX_WRAP_CLAMP:
      return AGX_WRAP_CLAMP_GL;
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE:
      return AGX_WRAP_MIRRORED_CLAMP_TO_EDGE;
   default:
      UNREACHABLE("Invalid wrap mode");
   }
}

static enum agx_mip_filter
agx_mip_filter_from_pipe(enum pipe_tex_mipfilter in)
{
   switch (in) {
   case PIPE_TEX_MIPFILTER_NEAREST:
      return AGX_MIP_FILTER_NEAREST;
   case PIPE_TEX_MIPFILTER_LINEAR:
      return AGX_MIP_FILTER_LINEAR;
   case PIPE_TEX_MIPFILTER_NONE:
      return AGX_MIP_FILTER_NONE;
   }

   UNREACHABLE("Invalid mip filter");
}

static const enum agx_compare_func agx_compare_funcs[PIPE_FUNC_ALWAYS + 1] = {
   [PIPE_FUNC_NEVER] = AGX_COMPARE_FUNC_NEVER,
   [PIPE_FUNC_LESS] = AGX_COMPARE_FUNC_LESS,
   [PIPE_FUNC_EQUAL] = AGX_COMPARE_FUNC_EQUAL,
   [PIPE_FUNC_LEQUAL] = AGX_COMPARE_FUNC_LEQUAL,
   [PIPE_FUNC_GREATER] = AGX_COMPARE_FUNC_GREATER,
   [PIPE_FUNC_NOTEQUAL] = AGX_COMPARE_FUNC_NOT_EQUAL,
   [PIPE_FUNC_GEQUAL] = AGX_COMPARE_FUNC_GEQUAL,
   [PIPE_FUNC_ALWAYS] = AGX_COMPARE_FUNC_ALWAYS,
};

static const enum agx_filter agx_filters[] = {
   [PIPE_TEX_FILTER_LINEAR] = AGX_FILTER_LINEAR,
   [PIPE_TEX_FILTER_NEAREST] = AGX_FILTER_NEAREST,
};

static enum pipe_format
fixup_border_zs(enum pipe_format orig, union pipe_color_union *c)
{
   switch (orig) {
   case PIPE_FORMAT_Z24_UNORM_S8_UINT:
   case PIPE_FORMAT_Z24X8_UNORM:
      /* Z24 is internally promoted to Z32F via transfer_helper. These formats
       * are normalized so should get clamped, but Z32F does not get clamped, so
       * we clamp here.
       */
      c->f[0] = SATURATE(c->f[0]);
      return PIPE_FORMAT_Z32_FLOAT;

   case PIPE_FORMAT_X24S8_UINT:
   case PIPE_FORMAT_X32_S8X24_UINT:
      /* Separate stencil is internally promoted */
      return PIPE_FORMAT_S8_UINT;

   default:
      return orig;
   }
}

static void *
agx_create_sampler_state(struct pipe_context *pctx,
                         const struct pipe_sampler_state *state)
{
   struct agx_sampler_state *so = CALLOC_STRUCT(agx_sampler_state);
   so->base = *state;

   so->border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   const bool uses_border =
      state->wrap_s == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
      state->wrap_t == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
      state->wrap_r == PIPE_TEX_WRAP_CLAMP_TO_BORDER ||
      state->wrap_s == PIPE_TEX_WRAP_CLAMP ||
      state->wrap_t == PIPE_TEX_WRAP_CLAMP ||
      state->wrap_r == PIPE_TEX_WRAP_CLAMP;

   if (uses_border && state->border_color_format != PIPE_FORMAT_NONE) {
      union pipe_color_union border = state->border_color;
      enum pipe_format format =
         fixup_border_zs(state->border_color_format, &border);
      const bool integer = util_format_is_pure_integer(format);

      /* A fixed-point depth comparison uses the border depth clamped to
       * [0, 1]; a floating-point one uses it unclamped. Z24 is clamped
       * above, before its promotion to Z32F. */
      const struct util_format_description *zdesc =
         util_format_description(format);
      if (state->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE &&
          util_format_has_depth(zdesc) &&
          zdesc->channel[0].type != UTIL_FORMAT_TYPE_FLOAT)
         border.f[0] = SATURATE(border.f[0]);

      if (!agx_gallium_classify_standard_border(
             border.ui, integer, &so->border_colour)) {
         so->uses_custom_border = true;
         so->border_colour = AGX_BORDER_COLOUR_CUSTOM;
         agx_pack_border(&so->border, border.ui, format);
         memcpy(so->custom_border, border.ui, sizeof(so->custom_border));
      }
   }

   /* We report a max texture LOD bias of 16, so clamp appropriately */
   float lod_bias = CLAMP(state->lod_bias, -16.0, 16.0);
   so->lod_bias_as_fp16 = _mesa_float_to_half(lod_bias);

   agx_pack(&so->desc, SAMPLER, cfg) {
      cfg.minimum_lod = state->min_lod;
      cfg.maximum_lod = state->max_lod;
      cfg.maximum_anisotropy =
         util_next_power_of_two(MAX2(state->max_anisotropy, 1));
      cfg.magnify = agx_filters[state->mag_img_filter];
      cfg.minify = agx_filters[state->min_img_filter];
      cfg.mip_filter = agx_mip_filter_from_pipe(state->min_mip_filter);
      cfg.wrap_s = agx_wrap_from_pipe(state->wrap_s);
      cfg.wrap_t = agx_wrap_from_pipe(state->wrap_t);
      cfg.wrap_r = agx_wrap_from_pipe(state->wrap_r);
      cfg.pixel_coordinates = state->unnormalized_coords;
      cfg.compare_func = agx_compare_funcs[state->compare_func];
      cfg.compare_enable = state->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;
      cfg.seamful_cube_maps = !state->seamless_cube_map;
      cfg.border_colour = so->border_colour;
   }

   memcpy(&so->desc_without_custom_border, &so->desc, sizeof(so->desc));

   if (so->uses_custom_border) {
      /* Neutralize the bindless-safe descriptor. XXX: This is a hack. */
      so->desc_without_custom_border.opaque[1] &= ~(1u << 23);
   }

   return so;
}

static void
agx_delete_sampler_state(struct pipe_context *ctx, void *state)
{
   struct agx_sampler_state *so = state;
   FREE(so);
}

static void
agx_bind_sampler_states(struct pipe_context *pctx, mesa_shader_stage shader,
                        unsigned start, unsigned count, void **states)
{
   struct agx_context *ctx = agx_context(pctx);

   ctx->stage[shader].dirty |= AGX_STAGE_DIRTY_SAMPLER;
   if (shader == MESA_SHADER_VERTEX)
      ctx->dirty |= AGX_DIRTY_VS_PROG;
   else if (shader == MESA_SHADER_FRAGMENT)
      ctx->dirty |= AGX_DIRTY_FS_PROG;

   for (unsigned i = 0; i < count; i++) {
      unsigned p = start + i;
      ctx->stage[shader].samplers[p] = states ? states[i] : NULL;
      if (ctx->stage[shader].samplers[p])
         ctx->stage[shader].valid_samplers |= BITFIELD_BIT(p);
      else
         ctx->stage[shader].valid_samplers &= ~BITFIELD_BIT(p);
   }

   ctx->stage[shader].sampler_count =
      util_last_bit(ctx->stage[shader].valid_samplers);

   /* Recalculate whether we need custom borders */
   ctx->stage[shader].custom_borders = false;

   u_foreach_bit(i, ctx->stage[shader].valid_samplers) {
      if (ctx->stage[shader].samplers[i]->uses_custom_border)
         ctx->stage[shader].custom_borders = true;
   }
}

static enum agx_texture_dimension
agx_translate_tex_dim(enum pipe_texture_target dim, unsigned samples)
{
   assert(samples >= 1);

   switch (dim) {
   case PIPE_BUFFER:
   case PIPE_TEXTURE_1D:
      /* Lowered to 2D */
      assert(samples == 1);
      return AGX_TEXTURE_DIMENSION_2D;

   case PIPE_TEXTURE_RECT:
   case PIPE_TEXTURE_2D:
      return samples > 1 ? AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED
                         : AGX_TEXTURE_DIMENSION_2D;

   case PIPE_TEXTURE_1D_ARRAY:
      assert(samples == 1);
      /* Lowered to 2D */
      FALLTHROUGH;
   case PIPE_TEXTURE_2D_ARRAY:
      return samples > 1 ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                         : AGX_TEXTURE_DIMENSION_2D_ARRAY;

   case PIPE_TEXTURE_3D:
      assert(samples == 1);
      return AGX_TEXTURE_DIMENSION_3D;

   case PIPE_TEXTURE_CUBE:
      assert(samples == 1);
      return AGX_TEXTURE_DIMENSION_CUBE;

   case PIPE_TEXTURE_CUBE_ARRAY:
      assert(samples == 1);
      return AGX_TEXTURE_DIMENSION_CUBE_ARRAY;

   default:
      UNREACHABLE("Unsupported texture dimension");
   }
}

static bool
target_is_cube(enum pipe_texture_target target)
{
   return target == PIPE_TEXTURE_CUBE || target == PIPE_TEXTURE_CUBE_ARRAY;
}

static void NORETURN
agx_native_descriptor_rejected(struct agx_device *dev, const char *operation)
{
   mesa_loge("asahi: %s has no proven native descriptor encoding on %s",
             operation, agx_chip_name(dev->chip));
   /* Never submit a zero or partially initialized native descriptor. */
   abort();
}

/* A chip-level geometry that the driver does not know at all. This cannot
 * happen for a recognised AGX3 identity and is not recoverable per batch. */
static void
agx_native_descriptor_unsupported(struct agx_device *dev, const char *operation)
{
   mesa_loge("asahi: %s has no proven native descriptor encoding on %s",
             operation, agx_chip_name(dev->chip));
   assert(!agx_chip_driver_supported(dev->chip));
   UNREACHABLE("native descriptor encoding rejected");
}

/* A descriptor or state that has no proven native encoding on this chip yet.
 * Nothing from the batch may reach the hardware: a zero descriptor is not a
 * valid substitute for rejected state, so the batch is flagged and dropped at
 * submission instead of being encoded. The affected draws never execute; the
 * application observes wrong output, and a conformance run records the
 * failure against the case that hit the gap. Reported once per operation. */
static void
agx_native_descriptor_reject_batch(struct agx_batch *batch,
                                   const char *operation)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);

   if (!batch->native_rejected)
      mesa_loge("asahi: %s has no proven native descriptor encoding on %s",
                operation, agx_chip_name(dev->chip));
   batch->native_rejected = true;
}

/* The M5 Ioto generators take the chip they emit code for. G18G runs them as
 * G17G and re-encodes their output; see agx_g18g_reencode_program. */
static inline enum agx_chip
agx_ioto_chip(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G18G ? AGX_CHIP_G17G : dev->chip;
}

/* The CDM barrier that ends an M5-profile compute dispatch: G17G's captured
 * flags-5 word, but on G18G the flag-free word, matching this Gallium path's
 * own capture.  The flags-5 strong flush is grounded for G18G too
 * (agx3_cdm_barrier_flags_grounded); Vulkan's dispatch append uses it. */
static inline uint8_t
agx_m5_dispatch_barrier_flags(const struct agx_device *dev)
{
   return dev->chip == AGX_CHIP_G18G ? 0 : 5;
}

/* Can the CPU path (agx_swtnl.c) tessellate? The draw module tessellates only
 * through LLVM. */
static bool
agx_swtnl_can_tessellate(void)
{
   /* Draws on any context may ask first, so keep the cached answer atomic.
    * Racing first calls compute the same value. */
   static int can = -1;
   int value = p_atomic_read_relaxed(&can);
   if (unlikely(value < 0)) {
      value = draw_get_option_use_llvm();
      p_atomic_set(&can, value);
   }
   return value;
}

/* M5 tessellates on the GPU (agx_draw_patches), as hk does, when the CPU path
 * cannot: in a build without LLVM the draw module drops every patch. With
 * LLVM the CPU path is the qualified one. The GPU path still loses some
 * shapes, among them tessellation feeding a geometry shader, which M5 does
 * not rasterize natively yet.
 */
bool
agx_m5_gpu_tess(struct agx_context *ctx)
{
   if (!agx_chip_g17_m5(agx_device(ctx->base.screen)->chip))
      return false;

   return !agx_swtnl_can_tessellate();
}

/* Whether the M5 generators' programs are re-encoded as agx3d for dev. The
 * generators behind it are G18G's own (agx3_g18g_graphics_program_*() and the
 * G18 TPS envelope, which still key on AGX_CHIP_G18G): a later agx3d chip
 * needs its own before it takes these paths. */
static inline bool
agx_reencodes_agx3d(const struct agx_device *dev)
{
   return agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX3D;
}

/* Room for a program's agx3d re-encoding: twice the G17G bound, which covers
 * every witness (the largest growth is f32mul's 8-byte form becoming 16).
 * The callers still refuse a program longer than what they allocated. */
static inline size_t
agx_ioto_capacity(const struct agx_device *dev, size_t g17g_capacity)
{
   return agx_reencodes_agx3d(dev) ? 2 * g17g_capacity : g17g_capacity;
}

/*
 * G18G re-encodings go through a cache (agx3_reencode_cache.c).
 * ASAHI_MESA_DEBUG=verify checks every hit against a full re-encoding.
 */
static struct agx3_reencode_cache agx_g18g_reencode_cache =
   AGX3_REENCODE_CACHE_INITIALIZER;

/* Log a program the transcoder refused, and dump it under FAULTTRACE */
static void
agx_g18g_reencode_refused(const struct agx_device *dev,
                          const struct agx3_g17p_graphics_program *program,
                          const char *why, unsigned at)
{
   mesa_loge("asahi: G18G Ioto program not re-encoded: %s at +0x%x", why, at);
   if (dev->debug & AGX_DBG_FAULTTRACE) {
      fprintf(stderr, "G18G_IOTO_REFUSED length=%u data=", program->length);
      for (unsigned i = 0; i < program->length; ++i)
         fprintf(stderr, "%02x", program->bytes[i]);
      fputc('\n', stderr);
   }
}

/* G18G: re-encode a generated M5 Ioto program as agx3d in place. The G17G
 * programs embed no address of their own (agx_ioto_g17g_* only validate
 * program_va), so the longer agx3d text stays position-independent. */
static bool
agx_g18g_reencode_program(const struct agx_device *dev,
                          struct agx3_g17p_graphics_program *program,
                          bool api_program, unsigned shader_gprs)
{
   if (dev->chip != AGX_CHIP_G18G)
      return true;

   const struct agx3_transcode_ioto ioto = {
      .api_program = api_program,
      .shader_gprs = shader_gprs,
   };
   const char *why = NULL;
   unsigned at = 0;

   const bool verify = dev->debug & AGX_DBG_VERIFY;
   const unsigned length = program->length;
   struct agx3_g17p_graphics_program agx3c;

   if (verify)
      agx3c = *program;

   switch (agx3_reencode_cached(&agx_g18g_reencode_cache, program, &ioto, &why,
                                &at)) {
   case AGX3_REENCODE_HIT:
      if (verify && (!agx3_reencode_program(&agx3c, &ioto, &why, &at) ||
                     agx3c.length != program->length ||
                     memcmp(agx3c.bytes, program->bytes, program->length)))
         mesa_loge("asahi: G18G re-encoding cache hit differs from "
                   "a full re-encoding (length %u)", length);
      return true;
   case AGX3_REENCODE_MISS:
      return true;
   case AGX3_REENCODE_REFUSED:
      agx_g18g_reencode_refused(dev, program, why, at);
      return false;
   case AGX3_REENCODE_UNCACHEABLE:
      break;
   }

   if (agx3_reencode_program(program, &ioto, &why, &at))
      return true;
   agx_g18g_reencode_refused(dev, program, why, at);
   return false;
}

static struct asahi_bg_eot
agx_reject_g17g_tile(struct agx_batch *batch, const char *operation)
{
   agx_native_descriptor_reject_batch(batch, operation);
   return (struct asahi_bg_eot){0};
}

static enum agx_channel
agx_rgb32_buffer_fallback_channel(enum agx_channel channel)
{
   if (channel == AGX_CHANNEL_G || channel == AGX_CHANNEL_B)
      return AGX_CHANNEL_0;
   if (channel == AGX_CHANNEL_A)
      return AGX_CHANNEL_1;
   return channel;
}

static bool
agx_pack_texture(enum agx_descriptor_generation descriptor_generation,
                 struct agx_gallium_texture_descriptor *out,
                 struct agx_resource *rsrc,
                 enum pipe_format format /* override */,
                 const struct pipe_sampler_view *state)
{
   const struct util_format_description *desc = util_format_description(format);

   assert(ail_is_valid_pixel_format(format));

   uint8_t format_swizzle[4] = {
      desc->swizzle[0],
      desc->swizzle[1],
      desc->swizzle[2],
      desc->swizzle[3],
   };

   if (util_format_is_depth_or_stencil(format)) {
      assert(!util_format_is_depth_and_stencil(format) &&
             "separate stencil always used");

      /* Broadcast depth and stencil */
      format_swizzle[0] = 0;
      format_swizzle[1] = 0;
      format_swizzle[2] = 0;
      format_swizzle[3] = 0;
   }

   /* We only have a single swizzle for the user swizzle and the format fixup,
    * so compose them now. */
   uint8_t out_swizzle[4];
   uint8_t view_swizzle[4] = {state->swizzle_r, state->swizzle_g,
                              state->swizzle_b, state->swizzle_a};

   util_format_compose_swizzles(format_swizzle, view_swizzle, out_swizzle);

   unsigned first_layer =
      (state->target == PIPE_BUFFER) ? 0 : state->u.tex.first_layer;

   /* First construct generation-independent semantics.  The native packer
    * selects Gen7/8, Gen9, or Gen10 without allowing a legacy C type to choose
    * the heap stride.
    */
   struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
   {
      cfg.dimension = agx_translate_tex_dim(state->target,
                                            util_res_sample_count(&rsrc->base.b));
      cfg.layout = agx_translate_layout(rsrc->layout.tiling);
      cfg.channels = ail_pixel_format[format].channels;
      cfg.type = ail_pixel_format[format].type;
      cfg.swizzle_r = agx_channel_from_pipe(out_swizzle[0]);
      cfg.swizzle_g = agx_channel_from_pipe(out_swizzle[1]);
      cfg.swizzle_b = agx_channel_from_pipe(out_swizzle[2]);
      cfg.swizzle_a = agx_channel_from_pipe(out_swizzle[3]);

      if (state->target == PIPE_BUFFER) {
         unsigned size_el =
            agx_texture_buffer_size_el(format, state->u.buf.size);

         /* Use a 2D texture to increase the maximum size */
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(size_el, cfg.width);
         cfg.first_level = cfg.last_level = 0;
         cfg.buffer_size_sw = size_el;
         cfg.buffer_offset_sw = 0;
      } else {
         cfg.width = rsrc->base.b.width0;
         cfg.height = rsrc->base.b.height0;
         cfg.first_level = state->u.tex.first_level;
         cfg.last_level = state->u.tex.last_level;
      }

      cfg.srgb = (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB);
      cfg.unk_mipmapped = rsrc->mipmapped;
      cfg.srgb_2_channel = cfg.srgb && util_format_colormask(desc) == 0x3;

      if (rsrc->layout.compressed &&
          !ail_is_view_compatible(&rsrc->layout, format))
         return false;

      cfg.compressed = rsrc->layout.compressed;
      cfg.extended = cfg.compressed;

      cfg.address = agx_map_texture_gpu(rsrc, first_layer);

      if (state->target == PIPE_BUFFER)
         cfg.address += state->u.buf.offset;

      if (rsrc->layout.compressed) {
         cfg.acceleration_buffer =
            agx_map_gpu(rsrc) + rsrc->layout.metadata_offset_B +
            (first_layer * rsrc->layout.compression_layer_stride_B);
      }

      /* Native Gen9 cube descriptors require one complete six-face range.
       * The shared extended-range address is filled for all applicable
       * dimensions below, after the selected layer count is known. */
      if (agx_chip_native_agx3(agx_device(rsrc->base.b.screen)->chip) &&
          state->target == PIPE_TEXTURE_CUBE) {
         const unsigned faces =
            state->u.tex.last_layer - state->u.tex.first_layer + 1;

         /* The recovered descriptor has one uncompressed six-face range.
          * Compression metadata or a different range would alias the same
          * secondary word, so reject those shapes instead of guessing. */
         if (rsrc->layout.compressed || faces != 6)
            return false;
      }

      if (state->target == PIPE_TEXTURE_3D) {
         cfg.depth = rsrc->base.b.depth0;
      } else if (state->target == PIPE_BUFFER) {
         cfg.depth = 1;
      } else {
         unsigned layers =
            state->u.tex.last_layer - state->u.tex.first_layer + 1;

         if (target_is_cube(state->target))
            layers /= 6;

         if (rsrc->layout.tiling == AIL_TILING_LINEAR &&
             (state->target == PIPE_TEXTURE_1D_ARRAY ||
              state->target == PIPE_TEXTURE_2D_ARRAY)) {

            cfg.depth_linear = layers;
            cfg.layer_stride_linear = (rsrc->layout.layer_stride_B - 0x80);
            cfg.extended = true;
         } else {
            assert((rsrc->layout.tiling != AIL_TILING_LINEAR) || (layers == 1));
            cfg.depth = layers;
         }
      }

      if (rsrc->base.b.nr_samples > 1)
         cfg.samples = agx_translate_sample_count(rsrc->base.b.nr_samples);

      if (state->target == PIPE_BUFFER) {
         cfg.stride = (cfg.width * util_format_get_blocksize(format)) - 16;
      } else if (rsrc->layout.tiling == AIL_TILING_LINEAR) {
         cfg.stride = ail_get_linear_stride_B(&rsrc->layout, 0) - 16;
      } else if (rsrc->layout.tiling == AIL_TILING_INTERCHANGE) {
         /* Macroblocks per row minus one in the linear stride field. */
         cfg.stride = (rsrc->layout.stride_el[0] / 16 - 1) << 4;
      } else {
         cfg.page_aligned_layers = rsrc->layout.page_aligned_layers;
      }

      /* A18 Gen9 publishes a nonlinear texture's exclusive end VA whenever
       * the descriptor is extended: one-level arrays/3D/cubes as well as
       * mipmapped 2D.  The exact captures advance by the complete selected
       * layer stride (or by the complete 3D miptree), independently of whether
       * the layers themselves are tightly or page aligned.  Compression keeps
       * using the same word for metadata instead. */
      const bool gen9_range =
         agx_chip_native_agx3(agx_device(rsrc->base.b.screen)->chip) &&
         !rsrc->layout.compressed &&
         rsrc->layout.tiling != AIL_TILING_LINEAR &&
         (state->target == PIPE_TEXTURE_2D_ARRAY ||
          state->target == PIPE_TEXTURE_3D ||
          state->target == PIPE_TEXTURE_CUBE ||
          rsrc->mipmapped ||
          state->u.tex.first_level != 0 || state->u.tex.last_level != 0);
      if (gen9_range) {
         unsigned layers =
            state->u.tex.last_layer - state->u.tex.first_layer + 1;
         const uint64_t range_B =
            state->target == PIPE_TEXTURE_3D
               ? rsrc->layout.size_B
               : rsrc->layout.layer_stride_B * layers;

         cfg.extended = true;
         cfg.acceleration_buffer = cfg.address + range_B;
      }
   }

   if (state->target == PIPE_BUFFER &&
       descriptor_generation == AGX_DESCRIPTOR_GENERATION_9 &&
       cfg.channels == AGX_CHANNELS_R32G32B32_EMULATED) {
      cfg.channels = AGX_CHANNELS_R32;
      cfg.stride = cfg.width * sizeof(uint32_t) - 16;
      cfg.swizzle_r = agx_rgb32_buffer_fallback_channel(cfg.swizzle_r);
      cfg.swizzle_g = agx_rgb32_buffer_fallback_channel(cfg.swizzle_g);
      cfg.swizzle_b = agx_rgb32_buffer_fallback_channel(cfg.swizzle_b);
      cfg.swizzle_a = agx_rgb32_buffer_fallback_channel(cfg.swizzle_a);
   }

   const bool packed =
      agx_gallium_pack_native_texture(descriptor_generation, out, &cfg);

   /* The buffer texture's element count for the shader's size query and
    * bounds check (agx_nir_lower_texture.c), in a word Gen10 leaves free. */
   if (packed && agx_chip_g17_m5(agx_device(rsrc->base.b.screen)->chip) && descriptor_generation == AGX_DESCRIPTOR_GENERATION_10 &&
       state->target == PIPE_BUFFER)
      out->opaque[AGX_GEN10_BUFFER_TEXTURE_SIZE_OFFSET / 4] =
         cfg.buffer_size_sw;

   /* An array texture's packed descriptor, to diff against the captured
    * Apple one byte for byte. A mipmapped array samples layer zero for every
    * layer, and every field examined so far matches the witness, so print
    * the whole record rather than the fields we already believe in. */
   if (packed && (agx_device(rsrc->base.b.screen)->debug & AGX_DBG_FAULTTRACE) &&
       cfg.depth > 1) {
      fprintf(stderr,
              "M5_TEXTURE %ux%ux%u levels=%u..%u mip=%u pal=%u ext=%u "
              "stride_B=%" PRIu64 " accel=%#" PRIx64 " ",
              cfg.width, cfg.height, cfg.depth, cfg.first_level,
              cfg.last_level, cfg.unk_mipmapped, cfg.page_aligned_layers,
              cfg.extended, (uint64_t)rsrc->layout.layer_stride_B,
              (uint64_t)cfg.acceleration_buffer);
      const uint8_t *bytes = (const uint8_t *)out->opaque;
      for (unsigned i = 0; i < 32; ++i)
         fprintf(stderr, "%02x", bytes[i]);
      fprintf(stderr, "\n");
   }
   if (packed && descriptor_generation == AGX_DESCRIPTOR_GENERATION_9 &&
       (agx_device(rsrc->base.b.screen)->debug &
        (AGX_DBG_TRACE | AGX_DBG_ASYNCTRACE))) {
      fprintf(stderr,
              "asahi: Gen9 texture %s %ux%u levels=%u..%u mipmapped=%u "
              "address=0x%" PRIx64 " secondary=0x%" PRIx64
              " size=0x%" PRIx64 " layer-stride=0x%" PRIx64
              " page-aligned=%u extended=%u compressed=%u words:",
              util_format_short_name(format), cfg.width, cfg.height,
              cfg.first_level, cfg.last_level, cfg.unk_mipmapped, cfg.address,
              cfg.acceleration_buffer, (uint64_t)rsrc->layout.size_B,
              (uint64_t)rsrc->layout.layer_stride_B, cfg.page_aligned_layers,
              cfg.extended, cfg.compressed);
      for (unsigned i = 0; i <
           agx_gallium_texture_descriptor_size(descriptor_generation) / 4; ++i)
         fprintf(stderr, " %08x", out->opaque[i]);
      fprintf(stderr, "\n");
   }
   return packed;
}

static struct pipe_sampler_view *
agx_create_sampler_view(struct pipe_context *pctx,
                        struct pipe_resource *orig_texture,
                        const struct pipe_sampler_view *state)
{
   struct agx_resource *rsrc = agx_resource(orig_texture);
   struct agx_sampler_view *so = CALLOC_STRUCT(agx_sampler_view);

   if (!so)
      return NULL;

   struct pipe_resource *texture = orig_texture;
   enum pipe_format format = state->format;

   const struct util_format_description *desc = util_format_description(format);

   /* Separate stencil always used on G13, so we need to fix up for Z32S8 */
   if (util_format_has_stencil(desc) && rsrc->separate_stencil) {
      if (util_format_has_depth(desc)) {
         /* Reinterpret as the depth-only part */
         format = util_format_get_depth_only(format);
      } else {
         /* Use the stencil-only-part */
         rsrc = rsrc->separate_stencil;
         texture = &rsrc->base.b;
         format = texture->format;
      }
   }

   /* Save off the resource that we actually use, with the stencil fixed up.
    * Defer agx_legalize_compression() to agx_set_sampler_views() because
    * u_threaded_context calls create_sampler_view directly on the frontend
    * thread where decompressing/flushing batches would race with the driver.
    */
   so->rsrc = rsrc;
   so->format = format;

   so->base = *state;
   so->base.texture = NULL;
   pipe_resource_reference(&so->base.texture, orig_texture);
   pipe_reference_init(&so->base.reference, 1);
   so->base.context = pctx;
   return &so->base;
}

static void
agx_set_sampler_views(struct pipe_context *pctx, mesa_shader_stage shader,
                      unsigned start, unsigned count,
                      unsigned unbind_num_trailing_slots,
                      struct pipe_sampler_view **views)
{
   struct agx_context *ctx = agx_context(pctx);
   unsigned new_nr = 0;
   unsigned i;

   assert(start == 0);

   if (!views)
      count = 0;

   for (i = 0; i < count; ++i) {
      if (views[i]) {
         struct agx_sampler_view *view = (struct agx_sampler_view *)views[i];
         agx_legalize_compression(ctx, view->rsrc, view->format);

         /* Gallium AGX3 does not use compressed cube descriptors (a 6-layer
          * compressed 2D array can be reinterpreted as a cube view via
          * glTextureView). Decompress before binding a cube sampler view. */
         if (view->rsrc->layout.compressed &&
             agx_gallium_descriptor_generation_is_agx3(
                agx_chip_descriptor_generation(agx_device(pctx->screen)->chip)) &&
             target_is_cube(view->base.target))
            agx_decompress(ctx, view->rsrc, "Compressed cube view");
      }
      pipe_sampler_view_reference(
         (struct pipe_sampler_view **)&ctx->stage[shader].textures[i],
         views[i]);
   }

   for (; i < count + unbind_num_trailing_slots; i++) {
      pipe_sampler_view_reference(
         (struct pipe_sampler_view **)&ctx->stage[shader].textures[i], NULL);
   }

   for (unsigned t = 0; t < MAX2(ctx->stage[shader].texture_count, count);
        ++t) {
      if (ctx->stage[shader].textures[t])
         new_nr = t + 1;
   }

   ctx->stage[shader].texture_count = new_nr;
   if (new_nr)
      ctx->stages_with_textures |= BITFIELD_BIT(shader);
   else
      ctx->stages_with_textures &= ~BITFIELD_BIT(shader);
   ctx->stage[shader].dirty |= AGX_STAGE_DIRTY_IMAGE;
}

static void
agx_sampler_view_destroy(struct pipe_context *ctx,
                         struct pipe_sampler_view *pview)
{
   struct agx_sampler_view *view = (struct agx_sampler_view *)pview;
   pipe_resource_reference(&view->base.texture, NULL);
   FREE(view);
}

static void
agx_set_polygon_stipple(struct pipe_context *pctx,
                        const struct pipe_poly_stipple *state)
{
   struct agx_context *ctx = agx_context(pctx);

   memcpy(ctx->poly_stipple, state->stipple, sizeof(ctx->poly_stipple));
   ctx->dirty |= AGX_DIRTY_POLY_STIPPLE;
}

static void
agx_set_sample_mask(struct pipe_context *pipe, unsigned sample_mask)
{
   struct agx_context *ctx = agx_context(pipe);

   /* Optimization: At most MSAA 4x supported, so normalize to avoid pointless
    * dirtying switching between e.g. 0xFFFF and 0xFFFFFFFF masks.
    */
   unsigned new_mask = sample_mask & BITFIELD_MASK(4);

   if (ctx->sample_mask != new_mask) {
      ctx->sample_mask = new_mask;
      ctx->dirty |= AGX_DIRTY_SAMPLE_MASK;
   }
}

static void
agx_set_scissor_states(struct pipe_context *pctx, unsigned start_slot,
                       unsigned num_scissors,
                       const struct pipe_scissor_state *scissor)
{
   struct agx_context *ctx = agx_context(pctx);

   STATIC_ASSERT(sizeof(ctx->scissor[0]) == sizeof(*scissor));
   assert(start_slot + num_scissors <= AGX_MAX_VIEWPORTS);

   memcpy(&ctx->scissor[start_slot], scissor, sizeof(*scissor) * num_scissors);
   ctx->dirty |= AGX_DIRTY_SCISSOR_ZBIAS;
}

static void
agx_set_stencil_ref(struct pipe_context *pctx,
                    const struct pipe_stencil_ref state)
{
   struct agx_context *ctx = agx_context(pctx);
   ctx->stencil_ref = state;
   ctx->dirty |= AGX_DIRTY_STENCIL_REF;
}

static void
agx_set_viewport_states(struct pipe_context *pctx, unsigned start_slot,
                        unsigned num_viewports,
                        const struct pipe_viewport_state *vp)
{
   struct agx_context *ctx = agx_context(pctx);

   STATIC_ASSERT(sizeof(ctx->viewport[0]) == sizeof(*vp));
   assert(start_slot + num_viewports <= AGX_MAX_VIEWPORTS);

   memcpy(&ctx->viewport[start_slot], vp, sizeof(*vp) * num_viewports);
   ctx->dirty |= AGX_DIRTY_VIEWPORT;
}

static void
agx_get_scissor_extents(const struct pipe_viewport_state *vp,
                        const struct pipe_scissor_state *ss,
                        const struct pipe_framebuffer_state *fb, unsigned *minx,
                        unsigned *miny, unsigned *maxx, unsigned *maxy)
{
   float trans_x = vp->translate[0], trans_y = vp->translate[1];
   float abs_scale_x = fabsf(vp->scale[0]), abs_scale_y = fabsf(vp->scale[1]);

   /* Calculate the extent of the viewport. Note if a particular dimension of
    * the viewport is an odd number of pixels, both the translate and the scale
    * will have a fractional part of 0.5, so adding and subtracting them yields
    * an integer. Therefore we don't need to round explicitly */
   *minx = CLAMP((int)(trans_x - abs_scale_x), 0, fb->width);
   *miny = CLAMP((int)(trans_y - abs_scale_y), 0, fb->height);
   *maxx = CLAMP((int)(trans_x + abs_scale_x), 0, fb->width);
   *maxy = CLAMP((int)(trans_y + abs_scale_y), 0, fb->height);

   if (ss) {
      *minx = MAX2(ss->minx, *minx);
      *miny = MAX2(ss->miny, *miny);
      *maxx = MIN2(ss->maxx, *maxx);
      *maxy = MIN2(ss->maxy, *maxy);
   }
}

static void
agx_upload_viewport_scissor(struct agx_pool *pool, struct agx_batch *batch,
                            uint8_t **out, const struct pipe_viewport_state *vp,
                            const struct pipe_scissor_state *ss,
                            bool clip_halfz, bool multi_viewport)
{
   /* Number of viewports/scissors isn't precisely determinable in Gallium, so
    * just key off whether we can write to anything other than viewport 0. This
    * could be tuned in the future.
    */
   unsigned count = multi_viewport ? AGX_MAX_VIEWPORTS : 1;

   /* Allocate scissor descriptors */
   unsigned index = batch->scissor.size / AGX_SCISSOR_LENGTH;
   struct agx_scissor_packed *scissors =
      util_dynarray_grow_bytes(&batch->scissor, count, AGX_SCISSOR_LENGTH);

   unsigned minx[AGX_MAX_VIEWPORTS], miny[AGX_MAX_VIEWPORTS];
   unsigned maxx[AGX_MAX_VIEWPORTS], maxy[AGX_MAX_VIEWPORTS];

   /* Upload each scissor */
   for (unsigned i = 0; i < count; ++i) {
      agx_get_scissor_extents(&vp[i], ss ? &ss[i] : NULL, &batch->key, &minx[i],
                              &miny[i], &maxx[i], &maxy[i]);

      float minz, maxz;
      util_viewport_zmin_zmax(vp, clip_halfz, &minz, &maxz);

      agx_pack(scissors + i, SCISSOR, cfg) {
         cfg.min_x = minx[i];
         cfg.min_y = miny[i];
         cfg.min_z = minz;
         cfg.max_x = maxx[i];
         cfg.max_y = maxy[i];
         cfg.max_z = maxz;
      }
   }

   /* Upload state */
   struct AGX_PPP_HEADER present = {
      .depth_bias_scissor = true,
      .region_clip = true,
      .viewport = true,
      .viewport_count = count,
   };

   size_t size = agx_ppp_update_size(&present);
   struct agx_ptr T =
      agx_pool_alloc_aligned(&batch->pool, size, AGX_PPP_HEADER_ALIGN);
   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

   agx_ppp_push(&ppp, DEPTH_BIAS_SCISSOR, cfg) {
      cfg.scissor = index;

      /* Use the current depth bias, we allocate linearly */
      unsigned count = batch->depth_bias.size / AGX_DEPTH_BIAS_LENGTH;
      cfg.depth_bias = count ? count - 1 : 0;
   };

   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
   }

   agx_ppp_push(&ppp, VIEWPORT_CONTROL, cfg)
      ;

   /* Upload viewports */
   for (unsigned i = 0; i < count; ++i) {
      agx_ppp_push(&ppp, VIEWPORT, cfg) {
         cfg.translate_x = vp[i].translate[0];
         cfg.translate_y = vp[i].translate[1];
         cfg.translate_z = vp[i].translate[2];
         cfg.scale_x = vp[i].scale[0];
         cfg.scale_y = vp[i].scale[1];
         cfg.scale_z = vp[i].scale[2];

         if (!clip_halfz) {
            cfg.translate_z -= cfg.scale_z;
            cfg.scale_z *= 2;
         }
      }
   }

   bool emitted = agx_gallium_ppp_fini(batch, out, &ppp);
   assert(emitted);
   (void)emitted;
}

static void
agx_upload_depth_bias(struct agx_batch *batch,
                      const struct pipe_rasterizer_state *rast)
{
   void *ptr =
      util_dynarray_grow_bytes(&batch->depth_bias, 1, AGX_DEPTH_BIAS_LENGTH);

   agx_pack(ptr, DEPTH_BIAS, cfg) {
      cfg.depth_bias = rast->offset_units * 2.0f;
      cfg.slope_scale = rast->offset_scale;
      cfg.clamp = rast->offset_clamp;
   }
}

/* A framebuffer state can be reused across batches, so it doesn't make sense
 * to add surfaces to the BO list here. Instead we added them when flushing.
 */


static bool
agx_fb_has_colour(const struct pipe_framebuffer_state *fb)
{
   for (unsigned i = 0; i < fb->nr_cbufs; ++i) {
      if (fb->cbufs[i].texture)
         return true;
   }
   return false;
}

/*
 * M5 keeps every colour attachment in the tilebuffer when its tile programs
 * move all of them in one render, as Borealis's M5 render passes do
 * (hk_m5_mrt_layout_supported, hk_build_m5_mrt_bg_eot): one background
 * program clears or reloads each attachment and one end-of-tile program
 * stores each through its own PBE (the multi-target programs of
 * agx_bg_eot.c). Their admission is the same here: a single-sample layout of
 * two or more attachments that all fit in the tilebuffer, each with a
 * multi-target end-of-tile store class, in a layout the multi-target
 * classifier names. Such a framebuffer is drawn and cleared once, as a
 * single-attachment one is, instead of an attachment at a time or with
 * spilled attachments loaded and stored by every fragment.
 *
 * The answer depends on the formats and the sample count alone, so the
 * batch's layout (agx_batch_init_state), its fragment epilogs and the
 * framebuffer fetches compiled for it (agx_compile_variant) all agree.
 */
static bool
agx_g17g_mrt_resident_layout(enum agx_chip chip,
                             const enum pipe_format *formats,
                             unsigned nr_cbufs, unsigned nr_samples)
{
   if (!agx_chip_g17_m5(chip) || nr_samples > 1)
      return false;

   unsigned active = 0;
   for (unsigned rt = 0; rt < nr_cbufs; ++rt)
      active += formats[rt] != PIPE_FORMAT_NONE;
   if (active < 2)
      return false;

   /* As agx_batch_init_state lays the attachments out. */
   const struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout_with_limits(
         formats, nr_cbufs, 1, false, 0, AGX_G17G_MAX_BYTES_PER_TILE,
         AGX_G17G_MAX_SAMPLES_PER_TILE);
   if (agx_tilebuffer_spills(&tib) || tib.tile_size != 32 * 32 ||
       agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_NONE)
      return false;

   for (unsigned rt = 0; rt < nr_cbufs; ++rt) {
      if (tib.logical_format[rt] == PIPE_FORMAT_NONE)
         continue;

      const uint8_t selector =
         agx_g17p_mrt_eot_selector(tib.logical_format[rt]);
      if (selector != 0x15 && selector != 0x35 && selector != 0x55 &&
          selector != 0x75 && selector != 0x95)
         return false;
   }

   return true;
}

/* A G17G framebuffer is drawn a layer at a time (agx_g17g_layered_draw), so
 * only a single-layer one is a batch's. */
static bool
agx_g17g_fb_mrt_resident(const struct agx_device *dev,
                         const struct pipe_framebuffer_state *fb)
{
   if (!agx_chip_g17_m5(dev->chip) || fb->nr_cbufs < 2 ||
       util_framebuffer_get_num_layers(fb) != 1)
      return false;

   enum pipe_format formats[PIPE_MAX_COLOR_BUFS] = {0};
   for (unsigned rt = 0; rt < fb->nr_cbufs; ++rt)
      formats[rt] = fb->cbufs[rt].format;

   return agx_g17g_mrt_resident_layout(dev->chip, formats, fb->nr_cbufs,
                                       util_framebuffer_get_num_samples(fb));
}

/* G17G's tile programs move one colour attachment and it renders one layer,
 * so it draws and clears a framebuffer with more colour attachments or layers
 * in passes that each bind part of it and then bind the whole framebuffer
 * again (agx_g17g_split_mrt, agx_g17g_layered_draw, agx_g17g_layered_clear).
 * Each of those binds ends a Z/S merge, after which the passes and the draws
 * that follow would write colour the application never bound. These are the
 * conditions under which agx_draw_vbo() and agx_clear() take those paths. */
static bool
agx_fb_drawn_in_passes(const struct agx_device *dev,
                       const struct pipe_framebuffer_state *fb)
{
   return agx_chip_g17_m5(dev->chip) &&
          (fb->nr_cbufs > 1 || util_framebuffer_get_num_layers(fb) > 1);
}

/* If the app binds a framebuffer with only a depth/stencil surface, and an
 * active render batch already renders to that same surface with the same
 * size, keep drawing into that batch: the draws are those of the batch's
 * framebuffer with every colour write masked. Each switch otherwise ends the
 * render pass and stores and reloads every attachment. */
static void
agx_try_zs_merge(struct agx_context *ctx,
                 const struct pipe_framebuffer_state *state)
{
   const struct agx_device *dev = agx_device(ctx->base.screen);

   if (!state->zsbuf.texture ||
       agx_fb_has_colour(state) || state->resolve || state->viewmask)
      return;

   struct agx_batch *best = NULL;
   for (unsigned idx = 0; idx < AGX_MAX_BATCHES; ++idx) {
      if (!BITSET_TEST(ctx->batches.active, idx))
         continue;
      struct agx_batch *b = &ctx->batches.slots[idx];
      if (agx_batch_is_compute(b) || !agx_fb_has_colour(&b->key) ||
          agx_fb_drawn_in_passes(dev, &b->key) ||
          b->key.resolve || b->key.viewmask ||
          b->key.width != state->width || b->key.height != state->height ||
          b->key.samples != state->samples ||
          b->key.layers != state->layers ||
          !pipe_surface_equal(&b->key.zsbuf, &state->zsbuf))
         continue;
      if (!best || b->seqnum > best->seqnum)
         best = b;
   }
   if (!best)
      return;

   util_copy_framebuffer_state(&ctx->zs_merge_real, state);
   util_copy_framebuffer_state(&ctx->framebuffer, &best->key);
   ctx->zs_merge = true;
}

/* A merged draw that samples or binds as an image one of the batch's colour
 * targets would read memory the batch has not stored yet: undo the merge so
 * the normal path ends the batch first. */
static void
agx_check_zs_merge(struct agx_context *ctx)
{
   if (!ctx->zs_merge)
      return;

   for (unsigned stage = 0; stage < ARRAY_SIZE(ctx->stage); ++stage) {
      for (unsigned i = 0; i < ctx->stage[stage].texture_count; ++i) {
         struct agx_sampler_view *tex = ctx->stage[stage].textures[i];
         if (!tex)
            continue;
         for (unsigned cb = 0; cb < ctx->framebuffer.nr_cbufs; ++cb) {
            if (ctx->framebuffer.cbufs[cb].texture == &tex->rsrc->base.b)
               goto unmerge;
         }
      }
      u_foreach_bit64(i, ctx->stage[stage].image_mask) {
         struct pipe_resource *r = ctx->stage[stage].images[i].resource;
         for (unsigned cb = 0; cb < ctx->framebuffer.nr_cbufs; ++cb) {
            if (r && ctx->framebuffer.cbufs[cb].texture == r)
               goto unmerge;
         }
      }
   }
   return;

unmerge:
   util_copy_framebuffer_state(&ctx->framebuffer, &ctx->zs_merge_real);
   ctx->zs_merge = false;
   ctx->batch = NULL;
   agx_dirty_all(ctx);
}

static struct agx_tilebuffer_layout
agx_gallium_tilebuffer_layout(const struct agx_device *dev,
                              const enum pipe_format *formats,
                              unsigned nr_cbufs, unsigned nr_samples,
                              bool layered);

static void
agx_set_framebuffer_state(struct pipe_context *pctx,
                          const struct pipe_framebuffer_state *state)
{
   struct agx_context *ctx = agx_context(pctx);

   if (!state)
      return;

   /* On chips without in-place decompression (AGX3), reallocate any
    * compressed render target that this framebuffer's tilebuffer layout will
    * spill before binding `state` into `ctx->framebuffer`. Running
    * agx_decompress() here avoids re-entering with the spilling framebuffer
    * when the blitter restores `ctx->framebuffer`, and guarantees no batch
    * reaches AGX2-only agx_decompress_inplace().
    */
   const struct agx_device *dev = agx_device(pctx->screen);
   if (!agx_gallium_shared_compression_supported(
          agx_chip_descriptor_generation(dev->chip)) &&
       state->nr_cbufs > 1) {
      bool any_compressed = false;
      enum pipe_format formats[PIPE_MAX_COLOR_BUFS] = {0};

      for (unsigned i = 0; i < state->nr_cbufs; ++i) {
         formats[i] = state->cbufs[i].format;
         if (state->cbufs[i].texture &&
             agx_resource(state->cbufs[i].texture)->layout.compressed)
            any_compressed = true;
      }

      if (any_compressed) {
         struct agx_tilebuffer_layout tib = agx_gallium_tilebuffer_layout(
            dev, formats, state->nr_cbufs,
            util_framebuffer_get_num_samples(state),
            util_framebuffer_get_num_layers(state) > 1);
         if (agx_tilebuffer_spills(&tib)) {
            for (unsigned i = 0; i < state->nr_cbufs; ++i) {
               if (!tib.spilled[i] || !state->cbufs[i].texture)
                  continue;

               struct agx_resource *rsrc = agx_resource(state->cbufs[i].texture);
               if (!agx_resource_is_shared(rsrc) &&
                   rsrc->layout.compressed)
                  agx_decompress(ctx, rsrc, "Render target spilled");
            }
         }
      }
   }

   util_copy_framebuffer_state(&ctx->framebuffer, state);
   ctx->g17g_mrt_spills_known = false;
   ctx->g17g_mrt_resident_known = false;
   ctx->zs_merge = false;

   /* The frontend still reads and resolves the window's multisample colour
    * attachment. G17G/G18G's direct EOT resolve only writes the single-sample
    * target, losing that source. Preserve the samples and let the frontend's
    * normal blit perform the resolve, as for explicit framebuffer blits.
    */
   if (agx_chip_hal300_path(agx_device(pctx->screen)->chip))
      pipe_resource_reference(&ctx->framebuffer.resolve, NULL);

   for (unsigned i = 0; i < state->nr_cbufs; ++i) {
      agx_legalize_compression(ctx,
                               agx_resource(ctx->framebuffer.cbufs[i].texture),
                               ctx->framebuffer.cbufs[i].format);
   }

   if (state->resolve)
      agx_legalize_compression(ctx, agx_resource(state->resolve),
                               state->resolve->format);

   agx_legalize_compression(ctx, agx_resource(ctx->framebuffer.zsbuf.texture),
                            ctx->framebuffer.zsbuf.format);

   agx_try_zs_merge(ctx, state);

   ctx->batch = NULL;
   agx_dirty_all(ctx);
}

/*
 * To write out render targets, each render target surface is bound as a
 * writable shader image, written with the end-of-tile program. This helper
 * constructs the internal pipe_image_view used.
 */
static struct pipe_image_view
image_view_for_surface(const struct pipe_surface *surf)
{
   return (struct pipe_image_view){
      .resource = surf->texture,
      .format = surf->format,
      .access = PIPE_IMAGE_ACCESS_READ_WRITE,
      .shader_access = PIPE_IMAGE_ACCESS_READ_WRITE,
      .u.tex.single_layer_view = surf->first_layer == surf->last_layer,
      .u.tex.first_layer = surf->first_layer,
      .u.tex.last_layer = surf->last_layer,
      .u.tex.level = surf->level,
   };
}

static struct agx_tilebuffer_layout
agx_gallium_tilebuffer_layout(const struct agx_device *dev,
                              const enum pipe_format *formats,
                              unsigned nr_cbufs, unsigned nr_samples,
                              bool layered)
{
   if (dev->chip != AGX_CHIP_G18G) {
      const bool imageblock = agx_chip_native_imageblock(dev->chip);
      return agx_build_tilebuffer_layout_with_limits(
         formats, nr_cbufs, nr_samples, layered,
         imageblock ? AGX_G17P_MAX_BYTES_PER_SAMPLE : 0,
         imageblock ? 32768 : 0, agx_chip_native_agx3(dev->chip) ? 2048 : 0);
   }

   /* Native G18 retains three RGBA32 targets at 2x/4x: 96 KiB and 2048
    * raster samples, with the existing 32-KiB/1024-sample limits at 1x. */
   const bool g18_native = dev->chip == AGX_CHIP_G18G &&
      (nr_samples == 1 || nr_samples == 2 || nr_samples == 4);
   /* Native1044 keeps up to eight RGBA32 outputs resident, with a 128-byte
    * sample and 32x32 tile. Native855 (verified in 1076) shows identical
    * fragment stores, background, end-of-tile and setup programs for
    * RGBA32Uint, RGBA32Sint and RGBA32Float in every mode, so the targets
    * need not share one type. Preserve other layout budgets. */
   bool raw32 = g18_native && nr_samples == 1 && !layered;
   unsigned active = 0;
   for (unsigned rt = 0; rt < nr_cbufs; ++rt) {
      enum pipe_format f = formats[rt];
      if (f == PIPE_FORMAT_NONE)
         continue;
      raw32 &= f == PIPE_FORMAT_R32G32B32A32_FLOAT ||
               f == PIPE_FORMAT_R32G32B32A32_SINT ||
               f == PIPE_FORMAT_R32G32B32A32_UINT;
      active++;
   }
   raw32 &= active >= 2 && active <= 8;
   const bool kernel_native_samples =
      !!(dev->params.features & DRM_ASAHI_FEATURE_G18_NATIVE_TILE_SAMPLES);
   /* Native1130 (run 1139): Apple packs a single-sample mixed-format layout
    * in index order, each attachment aligned to its lane width, in a 32x32
    * tile of its whole sample -- the background stores of all 39 workloads of
    * capture 1130 sit at exactly those offsets. The fragment epilog builds
    * the same layout from the same formats (agx_nir_fs_epilog). */
   if (g18_native && !raw32 &&
       agx_g18g_native_mixed_tile_bytes(formats, nr_cbufs, nr_samples, layered,
                                        kernel_native_samples))
      return agx_build_tilebuffer_layout_g18_mixed(formats, nr_cbufs);

   return agx_build_tilebuffer_layout_with_limits(
      formats, nr_cbufs, nr_samples, layered, raw32 ? 128 : 0,
      raw32 ? 131072 : g18_native ? (nr_samples == 1 ? 32768 : 98304) : 0,
      g18_native ? (nr_samples == 1 ? 1024 : 2048) : 0);
}

static bool
agx_g18_msaa4_color_format(enum pipe_format format)
{
   /* Native four-sample clear, sample-ID and reload programs use the same
    * class-12 transaction for these W8, W16 and packed-W32 color lanes. */
   switch (format) {
   case PIPE_FORMAT_R8_UNORM:
   case PIPE_FORMAT_R8G8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_UNORM:
   case PIPE_FORMAT_B8G8R8A8_UNORM:
   case PIPE_FORMAT_R8G8B8X8_UNORM:
   case PIPE_FORMAT_B8G8R8X8_UNORM:
   case PIPE_FORMAT_R8G8B8A8_SRGB:
   case PIPE_FORMAT_B8G8R8A8_SRGB:
   case PIPE_FORMAT_R8G8B8X8_SRGB:
   case PIPE_FORMAT_B8G8R8X8_SRGB:
   case PIPE_FORMAT_R4G4B4A4_UNORM:
   case PIPE_FORMAT_B4G4R4A4_UNORM:
   case PIPE_FORMAT_A4B4G4R4_UNORM:
   case PIPE_FORMAT_A4R4G4B4_UNORM:
   case PIPE_FORMAT_R5G5B5A1_UNORM:
   case PIPE_FORMAT_B5G5R5A1_UNORM:
   case PIPE_FORMAT_A1R5G5B5_UNORM:
   case PIPE_FORMAT_A1B5G5R5_UNORM:
   case PIPE_FORMAT_R16_FLOAT:
   case PIPE_FORMAT_R16G16_FLOAT:
   /* Native334 verifies the complete four-half-lane F16 store and native330
    * supplies both 2x/4x fragment reload/sample-ID pipelines (mask 0xf). */
   case PIPE_FORMAT_R16G16B16A16_FLOAT:
   case PIPE_FORMAT_R10G10B10A2_UNORM:
   case PIPE_FORMAT_B10G10R10A2_UNORM:
   case PIPE_FORMAT_R11G11B10_FLOAT:
      return true;
   /* Native G18 2x/4x typed clear, sample-ID and reload programs retain the
    * class-12 transaction for these formats. The storage and tile predicates
    * below still require uncompressed, non-layered supported sample storage. */
   case PIPE_FORMAT_B5G5R5X1_UNORM:
   case PIPE_FORMAT_B5G6R5_UNORM:
   case PIPE_FORMAT_R10G10B10A2_UINT:
   case PIPE_FORMAT_R10G10B10X2_UNORM:
   case PIPE_FORMAT_R16G16B16A16_SINT:
   case PIPE_FORMAT_R16G16B16A16_SNORM:
   case PIPE_FORMAT_R16G16B16A16_UINT:
   case PIPE_FORMAT_R16G16B16A16_UNORM:
   case PIPE_FORMAT_R16G16B16X16_FLOAT:
   case PIPE_FORMAT_R16G16B16X16_SINT:
   case PIPE_FORMAT_R16G16B16X16_UINT:
   case PIPE_FORMAT_R16G16_SINT:
   case PIPE_FORMAT_R16G16_SNORM:
   case PIPE_FORMAT_R16G16_UINT:
   case PIPE_FORMAT_R16G16_UNORM:
   case PIPE_FORMAT_R16_SINT:
   case PIPE_FORMAT_R16_SNORM:
   case PIPE_FORMAT_R16_UINT:
   case PIPE_FORMAT_R16_UNORM:
   case PIPE_FORMAT_R32G32_FLOAT:
   case PIPE_FORMAT_R32G32_SINT:
   case PIPE_FORMAT_R32G32_UINT:
   case PIPE_FORMAT_R32_FLOAT:
   case PIPE_FORMAT_R32_SINT:
   case PIPE_FORMAT_R32_UINT:
   /* Native331 binds all four raw 32-bit lanes at 2x/4x, including signed
    * and high-bit integer values. RGB uses the same RGBX storage family. */
   case PIPE_FORMAT_R32G32B32A32_FLOAT:
   case PIPE_FORMAT_R32G32B32X32_FLOAT:
   case PIPE_FORMAT_R32G32B32A32_SINT:
   case PIPE_FORMAT_R32G32B32X32_SINT:
   case PIPE_FORMAT_R32G32B32A32_UINT:
   case PIPE_FORMAT_R32G32B32X32_UINT:
   case PIPE_FORMAT_R8G8B8A8_SINT:
   case PIPE_FORMAT_R8G8B8A8_SNORM:
   case PIPE_FORMAT_R8G8B8A8_UINT:
   case PIPE_FORMAT_R8G8B8X8_SINT:
   case PIPE_FORMAT_R8G8B8X8_SNORM:
   case PIPE_FORMAT_R8G8B8X8_UINT:
   case PIPE_FORMAT_R8G8_SINT:
   case PIPE_FORMAT_R8G8_SNORM:
   case PIPE_FORMAT_R8G8_UINT:
   case PIPE_FORMAT_R8_SINT:
   case PIPE_FORMAT_R8_SNORM:
   case PIPE_FORMAT_R8_UINT:
      return true;
   default:
      return false;
   }
}

/* Apply the same storage contract to writers, reloads and later blits.
 * Compression remains separate; layered bindings are checked below. */
bool
agx_g18_msaa4_storage(const struct agx_device *dev,
                      const struct agx_resource *rsrc, enum pipe_format format)
{
   bool format_ok = agx_g18_msaa4_color_format(format);
   if (rsrc->layout.format == PIPE_FORMAT_Z32_FLOAT) {
      format_ok = format == PIPE_FORMAT_Z32_FLOAT ||
                  format == PIPE_FORMAT_R32_FLOAT ||
                  format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
                  format == PIPE_FORMAT_Z24_UNORM_S8_UINT ||
                  format == PIPE_FORMAT_Z24X8_UNORM;
   } else if (rsrc->layout.format == PIPE_FORMAT_Z16_UNORM) {
      /* Native321 binds D16 writes, reloads and sampled readback at 4x. */
      format_ok = format == PIPE_FORMAT_Z16_UNORM ||
                  format == PIPE_FORMAT_R16_UNORM;
   } else if (rsrc->layout.format == PIPE_FORMAT_S8_UINT) {
      format_ok = format == PIPE_FORMAT_S8_UINT ||
                  format == PIPE_FORMAT_R8_UINT;
   }

   /* A non-layered attachment to an array selects one layer through the
    * existing 2D load/store views. Both descriptors use the selected layer's
    * base address; the tile program does not consume an array-layer register.
    * Native AIL comparison312 covers the allocation and layer stride.
    * Layered bindings use the array BG/EOT programs under the tile-layout
    * and draw predicates below.
    */
   const bool target_ok =
      (rsrc->base.b.target == PIPE_TEXTURE_2D && rsrc->base.b.array_size == 1) ||
      (rsrc->base.b.target == PIPE_TEXTURE_2D_ARRAY && rsrc->base.b.array_size >= 1);

   return dev->chip == AGX_CHIP_G18G &&
          (rsrc->base.b.nr_samples == 2 || rsrc->base.b.nr_samples == 4) &&
          target_ok && rsrc->base.b.last_level == 0 &&
          rsrc->base.b.depth0 == 1 && !rsrc->layout.compressed &&
          rsrc->layout.tiling == AIL_TILING_GPU &&
          format_ok;
}

static bool
agx_g18_msaa4_zs_surface(const struct agx_device *dev,
                          const struct pipe_surface *surf)
{
   if (!surf->texture)
      return true;

   struct agx_resource *rsrc = agx_resource(surf->texture);
   struct agx_resource *stencil = rsrc->separate_stencil;
   /* Native1094 msdepth (Apple M6, verified): a colour-free 2x/4x render
    * into layer 3 of an eight-layer Depth32Float 2D-multisample array
    * differs from the layer-0 render only in the depth load/store bases
    * (0x15329/0x15331, +3 layer strides) and the compression-metadata
    * bases (0x153c1/0x153c9), with the same imageblock (specLM) and empty
    * background/end-of-tile programs. A single non-zero layer of a D32
    * array therefore selects its base address only, as colour does. */
   const bool single_d32_layer =
      surf->first_layer == surf->last_layer &&
      rsrc->layout.format == PIPE_FORMAT_Z32_FLOAT && !stencil &&
      rsrc->base.b.target == PIPE_TEXTURE_2D_ARRAY;
   return surf->level == 0 &&
          ((surf->first_layer == 0 && surf->last_layer == 0) ||
           single_d32_layer) &&
          (rsrc->layout.format == PIPE_FORMAT_Z32_FLOAT ||
           rsrc->layout.format == PIPE_FORMAT_Z16_UNORM ||
           rsrc->layout.format == PIPE_FORMAT_S8_UINT) &&
          agx_g18_msaa4_storage(dev, rsrc, rsrc->layout.format) &&
          (!stencil ||
           (stencil->layout.format == PIPE_FORMAT_S8_UINT &&
            stencil->base.b.width0 == rsrc->base.b.width0 &&
            stencil->base.b.height0 == rsrc->base.b.height0 &&
            agx_g18_msaa4_storage(dev, stencil, PIPE_FORMAT_S8_UINT)));
}

/* Native326's two-sample tile is 32x32; four samples use 32x16.
 * Native331 adds sixteen-byte samples with the same geometry and bounded
 * single-target protocol, occupying the full 32-KiB imageblock.
 * Native315 also binds D32/S8 planes; colour-free renders retain this tile
 * geometry with empty colour programs. Native598/618/639 add layered
 * four-sample colour; native673 compares the two-sample array geometry and
 * all five register lists. Both retain up to sixteen bytes per sample;
 * spills stay separate. */
static bool
agx_g18_msaa4_tile_layout(const struct agx_device *dev,
                          const struct agx_tilebuffer_layout *tib)
{
   if (dev->chip == AGX_CHIP_G18G && agx_g18g_ms_mrt_mask(tib))
      return true;

   if (dev->chip != AGX_CHIP_G18G ||
       (tib->nr_samples != 2 && tib->nr_samples != 4) ||
       (tib->sample_size_B != 8 && tib->sample_size_B != 16) ||
       tib->tile_size != (tib->nr_samples == 2 ? 32 * 32 : 32 * 16) ||
       agx_tilebuffer_spills(tib) ||
       agx_tilebuffer_offset_B(tib, 0) != 0)
      return false;

   const enum pipe_format format = tib->logical_format[0];
   /* Stencil replication writes the S8 plane through an R8_UINT color view.
    * Its raw W8 tile lane uses the existing class-12 protocol. The separate
    * storage predicate still requires a supported backing resource. */
   if (format != PIPE_FORMAT_NONE && format != PIPE_FORMAT_R8_UINT &&
       !agx_g18_msaa4_color_format(format))
      return false;

   for (unsigned rt = 1; rt < ARRAY_SIZE(tib->logical_format); ++rt) {
      if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
         return false;
   }
   return true;
}

static bool
agx_g18_msaa_color_surfaces(const struct agx_batch *batch)
{
   const struct agx_device *dev = agx_device(batch->ctx->base.screen);
   unsigned mask = agx_g18g_ms_mrt_mask(&batch->tilebuffer_layout);
   if (!mask) {
      if (batch->key.nr_cbufs != 1)
         return false;
      mask = 1;
   }
   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      const struct pipe_surface *surf = &batch->key.cbufs[rt];
      if (!(mask & BITFIELD_BIT(rt))) {
         if (surf->texture)
            return false;
         continue;
      }
      if (!surf->texture || !agx_g18_msaa4_storage(
             dev, agx_resource(surf->texture), surf->format))
         return false;
   }
   return true;
}

static bool
agx_g18_native_ms_spills(const struct agx_batch *batch)
{
   const struct agx_device *dev = agx_device(batch->ctx->base.screen);
   const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;
   return dev->chip == AGX_CHIP_G18G && !tib->layered &&
          (tib->nr_samples == 2 || tib->nr_samples == 4) &&
          agx_tilebuffer_spills(tib);
}

/* The Gen9 path proven by Honeykrisp resolves a 2x/4x colour tile directly
 * into an ordinary single-sample PBE target. Gallium exposes exactly that WSI
 * contract through pipe_framebuffer_state::resolve: the multisample colour
 * attachment is transient and resolve names the image presented by the
 * loader. Keep every other resolve shape on the existing fail-closed path.
 */
static bool
agx_g17p_direct_resolve_surface(struct agx_batch *batch,
                                struct pipe_surface *out)
{
   const struct pipe_framebuffer_state *fb = &batch->key;
   const struct pipe_surface *source = &fb->cbufs[0];
   struct pipe_resource *target = fb->resolve;
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   /* msaa163: G18G resolves its witnessed four-sample tile the same way. */
   const bool g18_msaa4 =
      agx_g18_msaa4_tile_layout(dev, &batch->tilebuffer_layout);
   if ((!agx_chip_native_agx3(dev->chip) && !g18_msaa4 &&
        !agx_chip_g17_m5(dev->chip)) ||
       fb->nr_cbufs != 1 || !source->texture || !target ||
       (util_res_sample_count(source->texture) != 2 &&
        util_res_sample_count(source->texture) != 4) ||
       util_res_sample_count(target) != 1 || source->level != 0 ||
       source->first_layer != 0 || source->last_layer != 0 ||
       (target->target != PIPE_TEXTURE_2D &&
        target->target != PIPE_TEXTURE_RECT) ||
       target->last_level != 0 || target->array_size != 1 ||
       target->depth0 != 1 || target->format != source->format ||
       target->width0 != fb->width || target->height0 != fb->height ||
       source->texture->width0 != fb->width ||
       source->texture->height0 != fb->height ||
       (!agx_chip_hal300_path(dev->chip) &&
        !agx_g17p_rt0_tile_layout_supported(&batch->tilebuffer_layout)))
      return false;

   *out = (struct pipe_surface){
      .format = source->format,
      .first_layer = 0,
      .last_layer = 0,
      .level = 0,
      .texture = target,
   };
   return true;
}
/* Similarly, to read render targets, surfaces are bound as textures */
static struct pipe_sampler_view
sampler_view_for_surface(const struct pipe_surface *surf)
{
   bool layered = surf->last_layer > surf->first_layer;

   return (struct pipe_sampler_view){
      /* To reduce shader variants, we always use a 2D texture. For reloads of
       * arrays and cube maps, we map a single layer as a 2D image.
       */
      .target = layered ? PIPE_TEXTURE_2D_ARRAY : PIPE_TEXTURE_2D,
      .swizzle_r = PIPE_SWIZZLE_X,
      .swizzle_g = PIPE_SWIZZLE_Y,
      .swizzle_b = PIPE_SWIZZLE_Z,
      .swizzle_a = PIPE_SWIZZLE_W,
      .u.tex =
         {
            .first_layer = surf->first_layer,
            .last_layer = surf->last_layer,
            .first_level = surf->level,
            .last_level = surf->level,
         },
   };
}

static bool
target_is_array(enum pipe_texture_target target)
{
   switch (target) {
   case PIPE_TEXTURE_3D:
   case PIPE_TEXTURE_CUBE:
   case PIPE_TEXTURE_1D_ARRAY:
   case PIPE_TEXTURE_2D_ARRAY:
   case PIPE_TEXTURE_CUBE_ARRAY:
      return true;
   default:
      return false;
   }
}

static bool
agx_batch_upload_pbe_internal(struct agx_batch *batch,
                     struct agx_gallium_pbe_descriptor *out,
                     struct pipe_image_view *view, bool block_access,
                     bool arrays_as_2d, bool force_2d_array, bool emrt,
                     bool native_storage_write, uint8_t *native_emrt)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);
   const unsigned descriptor_size =
      agx_gallium_pbe_descriptor_size(descriptor_generation);

   if (!descriptor_size)
      return false;

   if (!agx_gallium_native_pbe_supported(descriptor_generation)) {
      memset(out, 0, descriptor_size);
      return false;
   }

   struct agx_resource *tex = agx_resource(view->resource);
   const struct util_format_description *desc =
      util_format_description(view->format);
   enum pipe_texture_target target = tex->base.b.target;
   bool is_buffer = (target == PIPE_BUFFER);

   if (!is_buffer && view->u.tex.single_layer_view)
      target = PIPE_TEXTURE_2D;

   arrays_as_2d |= (view->access & PIPE_IMAGE_ACCESS_DRIVER_INTERNAL);

   /* To reduce shader variants, spilled layered render targets are accessed as
    * 2D Arrays regardless of the actual target, so force in that case.
    *
    * Likewise, cubes are accessed as arrays for consistency with NIR.
    */
   if ((arrays_as_2d && target_is_array(target)) || target_is_cube(target) ||
       force_2d_array)
      target = PIPE_TEXTURE_2D_ARRAY;

   unsigned level = is_buffer ? 0 : view->u.tex.level;
   unsigned layer = is_buffer ? 0 : view->u.tex.first_layer;

   /* Build the SEMANTIC descriptor and let the native packer decide how to
    * encode it. The AGX2 packet is a union -- stride aliases levels/layers,
    * and the software addressing sideband aliases the extended word -- so
    * only this producer knows which arm is live, and packing here would
    * throw that away before the AGX3 encoder could check it.
    */
   struct AGX_PBE cfg = {AGX_PBE_header};
   {
      cfg.dimension =
         agx_translate_tex_dim(target, util_res_sample_count(&tex->base.b));
      /* G17G shaders store to layered images as 3D (agx3_pack_binary.c);
       * G17C stores keep the array (agx_chip_m5_layer_operand()). */
      if (descriptor_generation == AGX_DESCRIPTOR_GENERATION_10 &&
          !agx_chip_m5_layer_operand(dev->chip) &&
          !block_access && cfg.dimension == AGX_TEXTURE_DIMENSION_2D_ARRAY)
         cfg.dimension = AGX_TEXTURE_DIMENSION_3D;
      cfg.layout = agx_translate_layout(tex->layout.tiling);
      cfg.channels = ail_pixel_format[view->format].channels;
      cfg.type = ail_pixel_format[view->format].type;
      cfg.srgb = util_format_is_srgb(view->format);

      assert(desc->nr_channels >= 1 && desc->nr_channels <= 4);

      for (unsigned i = 0; i < desc->nr_channels; ++i) {
         if (desc->swizzle[i] == 0)
            cfg.swizzle_r = i;
         else if (desc->swizzle[i] == 1)
            cfg.swizzle_g = i;
         else if (desc->swizzle[i] == 2)
            cfg.swizzle_b = i;
         else if (desc->swizzle[i] == 3)
            cfg.swizzle_a = i;
      }

      /* RGBX/BGRX: the fourth memory channel is void, so the loop leaves
       * swizzle_a at 0 and the channel takes tile channel 0 (red).  Point it at
       * the tile's alpha instead, which agx_nir_lower_tilebuffer fills with
       * 1.0 for these formats, so X scans out as opaque all-ones. On G17P/M5
       * this covers 10-bit RGB10X2/BGR10X2 whose packed RGB10A2 tile word
       * always carries 1.0 in channel 3.
       */
      if ((dev->chip == AGX_CHIP_G18G ||
           (agx_chip_native_or_m5(dev->chip) &&
            cfg.channels == AGX_CHANNELS_R10G10B10A2)) &&
          desc->nr_channels == 4 &&
          desc->channel[3].type == UTIL_FORMAT_TYPE_VOID)
         cfg.swizzle_a = 3;

      cfg.buffer = agx_map_texture_gpu(tex, layer);
      cfg.unk_mipmapped = tex->mipmapped;

      if (is_buffer) {
         unsigned size_el =
            agx_texture_buffer_size_el(view->format, view->u.buf.size);

         /* Buffers uniquely have offsets (in bytes, not texels) */
         cfg.buffer += view->u.buf.offset;

         /* Use a 2D texture to increase the maximum size */
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(size_el, cfg.width);
         cfg.level = 0;
         cfg.stride = (cfg.width * util_format_get_blocksize(view->format)) - 4;
         cfg.layers = 1;
         cfg.levels = 1;
      } else if (util_res_sample_count(&tex->base.b) > 1 && !block_access) {
         /* Multisampled images are bound like buffer textures, with
          * addressing arithmetic to determine the texel to write.
          *
          * Note that the end-of-tile program uses real multisample images with
          * image_write_block instructions.
          */
         unsigned blocksize_B = util_format_get_blocksize(view->format);
         unsigned size_px =
            (tex->layout.size_B - tex->layout.layer_stride_B * layer) /
            blocksize_B;

         cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
         cfg.layout = AGX_LAYOUT_LINEAR;
         cfg.width = AGX_TEXTURE_BUFFER_WIDTH;
         cfg.height = DIV_ROUND_UP(size_px, cfg.width);
         cfg.stride = (cfg.width * blocksize_B) - 4;
         cfg.layers = 1;
         cfg.levels = 1;

         cfg.buffer += tex->layout.level_offsets_B[level];
         cfg.level = 0;
      } else {
         cfg.width = view->resource->width0;
         cfg.height = view->resource->height0;
         cfg.level = level;

         unsigned layers = view->u.tex.last_layer - layer + 1;

         if (tex->layout.tiling == AIL_TILING_LINEAR &&
             (target == PIPE_TEXTURE_1D_ARRAY ||
              target == PIPE_TEXTURE_2D_ARRAY)) {

            cfg.depth_linear = layers;
            cfg.layer_stride_linear = (tex->layout.layer_stride_B - 0x80);
            cfg.extended = true;
         } else {
            assert((tex->layout.tiling != AIL_TILING_LINEAR) || (layers == 1));
            cfg.layers = layers;
         }

         if (tex->layout.tiling == AIL_TILING_LINEAR) {
            cfg.stride = ail_get_linear_stride_B(&tex->layout, level) - 4;
            cfg.levels = 1;
         } else if (tex->layout.tiling == AIL_TILING_INTERCHANGE) {
            /* The field encodes stride_B / 16 - 1, in macroblocks. */
            cfg.stride = (tex->layout.stride_el[0] / 16) * 16 - 4;
            cfg.levels = 1;
         } else {
            cfg.page_aligned_layers = tex->layout.page_aligned_layers;
            cfg.levels = tex->base.b.last_level + 1;

            /* The M5 allocator's flag describes padding, so it is clear when
             * a layer is exactly one page. G18 block stores still need the
             * page-addressed array mode for that already-aligned backing.
             * This also applies to multisample layers: sample count changes
             * the footprint, not the unit used to address the next layer.
             * Do not select it for compact subpage or mipmapped layouts.
             */
            if (dev->chip == AGX_CHIP_G18G && block_access &&
                target == PIPE_TEXTURE_2D_ARRAY && layers > 1 &&
                cfg.levels == 1 && !tex->layout.compressed &&
                tex->layout.layer_stride_B != 0 &&
                (tex->layout.layer_stride_B % AIL_PAGESIZE) == 0)
               cfg.page_aligned_layers = true;
         }

         /* G17G's Gen10 store packet is witnessed for one-level images only;
          * the Gen9 mip form has no G17G witness yet. Storing into level 0 of
          * a mipmapped twiddled 2D texture is the same write as storing into
          * a one-level image of level 0's size at level 0's address: AIL
          * places level 0 first, at offset zero, with the twiddling of a
          * standalone image of that size, and the store never touches the
          * other levels. Describe exactly that write. Other levels, arrays and
          * compressed layouts keep waiting for their witnessed forms. */
         /* AIL tiles every level as a standalone image of that level's
          * size, so a store into level N is likewise a one-level image of
          * level N's size at level N's offset (blits into mip levels,
          * glCopyTexImage2D, rendering to a level). */
         if (descriptor_generation == AGX_DESCRIPTOR_GENERATION_10 &&
             cfg.levels > 1 && layers == 1 &&
             cfg.dimension == AGX_TEXTURE_DIMENSION_2D &&
             tex->layout.tiling != AIL_TILING_LINEAR &&
             !tex->layout.compressed) {
            if (level > 0) {
               cfg.buffer += tex->layout.level_offsets_B[level];
               cfg.width = u_minify(view->resource->width0, level);
               cfg.height = u_minify(view->resource->height0, level);
               cfg.level = 0;
            }
            cfg.levels = 1;
            cfg.unk_mipmapped = false;
         }

         if (tex->base.b.nr_samples > 1)
            cfg.samples = agx_translate_sample_count(tex->base.b.nr_samples);
      }

      if (tex->layout.compressed && !emrt) {
         if (!ail_is_view_compatible(&tex->layout, view->format))
            return false;

         cfg.compressed = true;
         cfg.extended = true;

         cfg.acceleration_buffer =
            agx_map_gpu(tex) + tex->layout.metadata_offset_B +
            (layer * tex->layout.compression_layer_stride_B);
      }

      /* When the descriptor isn't extended architecturally, we can use the last
       * 8 bytes as a sideband. We use it to provide metadata for image atomics.
       *
       * Only libagx_image_texel_address() and libagx_buffer_texel_address()
       * ever read it, and both agx_unpack() the 24-byte AGX2 packet, so
       * neither can read an AGX3 descriptor at all. A block-access image is
       * addressed by the hardware and never consults them, so on AGX3 the
       * sideband would be bits with no Gen9/Gen10 home. Leave it out there;
       * every other AGX3 descriptor that would carry one stays refused by
       * agx_gallium_pack_native_pbe(), which is the honest answer until that
       * ABI is recovered. G18G (Gen10) leaves it out only for block access
       * and the bounded native write-only storage path.
       */
      /* G17G keeps its image-atomic word in the native PBE's free tail
       * (below), not in this AGX2 sideband. */
      const bool sideband_has_no_reader =
         agx_chip_g17_m5(dev->chip) ?
         (block_access &&
          agx_gallium_descriptor_generation_is_agx3(descriptor_generation)) ||
         descriptor_generation == AGX_DESCRIPTOR_GENERATION_10 :
         (dev->chip != AGX_CHIP_G18G ||
          descriptor_generation == AGX_DESCRIPTOR_GENERATION_9 ||
          block_access || native_storage_write) &&
         agx_gallium_descriptor_generation_is_agx3(descriptor_generation);

      if (!cfg.extended && (tex->layout.writeable_image || emrt) &&
          tex->base.b.target != PIPE_BUFFER && !sideband_has_no_reader) {

         if (util_res_sample_count(&tex->base.b) > 1) {
            cfg.aligned_width_msaa_sw =
               align(u_minify(view->resource->width0, level),
                     tex->layout.tilesize_el[level].width_el);

            cfg.sample_count_log2_sw = util_logbase2(tex->base.b.nr_samples);
         } else {
            cfg.level_offset_sw =
               ail_get_level_offset_B(&tex->layout, cfg.level);
         }

         if (tex->layout.tiling != AIL_TILING_LINEAR || emrt) {
            struct ail_tile tile_size = tex->layout.tilesize_el[level];
            cfg.tile_width_sw = tile_size.width_el;
            cfg.tile_height_sw = tile_size.height_el;

            cfg.layer_stride_sw = tex->layout.layer_stride_B;
         }
      }
   }

   if (native_emrt) {
      /* Keep software addressing out of the Gen10 packet. The shader reads
       * this private tail through the same96-byte eMRT ABI as Gen9, while
       * the texture and PBE themselves retain their Gen10 encodings. */
      if (dev->chip != AGX_CHIP_G18G || !emrt || block_access ||
          native_storage_write || cfg.extended || tex->layout.compressed ||
          cfg.level_offset_sw || !cfg.aligned_width_msaa_sw ||
          !util_is_power_of_two_nonzero(cfg.tile_width_sw) ||
          !util_is_power_of_two_nonzero(cfg.tile_height_sw) ||
          util_logbase2(cfg.tile_width_sw) > 15 ||
          util_logbase2(cfg.tile_height_sw) > 15 ||
          !cfg.layer_stride_sw || (cfg.layer_stride_sw & 127) ||
          (cfg.layer_stride_sw >> 7) > 0x7ffffffu ||
          (cfg.sample_count_log2_sw != 1 && cfg.sample_count_log2_sw != 2))
         return false;

      uint64_t sw = ((uint64_t)util_logbase2(cfg.tile_width_sw) << 27) |
                    ((uint64_t)util_logbase2(cfg.tile_height_sw) << 31) |
                    ((uint64_t)(cfg.layer_stride_sw >> 7) << 35) |
                    ((uint64_t)cfg.sample_count_log2_sw << 62);
      uint32_t width = cfg.aligned_width_msaa_sw;
      memcpy(native_emrt + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET, &sw, sizeof(sw));
      memcpy(native_emrt + AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET,
             &width, sizeof(width));
      cfg.aligned_width_msaa_sw = 0;
      cfg.sample_count_log2_sw = 0;
      /* These log2-encoded dimensions have neutral semantic value1. */
      cfg.tile_width_sw = 1;
      cfg.tile_height_sw = 1;
      cfg.layer_stride_sw = 0;
   }

   const bool packed = dev->chip == AGX_CHIP_G18G
      ? agx_gallium_pack_g18_native_pbe(out, &cfg)
      : agx_gallium_pack_native_pbe(descriptor_generation, out, &cfg);
   if (!packed && dev->chip == AGX_CHIP_G18G &&
       (dev->debug & AGX_DBG_FAULTTRACE)) {
      fprintf(stderr,
              "PBE_REFUSED format=%s target=%u dim=%u layout=%u level=%u "
              "levels=%u mip=%u layers=%u samples=%u %ux%u srgb=%u "
              "compressed=%u extended=%u stride=%u sw=%" PRIu64 ",%u,%u,%u,%"
              PRIu64 "\n",
              util_format_short_name(view->format), tex->base.b.target,
              cfg.dimension, cfg.layout, cfg.level, cfg.levels,
              cfg.unk_mipmapped, cfg.layers, cfg.samples, cfg.width,
              cfg.height, cfg.srgb, cfg.compressed, cfg.extended, cfg.stride,
              cfg.level_offset_sw, cfg.aligned_width_msaa_sw,
              cfg.tile_width_sw, cfg.tile_height_sw, cfg.layer_stride_sw);
   }
   /* G17G image atomics address the texel from this PBE and the G17P
    * software word (libagx_image_texel_address_gen9) kept in the PBE's free
    * last eight bytes. A buffer's word is zero; a multisampled image is a
    * linear window and has no atomic addressing yet. */
   if (packed && agx_chip_g17_m5(dev->chip) && descriptor_generation == AGX_DESCRIPTOR_GENERATION_10 &&
       !block_access && !emrt && tex->layout.writeable_image &&
       util_res_sample_count(&tex->base.b) <= 1) {
      uint64_t sw = 0;
      if (!is_buffer && tex->layout.tiling != AIL_TILING_LINEAR) {
         const struct ail_tile tile = tex->layout.tilesize_el[level];
         const uint64_t level_offset_B =
            cfg.level ? ail_get_level_offset_B(&tex->layout, cfg.level) : 0;
         sw = (level_offset_B >> 7) |
              ((uint64_t)util_logbase2(tile.width_el) << 27) |
              ((uint64_t)util_logbase2(tile.height_el) << 31) |
              ((uint64_t)(tex->layout.layer_stride_B >> 7) << 35);
      }
      out->opaque[AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET / 4] = (uint32_t)sw;
      out->opaque[AGX_GEN10_IMAGE_ATOMIC_SW_OFFSET / 4 + 1] =
         (uint32_t)(sw >> 32);
   }

   if (packed && (dev->debug & AGX_DBG_ASYNCTRACE) &&
       util_res_sample_count(&tex->base.b) > 1) {
      agx_msg("MSAA PBE: format=%s dimension=%u samples=%u address=0x%llx "
              "words: %08x %08x %08x %08x %08x %08x %08x %08x\n",
              util_format_short_name(view->format), cfg.dimension,
              util_res_sample_count(&tex->base.b), (unsigned long long)cfg.buffer,
              out->opaque[0], out->opaque[1], out->opaque[2], out->opaque[3],
              out->opaque[4], out->opaque[5], out->opaque[6], out->opaque[7]);
   }
   return packed;
}

static bool
agx_batch_upload_pbe(struct agx_batch *batch,
                     struct agx_gallium_pbe_descriptor *out,
                     struct pipe_image_view *view, bool block_access,
                     bool arrays_as_2d, bool force_2d_array, bool emrt)
{
   return agx_batch_upload_pbe_internal(batch, out, view, block_access,
      arrays_as_2d, force_2d_array, emrt, false, NULL);
}

/*
 * Whether a render into one level and layer of a texture can be stored: a
 * dry run of the end-of-tile PBE the tile program would use
 * (agx_build_native_g17g_bg_eot). A render whose PBE has no encoding is
 * dropped while the draw reports success, so callers that can do the work
 * another way ask first.
 */
bool
agx_level_renderable(struct agx_context *ctx, struct pipe_resource *prsc,
                     enum pipe_format format, unsigned level,
                     unsigned layer)
{
   /* The packer only reads the device through the batch. */
   struct agx_batch scratch = {.ctx = ctx};
   struct agx_gallium_pbe_descriptor pbe;
   struct pipe_image_view view = {
      .resource = prsc,
      .format = util_format_linear(format),
      .access = PIPE_IMAGE_ACCESS_READ_WRITE,
      .shader_access = PIPE_IMAGE_ACCESS_READ_WRITE,
      .u.tex.single_layer_view = true,
      .u.tex.first_layer = layer,
      .u.tex.last_layer = layer,
      .u.tex.level = level,
   };

   return agx_batch_upload_pbe(&scratch, &pbe, &view, true, true, false,
                               false);
}

/* Likewise constant buffers, textures, and samplers are handled in a common
 * per-draw path, with dirty tracking to reduce the costs involved.
 */

static void
agx_set_constant_buffer(struct pipe_context *pctx, mesa_shader_stage shader,
                        uint index, const struct pipe_constant_buffer *cb)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_stage *s = &ctx->stage[shader];
   struct pipe_constant_buffer *constants = &s->cb[index];
   const unsigned mask = 1u << index;

   if (cb && !cb->user_buffer && cb->buffer &&
       cb->buffer == constants->buffer && !constants->user_buffer) {
      const unsigned clamped_size = agx_buffer_range_in_resource(
         cb->buffer, cb->buffer_offset, cb->buffer_size);
      if ((s->cb_mask & mask) &&
          constants->buffer_offset == cb->buffer_offset &&
          constants->buffer_size == clamped_size)
         return;

      constants->buffer_offset = cb->buffer_offset;
      constants->buffer_size = clamped_size;
      s->cb_mask |= mask;
      ctx->stage[shader].dirty |= AGX_STAGE_DIRTY_CONST;
      return;
   }

   util_copy_constant_buffer(&s->cb[index], cb);

   /* Upload user buffer immediately */
   if (constants->user_buffer && !constants->buffer) {
      u_upload_data_ref(ctx->base.const_uploader, 0, constants->buffer_size, 64,
                        constants->user_buffer, &constants->buffer_offset,
                        &constants->buffer);
   }

   /* As for shader buffers: only the part of the range inside the buffer. */
   if (constants->buffer) {
      constants->buffer_size = agx_buffer_range_in_resource(
         constants->buffer, constants->buffer_offset, constants->buffer_size);
   }

   if (cb)
      s->cb_mask |= mask;
   else
      s->cb_mask &= ~mask;

   ctx->stage[shader].dirty |= AGX_STAGE_DIRTY_CONST;
}

static void
agx_delete_state(struct pipe_context *ctx, void *state)
{
   FREE(state);
}

/* BOs added to the batch in the uniform upload path */

static void
agx_set_vertex_buffers(struct pipe_context *pctx, unsigned count,
                       const struct pipe_vertex_buffer *buffers)
{
   struct agx_context *ctx = agx_context(pctx);

   if (buffers && !(ctx->vb_mask & ~BITFIELD_MASK(count))) {
      bool same = true;
      for (unsigned i = 0; i < count; ++i) {
         const struct pipe_vertex_buffer *a = &ctx->vertex_buffers[i];
         const struct pipe_vertex_buffer *b = &buffers[i];
         if (a->is_user_buffer != b->is_user_buffer ||
             a->buffer_offset != b->buffer_offset ||
             (a->is_user_buffer ? (a->buffer.user != b->buffer.user)
                                : (a->buffer.resource != b->buffer.resource))) {
            same = false;
            break;
         }
      }
      if (same)
         return;
   }

   util_set_vertex_buffers_mask(ctx->vertex_buffers, &ctx->vb_mask, buffers,
                                count);

   ctx->dirty |= AGX_DIRTY_VERTEX;
}

static void *
agx_create_vertex_elements(struct pipe_context *ctx, unsigned count,
                           const struct pipe_vertex_element *state)
{
   assert(count <= AGX_MAX_ATTRIBS);

   struct agx_vertex_elements *so = calloc(1, sizeof(*so));
   so->num_attribs = count;

   for (unsigned i = 0; i < count; ++i) {
      const struct pipe_vertex_element ve = state[i];

      const struct util_format_description *desc =
         util_format_description(ve.src_format);
      unsigned chan_size = desc->channel[0].size / 8;
      assert((ve.src_offset & (chan_size - 1)) == 0);

      so->buffers[i] = ve.vertex_buffer_index;
      so->src_offsets[i] = ve.src_offset;
      so->pipe[i] = ve;

      so->key[i] = (struct agx_velem_key){
         .stride = ve.src_stride,
         .format = ve.src_format,
         .divisor = ve.instance_divisor,
         .instanced = ve.instance_divisor > 0,
      };
   }

   return so;
}

static void
agx_bind_vertex_elements_state(struct pipe_context *pctx, void *cso)
{
   struct agx_context *ctx = agx_context(pctx);
   if (ctx->attributes == cso)
      return;
   ctx->attributes = cso;
   ctx->dirty |= AGX_DIRTY_VERTEX;
}

DERIVE_HASH_TABLE(asahi_vs_shader_key);
DERIVE_HASH_TABLE(asahi_fs_shader_key);
DERIVE_HASH_TABLE(asahi_cs_shader_key);
DERIVE_HASH_TABLE(asahi_gs_shader_key);
DERIVE_HASH_TABLE(agx_fast_link_key);

/* Outputs the rasterizer consumes itself, whatever the fragment shader
 * reads: position, point size, clip/cull distances, layer and viewport, and
 * the primitive ID a fragment shader reads as a system value. */
static uint64_t
agx_rast_system_outputs(void)
{
   return VARYING_BIT_POS | VARYING_BIT_PSIZ | VARYING_BIT_CLIP_DIST0 |
          VARYING_BIT_CLIP_DIST1 | VARYING_BIT_CULL_DIST0 |
          VARYING_BIT_CULL_DIST1 | VARYING_BIT_LAYER | VARYING_BIT_VIEWPORT |
          VARYING_BIT_PRIMITIVE_ID | VARYING_BIT_CLIP_VERTEX;
}

/* Drop a GS copy program's rasterizer export of a varying the fragment
 * shader does not read (asahi_gs_shader_key::rast_varyings). */
static bool
agx_nir_trim_rast_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;
   const uint64_t keep = *(const uint64_t *)data | agx_rast_system_outputs();
   const unsigned location = nir_intrinsic_io_semantics(intr).location;
   if (location >= 64 || (keep & BITFIELD64_BIT(location)))
      return false;
   nir_instr_remove(&intr->instr);
   return true;
}

/*
 * G17G draws a layered framebuffer a layer at a time (agx_g17g_bind_layer),
 * where the tile's own layer is always zero, so a fragment gl_Layer read
 * takes the layer the split bound from the draw uniforms instead.
 */
static bool
agx_nir_lower_g17g_fs_layer(nir_builder *b, nir_intrinsic_instr *intr,
                            UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input ||
       nir_intrinsic_io_semantics(intr).location != VARYING_SLOT_LAYER)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   struct agx_draw_uniforms *u = NULL;
   nir_def *layer =
      nir_load_sysval_agx(b, 1, 16, .desc_set = AGX_SYSVAL_TABLE_ROOT,
                          .binding = (uintptr_t)&u->layer_id);
   nir_def_replace(&intr->def, nir_u2uN(b, layer, intr->def.bit_size));
   return true;
}

/*
 * To implement point sprites, we'll replace TEX0...7 with point coordinate
 * reads as required. However, the .zw needs to read back 0.0/1.0. This pass
 * fixes up TEX loads of Z and W according to a uniform passed in a sideband,
 * eliminating shader variants.
 */
static bool
agx_nir_lower_point_sprite_zw(nir_builder *b, nir_intrinsic_instr *intr,
                              UNUSED void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_input &&
       intr->intrinsic != nir_intrinsic_load_interpolated_input)
      return false;

   gl_varying_slot loc = nir_intrinsic_io_semantics(intr).location;
   if (!(loc >= VARYING_SLOT_TEX0 && loc <= VARYING_SLOT_TEX7))
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   unsigned component = nir_intrinsic_component(intr);

   nir_def *mask = nir_load_tex_sprite_mask_agx(b);
   nir_def *location = nir_iadd_imm(b, nir_get_io_offset_src(intr)->ssa,
                                    loc - VARYING_SLOT_TEX0);
   nir_def *bit = nir_ishl(b, nir_imm_intN_t(b, 1, 16), location);
   nir_def *replace = nir_i2b(b, nir_iand(b, mask, bit));

   nir_def *vec = nir_pad_vec4(b, &intr->def);
   nir_def *chans[4] = {NULL, NULL, nir_imm_floatN_t(b, 0.0, vec->bit_size),
                        nir_imm_floatN_t(b, 1.0, vec->bit_size)};

   for (unsigned i = 0; i < 4; ++i) {
      nir_def *chan = nir_channel_or_undef(b, vec, i - component);
      chans[i] = chans[i] ? nir_bcsel(b, replace, chans[i], chan) : chan;
   }

   nir_def *new_vec = nir_vec(b, &chans[component], intr->def.num_components);
   nir_def_rewrite_uses_after(&intr->def, new_vec);
   return true;
}

/* Gallium's software image handle carries a byte offset into texture_base.
 * Native SMP and IMGWCI index the same table in descriptor units. Keep the
 * byte handle for descriptor crawls and replace only the image load/store use.
 * Reads take a zero-extended 64-bit index; stores retain the 32-bit index ABI
 * and explicit heap. Already-native 64-bit read handles need no conversion.
 */
static bool
agx_nir_lower_g18_image_index(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   const bool store = intr->intrinsic == nir_intrinsic_bindless_image_store;
   if (!store && intr->intrinsic != nir_intrinsic_bindless_image_load &&
       intr->intrinsic != nir_intrinsic_bindless_image_sparse_load)
      return false;

   nir_intrinsic_instr *handle =
      nir_def_as_intrinsic_or_null(intr->src[0].ssa);
   if (!handle || handle->intrinsic != nir_intrinsic_bindless_image_agx ||
       nir_src_bit_size(handle->src[0]) != 32)
      return false;

   const unsigned descriptor_size = *(const unsigned *)data;
   assert(descriptor_size == AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE);
   b->cursor = nir_before_instr(&intr->instr);
   nir_def *index = nir_ushr_imm(b, handle->src[0].ssa,
                                util_logbase2(descriptor_size));
   nir_def *native = nir_bindless_image_agx(
      b, store ? index : nir_u2u64(b, index),
      .desc_set = store ? nir_intrinsic_desc_set(handle) : 0);
   nir_src_rewrite(&intr->src[0], native);
   return true;
}

/* hk_g17p_native_cprog_qualified(): the bounded native protocol launches it */
static bool
agx_g17p_fs_cprog_qualified(const struct agx_shader_info *info)
{
   return info->has_preamble && info->preamble_size &&
          info->agx3_preamble_staging_known && info->nr_preamble_gprs &&
          info->nr_preamble_gprs <= 256 &&
          (info->preamble_scratch_size ==
              info->agx3_preamble_inputs.storage_bytes ||
           agx3_cprog_static_spill_info_valid(info)) &&
          (!!info->agx3_preamble_inputs.count ==
           info->agx3_preamble_tps_inputs);
}

/* hk_agx3_compute_cprog_supported(): native AGX3 compute constant program
 * launch contract (G17P). */
static bool
agx_g17p_cs_cprog_qualified(const struct agx_shader_info *info)
{
   return agx_g17p_fs_cprog_qualified(info) && info->agx3_staging_known &&
          MAX2(info->scratch_size, info->agx3_spectpr_scratch_size) <=
             AGX3_G17P_MAX_SPILL_B &&
          info->local_size <= 32768 &&
          info->preamble_scratch_size <= AGX3_G17P_MAX_SPILL_B &&
          info->preamble_scratch_size >=
             info->agx3_preamble_inputs.storage_bytes &&
          info->texture_state_count == 0 && !info->uses_sampler_heap &&
          info->sampler_state_count == 0 && !info->uses_ray_traversal &&
          !info->uses_resource_info_query && !(info->preamble_offset & 1) &&
          !(info->preamble_size & 1) &&
          info->preamble_offset <= info->binary_size &&
          info->preamble_size <= info->binary_size - info->preamble_offset;
}

/*
 * Compile a NIR shader. The only lowering left at this point is sysvals. The
 * shader key should have already been applied. agx_compile_variant may call
 * this multiple times if there are auxiliary shaders.
 */
static struct agx_compiled_shader *
agx_compile_nir_tile(struct agx_device *dev, nir_shader *nir,
                     struct util_debug_callback *debug, mesa_shader_stage stage,
                     bool internal_kernel, bool terminal, bool secondary,
                     unsigned cf_base, unsigned nr_samples,
                     const struct agx_fs_epilog_key *epilog,
                     BITSET_WORD *attrib_components_read,
                     bool g18_native_mixed);

static struct agx_compiled_shader *
agx_compile_nir(struct agx_device *dev, nir_shader *nir,
                struct util_debug_callback *debug, mesa_shader_stage stage,
                bool internal_kernel, bool terminal, bool secondary,
                unsigned cf_base, unsigned nr_samples,
                const struct agx_fs_epilog_key *epilog,
                BITSET_WORD *attrib_components_read)
{
   return agx_compile_nir_tile(dev, nir, debug, stage, internal_kernel,
                               terminal, secondary, cf_base, nr_samples,
                               epilog, attrib_components_read, false);
}

/* Set key->g17p_tile_partitions to the colour tile partitions a fragment
 * epilog's tile accesses address. */
static void
agx_epilog_tile_partitions(const struct agx_fs_epilog_key *epilog,
                           unsigned nr_samples, struct agx_shader_key *key)
{
   const struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout_with_limits(
         epilog->rt_formats, ARRAY_SIZE(epilog->rt_formats), nr_samples, true,
         epilog->max_sample_size_B, epilog->max_bytes_per_tile,
         epilog->max_samples_per_tile);

   BITSET_ZERO(key->g17p_tile_partitions);
   for (unsigned rt = 0; rt < ARRAY_SIZE(epilog->rt_formats); ++rt) {
      if (tib.logical_format[rt] != PIPE_FORMAT_NONE && !tib.spilled[rt])
         BITSET_SET(key->g17p_tile_partitions, tib._offset_B[rt]);
   }
   if (BITSET_COUNT(key->g17p_tile_partitions) < 2)
      BITSET_ZERO(key->g17p_tile_partitions);
}

/* Keep a fragment main's NIR and backend key as its backend compile takes
 * them, for agx_compile_inline_blend_fs(). Without memory the main simply
 * keeps blending in its split epilog. */
static void
agx_keep_inline_blend_nir(struct agx_compiled_shader *compiled,
                          const nir_shader *nir,
                          const struct agx_shader_key *key)
{
   struct blob blob;
   blob_init(&blob);
   nir_serialize(&blob, nir, true);
   if (blob.out_of_memory) {
      blob_finish(&blob);
      return;
   }

   blob_finish_get_buffer(&blob, &compiled->inline_blend_nir,
                          &compiled->inline_blend_nir_size);
   compiled->inline_blend_key = *key;
}

/* As agx_compile_nir(); g18_native_mixed marks a fragment epilog of a G18
 * single-sample mixed-format tile (run 1139) for the backend's tile protocol. */
static struct agx_compiled_shader *
agx_compile_nir_tile(struct agx_device *dev, nir_shader *nir,
                     struct util_debug_callback *debug, mesa_shader_stage stage,
                     bool internal_kernel, bool terminal, bool secondary,
                     unsigned cf_base, unsigned nr_samples,
                     const struct agx_fs_epilog_key *epilog,
                     BITSET_WORD *attrib_components_read,
                     bool g18_native_mixed)
{
   struct agx_compiled_shader *compiled = CALLOC_STRUCT(agx_compiled_shader);
   compiled->stage = stage;
   if (attrib_components_read)
      BITSET_COPY(compiled->attrib_components_read, attrib_components_read);

   struct agx_shader_key key = {
      .dev = agx_gather_device_key(dev),
      .has_scratch = !secondary,
      /* Apple's G17P constant-colour fragment keeps literals in the main
       * program (the byte-exact fs-flat-colour compiler ratchet covers that
       * sequence). A promoted literal instead requires an Ioto direct-uniform
       * range before a fragment with no asynchronous staging; that unwitnessed
       * combination retires without shading on hardware. Keep ordinary UBO
       * uniforms and proven compute/vertex promotion, but materialize fragment
       * literals through the native immediate path. */
      .promote_constants =
         !(agx_chip_native_or_m5(dev->chip) &&
           stage == MESA_SHADER_FRAGMENT),
      .no_stop = !terminal,
      .secondary = secondary,
      .g17p_emrt = epilog && epilog->native_spills,
       /* G17P binds the linked executable through a native Ioto program with
        * no executable preshader contract. Keep all stages monolithic,
        * including geometry/tessellation lowered to compute: `stage` still
        * names their original API stage here. G18G likewise: its draws and
        * dispatches refuse a preamble (the M5 draw profile's "vertex
        * preamble", agx_g17p_append_compute()) and it has no AGX2 fallback
        * to run one, so a tessellation or geometry stage with a preamble
        * could never launch there. M5 likewise, now that its tessellation
        * and geometry stages run on the GPU too (hk refuses their
        * preambles on M5 as well). */
       .no_preamble = agx_chip_agx3_path(dev->chip),
      .g18_native_mixed = g18_native_mixed &&
         agx_chip_has_cap(dev->chip, AGX_CAP_NATIVE_MIXED_EPILOG),
   };

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* G17P fragment Layer is a flat coefficient, as in Honeykrisp. The
       * compute grid-Z register is not a fragment Layer system value. */
      NIR_PASS(_, nir, agx_nir_lower_interpolation,
               agx_chip_native_or_g18g(dev->chip));
   }

   /* Coverage updates (gl_SampleMask writes, discards) are compiled for the
    * rasterization sample count: at one sample G17G reads SR2 and keeps one
    * sample, at two and four it reads SR39 (agx_compile.c sample_mask).
    * Without the count a 2x/4x fragment masked its kills to sample 0 and
    * stored every other sample (sample_variables.sample_mask.*). */
   if (nir->info.stage == MESA_SHADER_FRAGMENT && nr_samples > 1)
      key.fs.nr_samples = nr_samples;

   /* We always use dynamic sample shading in the GL driver. Indicate that. */
   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       nir->info.fs.uses_sample_shading)
      key.fs.inside_sample_loop = true;

   const bool g18_shared_uniform_base = dev->chip == AGX_CHIP_G18G &&
      nir->info.stage == MESA_SHADER_COMPUTE && nir->info.shared_size;

   if (internal_kernel) {
      key.reserved_preamble = 8;
   } else if (!secondary) {
      /* G18G takes the same state-0 write: native run112 kick 19's
       * write-only texture2d kernel addresses its PBE through uniform halves
       * 0..3 with descriptor index 0 (imgwci f79 0, f91 0), exactly this
       * lowering's contract; only its root differs (agx_launch). */
      compiled->g17p_storage_write =
         dev->chip == AGX_CHIP_G18G && stage == MESA_SHADER_COMPUTE &&
         agx_nir_lower_g17p_gallium_storage_write(nir, 1);
      const enum agx_descriptor_generation descriptor_generation =
         agx_chip_descriptor_generation(dev->chip);
      const unsigned texture_descriptor_size =
         agx_gallium_texture_descriptor_size(descriptor_generation);
      if (!texture_descriptor_size)
         agx_native_descriptor_rejected(dev, "shader descriptor geometry");

      NIR_PASS(_, nir, agx_nir_lower_sysvals, stage, true,
               texture_descriptor_size);
      if (dev->chip == AGX_CHIP_G18G) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  agx_nir_lower_g18_image_index, nir_metadata_control_flow,
                  (void *)&texture_descriptor_size);
      }
      /* Every stage but the fragment one reaches the GPU as a compute
       * dispatch or a hardware vertex program on a native AGX3 chip --
       * geometry and tessellation control run as compute, tessellation
       * evaluation as the vertex shader (keeping its API stage for sysvals)
       * -- and both carry the texture and sampler table pointers in uniform
       * halves 0..7. G17G publishes the same table for every stage but the
       * fragment one too. */
      compiled->g17p_texture_resources =
         agx_chip_agx3_path(dev->chip) &&
         stage != MESA_SHADER_FRAGMENT &&
         (BITSET_LAST_BIT(nir->info.textures_used) != 0 ||
          BITSET_LAST_BIT(nir->info.images_used) != 0);
      unsigned uniform_base =
         compiled->g17p_storage_write
            ? (dev->chip == AGX_CHIP_G18G ? AGX_G18G_STORAGE_ROOT_HALFS
                                          : AGX_G17P_STORAGE_ROOT_HALFS)
            : compiled->g17p_texture_resources
                 ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
                 : 0;
      /* The shared-memory base is hardware-preloaded at half 8. A pushed
       * constant-data pointer after the eight-half texture prefix otherwise
       * overwrites it, making local stores disappear and loads return zero. */
      if (g18_shared_uniform_base)
         uniform_base = MAX2(uniform_base, AGX_ABI_G18_SHARED_UNIFORM_HALFS);

      /* The native Ioto loaders (G17P, the M5 family, G18G) move uniform
       * dwords: lay the ranges out so that each starts at an even half and
       * covers an even count. The M5 family's program builder refuses any
       * other shape, which dropped most GL fragment programs there. */
      NIR_PASS(_, nir, agx_nir_layout_uniforms, compiled,
               &key.reserved_preamble, uniform_base,
               agx_chip_agx3_path(dev->chip));
      if (agx_chip_native_agx3(dev->chip))
         NIR_PASS(_, nir, agx_nir_lower_g17p_storage_records,
                  texture_descriptor_size);
   }

   /* Internal programs also need to keep preamble exports above the base. */
   if (g18_shared_uniform_base)
      key.reserved_preamble =
         MAX2(key.reserved_preamble, AGX_ABI_G18_SHARED_UNIFORM_HALFS);

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      key.fs.cf_base = cf_base;
      key.fs.nr_samples = nr_samples;
   }

   if (epilog)
      agx_epilog_tile_partitions(epilog, nr_samples, &key);

   if (key.g17p_emrt) {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_metadata_require(impl, nir_metadata_dominance);
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            const nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            if ((op == nir_intrinsic_store_local_pixel_agx ||
                 op == nir_intrinsic_load_local_pixel_agx) &&
                !nir_block_dominates(block, nir_impl_last_block(impl))) {
               key.g17p_emrt = false;
               mesa_loge("asahi: G17P eMRT has conditional resident tile access");
            }
         }
      }
   }

   /* A native AGX3 fragment main that a colour epilog follows may later be
    * compiled together with it (agx_compile_inline_blend_fs()), so keep it as
    * the backend takes it. A main that runs per sample or reads the tile
    * itself always keeps its split epilog. */
   if (agx_chip_native_agx3(dev->chip) && agx_chip_native_preamble(dev->chip) &&
       stage == MESA_SHADER_FRAGMENT &&
       nir->info.stage == MESA_SHADER_FRAGMENT && !terminal && !secondary &&
       !internal_kernel && !nir->info.internal &&
       !nir->info.fs.uses_sample_shading && !nir->info.fs.uses_fbfetch_output)
      agx_keep_inline_blend_nir(compiled, nir, &key);

   /* G17P fragment and compute constant programs, mirroring Borealis: a
    * shader that reads descriptor-backed memory compiles a clone with the
    * native constant program and keeps it only when the native launch protocol
    * admits it; anything else compiles as before, without a preamble. */
   bool cprog_done = false;
   const bool try_fs_cprog =
      agx_chip_native_preamble(dev->chip) && stage == MESA_SHADER_FRAGMENT &&
      !secondary && !internal_kernel && key.no_preamble &&
      (nir->info.num_ubos || nir->info.num_ssbos || nir->info.num_textures ||
       nir->info.num_images);
   const bool try_cs_cprog =
      agx_chip_native_agx3(dev->chip) && agx_chip_native_preamble(dev->chip) &&
      stage == MESA_SHADER_COMPUTE && nir->info.stage == MESA_SHADER_COMPUTE &&
      !secondary && !internal_kernel && key.no_preamble &&
      (nir->info.num_ubos || nir->info.num_ssbos || nir->info.num_textures ||
       nir->info.num_images);
   if (try_fs_cprog || try_cs_cprog) {
      nir_shader *attempt = nir_shader_clone(NULL, nir);
      struct agx_shader_key ckey = key;
      ckey.no_preamble = false;
      ckey.g17p_native_uniforms = true;
      if (try_cs_cprog)
         ckey.agx3_native_cprog = true;
      const struct agx_shader_key cprog_key = ckey;
      if (attempt && agx_compile_shader_nir(attempt, &ckey, &compiled->b)) {
         const bool qualified =
            !compiled->b.info.has_preamble ||
            (try_fs_cprog
                ? agx_g17p_fs_cprog_qualified(&compiled->b.info)
                : (agx_g17p_cs_cprog_qualified(&compiled->b.info) &&
                   (!compiled->b.info.agx3_preamble_tps_inputs ||
                    compiled->push_range_count ||
                    compiled->g17p_texture_resources ||
                    compiled->b.info.rodata.size_16)));
         if (qualified) {
            cprog_done = true;
            if (try_fs_cprog)
               compiled->inline_blend_key = cprog_key;
         } else {
            free(compiled->b.binary);
            memset(&compiled->b, 0, sizeof(compiled->b));
         }
      } else {
         memset(&compiled->b, 0, sizeof(compiled->b));
      }
      ralloc_free(attempt);
   }

   if (!cprog_done && !agx_compile_shader_nir(nir, &key, &compiled->b)) {
      /* Fail closed and say so. agx_compile_shader_nir() zeroes its output on
       * refusal, so what comes back here is an empty shader part with no
       * binary and no BO. Record that on the variant: the draw and dispatch
       * paths must skip it rather than dereference the missing BO.
       *
       * Which instruction was refused is only printed under
       * AGX_MESA_DEBUG=shaders, because only the operand shape makes it
       * actionable; name the stage unconditionally so the gap is visible
       * without a rebuild.
       */
      compiled->failed = true;
      const char *refusal = agx_last_compile_refusal();
      /* A geometry shader is lowered into several programs that all report
       * this stage, so name the shader too: which one refused decides where
       * the gap is. */
      mesa_loge("asahi: the AGX backend cannot encode this %s shader (%s) for "
                "%s; draws and dispatches using it will be skipped: %s "
                "(AGX_MESA_DEBUG=shaders names the instruction)",
                _mesa_shader_stage_to_abbrev(stage),
                nir->info.name ? nir->info.name : "unnamed",
                agx_chip_name(dev->chip),
                refusal && refusal[0] ? refusal : "no reason recorded");
      return compiled;
   }

   agx2_stats_util_debug(debug, _mesa_shader_stage_to_abbrev(nir->info.stage),
                         &compiled->b.info.stats);

   if (compiled->b.info.binary_size && !secondary) {
      compiled->bo = agx_bo_create(dev, compiled->b.info.binary_size, 0,
                                   AGX_BO_EXEC | AGX_BO_LOW_VA, "Executable");

      memcpy(agx_bo_map(compiled->bo), compiled->b.binary,
             compiled->b.info.binary_size);
   }

   agx_compute_shader_upload_sizes(compiled);
   return compiled;
}

static struct agx_compiled_shader *
agx_build_meta_shader_internal(struct agx_context *ctx,
                               meta_shader_builder_t builder, void *data,
                               size_t data_size, bool prolog, bool epilog,
                               unsigned cf_base, unsigned nr_samples,
                               bool internal_kernel);

static void
agx_gallium_preprocess_nir(struct agx_device *dev, nir_shader *nir)
{
   if (dev->chip == AGX_CHIP_G18G)
      agx_preprocess_nir_g18g(nir);
   else
      agx_preprocess_nir_for_encoding(nir, agx_chip_isa_encoding(dev->chip));
}

/* M5 draws a patch draw too large for one render in parts
 * (agx_draw_patches_split). Each part numbers its patches from its first
 * one, so the API patch ID adds poly_tess_params::first_patch: in the
 * control and evaluation shaders' own reads, and in the patch ID an
 * evaluation shader hands a geometry shader.
 */
static nir_def *
agx_nir_load_first_patch(nir_builder *b)
{
   nir_def *addr = nir_iadd_imm(b, nir_load_tess_param_buffer_poly(b),
                                offsetof(struct poly_tess_params, first_patch));
   return nir_load_global_constant(b, 1, 32, addr, .align_mul = 4);
}

static bool
agx_nir_offset_patch_id(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic == nir_intrinsic_load_primitive_id) {
      b->cursor = nir_after_instr(&intr->instr);
      nir_def *id = nir_iadd(b, &intr->def, agx_nir_load_first_patch(b));
      nir_def_rewrite_uses_after(&intr->def, id);
      return true;
   }

   if (intr->intrinsic == nir_intrinsic_store_output &&
       nir_intrinsic_io_semantics(intr).location == VARYING_SLOT_PRIMITIVE_ID) {
      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(&intr->src[0], nir_iadd(b, intr->src[0].ssa,
                                              agx_nir_load_first_patch(b)));
      return true;
   }

   return false;
}

/* Likewise the API instance ID of a software VS adds
 * poly_vertex_params::instance_offset. Its output layout keeps the
 * dispatch's instance ID (poly_nir_lower_vs_before_gs), which the prolog
 * exports unchanged.
 */
static bool
agx_nir_offset_instance_id(nir_builder *b, nir_intrinsic_instr *intr,
                           void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_instance_id)
      return false;

   b->cursor = nir_after_instr(&intr->instr);
   nir_def *id = nir_iadd(b, &intr->def, poly_nir_load_instance_offset(b));
   nir_def_rewrite_uses_after(&intr->def, id);
   return true;
}

/* A sampler's LOD bias query that still names a fixed sampler slot directly
 * after binding lowering. nir_build_texture_query(..., nir_texop_lod_bias,
 * false, false) only copies binding sources (derefs, offsets, handles), and
 * agx_nir_lower_bindings() may replace dynamic offsets with texture/sampler
 * handles. Requiring zero sources rejects offsets, handles, derefs, and any
 * backend binding source so only direct fixed-slot queries are keyed/folded. */
static bool
agx_direct_lod_bias_query(const nir_tex_instr *tex)
{
   return tex->op == nir_texop_lod_bias &&
          tex->num_srcs == 0 &&
          tex->sampler_index < 32;
}

uint32_t
agx_lod_bias_samplers(nir_shader *nir)
{
   uint32_t mask = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_tex)
               continue;
            nir_tex_instr *tex = nir_instr_as_tex(instr);
            if (agx_direct_lod_bias_query(tex))
               mask |= BITFIELD_BIT(tex->sampler_index);
         }
      }
   }
   return mask;
}

/* Replace a sampler's LOD bias query by the constant zero the sampler holds.
 * The query would load the same half-float value, so the program computes
 * exactly what it would with the bias added in. Constant folding then turns
 * txb(0) back into tex, which samples at the implicit level. */
static bool
agx_fold_zero_lod_bias(nir_builder *b, nir_instr *instr, void *data)
{
   const uint32_t zero = *(const uint32_t *)data;
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (!agx_direct_lod_bias_query(tex) ||
       !(zero & BITFIELD_BIT(tex->sampler_index)))
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def_replace(&tex->def, nir_imm_zero(b, 1, tex->def.bit_size));
   return true;
}

bool
agx_nir_fold_zero_lod_bias(nir_shader *nir, uint32_t zero_lod_bias)
{
   return nir_shader_instructions_pass(nir, agx_fold_zero_lod_bias,
                                       nir_metadata_control_flow,
                                       &zero_lod_bias);
}

/* Does not take ownership of key. Clones if necessary. */

static struct agx_compiled_shader *
agx_compile_variant(struct agx_device *dev,
                    struct util_debug_callback *debug_cb,
                    struct agx_uncompiled_shader *so,
                    union asahi_shader_key *key_)
{
   struct blob_reader reader;
   blob_reader_init(&reader, so->serialized_nir.data, so->serialized_nir.size);
   nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);

   /* Compiler statistics are GL_DEBUG_SOURCE_SHADER_COMPILER notifications.
    * Report them for application shaders only. Utility shaders (u_blitter,
    * state-tracker and driver meta/passthrough shaders) are created lazily and
    * cached per context, so their notifications would appear inside unrelated
    * later API calls, once per context, and the count an application sees
    * would depend on earlier work rather than on its own commands. */
   struct util_debug_callback *debug =
      debug_cb && (dev->chip != AGX_CHIP_G18G || so->report_shader_info)
         ? debug_cb
         : NULL;

   /* Auxiliary programs */
   struct poly_gs_info gs_info = {0};
   uint64_t outputs = 0;
   struct agx_fs_epilog_link_info epilog_key = {false};
   bool g17g_samples_unrolled = false;
   nir_shader *gs_count = NULL;
   nir_shader *gs_copy = NULL;
   nir_shader *pre_gs = NULL;
   BITSET_DECLARE(attrib_components_read, VERT_ATTRIB_MAX * 4) = {0};

   /* This can happen at inopportune times and cause jank, log it */
   perf_debug(dev, "Compiling %s shader variant #%u",
              _mesa_shader_stage_to_abbrev(so->type),
              _mesa_hash_table_num_entries(so->variants));

   /* G17G has no proven custom border colour either; it shares G17P's
    * sampler table with the clamp-to-zero bank. */
   const bool custom_borders =
      (agx_chip_native_or_g18g(dev->chip) &&
       ((so->type == MESA_SHADER_FRAGMENT && key_->fs.custom_borders) ||
        ((so->type == MESA_SHADER_VERTEX ||
          so->type == MESA_SHADER_TESS_EVAL) &&
         key_->vs.custom_borders) ||
        (so->type == MESA_SHADER_COMPUTE && key_->cs.custom_borders))) ||
      (agx_chip_g17_m5(dev->chip) &&
       ((so->type == MESA_SHADER_FRAGMENT && key_->fs.custom_borders) ||
        ((so->type == MESA_SHADER_VERTEX ||
          so->type == MESA_SHADER_TESS_EVAL) &&
         key_->vs.custom_borders)));
   bool custom_border_supported = true;
   if (custom_borders && agx_chip_g17_m5(dev->chip)) {
      NIR_PASS(_, nir, agx_nir_lower_custom_border_g17g);
   } else if (custom_borders) {
      NIR_PASS(_, nir, agx_nir_lower_custom_border_native,
               agx_shadow_gather_custom_border(dev->chip),
               dev->chip == AGX_CHIP_G18G,
               &custom_border_supported);
   }
   struct agx_unlinked_uvs_layout uvs = {0};
   bool translucent = false;

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      struct asahi_vs_shader_key *key = &key_->vs;

      if (nir->info.vs.tes_poly) {
         const bool split = agx_chip_g17_m5(dev->chip);
         if (split)
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     agx_nir_offset_patch_id, nir_metadata_control_flow, NULL);
         NIR_PASS(_, nir, poly_nir_lower_tes, key->hw);
         /* The patch ID stored for a geometry shader */
         if (split && !key->hw)
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     agx_nir_offset_patch_id, nir_metadata_control_flow, NULL);
      } else {
         NIR_PASS(_, nir, agx_nir_gather_vs_inputs, attrib_components_read);
         NIR_PASS(_, nir, agx_nir_lower_vs_input_to_prolog);
      }

      if (key->hw) {
         if (key->omit_point_size) {
            /* Point size is ignored for lines. Keep the native G18 line
             * output layout instead of inserting an unused point-size slot.
             * XFB stores are handled before this rasterizer-only lowering.
             */
            NIR_PASS(_, nir, nir_remove_outputs, MESA_SHADER_FRAGMENT, 0,
                     VARYING_BIT_PSIZ);
            nir->info.outputs_written &= ~VARYING_BIT_PSIZ;
         } else {
            NIR_PASS(_, nir, agx_nir_lower_point_size, true);
         }
         NIR_PASS(_, nir, agx_nir_lower_clip_m1_1);

         NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
                  NULL);
         NIR_PASS(_, nir, agx_nir_lower_cull_distance_vs);
         if (key->g17g_link_uvs) {
            nir->info.known_interpolation_qualifiers = true;
            nir->info.linear_varyings = key->linear_varyings;
            nir->info.perspective_varyings =
               ~(key->flat_varyings | key->linear_varyings);
         }
         if (key->g17g_fs_inputs) {
            uint64_t read = key->g17g_fs_inputs;
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     agx_drop_unread_varying, nir_metadata_control_flow,
                     &read);
            NIR_PASS(_, nir, nir_opt_dce);
            nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
         }
         if (dev->chip == AGX_CHIP_G18G) {
            if (key->layer_as_varying)
               NIR_PASS(_, nir, agx_nir_lower_default_layer);
            NIR_PASS(_, nir, agx_nir_lower_uvs, &uvs, key->layer_as_varying,
                     false, -1);
         } else {
            /* The M5 profile renders one layer per draw: a layered
             * framebuffer is drawn a layer at a time, a vertex shader's
             * gl_Layer into it takes the CPU vertex stages
             * (agx_swtnl_applies), and a fragment shader reads gl_Layer
             * from the draw uniforms (agx_nir_lower_g17g_fs_layer). So a
             * hardware vertex shader's layer export reaches nothing, while
             * the layer/viewport UVS group it adds is outside every shape
             * the M5 draw profile admits. st/mesa's PBO vertex shader
             * (st_pbo_create_vs) writes gl_Layer for every upload and
             * download, which dropped all of those draws.
             */
            if (agx_chip_g17_m5(dev->chip) &&
                (nir->info.outputs_written & VARYING_BIT_LAYER)) {
               NIR_PASS(_, nir, nir_remove_outputs, MESA_SHADER_FRAGMENT,
                        VARYING_BIT_LAYER, VARYING_BIT_LAYER);
               nir->info.outputs_written &= ~VARYING_BIT_LAYER;
            }
            NIR_PASS(_, nir, agx_nir_lower_uvs, &uvs,
                     agx_chip_native_agx3(dev->chip), false, -1);
         }
      } else {
         if (agx_chip_g17_m5(dev->chip) && !nir->info.vs.tes_poly)
            NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                     agx_nir_offset_instance_id, nir_metadata_control_flow,
                     NULL);
         NIR_PASS(_, nir, poly_nir_lower_vs_before_gs);

         /* Turn into a compute shader now that we're free of vertexisms */
         nir->info.stage = MESA_SHADER_COMPUTE;
         memset(&nir->info.cs, 0, sizeof(nir->info.cs));
         nir->xfb_info = NULL;
         outputs = nir->info.outputs_written;
      }
   } else if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
      /* The dispatch uses one workgroup per patch and one invocation per
       * output vertex. In particular, nir_create_passthrough_tcs() inherits
       * the builder's default 1x1x1 size; leaving that in place lets poly
       * fold every invocation ID to zero, so only control point zero is
       * copied. Declare the same shape that agx_draw_patches() launches. */
      nir->info.workgroup_size[0] = nir->info.tess.tcs_vertices_out;
      nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
      if (agx_chip_g17_m5(dev->chip))
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_nir_offset_patch_id,
                  nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, poly_nir_lower_tcs, true);
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      NIR_PASS(_, nir, poly_nir_lower_gs, &gs_count, &gs_copy, &pre_gs,
               &gs_info);

      agx_gallium_preprocess_nir(dev, gs_count);
      agx_gallium_preprocess_nir(dev, gs_copy);
      agx_gallium_preprocess_nir(dev, pre_gs);
   } else if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      struct asahi_fs_shader_key *key = &key_->fs;
      const bool writes_sample_mask =
         nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);

      if (key->zero_lod_bias)
         NIR_PASS(_, nir, agx_nir_fold_zero_lod_bias, key->zero_lod_bias);

      /* A split G17G draw keeps one colour output, at location 0, so the
       * program is a single-attachment one (agx_g17g_split_mrt). */
      if (key->g17g_mrt_output) {
         unsigned keep = key->g17g_mrt_output - 1;
         NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_keep_one_fs_output,
                  nir_metadata_control_flow, &keep);
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      }

      const bool native_msaa_fbfetch =
         agx_chip_native_agx3(dev->chip) && key->nr_samples > 1 &&
         nir->info.fs.uses_fbfetch_output;
      if (native_msaa_fbfetch)
         nir->info.fs.uses_sample_shading = true;

      /* Discards must be lowering before lowering MSAA to handle discards */
      NIR_PASS(_, nir, agx_nir_lower_discard_zs_emit);

      /* The zero-key precompile uses the same single-sample specialization
       * as the explicit one-sample draw variant. */
      const unsigned keyed_samples = key->nr_samples ? key->nr_samples : 1;
      /* G17P has no recovered samples-log2 special register, and the variant
       * key already carries the raster sample count, so fold it into the
       * shader as hk_shader.c does. gl_SampleMaskIn, gl_SampleID and
       * interpolateAtSample() all read it. */
      if (agx_chip_native_agx3(dev->chip) && agx_nir_needs_sample_count(nir))
         NIR_PASS(_, nir, agx_nir_specialize_sample_count,
                  MAX2(key->nr_samples, 1));

      /* At one rasterization sample, per-sample and per-pixel invocation are
       * the same thing, so collapsing sample shading to per-pixel is exact --
       * and it is what has to happen here, because G17P has no sample loop
       * for the linker to splice and asking for one would drop the draw
       * outright. hk_shader.c does the same on the Vulkan side. Do it before
       * the epilog split, so the link key records the collapse too.
       * G18G has no sample loop for the linker to splice either; there a
       * multisample stencil resolve still rasterizes into a single-sample
       * destination, and its tile stores use the whole-pixel protocol. */
      if ((((agx_chip_native_agx3(dev->chip) && key->nr_samples <= 1) ||
           (dev->chip == AGX_CHIP_G18G && keyed_samples == 1)) &&
          nir->info.fs.uses_sample_shading) ||
          (agx_chip_g17_m5(dev->chip) && key->nr_samples == 1)) {
         NIR_PASS(_, nir, agx_nir_lower_g17p_1x_sample_shading);
      }

      /* Keep helper invocation intact until one-sample specialization has
       * removed per-sample execution. Lowering it in the serialized shader
       * would leave a multisample coverage test after that collapse. */
      if (dev->chip == AGX_CHIP_G18G && so->info.uses_sample_shading)
         NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, true);

      /* AGX3 has no equivalent of the AGX2 machine code agx_fast_link()
       * splices around a linked fragment program to run it once per sample.
       * Apple's own compiler unrolls the body instead, once per sample, in one
       * program.
       *
       * An unrolled main cannot carry a separate epilog: the stores the split
       * moves out -- colour, and depth/stencil, which it defers past discards
       * -- would run once for a body that ran per sample. So keep the whole
       * fragment in one program. Only the shape with nothing but depth and
       * stencil to keep is taken here, which is what a multisample depth blit
       * needs; a per-sample colour shader would also need its tilebuffer and
       * blending built into the main, and keeps its existing named refusal.
       * G18 colour shaders instead expand both linked parts with the same
       * per-sample export layout below.
       */
      if (dev->chip == AGX_CHIP_G18G &&
          (keyed_samples == 1 || keyed_samples == 2 || keyed_samples == 4) &&
          agx_nir_needs_sample_count(nir))
         NIR_PASS(_, nir, agx_nir_specialize_sample_count, keyed_samples);

      const uint64_t zs_outputs = BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                  BITFIELD64_BIT(FRAG_RESULT_STENCIL);
      const bool unrolled_samples =
         ((agx_chip_native_agx3(dev->chip) && key->nr_samples > 1) ||
          (dev->chip == AGX_CHIP_G18G &&
           (key->nr_samples == 2 || key->nr_samples == 4))) &&
         nir->info.fs.uses_sample_shading &&
         !(nir->info.outputs_written & ~zs_outputs);

      if (!unrolled_samples) {
         NIR_PASS(_, nir, agx_nir_lower_fs_output_to_epilog_for_device,
                  dev->chip, &epilog_key);
      } else {
         /* The stores stay here, but the draw-time state still has to know
          * the fragment writes them. */
         epilog_key.write_z =
            !!(nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH));
         epilog_key.write_s =
            !!(nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_STENCIL));
      }

      if (nir->info.fs.uses_fbfetch_output) {
         bool rt0_resident = false;
         for (unsigned rt = 1; rt < ARRAY_SIZE(key->rt_formats); ++rt)
            rt0_resident |= key->rt_formats[rt] != PIPE_FORMAT_NONE;
         /* As the batch decides (agx_g17g_fb_mrt_resident), from the same
          * formats and sample count. */
         rt0_resident &= agx_chip_g17_m5(dev->chip) &&
                         !agx_g17g_mrt_resident_layout(
                            dev->chip, key->rt_formats,
                            ARRAY_SIZE(key->rt_formats), key->nr_samples);

         /* As the batch lays the attachments out (agx_batch_init_state). */
         struct agx_tilebuffer_layout tib =
            agx_chip_g17_m5(dev->chip)
               ? (rt0_resident
                     ? agx_build_tilebuffer_layout_rt0_resident(
                          key->rt_formats, ARRAY_SIZE(key->rt_formats),
                          key->nr_samples, AGX_G17G_MAX_BYTES_PER_TILE,
                          AGX_G17G_MAX_SAMPLES_PER_TILE)
                     : agx_build_tilebuffer_layout_with_limits(
                          key->rt_formats, ARRAY_SIZE(key->rt_formats),
                          key->nr_samples, true, 0,
                          AGX_G17G_MAX_BYTES_PER_TILE,
                          AGX_G17G_MAX_SAMPLES_PER_TILE))
               : agx_gallium_tilebuffer_layout(
                    dev, key->rt_formats, ARRAY_SIZE(key->rt_formats),
                    key->nr_samples, true);

         if (dev->debug & AGX_DBG_SMALLTILE)
            tib.tile_size = 16 * 16;

         /* XXX: don't replicate this all over the driver */
         unsigned rt_spill_base = BITSET_LAST_BIT(nir->info.textures_used) +
                                  (2 * BITSET_LAST_BIT(nir->info.images_used));
         unsigned rt_spill = rt_spill_base;
         NIR_PASS(_, nir, agx_nir_lower_tilebuffer_with_conversions, &tib, NULL,
                  &rt_spill, NULL, &translucent, false,
                  dev->chip == AGX_CHIP_G18G,
                  dev->chip == AGX_CHIP_G18G,
                  dev->chip == AGX_CHIP_G18G,
                  agx_chip_native_or_m5(dev->chip),
                  dev->chip == AGX_CHIP_G18G);
         if (rt0_resident)
            NIR_PASS(_, nir, agx_nir_rebind_spilled_rts);
      }

      /* G17G has no external sample loop either. Repeat the body once per
       * sample instead, each copy with its own colour exports, as
       * Honeykrisp's G17P path and Apple's compiler do; the epilog repeats
       * itself the same way (build_g17p_fs_epilog). A shader whose
       * per-sample exports would not fit the register file keeps the
       * per-pixel form, which the link refuses. */
      /* The general register file the exports must fit: 192 halves, as
       * the compiler bounds G17P (AGX3B_NUM_REGS). */
      const unsigned export_halves = 192;
      struct agx_g17p_colour_layout sample_layout;
      bool g17g_unroll = false;
      if (!unrolled_samples && agx_chip_g17_m5(dev->chip) &&
          nir->info.fs.uses_sample_shading &&
          (key->nr_samples == 2 || key->nr_samples == 4)) {
         agx_g17p_colour_layout_init(&sample_layout, epilog_key.loc_written,
                                     epilog_key.size_32,
                                     epilog_key.colour_comps, false);
         if (agx_g17p_sample_exports_end(&sample_layout, key->nr_samples) >
             export_halves) {
            NIR_PASS(_, nir, agx_nir_pack_g17p_colour_exports, &epilog_key);
            agx_g17p_colour_layout_init(&sample_layout,
                                        epilog_key.loc_written,
                                        epilog_key.size_32,
                                        epilog_key.colour_comps, true);
         }
         g17g_unroll =
            agx_g17p_sample_exports_end(&sample_layout, key->nr_samples) <=
            export_halves;
      }

      if (unrolled_samples) {
         if (dev->chip == AGX_CHIP_G18G) {
            /* Keep memory side effects and coverage predicates in each
             * statically specialized body. The AGX2 loop otherwise depends
             * on optimizer unrolling and can reach unsupported loop forms. */
            NIR_PASS(_, nir, agx_nir_lower_sample_mask);
            NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
            NIR_PASS(_, nir, agx_nir_unroll_g17p_samples,
                     key->nr_samples, 0);
         } else {
            NIR_PASS(_, nir, agx_nir_lower_monolithic_msaa, key->nr_samples);
         }

         /* Nothing is left for the linker to wrap. */
         nir->info.fs.uses_sample_shading = false;
         epilog_key.sample_shading = false;
      } else if (dev->chip == AGX_CHIP_G18G &&
                 (key->nr_samples == 2 || key->nr_samples == 4) &&
                 nir->info.fs.uses_sample_shading && !nir->info.writes_memory) {
         /* Each invocation exports its own colours. The epilog repeats the
          * same layout and count, so no AGX2 machine-code loop is needed.
          * Shader memory side effects still require coverage predication. */
         struct agx_g17p_colour_layout layout;
         agx_g17p_colour_layout_init(&layout, epilog_key.loc_written,
                                    epilog_key.size_32, epilog_key.colour_comps,
                                    false);
         /* G18 has 252 usable register halves. The allocator rounds its
          * available file down to a whole eight-half colour slot. */
         const unsigned limit = 248;
         if (agx_g17p_sample_exports_end(&layout, key->nr_samples) > limit) {
            NIR_PASS(_, nir, agx_nir_pack_g17p_colour_exports, &epilog_key);
            agx_g17p_colour_layout_init(&layout, epilog_key.loc_written,
                                       epilog_key.size_32,
                                       epilog_key.colour_comps, true);
         }
         if (agx_g17p_sample_exports_end(&layout, key->nr_samples) > limit) {
            mesa_loge("asahi: G18 per-sample colour exports exceed %u halves",
                      limit);
            ralloc_free(nir);
            struct agx_compiled_shader *failed =
               CALLOC_STRUCT(agx_compiled_shader);
            failed->stage = so->type;
            failed->so = so;
            failed->failed = true;
            return failed;
         }

         NIR_PASS(_, nir, agx_nir_lower_sample_mask);
         NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
         NIR_PASS(_, nir, agx_nir_unroll_g17p_samples_layout,
                  key->nr_samples, &layout);
         nir->info.fs.uses_sample_shading = false;
      } else if (g17g_unroll) {
         NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
         NIR_PASS(_, nir, agx_nir_unroll_g17p_samples_layout,
                  key->nr_samples, &sample_layout);
         /* epilog_key.sample_shading stays set: the epilog unrolls too. */
         nir->info.fs.uses_sample_shading = false;
         g17g_samples_unrolled = true;
      } else if (nir->info.fs.uses_sample_shading &&
                 agx_chip_native_agx3(dev->chip)) {
         /* Use the same export layout and static sample copies as Vulkan.
          * The epilog consumes each sample's exports in the matching order.
          */
         /* A fragment tests its coverage once. Lowered before the unroll,
          * the testing update that follows the last discard is cloned with
          * the body, and the fragment then tests once per sample; a pass of
          * such fragments can stall the GPU when another context's fragment
          * work runs beside it. So unroll the demotes -- each copy keeps its
          * own sample's -- and lower them afterwards: every copy kills its
          * sample where it discards, and one update after the last copy
          * tests whatever is left of the pixel, as in a per-pixel shader
          * that discards several times.
          *
          * Forced-early fragments test once at the start already (the
          * unroll hoists that update ahead of the copies) and keep their
          * order. */
         const bool test_once = !nir->info.fs.early_fragment_tests;
         if (!test_once)
            NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred);
         if (!agx_nir_lower_g17p_sample_shading(nir, &epilog_key,
                                                key->nr_samples)) {
            mesa_loge("asahi: G17P per-sample colour exports exceed the register file");
            struct agx_compiled_shader *failed = CALLOC_STRUCT(agx_compiled_shader);
            failed->stage = so->type;
            failed->so = so;
            failed->failed = true;
            ralloc_free(nir);
            return failed;
         }
         if (test_once) {
            /* The copies' one-hot sample masks, so that the lowering sees
             * each demote's constant sample. */
            NIR_PASS(_, nir, nir_opt_constant_folding);
            /* Kills go into the one testing update where they can: a
             * kill-only update of part of a multisampled pixel loses the
             * killed samples' previous contents. Depth/stencil exports and
             * memory writes keep the kills where they happen. */
            const uint64_t zs = BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                                BITFIELD64_BIT(FRAG_RESULT_STENCIL);
            if (!(nir->info.outputs_written & zs) && !nir->info.writes_memory)
               NIR_PASS(_, nir, agx_nir_lower_sample_mask_one_update);
            else
               NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred);
         }
      } else if (nir->info.fs.uses_sample_shading) {
         /* Ensure the sample ID is preserved in register */
         nir_builder b =
            nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
         nir_export_agx(
            &b,
            nir_load_exported_agx(&b, 1, 16, .base = AGX_ABI_FIN_SAMPLE_MASK),
            .base = AGX_ABI_FOUT_SAMPLE_MASK);

         NIR_PASS(_, nir, agx_nir_lower_to_per_sample);
      }

      /* native_msaa_fbfetch is G17P-only, but G17G's own per-sample unroll
       * folds the active-sample mask to a one-hot immediate as well. The
       * packer takes only the whole-pixel 0xff immediate or a live mask
       * register, so give the unrolled copies the register form too. */
      if (native_msaa_fbfetch ||
          (g17g_samples_unrolled && key->nr_samples > 1)) {
         NIR_PASS(_, nir, nir_opt_constant_folding);
         NIR_PASS(_, nir, agx_nir_lower_g17p_tile_samples, key->nr_samples);
      }

      /* Defer only discards: gl_SampleMask writes and per-sample fragments
       * reach this pass as demotes too and keep their ordinary lowering,
       * except M5's unrolled copies (below). */
      const uint64_t m5_zs = BITFIELD64_BIT(FRAG_RESULT_DEPTH) |
                             BITFIELD64_BIT(FRAG_RESULT_STENCIL);
      if (agx_chip_g17_m5(dev->chip) && !g17g_samples_unrolled &&
          !nir->info.fs.uses_sample_shading &&
          !writes_sample_mask)
         NIR_PASS(_, nir, agx_nir_lower_sample_mask_deferred_m5);
      else if (g17g_samples_unrolled && key->nr_samples > 1 &&
               !nir->info.fs.early_fragment_tests &&
               !(nir->info.outputs_written & m5_zs) &&
               !nir->info.writes_memory)
         /* The unrolled copies' kills go into the one testing update, as on
          * G17P, in M5's uniformly selected form (the draw's test selector
          * is a uniform here): a kill-only update of part of a multisampled
          * pixel loses the killed samples' previous contents. */
         NIR_PASS(_, nir, agx_nir_lower_sample_mask_one_update_m5);
      else
         NIR_PASS(_, nir, agx_nir_lower_sample_mask);
      if (dev->chip == AGX_CHIP_G18G &&
          (key->nr_samples == 2 || key->nr_samples == 4) &&
          !nir->info.fs.early_fragment_tests) {
         NIR_PASS(_, nir, agx_nir_mark_g18_sample_mask_tests);
      }
      NIR_PASS(_, nir, agx_nir_lower_fs_active_samples_to_register);

      /* G17G has no grounded samples_log2 read (gl_SampleMask writes test
       * it); its key always carries the count. */
      if (agx_chip_g17_m5(dev->chip) &&
          (key->nr_samples == 1 || key->nr_samples == 2 ||
           key->nr_samples == 4))
         NIR_PASS(_, nir, agx_nir_specialize_sample_count, key->nr_samples);
   }

   /* The Gen9 form addresses through the storage tail record that
    * agx_g17p_fill_storage_tail() writes. */
   /* G18G adds AGX_NIR_TEXTURE_GENERATION_G18G to keep its own forms. */
   NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store_for_generation,
            agx_chip_descriptor_generation(dev->chip) |
               (dev->chip == AGX_CHIP_G18G ? AGX_NIR_TEXTURE_GENERATION_G18G
                                           : 0));

   struct agx_compiled_shader *compiled = agx_compile_nir(
      dev, nir, debug, so->type, false, so->type != MESA_SHADER_FRAGMENT,
      false, 0, so->type == MESA_SHADER_FRAGMENT ? key_->fs.nr_samples : 0,
      NULL, attrib_components_read);
   compiled->g17p_custom_border_emulation =
      custom_borders &&
      (dev->chip != AGX_CHIP_G18G || custom_border_supported);

   if (so->type == MESA_SHADER_FRAGMENT) {
      /* XXX: don't replicate this all over the driver */
      epilog_key.rt_spill_base =
         BITSET_LAST_BIT(nir->info.textures_used) +
         (so->info.image_records * BITSET_LAST_BIT(nir->info.images_used));

      compiled->epilog_key = epilog_key;
      compiled->g17g_samples_unrolled = g17g_samples_unrolled;
      compiled->b.info.reads_tib |= translucent;
   }

   compiled->so = so;
   compiled->uvs = uvs;

   /* Compile auxiliary programs */
   if (gs_count) {
      compiled->gs_count = agx_compile_nir(
         dev, gs_count, debug, so->type, false, true, false, 0, 0, NULL, NULL);
      compiled->gs_count->so = so;
   }

   if (pre_gs) {
      compiled->pre_gs =
         agx_compile_nir(dev, pre_gs, debug, MESA_SHADER_COMPUTE, false,
                         true, false, 0, 0, NULL, NULL);
   }

   if (gs_copy) {
      /* poly_nir_lower_gs writes the default point size into a point-output
       * copy program only when it also writes a position. A geometry shader
       * with points out that never writes gl_Position (for example one that
       * emits no vertex) still draws points, and the native point profile
       * exports a point size with every point draw. Give it the GL/GLES
       * default of 1.0 (nir_lower_default_point_size, the GLES geometry
       * shader rule); the replacement below then clamps it and substitutes
       * the API point size whenever the size is not per-vertex.
       */
      if (agx_chip_hal300_path(dev->chip) && so->gs_mode == MESA_PRIM_POINTS)
         NIR_PASS(_, gs_copy, nir_lower_default_point_size);

      /* Replace the point size write if present, but do not insert a write:
       * the GS rast program writes point size iff we have points.
       */
      NIR_PASS(_, gs_copy, agx_nir_lower_point_size, false);

      NIR_PASS(_, gs_copy, agx_nir_lower_clip_m1_1);

      NIR_PASS(_, gs_copy, nir_lower_io_to_scalar, nir_var_shader_out, NULL,
               NULL);
      /* HAL300 chips trim unused geometry copy outputs via asahi_gs_shader_key
       * so the rasterization copy program only exports fragment-read varyings. */
      const bool hal300_gs_key = agx_chip_hal300_path(dev->chip);
      if (hal300_gs_key && key_->gs.trim_rast_varyings) {
         uint64_t keep = key_->gs.rast_varyings;
         NIR_PASS(_, gs_copy, nir_shader_intrinsics_pass,
                  agx_nir_trim_rast_output, nir_metadata_control_flow, &keep);
         gs_copy->info.outputs_written &=
            keep | agx_rast_system_outputs();
      }
      NIR_PASS(_, gs_copy, agx_nir_lower_cull_distance_vs);

      /* agx_nir_lower_uvs inlines the UVS indices when the interpolation
       * qualifiers are known; the key carries the bound fragment shader's
       * masks in the form agx_assign_uvs takes at draw time. */
      if (key_->gs.link_uvs) {
         gs_copy->info.known_interpolation_qualifiers = true;
         gs_copy->info.linear_varyings = key_->gs.linear_varyings;
         gs_copy->info.perspective_varyings =
            ~(key_->gs.flat_varyings | key_->gs.linear_varyings);
      }

      struct agx_unlinked_uvs_layout uvs = {0};
      if (dev->chip == AGX_CHIP_G18G) {
         const bool layer_as_varying =
            key_->gs.flat_varyings & VARYING_BIT_LAYER;
         if (layer_as_varying)
            NIR_PASS(_, gs_copy, agx_nir_lower_default_layer);
         NIR_PASS(_, gs_copy, agx_nir_lower_uvs, &uvs, layer_as_varying,
                  false, -1);
      } else {
         NIR_PASS(_, gs_copy, agx_nir_lower_uvs, &uvs,
                  agx_chip_native_agx3(dev->chip), false, -1);
      }

      compiled->gs_copy =
         agx_compile_nir(dev, gs_copy, debug, MESA_SHADER_GEOMETRY,
                         false, true, false, 0, 0, NULL, NULL);
      compiled->gs_copy->so = so;
      compiled->gs_copy->stage = so->type;
      compiled->gs_copy->uvs = uvs;
   }

   compiled->gs = gs_info;
   compiled->b.info.outputs = outputs;

   /* An auxiliary program is as load-bearing as the main one. A geometry
    * variant whose count, pre-GS or copy program was refused cannot run, so
    * the variant as a whole is unusable.
    */
   if ((compiled->gs_count && compiled->gs_count->failed) ||
       (compiled->pre_gs && compiled->pre_gs->failed) ||
       (compiled->gs_copy && compiled->gs_copy->failed))
      compiled->failed = true;

   ralloc_free(nir);
   ralloc_free(pre_gs);
   ralloc_free(gs_count);
   ralloc_free(gs_copy);
   return compiled;
}

static struct agx_compiled_shader *
agx_get_shader_variant(struct agx_screen *screen,
                       struct util_debug_callback *debug_cb,
                       struct agx_uncompiled_shader *so,
                       union asahi_shader_key *key)
{
   struct agx_compiled_shader *compiled =
      agx_disk_cache_retrieve(screen, so, key);

   if (!compiled) {
      compiled = agx_compile_variant(&screen->dev, debug_cb, so, key);

      /* The serialized form carries only the binary and its info, so a
       * refused variant would come back from the cache indistinguishable
       * from a legitimately empty one -- and silently, on a later run. Keep
       * refusals out of it; they are cheap to re-derive and must stay
       * visible.
       */
      if (!compiled->failed)
         agx_disk_cache_store(screen->disk_cache, so, key, compiled);
   }

   /* key may be destroyed after we return, so clone it before using it as a
    * hash table key. The clone is logically owned by the hash table.
    */
   union asahi_shader_key *cloned_key =
      rzalloc(so->variants, union asahi_shader_key);

   if (so->type == MESA_SHADER_FRAGMENT) {
      memcpy(cloned_key, key, sizeof(struct asahi_fs_shader_key));
   } else if (so->type == MESA_SHADER_VERTEX ||
              so->type == MESA_SHADER_TESS_EVAL) {
      memcpy(cloned_key, key, sizeof(struct asahi_vs_shader_key));
   } else if (so->type == MESA_SHADER_COMPUTE) {
      memcpy(cloned_key, key, sizeof(struct asahi_cs_shader_key));
   } else if (so->type == MESA_SHADER_GEOMETRY) {
      memcpy(cloned_key, key, sizeof(struct asahi_gs_shader_key));
   } else {
      /* No key */
   }

   _mesa_hash_table_insert(so->variants, cloned_key, compiled);

   return compiled;
}

/* Forward declared for the precompile-error path below. */
static void agx_delete_uncompiled_shader(struct agx_device *dev,
                                         struct agx_uncompiled_shader *so);

/*
 * st/mesa asks drivers to report failures from the default variant while it
 * is linking a GLSL program. The AGX compiler is deliberately fail-closed,
 * but merely retaining a failed variant here makes glLinkProgram() succeed
 * and postpones the error until the first draw is dropped. WebGL treats a
 * successful link as an executable program, so surface the refusal through
 * pipe_shader_state::error_message and return no CSO when requested.
 *
 * Non-link-time users keep the failed CSO. The draw/dispatch guards still
 * consume it safely and this preserves the existing behaviour for internal
 * shaders and late key-specific variants.
 */
static bool
agx_precompile_variant(struct agx_device *dev, struct pipe_context *pctx,
                       struct agx_uncompiled_shader *so,
                       const struct pipe_shader_state *cso,
                       union asahi_shader_key *key)
{
   struct agx_compiled_shader *compiled =
      agx_get_shader_variant(agx_screen(pctx->screen), &pctx->debug, so, key);

   if (!cso->report_compile_error || (compiled && !compiled->failed))
      return true;

   /* G17G refuses instruction shapes it has no grounded encoding for. The
    * program is valid GLSL, so link it: its interface can be queried, and
    * the draws using it are dropped with a named refusal. */
   if (agx_chip_g17_m5(dev->chip) && compiled)
      return true;

   char message[192];
   snprintf(message, sizeof(message),
            "asahi: the AGX backend cannot encode the default %s shader "
            "variant for %s",
            _mesa_shader_stage_to_string(so->type), agx_chip_name(dev->chip));
   ((struct pipe_shader_state *)cso)->error_message = strdup(message);

   agx_delete_uncompiled_shader(dev, so);
   return false;
}

static unsigned
glsl_type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

static bool
should_lower_robustness(const nir_intrinsic_instr *intr, const void *data)
{
   const bool *gl_robust = data;

   switch (intr->intrinsic) {
   /* The texture/PBE hardware is robust, but our buffer image implementation
    * is not. Lower robustness only for buffer images.
    */
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_store:
      return nir_intrinsic_image_dim(intr) == GLSL_SAMPLER_DIM_BUF;

   /* Image atomics are lowered to raw memory access */
   case nir_intrinsic_image_atomic:
   case nir_intrinsic_image_atomic_swap:
      return true;

   /* UBOs/SSBOs are lowered to raw pointers */
   case nir_intrinsic_load_ubo:
   case nir_intrinsic_load_ssbo:
   case nir_intrinsic_store_ssbo:
   case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap:
      return *gl_robust;

   default:
      return false;
   }
}

/* The image intrinsics that only read texels or query the texture
 * descriptor. Every other one -- stores, atomics, texel addresses (which
 * read the PBE sideband) and anything added later -- counts as a write. */
static bool
agx_image_intrinsic_only_reads(nir_intrinsic_op op)
{
   switch (op) {
   case nir_intrinsic_image_load:
   case nir_intrinsic_image_sparse_load:
   case nir_intrinsic_image_size:
   case nir_intrinsic_image_levels:
   case nir_intrinsic_image_samples:
   case nir_intrinsic_image_deref_load:
   case nir_intrinsic_image_deref_sparse_load:
   case nir_intrinsic_image_deref_size:
   case nir_intrinsic_image_deref_levels:
   case nir_intrinsic_image_deref_samples:
   case nir_intrinsic_bindless_image_load:
   case nir_intrinsic_bindless_image_sparse_load:
   case nir_intrinsic_bindless_image_size:
   case nir_intrinsic_bindless_image_levels:
   case nir_intrinsic_bindless_image_samples:
      return true;
   default:
      return false;
   }
}

static bool
agx_shader_writes_images(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_has_image_dim(intr) &&
                !agx_image_intrinsic_only_reads(intr->intrinsic))
               return true;
         }
      }
   }
   return false;
}

/* Whether some image intrinsic is neither a read nor a plain store: an
 * atomic, a texel address, or anything else whose lowering may read the
 * PBE's AGX2 software sideband (libagx_image_texel_address()). */
static bool
agx_shader_reads_image_sideband(nir_shader *nir)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (nir_intrinsic_has_image_dim(intr) &&
                !agx_image_intrinsic_only_reads(intr->intrinsic) &&
                intr->intrinsic != nir_intrinsic_image_store &&
                intr->intrinsic != nir_intrinsic_image_deref_store &&
                intr->intrinsic != nir_intrinsic_bindless_image_store)
               return true;
         }
      }
   }
   return false;
}

static nir_def *
agx_buffer_texture_metadata(nir_builder *b, nir_tex_instr *tex)
{
   const int src = nir_tex_instr_src_index(tex, nir_tex_src_texture_handle);
   assert(src >= 0);
   nir_intrinsic_instr *handle = nir_src_as_intrinsic(tex->src[src].src);
   assert(handle && handle->intrinsic == nir_intrinsic_load_texture_handle_agx);
   mesa_shader_stage stage = b->shader->info.stage;
   if (stage == MESA_SHADER_VERTEX && b->shader->info.vs.tes_poly)
      stage = MESA_SHADER_TESS_EVAL;
   nir_def *base = nir_load_sysval_agx(
      b, 1, 64, .desc_set = AGX_SYSVAL_STAGE(stage),
      .binding = offsetof(struct agx_stage_uniforms, buffer_texture_base));
   return nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(
      b, nir_u2u32(b, handle->src[0].ssa), AGX_G17P_EMRT_DESCRIPTOR_SIZE)));
}

/* G17G: a gather with a texel offset reads the wrong texels in the native
 * form. Fold the offset into the coordinate instead -- for a gather the
 * offset only shifts the 2x2 footprint, uv + offset / size -- so the
 * native implicit gather runs without one. */
/* G17G has no proven sample through a run-time texture or sampler index:
 * the native heap handle 4df8a4db53b tried read wrong texels and, after
 * enough earlier work, faulted the GPU. A dynamically uniform sampler array
 * index (GLSL ES 3.1) selects one of a few bindings, so test it against each
 * and sample that binding with constant indices. */
static bool
lower_g17g_dynamic_sampler(nir_builder *b, nir_instr *instr,
                           UNUSED void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;

   nir_tex_instr *tex = nir_instr_as_tex(instr);
   const int t = nir_tex_instr_src_index(tex, nir_tex_src_texture_offset);
   const int s = nir_tex_instr_src_index(tex, nir_tex_src_sampler_offset);
   if (t < 0 && s < 0)
      return false;

   /* GL's combined samplers offset the texture and the sampler by the same
    * index, though not always through the same SSA value. */
   nir_def *index = t >= 0 ? tex->src[t].src.ssa : tex->src[s].src.ssa;
   nir_def *sampler_index =
      t >= 0 && s >= 0 && tex->src[s].src.ssa != index ? tex->src[s].src.ssa
                                                         : NULL;

   const unsigned textures =
      MAX2(b->shader->info.num_textures,
           BITSET_LAST_BIT(b->shader->info.textures_used));
   const unsigned count =
      textures > tex->texture_index ? textures - tex->texture_index : 0;
   if (count == 0 || count > 32)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *result = nir_undef(b, tex->def.num_components, tex->def.bit_size);
   for (unsigned k = 0; k < count; ++k) {
      nir_def *hit = nir_ieq_imm(b, index, k);
      if (sampler_index)
         hit = nir_iand(b, hit, nir_ieq_imm(b, sampler_index, k));
      nir_push_if(b, hit);
      nir_tex_instr *sel = nir_instr_as_tex(nir_instr_clone(b->shader, instr));
      nir_builder_instr_insert(b, &sel->instr);
      int src = nir_tex_instr_src_index(sel, nir_tex_src_texture_offset);
      if (src >= 0)
         nir_tex_instr_remove_src(sel, src);
      src = nir_tex_instr_src_index(sel, nir_tex_src_sampler_offset);
      if (src >= 0)
         nir_tex_instr_remove_src(sel, src);
      if (t >= 0)
         sel->texture_index += k;
      if (s >= 0)
         sel->sampler_index += k;
      nir_pop_if(b, NULL);
      result = nir_if_phi(b, &sel->def, result);
   }

   nir_def_replace(&tex->def, result);
   return true;
}

static bool
lower_g17g_gather_offset(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *tex = nir_instr_as_tex(instr);
   if (tex->op != nir_texop_tg4 || tex->sampler_dim != GLSL_SAMPLER_DIM_2D)
      return false;
   const int offset_idx = nir_tex_instr_src_index(tex, nir_tex_src_offset);
   const int coord_idx = nir_tex_instr_src_index(tex, nir_tex_src_coord);
   if (offset_idx < 0 || coord_idx < 0)
      return false;

   b->cursor = nir_before_instr(instr);
   nir_def *coord = tex->src[coord_idx].src.ssa;
   nir_def *offset = nir_i2f32(b, nir_trim_vector(b, tex->src[offset_idx].src.ssa, 2));
   nir_def *size = nir_i2f32(b, nir_trim_vector(b, nir_get_texture_size(b, tex), 2));
   nir_def *xy = nir_fadd(b, nir_trim_vector(b, coord, 2), nir_fdiv(b, offset, size));
   nir_def *comps[4] = {nir_channel(b, xy, 0), nir_channel(b, xy, 1)};
   for (unsigned i = 2; i < coord->num_components; ++i)
      comps[i] = nir_channel(b, coord, i);
   nir_src_rewrite(&tex->src[coord_idx].src,
                   nir_vec(b, comps, coord->num_components));
   nir_tex_instr_remove_src(tex, offset_idx);
   return true;
}
/* Symbolically evaluate an inputless single-block vertex shader for
 * gl_VertexID in {0, 1, 2} to detect whether a 3-vertex direct draw covers
 * the full NDC square [-1, 1]^2. Returns 1 for CCW in NDC, 2 for CW in NDC,
 * or 0 otherwise. */
uint8_t
agx_nir_analyze_fullscreen_tri(nir_shader *nir, bool *out_z_in_01,
                               bool *out_z_in_m11)
{
   *out_z_in_01 = false;
   *out_z_in_m11 = false;

   if (nir->info.stage != MESA_SHADER_VERTEX || nir->info.inputs_read != 0 ||
       !(nir->info.outputs_written & VARYING_BIT_POS))
      return 0;

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl || impl->ssa_alloc == 0 || impl->ssa_alloc > 512 ||
       exec_list_length(&impl->body) != 1)
      return 0;

   nir_cf_node *node = exec_node_data(nir_cf_node, exec_list_get_head(&impl->body),
                                      node);
   if (node->type != nir_cf_node_block)
      return 0;

   nir_block *block = nir_cf_node_as_block(node);

   struct agx_const_eval_val {
      bool valid;
      nir_const_value c[4];
   };
   struct agx_const_eval_val *vals = calloc(impl->ssa_alloc, sizeof(*vals));
   if (!vals)
      return 0;

   float pos[3][4] = {{0}};
   bool ok = true;

   for (unsigned vid = 0; vid < 3 && ok; ++vid) {
      memset(vals, 0, impl->ssa_alloc * sizeof(*vals));
      uint8_t pos_mask = 0;

      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_load_const) {
            nir_load_const_instr *lc = nir_instr_as_load_const(instr);
            if (lc->def.index < impl->ssa_alloc &&
                lc->def.num_components <= 4) {
               vals[lc->def.index].valid = true;
               for (unsigned i = 0; i < lc->def.num_components; ++i)
                  vals[lc->def.index].c[i] = lc->value[i];
            }
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            switch (intr->intrinsic) {
            case nir_intrinsic_load_vertex_id:
            case nir_intrinsic_load_vertex_id_zero_base:
               if (intr->def.index < impl->ssa_alloc) {
                  vals[intr->def.index].valid = true;
                  vals[intr->def.index].c[0].u32 = vid;
               }
               break;
            case nir_intrinsic_load_base_vertex:
            case nir_intrinsic_load_first_vertex:
               if (intr->def.index < impl->ssa_alloc) {
                  vals[intr->def.index].valid = true;
                  vals[intr->def.index].c[0].u32 = 0;
               }
               break;
            case nir_intrinsic_store_output: {
               nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
               if (sem.location != VARYING_SLOT_POS)
                  break;
               if (!nir_src_is_const(intr->src[1]) ||
                   nir_src_as_uint(intr->src[1]) != 0) {
                  pos_mask = 0;
                  break;
               }
               unsigned comp = nir_intrinsic_component(intr);
               unsigned wmask = nir_intrinsic_write_mask(intr);
               nir_def *src = intr->src[0].ssa;
               bool src_valid = src->index < impl->ssa_alloc &&
                                src->bit_size == 32 && vals[src->index].valid;
               for (unsigned i = 0; i < src->num_components; ++i) {
                  if (!(wmask & BITFIELD_BIT(i)))
                     continue;
                  unsigned dst_comp = comp + i;
                  if (dst_comp >= 4)
                     continue;
                  if (src_valid) {
                     pos[vid][dst_comp] = vals[src->index].c[i].f32;
                     pos_mask |= BITFIELD_BIT(dst_comp);
                  } else {
                     pos_mask &= ~BITFIELD_BIT(dst_comp);
                  }
               }
               break;
            }
            default:
               break;
            }
         } else if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            const nir_op_info *op_info = &nir_op_infos[alu->op];
            if (alu->def.index >= impl->ssa_alloc ||
                alu->def.num_components > 4 ||
                op_info->num_inputs > NIR_ALU_MAX_INPUTS)
               continue;

            nir_const_value src_vals[NIR_ALU_MAX_INPUTS][4];
            nir_const_value *src_ptrs[NIR_ALU_MAX_INPUTS];
            bool all_valid = true;

            for (unsigned i = 0; i < op_info->num_inputs; ++i) {
               nir_def *s = alu->src[i].src.ssa;
               unsigned ncomp = nir_ssa_alu_instr_src_components(alu, i);
               if (s->index >= impl->ssa_alloc || !vals[s->index].valid ||
                   ncomp > 4) {
                  all_valid = false;
                  break;
               }
               for (unsigned j = 0; j < ncomp; ++j) {
                  unsigned swiz = alu->src[i].swizzle[j];
                  if (swiz >= 4) {
                     all_valid = false;
                     break;
                  }
                  src_vals[i][j] = vals[s->index].c[swiz];
               }
               if (!all_valid)
                  break;
               src_ptrs[i] = src_vals[i];
            }

            if (!all_valid)
               continue;

            unsigned bit_size = 0;
            if (!nir_alu_type_get_type_size(op_info->output_type)) {
               bit_size = alu->def.bit_size;
            } else {
               for (unsigned i = 0; i < op_info->num_inputs; ++i) {
                  if (!nir_alu_type_get_type_size(op_info->input_types[i])) {
                     bit_size = alu->src[i].src.ssa->bit_size;
                     break;
                  }
               }
            }
            if (bit_size == 0)
               bit_size = 32;

            nir_const_value dst_vals[4] = {0};
            uint16_t poison = 0;
            nir_eval_const_opcode(alu->op, dst_vals, &poison,
                                  alu->def.num_components, bit_size, src_ptrs,
                                  nir->info.float_controls_execution_mode);
            if (poison == 0) {
               vals[alu->def.index].valid = true;
               for (unsigned i = 0; i < alu->def.num_components; ++i)
                  vals[alu->def.index].c[i] = dst_vals[i];
            }
         }
      }

      if (pos_mask != 0xf)
         ok = false;
   }

   free(vals);
   if (!ok)
      return 0;

   float nx[3], ny[3], nz[3];
   bool z_in_01 = true;
   bool z_in_m11 = true;
   for (unsigned i = 0; i < 3; ++i) {
      for (unsigned c = 0; c < 4; ++c) {
         if (!isfinite(pos[i][c]))
            return 0;
      }
      if (!(pos[i][3] > 0.0f))
         return 0;
      nx[i] = pos[i][0] / pos[i][3];
      ny[i] = pos[i][1] / pos[i][3];
      nz[i] = pos[i][2] / pos[i][3];
      if (!isfinite(nx[i]) || !isfinite(ny[i]) || !isfinite(nz[i]))
         return 0;
      z_in_01 &= (nz[i] >= 0.0f && nz[i] <= 1.0f);
      z_in_m11 &= (nz[i] >= -1.0f && nz[i] <= 1.0f);
   }

   const float area2 =
      (nx[1] - nx[0]) * (ny[2] - ny[0]) - (ny[1] - ny[0]) * (nx[2] - nx[0]);
   if (!(fabsf(area2) > 1e-4f))
      return 0;

   const bool ccw = area2 > 0.0f;
   static const float corners[4][2] = {
      {-1.0f, -1.0f},
      {1.0f, -1.0f},
      {1.0f, 1.0f},
      {-1.0f, 1.0f},
   };

   for (unsigned c = 0; c < 4; ++c) {
      for (unsigned e = 0; e < 3; ++e) {
         const unsigned a = e;
         const unsigned b = (e + 1) % 3;
         const float edge = (nx[b] - nx[a]) * (corners[c][1] - ny[a]) -
                            (ny[b] - ny[a]) * (corners[c][0] - nx[a]);
         if (ccw ? !(edge >= 0.0f) : !(edge <= 0.0f))
            return 0;
      }
   }

   *out_z_in_01 = z_in_01;
   *out_z_in_m11 = z_in_m11;
   return ccw ? 1 : 2;
}

/* Determine the per-RT 4-bit RGBA channel masks a fragment shader writes
 * unconditionally along its top-level control flow, and whether the shader
 * can discard/demote/terminate fragments. */
void
agx_nir_analyze_fs_color_output_masks(nir_shader *nir,
                                      uint8_t out_masks[PIPE_MAX_COLOR_BUFS],
                                      bool *out_uses_discard)
{
   bool uses_discard = nir->info.fs.uses_discard;
   uint8_t rt_mask[PIPE_MAX_COLOR_BUFS] = {0};
   uint8_t broadcast_mask = 0;

   memset(out_masks, 0, PIPE_MAX_COLOR_BUFS * sizeof(out_masks[0]));

   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   if (!impl) {
      *out_uses_discard = true;
      return;
   }

   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_demote:
         case nir_intrinsic_demote_if:
         case nir_intrinsic_terminate:
         case nir_intrinsic_terminate_if:
            uses_discard = true;
            break;
         default:
            break;
         }
      }
   }

   *out_uses_discard = uses_discard;
   if (uses_discard)
      return;

   foreach_list_typed(nir_cf_node, cf, node, &impl->body) {
      if (cf->type != nir_cf_node_block)
         continue;
      nir_block *block = nir_cf_node_as_block(cf);
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_jump)
            return;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_output)
            continue;
         if (!nir_src_is_const(intr->src[1]) ||
             nir_src_as_uint(intr->src[1]) != 0)
            continue;
         nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
         unsigned comp = nir_intrinsic_component(intr);
         unsigned wmask = (nir_intrinsic_write_mask(intr) << comp) & 0xf;
         if (sem.location == FRAG_RESULT_COLOR) {
            broadcast_mask |= wmask;
         } else if (sem.location >= FRAG_RESULT_DATA0 &&
                    sem.location < FRAG_RESULT_DATA0 + PIPE_MAX_COLOR_BUFS) {
            rt_mask[sem.location - FRAG_RESULT_DATA0] |= wmask;
         }
      }
   }

   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt)
      out_masks[rt] = (rt_mask[rt] | broadcast_mask) & 0xf;
}

uint8_t
agx_nir_analyze_fs_color_outputs(nir_shader *nir, bool *out_uses_discard)
{
   uint8_t masks[PIPE_MAX_COLOR_BUFS];
   agx_nir_analyze_fs_color_output_masks(nir, masks, out_uses_discard);

   uint8_t full = 0;
   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (masks[rt] == 0xf)
         full |= BITFIELD_BIT(rt);
   }
   return full;
}

static void
agx_shader_initialize(struct agx_device *dev, struct agx_uncompiled_shader *so,
                      nir_shader *nir, bool support_lod_bias, bool robust)
{
   if (nir->info.stage == MESA_SHADER_KERNEL)
      nir->info.stage = MESA_SHADER_COMPUTE;

   blob_init(&so->early_serialized_nir);
   nir_serialize(&so->early_serialized_nir, nir, true);

   /* We need to lower robustness before bindings, since robustness lowering
    * affects the bindings used.
    */
   NIR_PASS(_, nir, nir_lower_robust_access, should_lower_robustness, &robust);

   /* Similarly, we need to do early texture lowering before bindings. */
   if (agx_chip_native_agx3(dev->chip)) {
      NIR_PASS(_, nir, agx_nir_lower_g17p_texture_early, support_lod_bias,
               agx_chip_sampler_has_lod_bias(dev->chip));
   } else {
      /* G17G's texel fetch has no witnessed integer-offset form either and
       * its packer drops the offset: WebRender's texelFetchOffset(uv, (1,0))
       * read texel uv, so every render task's device_pixel_scale came back
       * as task_rect.p0.x = 0 and all text quads went to infinity. A txf
       * offset is integer coordinate addition; lower it as G17P does.
       * textureGrad has no grounded G17G sample (coordinates and four
       * gradient scalars do not fit its staging block): sample at the
       * level the gradients select, as nir_lower_tex computes it. */
      if (agx_chip_g17_m5(dev->chip))
         NIR_PASS(_, nir, nir_lower_tex,
                  &(nir_lower_tex_options){.lower_txf_offset = true,
                                           .lower_txd = true});
      NIR_PASS(_, nir, agx_nir_lower_texture_early, support_lod_bias);
   }

   /* Keep API image bindings alive after atomics/stores become global memory
    * operations. Each lowering supplies the required image robustness. */
   unsigned nr_images = BITSET_LAST_BIT(nir->info.images_used);
   so->info.image_address_mask = 0;
   so->info.image_pbe_mask = 0;
   if (dev->chip == AGX_CHIP_G18G) {
      NIR_PASS(_, nir, agx_nir_lower_bound_image_atomics);
      NIR_PASS(_, nir, agx_nir_lower_bound_ms_image_stores);
   }

   /* We need to lower binding tables before calling agx_preprocess_nir, since
    * that does texture lowering that needs to know the binding model.
    */
   if (agx_chip_native_agx3(dev->chip)) {
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_tex)
                  continue;

               nir_tex_instr *tex = nir_instr_as_tex(instr);
               so->info.uses_buffer_textures |=
                  tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;
               if (tex->op == nir_texop_lod) {
                  /* G17P's native LOD query takes implicit derivatives and
                   * the sampler's descriptor bias, not a shader bias operand.
                   * The early pass above added only the API sampler bias;
                   * remove it here and bind a biased companion below. Keep
                   * ordinary sampling's software bias to avoid accumulating
                   * descriptor and sample-level quantization errors. */
                  nir_steal_tex_src(tex, nir_tex_src_bias);
                  so->info.uses_query_lod = true;
               }
            }
         }
      }
   }

   struct agx_lower_bindings_options binding_options = {
      .uses_bindless_samplers = &so->uses_bindless_samplers,
      .image_address_mask = dev->chip == AGX_CHIP_G18G
                              ? &so->info.image_address_mask : NULL,
      .image_pbe_mask = dev->chip == AGX_CHIP_G18G
                          ? &so->info.image_pbe_mask : NULL,
      .g17p_plain_txf_crawl = agx_chip_native_or_m5(dev->chip),
      .table_index_crawls = agx_chip_hal300_path(dev->chip),
      .native_buffer_metadata = dev->chip == AGX_CHIP_G18G,
      .g17p_native_sampler_handles =
         agx_chip_agx3_path(dev->chip),
      .g18 = dev->chip == AGX_CHIP_G18G,
      .g17p_bindless_storage = agx_chip_native_agx3(dev->chip),
      .image_records = agx_chip_native_agx3(dev->chip) ? 3 : 2,
      .bindful_sampler_count =
         agx_chip_agx3_path(dev->chip)
            ? AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS
            : 16,
   };
   /* Before the bindings lowering: the size query it adds is a descriptor
    * crawl, which has to be bindless by agx_nir_lower_texture(). */
   if (agx_chip_g17_m5(dev->chip))
      NIR_PASS(_, nir, nir_shader_instructions_pass, lower_g17g_gather_offset,
               nir_metadata_control_flow, NULL);
   if (agx_chip_g17_m5(dev->chip))
      NIR_PASS(_, nir, nir_shader_instructions_pass,
               lower_g17g_dynamic_sampler, nir_metadata_none, NULL);
   NIR_PASS(_, nir, agx_nir_lower_bindings, &binding_options);
   so->info.image_records = binding_options.image_records;

   /* We need to do some I/O lowering before lowering textures */
   so->info.nr_bindful_textures = BITSET_LAST_BIT(nir->info.textures_used);
   so->info.nr_bindful_images =
      dev->chip == AGX_CHIP_G18G
         ? MAX2(nr_images, BITSET_LAST_BIT(nir->info.images_used))
         : BITSET_LAST_BIT(nir->info.images_used);

   so->info.writes_images = agx_shader_writes_images(nir);
   so->info.reads_image_sideband = agx_shader_reads_image_sideband(nir);

   /* Native988: API fragment Layer is an ordinary flat coefficient. Internal
    * background programs bypass this API-shader initializer and keep their
    * distinct load_layer_id register contract. */
   if (dev->chip == AGX_CHIP_G18G && nir->info.stage == MESA_SHADER_FRAGMENT) {
      const nir_lower_sysvals_to_varyings_options layer = {.layer_id = true};
      NIR_PASS(_, nir, nir_lower_sysvals_to_varyings, &layer);
   }
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out,
            glsl_type_size,
            nir_lower_io_lower_64bit_to_32 |
               nir_lower_io_use_interpolated_input_intrinsics);

   /* Regather shader info after nir_lower_io. This recalculates interpolation
    * qualifiers which got lost when mesa/st lowered I/O back to vars.
    */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   /* Every stage: a G17G draw whose vertex shader writes memory is not
    * repeated per attachment either (agx_g17g_mrt_one_pass). */
   so->info.writes_memory = nir->info.writes_memory;

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      so->info.uses_fbfetch = nir->info.fs.uses_fbfetch_output;
      so->info.uses_sample_shading = nir->info.fs.uses_sample_shading;
      so->info.writes_memory = nir->info.writes_memory;
      so->info.writes_api_sample_mask =
         nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK);
      agx_nir_analyze_fs_color_output_masks(
         nir, so->info.fs_unconditional_color_masks, &so->info.fs_uses_discard);
      so->info.inputs_linear_shaded = nir->info.linear_varyings;
      so->info.inputs_read = nir->info.inputs_read;
      so->info.inputs_flat_shaded = nir->info.inputs_read &
                                    ~nir->info.linear_varyings &
                                    ~nir->info.perspective_varyings;
      so->info.inputs_read = nir->info.inputs_read;

      /* Interpolate varyings at fp16 and write to the tilebuffer at fp16. As an
       * exception, interpolate flat shaded at fp32. This works around a
       * hardware limitation. The resulting code (with an extra f2f16 at the end
       * if needed) matches what Metal produces.
       */
      if (likely(!(dev->debug & AGX_DBG_NO16))) {
         uint64_t texcoord = agx_gather_texcoords(nir);

         NIR_PASS(_, nir, nir_lower_mediump_io,
                  nir_var_shader_in | nir_var_shader_out,
                  ~(so->info.inputs_flat_shaded | texcoord), false);
      }
   } else if (nir->info.stage == MESA_SHADER_VERTEX ||
              nir->info.stage == MESA_SHADER_TESS_EVAL) {
      so->info.has_edgeflags = nir->info.outputs_written & VARYING_BIT_EDGE;
      so->info.writes_layer = nir->info.outputs_written & VARYING_BIT_LAYER;
      so->info.writes_viewport =
         nir->info.outputs_written & VARYING_BIT_VIEWPORT;
      so->info.writes_clip_distance =
         (nir->info.outputs_written &
          (VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1 |
           VARYING_BIT_CULL_DIST0 | VARYING_BIT_CULL_DIST1)) != 0 ||
         nir->info.clip_distance_array_size > 0;
      so->info.cull_distance_size = nir->info.cull_distance_array_size;
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      so->info.cull_distance_size = nir->info.cull_distance_array_size;
   }

   /* Shrink and vectorize SSBOs before lowering them, since it is harder to
    * optimize the lowered code.
    */
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_load_const_to_scalar);
   NIR_PASS(_, nir, agx_nir_cleanup_amul);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_cse);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_opt_shrink_vectors, true);
   NIR_PASS(_, nir, nir_opt_copy_prop);

   if (nir->info.stage == MESA_SHADER_VERTEX) {
      so->info.fullscreen_tri_winding = agx_nir_analyze_fullscreen_tri(
         nir, &so->info.fullscreen_tri_z_in_01,
         &so->info.fullscreen_tri_z_in_m11);
   }

   NIR_PASS(
      _, nir, nir_opt_load_store_vectorize,
      &(const nir_load_store_vectorize_options){
         .modes = nir_var_mem_global | nir_var_mem_constant | nir_var_mem_ssbo,
         .callback = agx_mem_vectorize_cb,
      });

   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);
   if (agx_chip_native_agx3(dev->chip))
   {
      /* Texel fetches get G17P's explicit out-of-view predicate only in
       * robust contexts: elsewhere an out-of-range texelFetch is undefined
       * (GL 4.6, 8.6) and the hardware fetch is memory-safe. Shader packs'
       * full-screen passes spent 5-7 instructions on every fetch. */
      NIR_PASS(_, nir, agx_nir_lower_texture_with_buffer_metadata_cb,
               descriptor_generation, agx_buffer_texture_metadata,
               robust);
   }
   else
      NIR_PASS(_, nir, agx_nir_lower_texture_for_generation,
               descriptor_generation |
                  (agx_chip_hal300_path(dev->chip) ? AGX_NIR_TEXTURE_GENERATION_G18G
                                              : 0));
   /* G18 indexed fetches also consume the native texture-table index, not
    * the base-plus-byte-offset handle retained for descriptor crawls. */
   if (agx_chip_agx3_path(dev->chip))
      NIR_PASS(_, nir, agx_nir_rebind_g17p_texture_crawls);
   if (agx_chip_g17_m5(dev->chip))
      NIR_PASS(_, nir, agx_nir_rebind_g17g_image_loads);
   NIR_PASS(_, nir, nir_lower_ssbo, NULL);

   /* Before agx_preprocess_nir: only the shader's own strict FMAs take the
    * explicit-RTE repair (agx_nir_lower_fma_rte()). */
   NIR_PASS(_, nir, agx_nir_lower_fma_rte, dev->chip);

   if (agx_chip_g17_m5(dev->chip)) {
      /* agx3c's name for the same preprocessing mode (false, true), which
       * theirs exposed as agx_preprocess_nir_g17g(). */
      agx_preprocess_nir_m5_chip(nir, dev->chip);
      if (nir->info.stage == MESA_SHADER_FRAGMENT &&
          (nir->info.inputs_read & VARYING_BIT_LAYER)) {
         NIR_PASS(_, nir, nir_shader_intrinsics_pass,
                  agx_nir_lower_g17g_fs_layer, nir_metadata_control_flow,
                  NULL);
         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
      }
   } else {
      agx_gallium_preprocess_nir(dev, nir);
   }

   /* The native stack protocol only encodes whole words; Honeykrisp gets
    * this from agx_preprocess_nir_g17p(). */
   if (agx_chip_native_agx3(dev->chip))
      agx3_nir_lower_scratch_words(nir);

   if (nir->info.stage == MESA_SHADER_FRAGMENT &&
       (nir->info.inputs_read & VARYING_BITS_TEX_ANY)) {

      NIR_PASS(_, nir, nir_shader_intrinsics_pass,
               agx_nir_lower_point_sprite_zw, nir_metadata_control_flow, NULL);
   }

   if (nir->info.stage == MESA_SHADER_FRAGMENT) {
      /* G18 sample-shading variants must specialize first. Their original
       * uses_sample_shading flag already keys the raster sample count. */
      if (dev->chip != AGX_CHIP_G18G || !nir->info.fs.uses_sample_shading)
         NIR_PASS(_, nir, agx_nir_lower_sample_intrinsics, true);
      so->info.uses_sample_shading = nir->info.fs.uses_sample_shading;
      so->info.needs_sample_count =
         agx_chip_native_or_g18g(dev->chip) &&
         agx_nir_needs_sample_count(nir);
   }

   so->type = nir->info.stage;
   so->info.lod_bias_samplers = agx_lod_bias_samplers(nir);

   if (nir->info.stage == MESA_SHADER_TESS_EVAL) {
      nir->info.stage = MESA_SHADER_VERTEX;
      nir->info.vs.tes_poly = true;
   }

   blob_init(&so->serialized_nir);
   nir_serialize(&so->serialized_nir, nir, true);
   _mesa_blake3_compute(so->serialized_nir.data, so->serialized_nir.size,
                      so->nir_blake3);
   so->source_hash = (uint32_t)nir->info.source_blake3[0] << 24 |
                     (uint32_t)nir->info.source_blake3[1] << 16 |
                     (uint32_t)nir->info.source_blake3[2] << 8 |
                     nir->info.source_blake3[3];

   so->has_xfb_info = (nir->xfb_info != NULL);

   static_assert(
      ARRAY_SIZE(so->xfb_strides) == ARRAY_SIZE(nir->info.xfb_stride),
      "known target count");

   if (so->has_xfb_info) {
      struct nir_xfb_info *xfb = nir->xfb_info;

      for (unsigned i = 0; i < ARRAY_SIZE(so->xfb_strides); ++i) {
         so->xfb_strides[i] = xfb->buffers[i].stride;
      }
   }
}

static bool
agx_shader_compile_sync(struct agx_screen *screen, struct pipe_context *pctx,
                        bool report_compile_error)
{
   if (!util_queue_is_initialized(&screen->shader_compiler_queue))
      return true;

   /* G18G surfaces backend encoding refusals as link-time errors when st/mesa
    * requests compile-error reporting (WebGL). */
   if (screen->dev.chip == AGX_CHIP_G18G && report_compile_error)
      return true;

   if (pctx->debug.debug_message && !pctx->debug.async)
      return true;

   if (screen->dev.debug & AGX_DBG_PRECOMPILE)
      return true;

   if (agx_get_compiler_debug() & (AGX_DBG_SHADERS | AGX_DBG_SHADERDB |
                                   AGX_DBG_VERBOSE | AGX_DBG_INTERNAL))
      return true;

   return false;
}

static unsigned
agx_fill_precompile_keys(const struct agx_device *dev,
                         const struct agx_uncompiled_shader *so,
                         mesa_shader_stage next_stage,
                         union asahi_shader_key keys[2])
{
   /* G17G runs geometry and tessellation stages on the CPU (agx_swtnl.c);
    * their GPU variants are only built if a draw needs them. */
   const bool cpu_stage = agx_chip_g17_m5(dev->chip) &&
                          (so->type == MESA_SHADER_TESS_CTRL ||
                           so->type == MESA_SHADER_GEOMETRY);
   if (!cpu_stage && ((so->type == MESA_SHADER_TESS_CTRL) ||
                      (so->type == MESA_SHADER_GEOMETRY) ||
                      (so->type == MESA_SHADER_FRAGMENT &&
                       !so->info.uses_fbfetch))) {
      union asahi_shader_key key = {0};
      if (so->type == MESA_SHADER_FRAGMENT) {
         /* A GL sampler's LOD bias defaults to zero. */
         key.fs.zero_lod_bias = so->info.lod_bias_samplers;
         /* Match agx_update_fs()'s single-sample key so the precompiled
          * variant hits on the first 1x draw instead of recompiling. */
         if (agx_chip_agx3_path(dev->chip) || so->info.needs_sample_count ||
             so->info.uses_sample_shading)
            key.fs.nr_samples = 1;
      }
      keys[0] = key;
      return 1;
   }

   if (so->type == MESA_SHADER_VERTEX) {
      keys[0] = (union asahi_shader_key){
         .vs.hw = next_stage == MESA_SHADER_FRAGMENT,
      };
      if (next_stage == MESA_SHADER_NONE) {
         keys[1] = (union asahi_shader_key){
            .vs.hw = true,
         };
         return 2;
      }
      return 1;
   }

   return 0;
}

static void
agx_precompile_shader_async(void *job, void *gdata, int thread_index)
{
   struct agx_uncompiled_shader *so = job;
   struct agx_screen *screen = so->precompile.screen;
   struct agx_device *dev = &screen->dev;
   struct util_debug_callback *debug_cb =
      so->precompile.debug.debug_message ? &so->precompile.debug : NULL;

   if (p_atomic_read(&so->precompile.cancelled)) {
      ralloc_free(so->precompile.nir);
      so->precompile.nir = NULL;
      return;
   }

   if (so->precompile.nir) {
      agx_shader_initialize(dev, so, so->precompile.nir,
                            so->precompile.support_lod_bias,
                            so->precompile.robust);
      ralloc_free(so->precompile.nir);
      so->precompile.nir = NULL;
   }

   if (p_atomic_read(&so->precompile.cancelled))
      return;

   if (so->type == MESA_SHADER_COMPUTE) {
      union asahi_shader_key key = {0};
      agx_get_shader_variant(screen, debug_cb, so, &key);
      return;
   }

   union asahi_shader_key keys[2];
   const unsigned num_keys =
      agx_fill_precompile_keys(dev, so, so->precompile.next_stage, keys);
   for (unsigned i = 0; i < num_keys; ++i) {
      if (p_atomic_read(&so->precompile.cancelled))
         return;
      agx_get_shader_variant(screen, debug_cb, so, &keys[i]);
   }
}

static void *
agx_create_shader_state(struct pipe_context *pctx,
                        const struct pipe_shader_state *cso)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_uncompiled_shader *so =
      rzalloc(NULL, struct agx_uncompiled_shader);
   struct agx_screen *screen = agx_screen(pctx->screen);
   struct agx_device *dev = &screen->dev;

   if (!so)
      return NULL;

   util_queue_fence_init(&so->ready);
   so->base = *cso;

   nir_shader *nir = cso->type == PIPE_SHADER_IR_NIR
                        ? cso->ir.nir
                        : tgsi_to_nir(cso->tokens, pctx->screen, false);

   so->type = nir->info.stage == MESA_SHADER_KERNEL ? MESA_SHADER_COMPUTE
                                                    : nir->info.stage;

   /* TGSI reaches this driver only from gallium and state-tracker utilities
    * (u_blitter, u_simple_shaders); their NIR shaders come from nir_builder,
    * which marks them internal. */
   so->report_shader_info =
      cso->type == PIPE_SHADER_IR_NIR && !nir->info.internal;

   if (nir->info.stage == MESA_SHADER_VERTEX ||
       nir->info.stage == MESA_SHADER_TESS_EVAL) {
      so->variants = asahi_vs_shader_key_table_create(so);
      so->linked_shaders = agx_fast_link_key_table_create(so);
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      so->variants = asahi_gs_shader_key_table_create(so);
      /* The geometry copy program links behind a vertex prolog. */
      so->linked_shaders = agx_fast_link_key_table_create(so);
   } else if (nir->info.stage == MESA_SHADER_TESS_CTRL) {
      /* No variants */
      so->variants = _mesa_hash_table_create(NULL, asahi_cs_shader_key_hash,
                                             asahi_cs_shader_key_equal);

      /* The geometry copy program is a hardware vertex program and on G17P
       * it links behind a vertex prolog like one, see
       * agx_update_gs_copy_link().
       */
      if (nir->info.stage == MESA_SHADER_GEOMETRY)
         so->linked_shaders = agx_fast_link_key_table_create(so);
   } else {
      so->variants = asahi_fs_shader_key_table_create(so);
      so->linked_shaders = agx_fast_link_key_table_create(so);
   }

   if (nir->info.stage == MESA_SHADER_TESS_EVAL ||
       nir->info.stage == MESA_SHADER_TESS_CTRL) {

      so->tess.ccw = nir->info.tess.ccw;
      so->tess.point_mode = nir->info.tess.point_mode;
      so->tess.spacing = nir->info.tess.spacing;
      so->tess.output_patch_size = nir->info.tess.tcs_vertices_out;
      so->tess.primitive = nir->info.tess._primitive_mode;
      so->tess.per_vertex_outputs = poly_tcs_per_vertex_outputs(nir);
      so->tess.nr_patch_outputs =
         util_last_bit(nir->info.patch_outputs_written);
      if (nir->info.stage == MESA_SHADER_TESS_CTRL)
         so->tess.output_stride = poly_tcs_output_stride(nir);
   } else if (nir->info.stage == MESA_SHADER_GEOMETRY) {
      so->gs_mode = nir->info.gs.output_primitive;
   }

   if (agx_chip_g17_m5(dev->chip))
      agx_xfb_cpu_create_vs(ctx, so, cso, nir);

   if (!agx_shader_compile_sync(screen, pctx, cso->report_compile_error)) {
      so->precompile.screen = screen;
      so->precompile.nir = nir;
      if (pctx->debug.debug_message)
         so->precompile.debug = pctx->debug;
      so->precompile.next_stage = nir->info.next_stage;
      so->precompile.support_lod_bias = ctx->support_lod_bias;
      so->precompile.robust = ctx->robust;
      util_queue_add_job(&screen->shader_compiler_queue, so, &so->ready,
                         agx_precompile_shader_async, NULL, 0);
      return so;
   }

   agx_shader_initialize(dev, so, nir, ctx->support_lod_bias, ctx->robust);
   mesa_shader_stage next_stage = nir->info.next_stage;

   /* We're done with the NIR, throw it away */
   ralloc_free(nir);
   nir = NULL;

   /* Precompile shaders that have a small key. For shader-db, precompile a
    * shader with a default key. This could be improved but hopefully this is
    * acceptable for now.
    */
   union asahi_shader_key keys[2];
   const unsigned num_keys =
      agx_fill_precompile_keys(dev, so, next_stage, keys);
   if (num_keys) {
      for (unsigned i = 0; i < num_keys; ++i) {
         if (!agx_precompile_variant(dev, pctx, so, cso, &keys[i]))
            return NULL;
      }
   } else if (dev->debug & AGX_DBG_PRECOMPILE) {
      union asahi_shader_key key = {0};

      switch (so->type) {
      case MESA_SHADER_TESS_EVAL:
         /* TODO: Tessellation shaders with shader-db */
         return so;

      case MESA_SHADER_FRAGMENT:
         key.fs.nr_samples = 1;
         key.fs.zero_lod_bias = so->info.lod_bias_samplers;

         /* For fbfetch */
         for (unsigned i = 0; i < ARRAY_SIZE(key.fs.rt_formats); ++i) {
            key.fs.rt_formats[i] = PIPE_FORMAT_R8G8B8A8_UNORM;
         }
         break;
      default:
         UNREACHABLE("Unknown shader stage in shader-db precompile");
      }

      agx_compile_variant(dev, &pctx->debug, so, &key);
   }

   return so;
}

static void *
agx_create_compute_state(struct pipe_context *pctx,
                         const struct pipe_compute_state *cso)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_screen *screen = agx_screen(pctx->screen);
   struct agx_device *dev = &screen->dev;
   struct agx_uncompiled_shader *so =
      rzalloc(NULL, struct agx_uncompiled_shader);

   if (!so)
      return NULL;

   util_queue_fence_init(&so->ready);
   so->type = MESA_SHADER_COMPUTE;
   so->variants = _mesa_hash_table_create(so, asahi_cs_shader_key_hash,
                                          asahi_cs_shader_key_equal);

   assert(cso->ir_type == PIPE_SHADER_IR_NIR && "TGSI kernels unsupported");
   nir_shader *nir = (void *)cso->prog;
   so->report_shader_info = !nir->info.internal;

   if (!agx_shader_compile_sync(screen, pctx, false)) {
      so->precompile.screen = screen;
      so->precompile.nir = nir;
      if (pctx->debug.debug_message)
         so->precompile.debug = pctx->debug;
      so->precompile.support_lod_bias = ctx->support_lod_bias;
      so->precompile.robust = ctx->robust;
      util_queue_add_job(&screen->shader_compiler_queue, so, &so->ready,
                         agx_precompile_shader_async, NULL, 0);
      return so;
   }

   union asahi_shader_key key = {0};

   agx_shader_initialize(dev, so, nir, ctx->support_lod_bias, ctx->robust);
   agx_get_shader_variant(screen, &pctx->debug, so, &key);

   /* We're done with the NIR, throw it away */
   ralloc_free(nir);
   return so;
}

static void
agx_get_compute_state_info(struct pipe_context *pctx, void *cso,
                           struct pipe_compute_state_object_info *info)
{
   struct agx_uncompiled_shader *uncompiled = cso;
   agx_shader_wait_ready(uncompiled);

   union asahi_shader_key key = {0};
   struct hash_entry *variant =
      _mesa_hash_table_search(uncompiled->variants, &key);
   struct agx_compiled_shader *so =
      variant ? variant->data
              : agx_get_shader_variant(agx_screen(pctx->screen), &pctx->debug,
                                       uncompiled, &key);

   info->max_threads =
      agx_occupancy_for_register_count(so->b.info.nr_gprs).max_threads;
   info->private_memory = 0;
   info->preferred_simd_size = 32;
   info->simd_sizes = 32;
}

/* Does not take ownership of key. Clones if necessary. */
static bool
agx_update_shader(struct agx_context *ctx, struct agx_compiled_shader **out,
                  mesa_shader_stage stage, union asahi_shader_key *key)
{
   struct agx_uncompiled_shader *so = ctx->stage[stage].shader;
   assert(so != NULL);
   agx_shader_wait_ready(so);

   struct hash_entry *he = _mesa_hash_table_search(so->variants, key);

   if (he) {
      if ((*out) == he->data)
         return false;

      *out = he->data;
      return true;
   }

   struct agx_screen *screen = agx_screen(ctx->base.screen);
   *out = agx_get_shader_variant(screen, &ctx->base.debug, so, key);
   return true;
}

/*
 * Whether a bound program can actually be issued.
 *
 * agx_compile_shader_nir() is fail-closed: a shape it cannot encode comes
 * back as a zeroed agx_shader_part, so the variant has an empty binary and no
 * executable BO. Consuming that is a null dereference inside a draw, which
 * for a compositor loses the whole session rather than one frame. A refused
 * program must therefore drop the work that needs it.
 *
 * `bo == NULL` is deliberately not the test. A fragment main whose entire
 * body lives in the epilog has a legitimately empty binary, and prologs,
 * epilogs and every other secondary part never own a BO at all.
 */
static bool
agx_program_usable(const struct agx_compiled_shader *cs)
{
   return cs != NULL && !cs->failed;
}

/*
 * Drop a draw or dispatch whose program is unusable, and say so once per
 * program rather than once per draw -- a refused shader in a compositor is
 * hit every frame.
 */
static bool
agx_drop_for_unusable_program(struct agx_context *ctx,
                              struct agx_compiled_shader *cs, bool linked,
                              const char *stage)
{
   if (agx_program_usable(cs) && linked)
      return false;

   ctx->skipped_no_program++;

   if (cs == NULL || !cs->refusal_reported) {
      if (cs != NULL)
         cs->refusal_reported = true;

      mesa_loge("asahi: dropping work: the %s program is unusable (%s). This "
                "is a driver gap; run with AGX_MESA_DEBUG=shaders to see the "
                "instruction the backend refused.",
                stage,
                cs == NULL     ? "no compiled variant"
                : cs->failed   ? "the backend refused to encode it"
                               : "it could not be linked");
   }

   return true;
}

/*
 * Refuse a draw whose G17P program address came back null.
 *
 * agx_build_g17p_graphics_program() is fail-closed and returns 0 from a dozen
 * places -- an unlinked shader, a preamble it cannot place, a sampling stage
 * whose texture/sampler pointer pair is missing, a program that overflows its
 * pool.  Every one of those used to be written straight into
 * agx_vdm_state_update_info.pipeline and submitted, so the shape of the bug on
 * hardware was a GPU fault at address 0 with nothing in the log connecting it
 * to the shader that caused it.  One such gap (a compiled-shader property lost
 * across the disk cache) took a hardware cycle to find for exactly that reason.
 *
 * Drop the work instead, and say which stage once per program rather than once
 * per draw -- a refused shader in a compositor is hit every frame.
 */
static bool
agx_drop_for_null_g17p_program(struct agx_context *ctx,
                               struct agx_compiled_shader *cs,
                               uint64_t program, const char *stage,
                               const char *why)
{
   if (program)
      return false;

   ctx->skipped_no_program++;

   if (cs == NULL || !cs->refusal_reported) {
      if (cs != NULL)
         cs->refusal_reported = true;

      if (why) {
         /* G18G names the refusal (agx_build_agx3_graphics_program()). */
         mesa_loge("asahi: dropping work: no G17P %s program could be built "
                   "for this draw: %s. This is a driver gap; submitting a "
                   "null program address faults the GPU.",
                   stage, why);
      } else {
         mesa_loge("asahi: dropping work: no G17P %s program could be built for "
                   "this draw. This is a driver gap; submitting a null program "
                   "address faults the GPU.",
                   stage);
      }
   }

   return true;
}

static enum mesa_prim
rast_prim(enum mesa_prim mode, unsigned fill_mode)
{
   if (u_reduced_prim(mode) == MESA_PRIM_TRIANGLES) {
      if (fill_mode == PIPE_POLYGON_MODE_POINT)
         return MESA_PRIM_POINTS;
      else if (fill_mode == PIPE_POLYGON_MODE_LINE)
         return MESA_PRIM_LINES;
   }

   return mode;
}

static bool
lower_fs_prolog_abi(nir_builder *b, nir_intrinsic_instr *intr, UNUSED void *_)
{
   if (intr->intrinsic != nir_intrinsic_load_polygon_stipple_agx &&
       intr->intrinsic != nir_intrinsic_load_stat_query_address_poly)
      return false;

   b->cursor = nir_before_instr(&intr->instr);
   nir_def *root = nir_load_preamble(b, 1, 64, .base = AGX_ABI_FUNI_ROOT);
   nir_def *repl;

   if (intr->intrinsic == nir_intrinsic_load_polygon_stipple_agx) {
      off_t stipple_offs = offsetof(struct agx_draw_uniforms, polygon_stipple);
      nir_def *stipple_ptr_ptr = nir_iadd_imm(b, root, stipple_offs);
      nir_def *base =
         nir_load_global_constant(b, 1, 64, stipple_ptr_ptr, .align_mul = 4);

      nir_def *row = intr->src[0].ssa;
      nir_def *addr = nir_iadd(b, base, nir_u2u64(b, nir_imul_imm(b, row, 4)));

      repl = nir_load_global_constant(b, 1, 32, addr);
   } else {
      off_t offs = offsetof(struct agx_draw_uniforms,
                            pipeline_statistics[nir_intrinsic_base(intr)]);
      repl = nir_load_global_constant(b, 1, 64, nir_iadd_imm(b, root, offs),
                                      .align_mul = 4);
   }

   nir_def_replace(&intr->def, repl);
   return true;
}

static void
build_fs_prolog(nir_builder *b, const void *key)
{
   agx_nir_fs_prolog(b, key);

   NIR_PASS(_, b->shader, nir_shader_intrinsics_pass, lower_fs_prolog_abi,
            nir_metadata_control_flow, NULL);
}

/* G17P addresses each tilebuffer sample explicitly, so a multisample epilog
 * writes one sample and leaves the rest untouched unless its tile stores are
 * expanded. Without this a four-sample draw reaches only sample zero, and
 * resolving it yields a quarter of the drawn colour.
 */
static void
build_g17p_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   if (key->link.sample_shading &&
       (key->nr_samples == 2 || key->nr_samples == 4)) {
      /* The main fragment exported every sample's colours (G17G's per-sample
       * unroll). Build the per-pixel epilog, then store and blend each sample
       * from its own colours, as Honeykrisp's build_g17p_fs_epilog does. */
      struct agx_fs_epilog_key pixel = *key;
      pixel.link.sample_shading = false;
      pixel.defer_monolithic_msaa = true;
      agx_nir_fs_epilog(b, &pixel);

      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, key->link.loc_written,
                                  key->link.size_32, key->link.colour_comps,
                                  key->link.packed_colour);

      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_unroll_g17p_samples_layout,
               key->nr_samples, &layout);
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, b->shader, nir_opt_constant_folding);
         NIR_PASS(progress, b->shader, nir_opt_algebraic);
      } while (progress);
      NIR_PASS(_, b->shader, agx_nir_hoist_g17p_zs_writes);
      return;
   }

   agx_nir_fs_epilog(b, key);

   NIR_PASS(_, b->shader, agx_nir_lower_g17p_tile_samples, key->nr_samples);
}

/* msaa163 P6: G18G's per-pixel four-sample epilog stores the
 * colour once through the covered-sample register, as Apple's G18 4x
 * fragment does (rld 210a42000000 ; lstcol 270605082901ad02000071440000,
 * native-msaa-reference-147). Without it the store keeps the 0xFF immediate,
 * which the backend now refuses at four samples instead of writing sample 0
 * only. */
static void
build_g18_msaa4_fs_epilog(nir_builder *b, const void *key_)
{
   const struct agx_fs_epilog_key *key = key_;
   if (key->link.sample_shading) {
      struct agx_fs_epilog_key pixel = *key;
      pixel.link.sample_shading = false;
      pixel.defer_monolithic_msaa = true;
      agx_nir_fs_epilog(b, &pixel);

      struct agx_g17p_colour_layout layout;
      agx_g17p_colour_layout_init(&layout, key->link.loc_written,
                                 key->link.size_32, key->link.colour_comps,
                                 key->link.packed_colour);
      NIR_PASS(_, b->shader, agx_nir_lower_to_per_sample);
      NIR_PASS(_, b->shader, agx_nir_unroll_g17p_samples_layout,
               key->nr_samples, &layout);
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, b->shader, nir_opt_constant_folding);
         NIR_PASS(progress, b->shader, nir_opt_algebraic);
      } while (progress);
      NIR_PASS(_, b->shader, agx_nir_hoist_g17p_zs_writes);
   } else {
      agx_nir_fs_epilog(b, key);
      /* The generated MSAA loop has exactly nr_samples iterations. Keep the
       * resident attachment transaction explicit across every sample even
       * when image-memory spills make the epilog too large for the general
       * unroll heuristic. This preserves the pixel-shaded export ABI. */
      nir_function_impl *impl = nir_shader_get_entrypoint(b->shader);
      foreach_list_typed(nir_cf_node, node, node, &impl->body) {
         if (node->type == nir_cf_node_loop)
            nir_cf_node_as_loop(node)->control = nir_loop_control_unroll;
      }
      /* Expose the one-hot constants before tile-sample lowering. Deferring
       * unrolling to the compiler would leave those new constants unmasked
       * by live coverage. This is the same ordering as G18 BG reloads. */
      bool progress;
      do {
         progress = false;
         NIR_PASS(progress, b->shader, nir_lower_vars_to_ssa);
         NIR_PASS(progress, b->shader, nir_opt_copy_prop);
         NIR_PASS(progress, b->shader, nir_opt_constant_folding);
         NIR_PASS(progress, b->shader, nir_opt_loop_unroll);
         NIR_PASS(progress, b->shader, nir_opt_dead_cf);
         NIR_PASS(progress, b->shader, nir_opt_dce);
      } while (progress);
   }

   NIR_PASS(_, b->shader, agx_nir_lower_g18_tile_samples, key->nr_samples);
}

/* The preparation each meta shader part gets between its builder and the
 * backend. Returns the sample count the part is compiled for. */
static unsigned
agx_prepare_meta_nir(struct agx_device *dev, nir_shader *nir,
                     meta_shader_builder_t builder, const void *data,
                     bool prolog, unsigned nr_samples)
{
   if (!prolog) {
      agx_gallium_preprocess_nir(dev, nir);
      NIR_PASS(_, nir, agx_nir_lower_texture);
      NIR_PASS(_, nir, agx_nir_lower_multisampled_image_store);
   }

   /* Match the compiler's sample protocol to the epilog's tile operands.
    * Preprocessing may expose literal masks while unrolling a sample loop;
    * lower them after that transformation, as the Vulkan path does.
    * G18G compiles its prolog and epilog for the sample count keyed here.
    */
   unsigned compile_samples = 0;
   if (dev->chip == AGX_CHIP_G18G) {
      compile_samples = nr_samples;
   } else if (builder == build_g17p_fs_epilog) {
      const struct agx_fs_epilog_key *fs = data;
      compile_samples = fs->nr_samples;
      if (!fs->link.sample_shading)
         NIR_PASS(_, nir, agx_nir_hoist_g17p_zs_writes);
      NIR_PASS(_, nir, agx_nir_lower_g17p_tile_samples, compile_samples);
   } else if (builder == build_fs_prolog &&
              agx_chip_native_or_m5(dev->chip)) {
      compile_samples = ((const struct agx_fs_prolog_key *)data)->nr_samples;
   }

   return compile_samples;
}

static void agx_delete_compiled_shader(struct agx_device *dev,
                                       struct agx_compiled_shader *so);

static bool
agx_g17p_fragment_blend_constant_base(enum agx_chip chip,
                                       const struct agx_compiled_shader *cs,
                                       uint16_t *base);

/* An epilog that reads uniforms (blend constants), images (spilled render
 * targets) or memory needs the split link's uniform layout. */
static bool
agx_inline_blend_epilog_admitted(nir_shader *epilog)
{
   if (epilog->info.writes_memory)
      return false;

   nir_foreach_function_impl(impl, epilog) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;

            switch (nir_instr_as_intrinsic(instr)->intrinsic) {
            case nir_intrinsic_load_preamble:
            case nir_intrinsic_load_blend_const_color_rgba:
            case nir_intrinsic_load_blend_const_color_r_float:
            case nir_intrinsic_load_blend_const_color_g_float:
            case nir_intrinsic_load_blend_const_color_b_float:
            case nir_intrinsic_load_blend_const_color_a_float:
            case nir_intrinsic_load_texture_handle_agx:
            case nir_intrinsic_bindless_image_agx:
            case nir_intrinsic_image_store:
            case nir_intrinsic_bindless_image_store:
               return false;
            default:
               break;
            }
         }
      }
   }

   return true;
}

/* The draw takes the fragment's resources, varyings and fragment state from
 * the API shader (ctx->fs) and only the uniform layout and constant program
 * from the program it launches, so the two must agree on the rest, as
 * Borealis requires of its inline-blend programs
 * (hk_g17p_mono_layout_mismatch()). */
static bool
agx_inline_blend_fs_launchable(const struct agx_compiled_shader *main,
                               const struct agx_shader_part *part)
{
   const struct agx_shader_info *a = &main->b.info, *m = &part->info;

   /* agx_build_g17p_graphics_program() launches only a constant program the
    * native protocol accepts, and agx_g17p_collect_graphics_uniform_ranges()
    * only rodata in whole dwords. Anything else would drop the draw that the
    * split link runs. */
   if ((m->has_preamble || m->preamble_scratch_size) &&
       !agx_g17p_fs_cprog_qualified(m))
      return false;
   if (m->has_preamble &&
       (m->preamble_offset > m->binary_size ||
        m->preamble_size > m->binary_size - m->preamble_offset ||
        (m->agx3_preamble_tps_inputs &&
         m->preamble_scratch_size < m->agx3_preamble_inputs.storage_bytes)))
      return false;
   if (m->rodata.size_16 != a->rodata.size_16 ||
       (m->rodata.size_16 && (m->rodata.base_uniform & 1)))
      return false;

   /* A program that spills where the main does not runs slower than the
    * split link. */
   if (m->scratch_size > a->scratch_size)
      return false;

   return m->texture_state_count == a->texture_state_count &&
          m->sampler_state_count == a->sampler_state_count &&
          m->uses_txf == a->uses_txf &&
          m->uses_sampler_heap == a->uses_sampler_heap &&
          m->g17p_sampled_fragment == a->g17p_sampled_fragment &&
          m->uses_draw_id == a->uses_draw_id &&
          m->uses_base_param == a->uses_base_param &&
          m->early_fragment_tests == a->early_fragment_tests &&
          m->needs_quad_derivatives == a->needs_quad_derivatives &&
          m->depth_layout == a->depth_layout &&
          !memcmp(&m->varyings.fs, &a->varyings.fs, sizeof(m->varyings.fs)) &&
          agx3_fragment_pixel_protocol_proven(part);
}

/*
 * Inline blending on native AGX3. The split link runs the API fragment main
 * and then its colour epilog, and hands the colours over in registers: the
 * main exports them and the epilog, a separately compiled part, blends them
 * with the tile. Borealis compiles the two as one program instead
 * (hk_g17p_inline_epilog_part()), as Apple's compiler does. The backend then
 * sees the whole blend: the terms that depend on uniforms only, such as a
 * uniform colour's clamps, 1 - alpha and the source products, join the main's
 * constant program and run once per draw instead of once per pixel.
 *
 * Compile the main's kept NIR (agx_keep_inline_blend_nir()) with the epilog
 * the split link would build appended (agx_nir_inline_fs_epilog()), under the
 * main's own backend key. The program launches with its own uniform layout
 * and constant program (agx_fs_launch_shader()) and with *blend_constant_base
 * placed after them. Return NULL, which keeps the split link, for any shape
 * this does not admit or that does not compile.
 */
static struct agx_compiled_shader *
agx_compile_inline_blend_fs(struct agx_context *ctx,
                            const struct agx_fast_link_key *key,
                            uint16_t *blend_constant_base)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   const struct agx_compiled_shader *main = key->main;
   const struct agx_fs_epilog_key *ekey = &key->epilog.fs;

   /* Borealis's admission: no sample shading, spilled render targets or
    * external sample loop, and a main compiled for the epilog's sample count.
    * A main that leaves a ray query's colour write wait to its epilog keeps
    * the split link, which rewrites that wait (agx_fast_link()), and so does
    * one with memory side effects, whose order against the tile accesses the
    * split link fixes. Inline blending pays for its compile through the
    * constant program, so chips without a native one keep the split link.
    */
   if (!agx_chip_native_agx3(dev->chip) ||
       !agx_chip_native_preamble(dev->chip) || !main->inline_blend_nir ||
       main->failed || main->g17g_samples_unrolled ||
       (main->so && main->so->info.writes_memory) ||
       key->nr_samples_shaded || ekey->link.sample_shading ||
       ekey->native_spills || ekey->link.rt_spill_base ||
       MAX2(ekey->nr_samples, 1) !=
          MAX2(main->inline_blend_key.fs.nr_samples, 1) ||
       main->b.info.agx3_colour_link.rt_query_write_acquired)
      return NULL;

   void *mem = ralloc_context(NULL);
   if (!mem)
      return NULL;

   struct agx_compiled_shader *fs = NULL;
   struct blob_reader reader;
   blob_reader_init(&reader, main->inline_blend_nir,
                    main->inline_blend_nir_size);
   nir_shader *nir = nir_deserialize(mem, &agx_nir_options, &reader);
   if (!nir || reader.overrun)
      goto out;

   /* The epilog exactly as the split link builds and compiles it
    * (asahi_fast_link(), agx_build_meta_shader_internal(),
    * agx_compile_nir_tile()). */
   nir_builder eb = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX inline blend epilog");
   ralloc_steal(mem, eb.shader);
   build_g17p_fs_epilog(&eb, ekey);
   const unsigned samples = agx_prepare_meta_nir(
      dev, eb.shader, build_g17p_fs_epilog, ekey, false, ekey->nr_samples);
   NIR_PASS(_, eb.shader, agx_nir_lower_interpolation,
            agx_chip_native_or_g18g(dev->chip));

   if (!agx_inline_blend_epilog_admitted(eb.shader) ||
       !agx_nir_inline_fs_epilog(nir, eb.shader))
      goto out;

   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);

   /* The main's key, now ending the program and addressing the epilog's
    * tile partitions. */
   struct agx_shader_key bkey = main->inline_blend_key;
   bkey.no_stop = false;
   agx_epilog_tile_partitions(ekey, samples, &bkey);

   fs = CALLOC_STRUCT(agx_compiled_shader);
   if (!fs)
      goto out;

   if (!agx_compile_shader_nir(nir, &bkey, &fs->b) ||
       !fs->b.info.binary_size || !agx_inline_blend_fs_launchable(main, &fs->b))
      goto fail;

   fs->so = main->so;
   fs->stage = main->stage;
   fs->g17p_texture_resources = main->g17p_texture_resources;
   fs->g17p_custom_border_emulation = main->g17p_custom_border_emulation;
   fs->push_range_count = main->push_range_count;
   memcpy(fs->push, main->push, sizeof(fs->push[0]) * main->push_range_count);
   fs->stage_uniforms_upload_size = main->stage_uniforms_upload_size;
   fs->root_uniforms_upload_size = main->root_uniforms_upload_size;
   fs->epilog_key = main->epilog_key;

   /* Blend constants are uploaded with every fragment program. A textured
    * one takes them after its whole uniform footprint (agx_update_fs()),
    * which here includes this program's constant program results. */
   *blend_constant_base = ekey->blend_constant_base;
   if (*blend_constant_base &&
       !agx_g17p_fragment_blend_constant_base(dev->chip, fs,
                                              blend_constant_base))
      goto fail;

   fs->bo = agx_bo_create(dev, fs->b.info.binary_size, 0,
                          AGX_BO_EXEC | AGX_BO_LOW_VA, "Executable");
   if (!fs->bo)
      goto fail;

   memcpy(agx_bo_map(fs->bo), fs->b.binary, fs->b.info.binary_size);
   goto out;

fail:
   agx_delete_compiled_shader(dev, fs);
   fs = NULL;
out:
   ralloc_free(mem);
   return fs;
}

static struct agx_linked_shader *
asahi_fast_link(struct agx_context *ctx, struct agx_uncompiled_shader *so,
                struct agx_fast_link_key *key)
{
   /* Try the cache */
   struct hash_entry *ent = _mesa_hash_table_search(so->linked_shaders, key);
   if (ent)
      return ent->data;

   /* Never link on top of a refused main program: agx_fast_link() would
    * happily concatenate an empty binary and produce a linked shader that
    * runs no API code at all. Refusing here keeps ctx->linked.* NULL, which
    * the draw path already treats as "this draw cannot be issued".
    */
   if (key->main == NULL || key->main->failed)
      return NULL;

   struct agx_compiled_shader *prolog = NULL, *epilog = NULL;
   struct agx_compiled_shader *inline_fs = NULL;
   uint16_t blend_constant_base = key->epilog.fs.blend_constant_base;

   /* Build the prolog/epilog now */
   if (so->type == MESA_SHADER_FRAGMENT) {
      prolog = agx_build_meta_shader_internal(
         ctx, build_fs_prolog, &key->prolog.fs, sizeof(key->prolog.fs), true,
         false, key->prolog.fs.cf_base,
         agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G
            ? key->epilog.fs.nr_samples : 0,
         false);

      /* Blend inline where the main and the epilog admit it, else link the
       * split epilog. */
      inline_fs = agx_compile_inline_blend_fs(ctx, key, &blend_constant_base);

      /* A multisampled G17G epilog stores through the coverage register as
       * G17P's does: Apple's 4x fragment reads SR39 and names it in LSTCOL
       * (gpu/g17g-witness/pos-25g72/ms4-fs_c). */
      const enum agx_chip chip = agx_device(ctx->base.screen)->chip;
      if (!inline_fs) {
         epilog = agx_build_meta_shader_internal(
            ctx,
            agx_chip_native_agx3(agx_device(ctx->base.screen)->chip) ||
                  (agx_chip_g17_m5(chip) && key->epilog.fs.nr_samples > 1)
               ? build_g17p_fs_epilog
            : agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G &&
                  (key->epilog.fs.nr_samples == 2 ||
                   key->epilog.fs.nr_samples == 4)
               ? build_g18_msaa4_fs_epilog
               : agx_nir_fs_epilog,
            &key->epilog.fs, sizeof(key->epilog.fs), false, true, 0,
            key->epilog.fs.nr_samples, false);
      }

   } else if (so->type == MESA_SHADER_TESS_EVAL) {
      /* No prolog/epilog needed */
   } else {
      /* A vertex shader, or a geometry shader's copy program: poly builds
       * that as a vertex-stage main, and on G17P it runs behind the same
       * vertex prolog as any hardware vertex shader. */
      assert(so->type == MESA_SHADER_VERTEX ||
             (so->type == MESA_SHADER_GEOMETRY &&
              key->main->b.info.stage == MESA_SHADER_VERTEX));

      /* G17P device loads return packed raw lanes. Its vertex prolog must
       * convert them rather than expect AGX2's formatted widening loads.
       * G17G fetches 8/16-bit normalized attributes only as raw integers
       * (a native S8NORM fetch was refused: draw.random), as Honeykrisp's
       * G17P prolog does, and converts them explicitly. */
      prolog = agx_build_meta_shader_internal(
         ctx,
         agx_chip_native_or_m5(agx_device(ctx->base.screen)->chip)
            ? agx_nir_vs_prolog_raw
            : agx_nir_vs_prolog,
         &key->prolog.vs, sizeof(key->prolog.vs), true,
         false, 0, 0, false);
   }

   struct agx_device *dev = agx_device(ctx->base.screen);

   /* A refused prolog or epilog is just as fatal as a refused main: the
    * linked program would be missing the code that feeds or drains it.
    */
   if ((prolog && prolog->failed) || (epilog && epilog->failed)) {
      if (inline_fs)
         agx_delete_compiled_shader(dev, inline_fs);
      return NULL;
   }

   /* Fast-link it all together */
   struct agx_linked_shader *linked =
      rzalloc(so->linked_shaders, struct agx_linked_shader);
   if (!agx_fast_link(linked, dev, so->type == MESA_SHADER_FRAGMENT,
                      inline_fs ? &inline_fs->b : &key->main->b, &prolog->b,
                      &epilog->b, key->nr_samples_shaded, NULL)) {
      if (inline_fs)
         agx_delete_compiled_shader(dev, inline_fs);
      ralloc_free(linked);
      return NULL;
   }

   if (so->type == MESA_SHADER_FRAGMENT) {
      linked->g17p_emrt_heap_base = key->epilog.fs.emrt_heap_base;
      linked->g17p_blend_constant_base = blend_constant_base;
      linked->inline_blend_fs = inline_fs;
   }

   /* Cache the fast linked program */
   union asahi_shader_key *cloned_key =
      ralloc_memdup(so->linked_shaders, key, sizeof(*key));
   _mesa_hash_table_insert(so->linked_shaders, cloned_key, linked);
   return linked;
}

static bool
agx_update_vs(struct agx_batch *batch, unsigned index_size_B)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);

   /* Only proceed if the shader or anything the key depends on changes
    *
    * vb_mask, attributes, vertex_buffers: VERTEX
    * G18G also keys the primitive (omit_point_size) and the fragment shader
    * (layer_as_varying): PRIM, FS_PROG
    */
   const bool g17g = agx_chip_g17_m5(dev->chip);
   const uint32_t key_dirty =
      AGX_DIRTY_VS_PROG | AGX_DIRTY_VERTEX | AGX_DIRTY_XFB |
      (g17g ? AGX_DIRTY_FS_PROG : 0) |
      (dev->chip == AGX_CHIP_G18G ? AGX_DIRTY_PRIM | AGX_DIRTY_FS_PROG : 0);
   if (!((ctx->dirty & key_dirty) ||
         ctx->stage[MESA_SHADER_TESS_EVAL].dirty ||
         ctx->stage[MESA_SHADER_GEOMETRY].dirty ||
         ctx->stage[MESA_SHADER_TESS_EVAL].shader ||
         ctx->stage[MESA_SHADER_GEOMETRY].shader || ctx->in_tess))
      return false;

   struct asahi_vs_shader_key key = {
      .hw = !((ctx->stage[MESA_SHADER_TESS_EVAL].shader && !ctx->in_tess) ||
              ctx->stage[MESA_SHADER_GEOMETRY].shader),
      .custom_borders = ctx->stage[MESA_SHADER_VERTEX].custom_borders,
   };
   struct agx_uncompiled_shader *fs = ctx->stage[MESA_SHADER_FRAGMENT].shader;
   if (g17g && key.hw && fs) {
      key.g17g_link_uvs = true;
      key.flat_varyings = fs->info.inputs_flat_shaded;
      key.linear_varyings = fs->info.inputs_linear_shaded;

      /* A split MRT pass's fragment program reads only its output's
       * varyings, and G17G's varying record admits exactly as many
       * fragment components as the vertex shader exports: export those
       * alone (fragment_out.random). Position keeps the key nonzero. */
      if (ctx->g17g_mrt_split)
         key.g17g_fs_inputs =
            agx_g17g_mrt_inputs(fs, ctx->g17g_mrt_rt) |
            BITFIELD64_BIT(VARYING_SLOT_POS);
   }

   key.omit_point_size = key.hw && dev->chip == AGX_CHIP_G18G &&
                         batch->reduced_prim == MESA_PRIM_LINES;
   key.layer_as_varying = key.hw && dev->chip == AGX_CHIP_G18G && fs &&
                         (fs->info.inputs_flat_shaded & VARYING_BIT_LAYER);

   agx_update_shader(ctx, &ctx->vs, MESA_SHADER_VERTEX,
                     (union asahi_shader_key *)&key);

   struct agx_fast_link_key link_key = {
      .prolog.vs.hw = key.hw,
      .prolog.vs.uniform_base =
         ctx->vs->g17p_texture_resources
            ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
            : 0,
      .prolog.vs.sw_index_size_B = key.hw ? 0 : index_size_B,
      /* M5 draws large patch draws in instance ranges */
      .prolog.vs.sw_instance_offset = g17g && !key.hw,

      .prolog.vs.robustness.level =
         ctx->robust ? AGX_ROBUSTNESS_GL : AGX_ROBUSTNESS_DISABLED,

      .prolog.vs.robustness.soft_fault = agx_has_soft_fault(dev),
      .main = ctx->vs,
   };

   STATIC_ASSERT(sizeof(link_key.prolog.vs.component_mask) ==
                 sizeof(ctx->vs->attrib_components_read));
   BITSET_COPY(link_key.prolog.vs.component_mask,
               ctx->vs->attrib_components_read);

   memcpy(link_key.prolog.vs.attribs, &ctx->attributes->key,
          sizeof(link_key.prolog.vs.attribs));

   void *old = ctx->linked.vs;

   ctx->linked.vs =
      asahi_fast_link(ctx, ctx->stage[MESA_SHADER_VERTEX].shader, &link_key);

   /* A refused variant owns no executable BO. Residency is decided per draw
    * in agx_draw_vbo(), which skips the draw entirely, so there is nothing
    * to make resident here and nothing to dereference.
    */
   if (ctx->vs->bo)
      agx_batch_add_bo(batch, ctx->vs->bo);
   if (ctx->linked.vs)
      agx_batch_add_bo(batch, ctx->linked.vs->bo);

   return old != ctx->linked.vs;
}

static bool
agx_update_tcs(struct agx_context *ctx, const struct pipe_draw_info *info)
{
   assert(info->mode == MESA_PRIM_PATCHES);

   struct agx_uncompiled_shader *tcs = ctx->stage[MESA_SHADER_TESS_CTRL].shader;
   agx_shader_wait_ready(tcs);

   struct hash_entry *he = _mesa_hash_table_next_entry(tcs->variants, NULL);
   if (he) {
      ctx->tcs = he->data;
      return true;
   }

   /* M5 compiles its only variant on first use (agx_create_shader_state). */
   union asahi_shader_key key = {0};
   agx_update_shader(ctx, &ctx->tcs, MESA_SHADER_TESS_CTRL, &key);
   return true;
}

/*
 * On G17P every hardware vertex program is a linked prolog plus main:
 * agx_build_g17p_graphics_program() has nothing to publish for a bare main.
 * The geometry copy program is exactly such a vertex program for a
 * geometry-assisted draw (transform feedback included), so link it like
 * agx_update_vs() links the vertex shader: hardware vertex ABI, no vertex
 * attributes, the usual vertex/instance ID exports. G18G builds its graphics
 * programs the same way (agx_build_agx3_graphics_program() takes only a
 * linked program), and the copy program is how its transform feedback is
 * written. Other chips execute the copy program's main directly and keep
 * ctx->linked.gs_copy NULL.
 */
static void
agx_update_gs_copy_link(struct agx_context *ctx, struct agx_batch *batch)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   ctx->linked.gs_copy = NULL;
   if (!ctx->gs || !ctx->gs->gs_copy ||
       !agx_chip_agx3_path(dev->chip))
      return;

   struct agx_compiled_shader *copy = ctx->gs->gs_copy;
   struct agx_fast_link_key link_key = {
      .prolog.vs.hw = true,
      .prolog.vs.uniform_base = copy->g17p_texture_resources
                                   ? AGX_ABI_G17P_TEXTURE_POINTER_HALFS
                                   : 0,
      .prolog.vs.robustness.level =
         ctx->robust ? AGX_ROBUSTNESS_GL : AGX_ROBUSTNESS_DISABLED,
      .prolog.vs.robustness.soft_fault = agx_has_soft_fault(dev),
      .main = copy,
   };

   ctx->linked.gs_copy =
      asahi_fast_link(ctx, ctx->stage[MESA_SHADER_GEOMETRY].shader, &link_key);
   if (ctx->linked.gs_copy)
      agx_batch_add_bo(batch, ctx->linked.gs_copy->bo);
}

static bool
agx_update_gs(struct agx_context *ctx, const struct pipe_draw_info *info,
              const struct pipe_draw_indirect_info *indirect)
{
   /* Only proceed if there is a geometry shader. Due to input assembly
    * dependence, we don't bother to dirty track right now.
    */
   if (!ctx->stage[MESA_SHADER_GEOMETRY].shader) {
      ctx->gs = NULL;
      return false;
   }

   /* Transform feedback always happens via the geometry shader, so look there
    * to get the XFB strides.
    */
   struct agx_uncompiled_shader *gs = ctx->stage[MESA_SHADER_GEOMETRY].shader;

   for (unsigned i = 0; i < ctx->streamout.num_targets; ++i) {
      struct agx_streamout_target *tgt =
         agx_so_target(ctx->streamout.targets[i]);

      if (tgt != NULL)
         tgt->stride = gs->xfb_strides[i];
   }

   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_uncompiled_shader *fs = ctx->stage[MESA_SHADER_FRAGMENT].shader;
   struct asahi_gs_shader_key key = {0};
   if (agx_chip_native_hal200(dev->chip) || agx_chip_hal300_path(dev->chip)) {
      key.link_uvs = true;
      if (fs) {
         key.flat_varyings = fs->info.inputs_flat_shaded;
         key.linear_varyings = fs->info.inputs_linear_shaded;
      }
      /* G17G/G18G rasterize at most the witnessed eight user varying
       * components, while transform feedback commonly captures far more
       * than the fragment shader reads (60 in
       * transform_feedback.capture_vertex_interleaved_test, read by no
       * fragment input). Rasterizing an output nothing reads is
       * unobservable, so the copy program exports only what the fragment
       * shader reads. */
      if (agx_chip_hal300_path(dev->chip)) {
         key.trim_rast_varyings = true;
         key.rast_varyings = fs ? fs->info.inputs_read : 0;
      }
      /* With no fragment shader bound there is no consumer for the user
       * varyings at all -- only transform feedback and the fixed-function
       * outputs are observable -- so how they would be interpolated is known
       * vacuously and every slot is trivially flat, which the zeroed masks
       * above already say. Withholding link_uvs here is not the conservative
       * choice: it denies agx_nir_lower_uvs the inlined UVS indices and
       * leaves the copy program storing varyings through the dynamic-index
       * form, which the AGX3 packer only accepts for one witnessed shape.
       * Every transform-feedback capture draws under GL_RASTERIZER_DISCARD
       * with no fragment shader, and each was refused at `st_vary` for
       * exactly this reason.
       */
   }

   agx_update_shader(ctx, &ctx->gs, MESA_SHADER_GEOMETRY,
                     (union asahi_shader_key *)&key);
   return true;
}

static enum pipe_blendfactor
optimize_blend_factor_w_1(enum pipe_blendfactor f)
{
   if (f == PIPE_BLENDFACTOR_SRC_ALPHA)
      return PIPE_BLENDFACTOR_ONE;
   else if (f == PIPE_BLENDFACTOR_INV_SRC_ALPHA)
      return PIPE_BLENDFACTOR_ZERO;
   else
      return f;
}

static uint32_t
agx_nr_tex_descriptors(struct agx_batch *batch,
                       struct agx_compiled_shader *cs);

/* Select a uniform range which cannot overlap either the native texture /
 * sampler pointer prefix or any main-shader uniform/constant range.  The
 * fragment epilog is linked against this same absolute half index. */
static bool
agx_g17p_fragment_blend_constant_base(enum agx_chip chip,
                                       const struct agx_compiled_shader *cs,
                                       uint16_t *base)
{
   if (!cs || !base)
      return false;

   unsigned occupied_halfs = cs->b.info.push_count;
   if (cs->b.info.rodata.size_16) {
      const unsigned rodata_end = (unsigned)cs->b.info.rodata.base_uniform +
                                  cs->b.info.rodata.size_16;
      occupied_halfs = MAX2(occupied_halfs, rodata_end);
   }

   return agx3_g17p_fragment_blend_constant_base(chip, occupied_halfs, base);
}

/* Tessellation ignored here due to the hard rebinding we do atm */
static struct agx_uncompiled_shader *
agx_last_uncompiled_vgt(struct agx_context *ctx)
{
   if (ctx->stage[MESA_SHADER_GEOMETRY].shader)
      return ctx->stage[MESA_SHADER_GEOMETRY].shader;
   else
      return ctx->stage[MESA_SHADER_VERTEX].shader;
}

/* The compiled shader whose uniform layout and constant program launch the
 * bound fragment program: an inline-blend program's own
 * (agx_compile_inline_blend_fs()), otherwise the API shader's. Everything
 * else about the fragment the draw takes from ctx->fs. */
static struct agx_compiled_shader *
agx_fs_launch_shader(const struct agx_context *ctx)
{
   if (ctx->linked.fs && ctx->linked.fs->inline_blend_fs)
      return ctx->linked.fs->inline_blend_fs;

   return ctx->fs;
}

static bool
agx_update_fs(struct agx_batch *batch)
{
   struct agx_context *ctx = batch->ctx;

   /* Only proceed if the shader or anything the key depends on changes
    *
    * batch->key: implicitly dirties everything, no explicit check
    * rast: RS
    * blend: BLEND
    * sample_mask: SAMPLE_MASK
    * reduced_prim: PRIM
    */
   if (!(ctx->dirty & (AGX_DIRTY_VS_PROG | AGX_DIRTY_FS_PROG | AGX_DIRTY_RS |
                       AGX_DIRTY_BLEND | AGX_DIRTY_SAMPLE_MASK |
                       AGX_DIRTY_PRIM | AGX_DIRTY_QUERY)) &&
       !ctx->stage[MESA_SHADER_GEOMETRY].dirty)
      return false;

   struct agx_device *dev = agx_device(ctx->base.screen);
   unsigned nr_samples = util_framebuffer_get_num_samples(&batch->key);

   /* Get main shader */
   struct asahi_fs_shader_key key = {0};
   key.custom_borders = ctx->stage[MESA_SHADER_FRAGMENT].custom_borders;

   /* Fold the LOD bias of each sampler holding zero, ±0 alike: the shader
    * adds the same half-float value the sampler state records. */
   const struct agx_stage *fs_stage = &ctx->stage[MESA_SHADER_FRAGMENT];
   u_foreach_bit(s, fs_stage->shader->info.lod_bias_samplers &
                       fs_stage->valid_samplers) {
      if (!(fs_stage->samplers[s]->lod_bias_as_fp16 & 0x7fff))
         key.zero_lod_bias |= BITFIELD_BIT(s);
   }

   /* G17P coverage updates use different native words at one and several
    * samples, including discard-only main shaders. Keep their raster sample
    * count in the main key, matching Honeykrisp and the linked shader parts.
    * Other generations need variants for sample-dependent shader operations.
    * A G17G fragment shader that runs per sample is unrolled over the sample
    * count, so the count belongs in its variant key too.
    */
   if (agx_chip_native_agx3(dev->chip) ||
       ctx->stage[MESA_SHADER_FRAGMENT].shader->info.needs_sample_count ||
       ctx->stage[MESA_SHADER_FRAGMENT].shader->info.uses_fbfetch ||
       ctx->stage[MESA_SHADER_FRAGMENT].shader->info.uses_sample_shading ||
       agx_chip_g17_m5(dev->chip)) {
      key.nr_samples = nr_samples;
   }
   /* G17G and G18G have no witnessed sample-count register: Apple compiles
    * get_num_samples() to a constant of the pipeline's raster sample count
    * (metal-nt applegpu_g17g/applegpu_g18g, m6-num-samples: ldimm 1 at 1x,
    * ldimm 4 at 4x). Key the count so the variant can do the same. */
   if (dev->chip == AGX_CHIP_G18G &&
       ctx->stage[MESA_SHADER_FRAGMENT].shader->info.needs_sample_count) {
      key.nr_samples = nr_samples;
   }

   /* G18 coverage updates have different encodings at 1x and 4x. Never
    * reuse a 1x discard/sample-mask program on a multisample attachment. */
   if (dev->chip == AGX_CHIP_G18G)
      key.nr_samples = nr_samples;

   if (ctx->stage[MESA_SHADER_FRAGMENT].shader->info.uses_fbfetch) {
      for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
         key.rt_formats[i] = batch->key.cbufs[i].format;
      }
   }

   if (ctx->g17g_mrt_split)
      key.g17g_mrt_output = ctx->g17g_mrt_rt + 1;

   agx_update_shader(ctx, &ctx->fs, MESA_SHADER_FRAGMENT,
                     (union asahi_shader_key *)&key);

   /* Fast link with prolog/epilog */
   bool msaa = ctx->rast->base.multisample;
   unsigned sample_mask = ctx->sample_mask & BITFIELD_MASK(nr_samples);
   const bool nontrivial_force_early =
      ctx->fs->b.info.early_fragment_tests &&
      (ctx->fs->b.info.writes_sample_mask ||
       (ctx->fs->so && ctx->fs->so->info.writes_memory));
   const bool defer_api_sample_mask = agx_chip_native_agx3(dev->chip) && msaa &&
      sample_mask != BITFIELD_MASK(nr_samples) && !nontrivial_force_early;

   struct agx_fast_link_key link_key = {
      .prolog.fs.statistics =
         ctx->pipeline_statistics[PIPE_STAT_QUERY_PS_INVOCATIONS],

      .prolog.fs.cull_distance_size =
         agx_last_uncompiled_vgt(ctx)->info.cull_distance_size,

      .prolog.fs.polygon_stipple =
         ctx->rast->base.poly_stipple_enable &&
         rast_prim(batch->reduced_prim, ctx->rast->base.fill_front) ==
            MESA_PRIM_TRIANGLES,

      .prolog.fs.api_sample_mask =
         (msaa && nr_samples > 1 && sample_mask != BITFIELD_MASK(nr_samples) &&
          !defer_api_sample_mask)
            ? sample_mask
            : 0xff,

      /* The coverage update (the API sample mask) is compiled for the
       * sample count on the native AGX3 parts and on G17G, as the main
       * fragment's. */
      .prolog.fs.nr_samples =
         agx_chip_native_or_m5(dev->chip)
            ? nr_samples
            : 0,
      .epilog.fs.nr_samples = nr_samples,
      .epilog.fs.max_sample_size_B = batch->tilebuffer_layout.max_sample_size_B,
      .epilog.fs.max_bytes_per_tile = batch->tilebuffer_layout.max_bytes_per_tile,
      .epilog.fs.max_samples_per_tile = batch->tilebuffer_layout.max_samples_per_tile,
      .epilog.fs.defer_api_sample_mask = defer_api_sample_mask,
      .epilog.fs.api_sample_mask = sample_mask,
      .epilog.fs.link = ctx->fs->epilog_key,
      .epilog.fs.force_small_tile = dev->debug & AGX_DBG_SMALLTILE,
      .epilog.fs.main_pixel_held =
         agx_chip_g17_m5(dev->chip) ? ctx->fs->b.info.agx3_pixel_held : 0,
      .epilog.fs.rt0_resident =
         agx_chip_g17_m5(dev->chip) && batch->key.nr_cbufs > 1 &&
         !batch->g17g_mrt_resident,

      .main = ctx->fs,
      .nr_samples_shaded = !agx_chip_native_agx3(dev->chip) &&
                           ctx->fs->epilog_key.sample_shading &&
                           !ctx->fs->g17g_samples_unrolled
                              ? nr_samples
                              : 0,
   };
   agx_fs_epilog_apply_device_policy(dev->chip, &link_key.epilog.fs);
   /* A fully masked G18 draw with no shader side effects uses the
    * colour-free ISP state (fragment_writes_nothing). Do not invent tile
    * loads/stores just to preserve its colour: those prevent the linker
    * from recognizing the no-op fragment, and the resulting multisample
    * preservation program cannot run for stencil-only browser draws.
    *
    * Keep preservation when any part can update coverage, depth, stencil,
    * memory or another colour attachment. Such programs are not no-ops.
    */
   if (dev->chip == AGX_CHIP_G18G &&
       !(ctx->blend->store &
         (PIPE_CLEAR_COLOR0 * BITFIELD_MASK(batch->key.nr_cbufs))) &&
       ctx->fs->b.info.tag_write_disable &&
       !link_key.prolog.fs.cull_distance_size &&
       !link_key.prolog.fs.polygon_stipple &&
       link_key.prolog.fs.api_sample_mask == 0xff &&
       !link_key.epilog.fs.defer_api_sample_mask &&
       !link_key.epilog.fs.link.write_z && !link_key.epilog.fs.link.write_s &&
       !link_key.epilog.fs.link.sample_shading &&
       !ctx->blend->key.alpha_to_coverage)
      link_key.epilog.fs.preserve_zero_mask = false;
   /* G17S (and G17C, the same core) share G17P's tile behaviour: a covered
    * fragment with no tile store still dirties the tile, which then stores
    * undefined colour at EOT. Measured on the M5 Pro: a draw with every
    * colour channel masked off left (0,0,0,0) where the clear had been. Keep
    * the destination with a load/store pair, as G17P does. G17G is left as
    * its validated line has it. */
   if (agx_chip_g17_m5(dev->chip) && dev->chip != AGX_CHIP_G17G)
      link_key.epilog.fs.preserve_zero_mask = true;

   if (agx_g18_native_ms_spills(batch)) {
      link_key.epilog.fs.native_spills = true;
      link_key.epilog.fs.native_spill_generation = 10;
      /* Native fragment programs already bind their texture heap at u0:u1. */
      link_key.epilog.fs.emrt_heap_base = 0;
   }

   /* Both colour parts have already been expanded over these samples. A
    * shader with memory writes retains the external-loop refusal until its
    * side effects can be predicated on each sample's live coverage. */
   if (dev->chip == AGX_CHIP_G18G && (nr_samples == 2 || nr_samples == 4) &&
       ctx->fs->so && !ctx->fs->so->info.writes_memory)
      link_key.nr_samples_shaded = 0;

   /* A depth-only sample-shaded G18 main already contains its statically
    * unrolled depth stores. Keep write_z in the compiled shader's state
    * metadata for PPP, but do not link another store from unused exports. */
   if (dev->chip == AGX_CHIP_G18G && (nr_samples == 2 || nr_samples == 4) &&
       ctx->fs->so && ctx->fs->so->info.uses_sample_shading &&
       !ctx->fs->epilog_key.sample_shading &&
       ctx->fs->b.info.writes_sample_mask && ctx->fs->epilog_key.write_z) {
      link_key.epilog.fs.link.write_z = false;
      link_key.epilog.fs.link.write_s = false;
      link_key.epilog.fs.link.already_ran_zs = true;
   }

   for (unsigned i = 0; i < PIPE_MAX_COLOR_BUFS; ++i) {
      link_key.epilog.fs.rt_formats[i] = batch->key.cbufs[i].format;
      link_key.epilog.fs.remap[i] =
         link_key.epilog.fs.link.broadcast_rt0 ? 0 : i;
   }
   /* Run 1139: the epilog builds the same index-order layout and stores
    * exactly what Apple's mixed-format fragments store. */
   link_key.epilog.fs.g18_native_mixed =
      agx_chip_has_cap(dev->chip, AGX_CAP_NATIVE_MIXED_EPILOG) &&
      agx_g18g_native_mixed_mask(&batch->tilebuffer_layout) != 0;

   memcpy(&link_key.epilog.fs.blend, &ctx->blend->key,
          sizeof(link_key.epilog.fs.blend));
   if (ctx->zs_merge) {
      for (unsigned i = 0; i < ARRAY_SIZE(link_key.epilog.fs.blend.rt); ++i)
         link_key.epilog.fs.blend.rt[i].colormask = 0;
   }

   /* A split G17G draw (agx_g17g_split_mrt) binds attachment g17g_mrt_rt
    * alone, at slot 0, and the variant moved its output to location 0:
    * blend it as that attachment. */
   if (ctx->g17g_mrt_split && ctx->g17g_mrt_rt)
      link_key.epilog.fs.blend.rt[0] = ctx->blend->key.rt[ctx->g17g_mrt_rt];

   /* HAL200 texture/sampler table pointers consume fragment uniform halves
    * 0..7, overlapping the legacy dynamic blend constants at halves4..11.
    * Link the epilog against a range after the main shader's complete uniform
    * footprint. The graphics program builder publishes that exact range.
    */
   /* G18G: agx_build_agx3_graphics_program() publishes a textured fragment's
    * blend constants at this same relocated base (the chip-independent G17P
    * uniform-file rule), so the epilog must be linked against it there too.
    * Without that the epilog read the legacy halves 4..11, which hold the
    * sampler pointer, and every constant-colour blend from a textured
    * fragment used a constant of zero (Qt Quick's default text material). */
   const bool g18g = dev->chip == AGX_CHIP_G18G;
   if ((agx_chip_native_or_m5(dev->chip) || g18g) &&
       agx_nr_tex_descriptors(batch, ctx->fs) != 0 &&
       !agx_g17p_fragment_blend_constant_base(
          g18g ? AGX_CHIP_G17P : dev->chip, ctx->fs,
          &link_key.epilog.fs.blend_constant_base)) {
      mesa_loge("asahi: fragment blend constants exceed the native "
                "uniform file");
      ctx->linked.fs = NULL;
      return true;
   }

   link_key.epilog.fs.native_spills = agx_chip_native_agx3(dev->chip) &&
                                      agx_tilebuffer_spills(&batch->tilebuffer_layout);
   if (link_key.epilog.fs.native_spills) {
      uint16_t base;
      if (!agx_g17p_fragment_blend_constant_base(dev->chip, ctx->fs, &base) ||
          ALIGN_POT(base, 4) > 512 - 12) {
         mesa_loge("asahi: G17P spill table exceeds the native uniform file");
         ctx->linked.fs = NULL;
         return true;
      }
      link_key.epilog.fs.emrt_heap_base = ALIGN_POT(base, 4);
      link_key.epilog.fs.blend_constant_base =
         link_key.epilog.fs.emrt_heap_base + 4;
      link_key.epilog.fs.link.rt_spill_base = 0;
   }
   /* Normalize */
   if (!agx_tilebuffer_spills(&batch->tilebuffer_layout))
      link_key.epilog.fs.link.rt_spill_base = 0;

   /* Try to disable blending to get rid of some fsats */
   if (link_key.epilog.fs.link.loc0_w_1) {
      struct agx_blend_rt_key *k = &link_key.epilog.fs.blend.rt[0];
      struct agx_blend_standard b = agx_unpack_blend_standard(k->mode);

      b.rgb_src_factor = optimize_blend_factor_w_1(b.rgb_src_factor);
      b.rgb_dst_factor = optimize_blend_factor_w_1(b.rgb_dst_factor);

      b.alpha_src_factor = optimize_blend_factor_w_1(b.alpha_src_factor);
      b.alpha_dst_factor = optimize_blend_factor_w_1(b.alpha_dst_factor);

      k->mode = agx_pack_blend_standard(b.rgb_func, b.rgb_src_factor,
                                        b.rgb_dst_factor, b.alpha_func,
                                        b.alpha_src_factor, b.alpha_dst_factor);
   }

   link_key.epilog.fs.blend.alpha_to_coverage &= msaa;

   /* The main shader must not run tests if the epilog will */
   bool epilog_discards = agx_fs_epilog_may_discard(
      &link_key.epilog.fs.link,
      link_key.epilog.fs.blend.alpha_to_coverage, nontrivial_force_early) ||
      defer_api_sample_mask;
   batch->uniforms.no_epilog_discard = !epilog_discards ? ~0 : 0;

   bool prolog_discards = (link_key.prolog.fs.api_sample_mask != 0xff ||
                           link_key.prolog.fs.cull_distance_size ||
                           link_key.prolog.fs.polygon_stipple);

   /* The prolog runs tests if neither the main shader nor epilog will */
   link_key.prolog.fs.run_zs_tests = !nontrivial_force_early &&
                                     !ctx->fs->b.info.writes_sample_mask &&
                                     !epilog_discards && prolog_discards;

   link_key.epilog.fs.link.already_ran_zs |= nontrivial_force_early;

   if (link_key.prolog.fs.cull_distance_size)
      link_key.prolog.fs.cf_base = ctx->fs->b.info.varyings.fs.nr_cf;

   void *old = ctx->linked.fs;

   ctx->linked.fs =
      asahi_fast_link(ctx, ctx->stage[MESA_SHADER_FRAGMENT].shader, &link_key);

   if (ctx->fs->bo)
      agx_batch_add_bo(batch, ctx->fs->bo);

   if (ctx->linked.fs) {
      agx_batch_add_bo(batch, ctx->linked.fs->bo);

      /* The launch holds its constant program and rodata. */
      if (ctx->linked.fs->inline_blend_fs)
         agx_batch_add_bo(batch, ctx->linked.fs->inline_blend_fs->bo);
   }

   return old != ctx->linked.fs;
}

static void
agx_bind_shader_state(struct pipe_context *pctx, void *cso,
                      mesa_shader_stage stage)
{
   struct agx_context *ctx = agx_context(pctx);

   if (stage == MESA_SHADER_VERTEX)
      ctx->dirty |= AGX_DIRTY_VS_PROG;
   else if (stage == MESA_SHADER_FRAGMENT)
      ctx->dirty |= AGX_DIRTY_FS_PROG;
   else
      ctx->stage[stage].dirty = ~0;

   struct agx_uncompiled_shader *old = ctx->stage[stage].shader;
   struct agx_uncompiled_shader *next = cso;
   if (stage == MESA_SHADER_FRAGMENT) {
      const bool old_ready =
         !old || util_queue_fence_is_signalled(&old->ready);
      const bool next_ready =
         !next || util_queue_fence_is_signalled(&next->ready);
      if (!old_ready || !next_ready ||
          (old && old->info.uses_query_lod) !=
             (next && next->info.uses_query_lod))
         ctx->stage[stage].dirty |= AGX_STAGE_DIRTY_SAMPLER;
   }

   /* A shader's ready fence is reset once, when its asynchronous compile is
    * queued at creation, and is only signalled after that. A shader that is
    * ready when it is bound therefore stays ready, and only the stages bound
    * while still compiling need a wait at the next draw. agx_launch_grid()
    * waits for its compute shader itself.
    */
   if (stage != MESA_SHADER_COMPUTE && next &&
       !util_queue_fence_is_signalled(&next->ready))
      ctx->stages_compiling |= BITFIELD_BIT(stage);

   ctx->stage[stage].shader = cso;
}

static void
agx_bind_vs_state(struct pipe_context *pctx, void *cso)
{
   agx_bind_shader_state(pctx, cso, MESA_SHADER_VERTEX);
}

static void
agx_bind_fs_state(struct pipe_context *pctx, void *cso)
{
   agx_bind_shader_state(pctx, cso, MESA_SHADER_FRAGMENT);
}

static void
agx_bind_gs_state(struct pipe_context *pctx, void *cso)
{
   agx_bind_shader_state(pctx, cso, MESA_SHADER_GEOMETRY);
}

static void
agx_bind_tcs_state(struct pipe_context *pctx, void *cso)
{
   agx_bind_shader_state(pctx, cso, MESA_SHADER_TESS_CTRL);
}

static void
agx_bind_tes_state(struct pipe_context *pctx, void *cso)
{
   agx_bind_shader_state(pctx, cso, MESA_SHADER_TESS_EVAL);
}

static void
agx_bind_cs_state(struct pipe_context *pctx, void *cso)
{
   agx_bind_shader_state(pctx, cso, MESA_SHADER_COMPUTE);
}

static void
agx_delete_compiled_shader(struct agx_device *dev,
                           struct agx_compiled_shader *so)
{
   if (so->gs_count)
      agx_delete_compiled_shader(dev, so->gs_count);

   if (so->pre_gs)
      agx_delete_compiled_shader(dev, so->pre_gs);

   if (so->gs_copy)
      agx_delete_compiled_shader(dev, so->gs_copy);

   free(so->b.binary);
   free(so->inline_blend_nir);
   agx_g17p_compute_templates_free(&so->g17p_compute_templates);
   agx_bo_unreference(dev, so->agx3_compute_bo);
   agx_bo_unreference(dev, so->bo);
   FREE(so);
}

static void
agx_delete_uncompiled_shader(struct agx_device *dev,
                             struct agx_uncompiled_shader *so)
{
   if (so->precompile.screen) {
      p_atomic_set(&so->precompile.cancelled, true);
      util_queue_drop_job(&so->precompile.screen->shader_compiler_queue,
                          &so->ready);
   }
   util_queue_fence_wait(&so->ready);
   ralloc_free(so->precompile.nir);
   so->precompile.nir = NULL;
   util_queue_fence_destroy(&so->ready);

   hash_table_foreach(so->variants, ent) {
      agx_delete_compiled_shader(dev, ent->data);
   }

   _mesa_hash_table_destroy(so->variants, NULL);

   if (so->linked_shaders) {
      hash_table_foreach(so->linked_shaders, ent) {
         struct agx_linked_shader *link = ent->data;
         agx_bo_unreference(dev, link->bo);
         if (link->inline_blend_fs)
            agx_delete_compiled_shader(dev, link->inline_blend_fs);
      }

      _mesa_hash_table_destroy(so->linked_shaders, NULL);
   }

   blob_finish(&so->serialized_nir);
   blob_finish(&so->early_serialized_nir);

   for (unsigned i = 0; i < MESA_PRIM_COUNT; ++i) {
      for (unsigned j = 0; j < 3; ++j) {
         for (unsigned k = 0; k < 2; ++k) {
            if (so->passthrough_progs[i][j][k])
               agx_delete_uncompiled_shader(dev,
                                            so->passthrough_progs[i][j][k]);
         }
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(so->passthrough_tcs); ++i) {
      if (so->passthrough_tcs[i])
         agx_delete_uncompiled_shader(dev, so->passthrough_tcs[i]);
   }

   ralloc_free(so);
}

static void
agx_delete_shader_state(struct pipe_context *ctx, void *cso)
{
   struct agx_device *dev = agx_device(ctx->screen);
   agx_xfb_cpu_delete_vs(agx_context(ctx), cso);
   agx_swtnl_delete_shader(agx_context(ctx), cso);
   agx_delete_uncompiled_shader(dev, cso);
}

struct agx_generic_meta_key {
   meta_shader_builder_t builder;
   size_t key_size;
   size_t nr_samples;
   uint8_t key[];
};

static uint32_t
meta_key_hash(const void *key_)
{
   const struct agx_generic_meta_key *key = key_;

   return _mesa_hash_data(key,
                          sizeof(struct agx_generic_meta_key) + key->key_size);
}

static bool
meta_key_equal(const void *a_, const void *b_)
{
   const struct agx_generic_meta_key *a = a_;
   const struct agx_generic_meta_key *b = b_;

   return a->builder == b->builder && a->key_size == b->key_size &&
          a->nr_samples == b->nr_samples &&
          memcmp(a->key, b->key, a->key_size) == 0;
}

void
agx_init_meta_shaders(struct agx_context *ctx)
{
   ctx->generic_meta =
      _mesa_hash_table_create(ctx, meta_key_hash, meta_key_equal);
}

static void
agx_destroy_compute_blitter(struct pipe_context *ctx, struct asahi_blitter *bl)
{
   hash_table_foreach(bl->blit_cs, ent) {
      ctx->delete_compute_state(ctx, ent->data);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(bl->g17p_mip_cs); ++i) {
      if (bl->g17p_mip_cs[i])
         ctx->delete_compute_state(ctx, bl->g17p_mip_cs[i]);
   }

   if (bl->g17p_block_mip_cs)
      ctx->delete_compute_state(ctx, bl->g17p_block_mip_cs);
   if (bl->g17p_block_mip_sampler)
      ctx->delete_sampler_state(ctx, bl->g17p_block_mip_sampler);

   for (unsigned i = 0; i < ARRAY_SIZE(bl->g17p_block_buf2img_cs); ++i) {
      for (unsigned j = 0; j < ARRAY_SIZE(bl->g17p_block_buf2img_cs[i]); ++j) {
         if (bl->g17p_block_buf2img_cs[i][j])
            ctx->delete_compute_state(ctx, bl->g17p_block_buf2img_cs[i][j]);
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(bl->g17p_tiled_buf2img_cs); ++i) {
      for (unsigned j = 0; j < ARRAY_SIZE(bl->g17p_tiled_buf2img_cs[i]); ++j) {
         if (bl->g17p_tiled_buf2img_cs[i][j])
            ctx->delete_compute_state(ctx, bl->g17p_tiled_buf2img_cs[i][j]);
      }
   }

   ctx->delete_sampler_state(ctx, bl->sampler[0]);
   ctx->delete_sampler_state(ctx, bl->sampler[1]);

   _mesa_hash_table_destroy(bl->blit_cs, NULL);
}

void
agx_destroy_meta_shaders(struct agx_context *ctx)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   hash_table_foreach(ctx->generic_meta, ent) {
      agx_delete_compiled_shader(dev, ent->data);
   }

   agx_destroy_compute_blitter(&ctx->base, &ctx->compute_blitter);
   _mesa_hash_table_destroy(ctx->generic_meta, NULL);
}

static struct agx_compiled_shader *
agx_build_meta_shader_internal(struct agx_context *ctx,
                               meta_shader_builder_t builder, void *data,
                               size_t data_size, bool prolog, bool epilog,
                               unsigned cf_base, unsigned nr_samples,
                               bool internal_kernel)
{
   /* Build the meta shader key */
   size_t total_key_size = sizeof(struct agx_generic_meta_key) + data_size;
   struct agx_generic_meta_key *key = alloca(total_key_size);

   *key = (struct agx_generic_meta_key){
      .builder = builder,
      .key_size = data_size,
      .nr_samples = nr_samples,
   };

   if (data_size)
      memcpy(key->key, data, data_size);

   /* Try to get the cached shader */
   struct hash_entry *ent = _mesa_hash_table_search(ctx->generic_meta, key);
   if (ent)
      return ent->data;

   /* Otherwise, compile the shader fresh */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "AGX meta shader");

   builder(&b, data);

   struct agx_device *dev = agx_device(ctx->base.screen);
   const unsigned compile_samples =
      agx_prepare_meta_nir(dev, b.shader, builder, data, prolog, nr_samples);

   /* Run 1139: a fragment epilog's key carries the mixed-format tile flag;
    * other meta shaders never set it. */
   const bool g18_native_mixed =
      epilog && b.shader->info.stage == MESA_SHADER_FRAGMENT &&
      data_size == sizeof(struct agx_fs_epilog_key) &&
      ((const struct agx_fs_epilog_key *)data)->g18_native_mixed;

   /* The compiler orders a fragment epilog's tile loads and stores by the
    * framebuffer's tile partitions, which it takes from the epilog key
    * (agx_compile_nir_tile). An M5 batch that keeps every colour attachment
    * in the tile (agx_g17g_mrt_resident) stores each through its own
    * partition. Its single-sample epilog needs the partitions as well:
    * without them, an epilog that leaves the attachment at byte 0 alone (a
    * masked attachment or a clear of another one) cannot be ordered and the
    * whole fragment program is refused. The epilog that keeps only
    * attachment 0 in the tile (rt0_resident) has a single partition.
    */
   const struct agx_fs_epilog_key *fs_epilog = NULL;
   if (builder == build_g17p_fs_epilog)
      fs_epilog = data;
   else if (builder == agx_nir_fs_epilog && agx_chip_g17_m5(dev->chip) &&
            !((const struct agx_fs_epilog_key *)data)->rt0_resident)
      fs_epilog = data;

   struct agx_compiled_shader *shader = agx_compile_nir_tile(
      dev, b.shader, NULL,
      dev->chip == AGX_CHIP_G18G ? MESA_SHADER_COMPUTE : b.shader->info.stage,
      internal_kernel,
      !prolog && !(!agx_chip_native_agx3(dev->chip) &&
                   b.shader->info.stage == MESA_SHADER_FRAGMENT &&
                   b.shader->info.fs.uses_sample_shading),
      prolog || epilog, cf_base, compile_samples, fs_epilog, NULL,
      g18_native_mixed);

   ralloc_free(b.shader);

   /* ..and cache it before we return. The key is on the stack right now, so
    * clone it before using it as a hash table key. The clone is logically owned
    * by the hash table.
    */
   void *cloned_key = rzalloc_size(ctx->generic_meta, total_key_size);
   memcpy(cloned_key, key, total_key_size);

   _mesa_hash_table_insert(ctx->generic_meta, cloned_key, shader);
   return shader;
}

struct agx_compiled_shader *
agx_build_meta_shader(struct agx_context *ctx, meta_shader_builder_t builder,
                      void *data, size_t data_size)
{
   return agx_build_meta_shader_internal(ctx, builder, data, data_size, false,
                                         false, 0, 0, false);
}

static unsigned
sampler_count(struct agx_context *ctx, mesa_shader_stage stage)
{
   /* We reserve sampler #0 for txf so add 1 to the API count */
   unsigned count = ctx->stage[stage].sampler_count + 1;
   /* Native staged MS reads name sampler one. Even an image-only shader
    * therefore needs that entry inside the declared sampler table. */
   return agx_chip_native_agx3(agx_device(ctx->base.screen)->chip)
             ? MAX2(count, 2u) : count;
}

static inline enum agx_sampler_states
translate_sampler_state_count(struct agx_context *ctx, mesa_shader_stage stage)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   /* Clamp to binding table maximum, anything larger will be bindless */
   return agx_translate_sampler_state_count(MIN2(sampler_count(ctx, stage), 16),
                                            !agx_chip_native_or_g18g(dev->chip) &&
                                               ctx->stage[stage].custom_borders);
}

static uint32_t
agx_nr_tex_descriptors_without_spilled_rts(const struct agx_compiled_shader *cs)
{
   if (!cs || !cs->so)
      return 0;

   /* image_records descriptors per image, 1 descriptor per texture */
   return cs->so->info.nr_bindful_textures +
          (cs->so->info.image_records * cs->so->info.nr_bindful_images);
}

static uint32_t
agx_nr_tex_descriptors(struct agx_batch *batch, struct agx_compiled_shader *cs)
{
   unsigned n = agx_nr_tex_descriptors_without_spilled_rts(cs);

   /* We add on texture/PBE descriptors for spilled render targets */
   bool spilled_rt = cs->stage == MESA_SHADER_FRAGMENT &&
                     agx_tilebuffer_spills(&batch->tilebuffer_layout);
   if (spilled_rt)
      n += batch->key.nr_cbufs * (agx_g18_native_ms_spills(batch) ? 3 : 2);

   return n;
}

/*
 * For spilled render targets, upload a texture/PBE pair for each surface to
 * allow loading/storing to the render target from the shader.
 */
static struct agx_gallium_texture_descriptor *
agx_texture_descriptor_at(void *base, unsigned stride, unsigned index)
{
   return (struct agx_gallium_texture_descriptor *)((uint8_t *)base +
                                                     (stride * index));
}

static bool
agx_gallium_g17p_tile_layout_supported(struct agx_tilebuffer_layout *tib)
{
   if (!tib)
      return false;

   /* Layer selection lives in the proven array BG/EOT programs and PBE
    * descriptor. It does not change the physical tile allocation, so validate
    * that allocation through the same bounded RT0 contract Honeykrisp uses. */
   struct agx_tilebuffer_layout physical = *tib;
   physical.layered = false;
   return agx_g17p_rt0_tile_layout_supported(&physical);
}

static struct agx_gallium_pbe_descriptor *
agx_pbe_descriptor_at(void *base, unsigned stride, unsigned index)
{
   return (struct agx_gallium_pbe_descriptor *)((uint8_t *)base +
                                                 (stride * index));
}

static void
agx_upload_spilled_rt_descriptors(void *out, struct agx_batch *batch,
                                  enum agx_descriptor_generation generation,
                                  unsigned heap_base, bool native_spills)
{
   const unsigned stride = agx_gallium_texture_descriptor_size(generation);
   const unsigned slots = native_spills ? 3 : 2;
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   assert(!native_spills || (agx_g18_native_ms_spills(batch) && stride == 32));

   for (unsigned rt = 0; rt < batch->key.nr_cbufs; ++rt) {
      uint8_t *entry = (uint8_t *)out + stride * slots * rt;
      if (native_spills)
         memset(entry, 0, AGX_G17P_EMRT_DESCRIPTOR_SIZE);
      struct agx_gallium_texture_descriptor *texture =
         agx_texture_descriptor_at(out, stride, slots * rt);
      struct agx_gallium_pbe_descriptor *pbe =
         agx_pbe_descriptor_at(out, stride, (slots * rt) + 1);

      const struct pipe_surface *surf = &batch->key.cbufs[rt];
      if (!surf->texture)
         continue;

      struct agx_resource *rsrc = agx_resource(surf->texture);
      if (native_spills &&
          (surf->level || surf->first_layer != surf->last_layer ||
           !agx_g18_msaa4_storage(dev, rsrc, surf->format))) {
         agx_native_descriptor_reject_batch(batch, "G18G spilled-RT storage");
         return;
      }
      struct pipe_image_view view = image_view_for_surface(surf);
      struct pipe_sampler_view sampler_view = sampler_view_for_surface(surf);
      sampler_view.target = PIPE_TEXTURE_2D_ARRAY;

      if (!agx_pack_texture(generation, texture, rsrc, surf->format,
                            &sampler_view))
         {
            agx_native_descriptor_reject_batch(batch, "spilled-RT texture");
            return;
         }

      if (!agx_batch_upload_pbe_internal(batch, pbe, &view, false, false,
                                         true, true, false,
                                         native_spills ? entry : NULL))
         {
            agx_native_descriptor_reject_batch(batch, "spilled-RT PBE");
            return;
         }

      if (native_spills) {
         uint32_t index = heap_base + slots * rt;
         memcpy(entry + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET, &index, sizeof(index));
         memcpy(entry + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET, &index, sizeof(index));
         if (dev->debug & AGX_DBG_FAULTTRACE) {
            uint64_t sw;
            uint32_t width;
            memcpy(&sw, entry + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET, sizeof(sw));
            memcpy(&width, entry + AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET, sizeof(width));
            fprintf(stderr, "G18_EMRT rt=%u heap=%u pbe=%u samples=%u "
                    "sw=%016llx aligned_width=%u layer=%u\n", rt, index,
                    index + 1, rsrc->base.b.nr_samples,
                    (unsigned long long)sw, width, surf->first_layer);
         }
      }
   }
}

/* The native PBE has no software atomic address fields. Describe the actual
 * uncompressed view allocation separately, after resource legalization. */
static bool
agx_image_address_metadata(uint32_t out[10], const struct pipe_image_view *view)
{
   struct agx_resource *rsrc = agx_resource(view->resource);
   const struct ail_layout *layout = &rsrc->layout;
   const bool buffer = rsrc->base.b.target == PIPE_BUFFER;
   const unsigned level = buffer ? 0 : view->u.tex.level;
   const unsigned layer = buffer ? 0 : view->u.tex.first_layer;
   const unsigned samples = util_res_sample_count(&rsrc->base.b);
   const unsigned bytes = util_format_get_blocksize(view->format);

   /* Bound atomics are raw32. MS stores additionally use two/four raw32
    * lanes; normalized/packed formats need their own conversion lowering. */
   if (samples > 1) {
      switch (view->format) {
      case PIPE_FORMAT_R32_FLOAT:
      case PIPE_FORMAT_R32_SINT:
      case PIPE_FORMAT_R32_UINT:
      case PIPE_FORMAT_R32G32_FLOAT:
      case PIPE_FORMAT_R32G32_SINT:
      case PIPE_FORMAT_R32G32_UINT:
      case PIPE_FORMAT_R32G32B32A32_FLOAT:
      case PIPE_FORMAT_R32G32B32A32_SINT:
      case PIPE_FORMAT_R32G32B32A32_UINT:
         break;
      default:
         return false;
      }
   }

   if (layout->compressed || layout->sample_count_sa != samples ||
       (samples != 1 &&
        ((samples != 2 && samples != 4) || buffer || level != 0 ||
         layout->levels != 1 || layout->tiling != AIL_TILING_GPU)) ||
       (samples == 1 && bytes != 4) ||
       util_format_get_blockwidth(layout->format) != 1 ||
       util_format_get_blockheight(layout->format) != 1 ||
       (!buffer && util_format_get_blocksize(layout->format) != bytes) ||
       (!buffer && layout->tiling != AIL_TILING_LINEAR &&
                   layout->tiling != AIL_TILING_GPU))
      return false;

   uint64_t address = agx_map_texture_gpu(rsrc, layer);
   address += buffer ? view->u.buf.offset : ail_get_level_offset_B(layout, level);
   const bool linear = buffer || layout->tiling == AIL_TILING_LINEAR;
   out[0] = address;
   out[1] = address >> 32;
   out[2] = buffer ? 4 : linear ? ail_get_linear_stride_B(layout, level) : 0;
   /* The address helper takes an aligned pixel pitch, while stride_el is the
    * logical width. Match the tile-row rounding used by CPU/GPU copies. */
   out[3] = linear ? 0 :
      align(layout->stride_el[level], layout->tilesize_el[level].width_el);
   out[4] = linear ? 1 : layout->tilesize_el[level].width_el;
   out[5] = linear ? 1 : layout->tilesize_el[level].height_el;
   out[6] = layout->layer_stride_B;
   out[7] = layout->layer_stride_B >> 32;
   out[8] = samples;
   out[9] = bytes;
   return true;
}

/*
 * The record after a G17P storage image's texture/PBE pair carries the
 * software words Honeykrisp's Gen9 storage descriptor keeps there, at the same
 * offsets from the texture descriptor: the native texture-table index of the
 * image's texture, the packed addressing word image atomics read through
 * libagx_image_texel_address_gen9(), and for multisampled images the aligned
 * width libagx_emrt_ms_index_gen9() needs. Gen9's PBE has no free bits to
 * carry these, which is why AGX2's in-descriptor sideband is left out of AGX3
 * PBEs. A view whose addressing does not fit the packed word keeps it zero.
 */
static void
agx_g17p_fill_storage_tail(uint8_t *tail, unsigned record_size_B,
                           const struct pipe_image_view *view,
                           unsigned texture_index, bool emrt)
{
   STATIC_ASSERT(AGX_G17P_IMAGE_ATOMIC_SW_OFFSET + sizeof(uint64_t) <=
                 3 * AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE);
   STATIC_ASSERT(AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET + sizeof(uint32_t) <=
                 3 * AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE);
   assert(record_size_B == AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE);
   const unsigned base = 2 * record_size_B;

   uint32_t heap_index = texture_index;
   memcpy(tail + AGX_G17P_STORAGE_HEAP_INDEX_OFFSET - base, &heap_index,
          sizeof(heap_index));

   struct agx_resource *rsrc = agx_resource(view->resource);
   if (rsrc->base.b.target == PIPE_BUFFER) {
      /* Same software texel-buffer metadata as Honeykrisp. The native
       * descriptors already include the view byte offset, so the extra
       * coordinate offset is zero; atomics use the exact bound byte address. */
      const uint32_t size =
         agx_texture_buffer_size_el(view->format, view->u.buf.size);
      const uint32_t offset = 0;
      const uint64_t address = agx_map_gpu(rsrc) + view->u.buf.offset;
      memcpy(tail + AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET - base, &size,
             sizeof(size));
      memcpy(tail + AGX_G17P_BUFFER_TEXTURE_TAIL_OFFSET - base, &offset,
             sizeof(offset));
      memcpy(tail + AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET - base, &address,
             sizeof(address));
      return;
   }
   if (!rsrc->layout.writeable_image && !emrt)
      return;

   const struct ail_layout *layout = &rsrc->layout;
   const unsigned samples = util_res_sample_count(&rsrc->base.b);
   const bool linear = layout->tiling == AIL_TILING_LINEAR;
   /* Multisampled storage is bound as a linear buffer starting at its level,
    * so its level offset is zero, as in hk's pack_pbe(). */
   const unsigned level = samples > 1 ? 0 : view->u.tex.level;

   if (samples > 1) {
      const uint32_t aligned_width =
         align(u_minify(rsrc->base.b.width0, view->u.tex.level),
               layout->tilesize_el[view->u.tex.level].width_el);
      memcpy(tail + AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET - base, &aligned_width,
             sizeof(aligned_width));
   }

   uint64_t sw = 0;
   if (agx_g17p_pack_image_atomic_sw(
          ail_get_level_offset_B(layout, level),
          linear ? 1 : layout->tilesize_el[view->u.tex.level].width_el,
          linear ? 1 : layout->tilesize_el[view->u.tex.level].height_el,
          layout->layer_stride_B, samples, &sw))
      memcpy(tail + AGX_G17P_IMAGE_ATOMIC_SW_OFFSET - base, &sw, sizeof(sw));
}

static uint64_t
agx_upload_g17p_spilled_rt_table(struct agx_batch *batch, const void *pairs,
                                 unsigned texture_base)
{
   uint8_t entries[PIPE_MAX_COLOR_BUFS][AGX_G17P_EMRT_DESCRIPTOR_SIZE] = {0};
   for (unsigned rt = 0; rt < batch->key.nr_cbufs; ++rt) {
      if (!batch->tilebuffer_layout.spilled[rt])
         continue;
      const struct pipe_surface *surf = &batch->key.cbufs[rt];
      if (!surf->texture)
         return 0;
      struct pipe_image_view view = image_view_for_surface(surf);
      const uint32_t texture_index = texture_base + 2 * rt;
      memcpy(entries[rt], (const uint8_t *)pairs + 64 * rt, 64);
      agx_g17p_fill_storage_tail(entries[rt] + 64, 32, &view, texture_index, true);
      memcpy(entries[rt] + AGX_G17P_SAMPLED_HEAP_INDEX_OFFSET,
             &texture_index, sizeof(texture_index));
   }
   const uint64_t address = agx_pool_upload_aligned(
      &batch->pool, entries, batch->key.nr_cbufs * sizeof(entries[0]), 32);
   return address ? agx_pool_upload_aligned(&batch->pool, &address, 8, 8) : 0;
}

static void
agx_upload_textures(struct agx_batch *batch, struct agx_compiled_shader *cs,
                    mesa_shader_stage stage)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_stage_uniforms *unif = &batch->stage_uniforms[stage];

   /* G18G: refresh the whole software table, including unbound entries,
    * whenever image/sampler-view state changes. Allocation padding is not
    * view size. */
   if (agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G) {
      memset(unif->texture_buffer_size, 0, sizeof(unif->texture_buffer_size));
      memset(unif->image_buffer_size, 0, sizeof(unif->image_buffer_size));
      memset(unif->texture_buffer_rgb32, 0, sizeof(unif->texture_buffer_rgb32));
      memset(unif->texture_buffer_address, 0,
             sizeof(unif->texture_buffer_address));
      memset(unif->image_address_metadata, 0,
             sizeof(unif->image_address_metadata));
      for (unsigned i = 0; i < ctx->stage[stage].texture_count; ++i) {
         struct agx_sampler_view *view = ctx->stage[stage].textures[i];
         if (view && view->base.target == PIPE_BUFFER) {
            unif->texture_buffer_size[i] =
               agx_texture_buffer_size_el(view->format, view->base.u.buf.size);
            unif->texture_buffer_rgb32[i] =
               ail_pixel_format[view->format].channels ==
               AGX_CHANNELS_R32G32B32_EMULATED;
            unif->texture_buffer_address[i] =
               agx_map_texture_gpu(view->rsrc, 0) + view->base.u.buf.offset;
         }
      }
      u_foreach_bit(i, ctx->stage[stage].image_mask) {
         const struct pipe_image_view *view = &ctx->stage[stage].images[i];
         if (view->resource && view->resource->target == PIPE_BUFFER)
            unif->image_buffer_size[i] =
               agx_texture_buffer_size_el(view->format, view->u.buf.size);
      }
   }

   /* This can occur for meta shaders */
   if (!cs->so) {
      batch->texture_count[stage] = 0;
      batch->stage_uniforms[stage].texture_base = 0;
      return;
   }

   unsigned nr_textures = cs->so->info.nr_bindful_textures;

   unsigned nr_active_textures = ctx->stage[stage].texture_count;
   unsigned nr_tex_descriptors = agx_nr_tex_descriptors(batch, cs);
   unsigned nr_images = cs->so->info.nr_bindful_images;

   struct agx_device *dev = agx_device(ctx->base.screen);
   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);
   const unsigned descriptor_size =
      agx_gallium_texture_descriptor_size(descriptor_generation);
   if (!descriptor_size)
      agx_native_descriptor_unsupported(dev, "texture heap geometry");

   struct agx_ptr T_tex = agx_pool_alloc_aligned(
      &batch->pool, descriptor_size * nr_tex_descriptors, 64);

   struct agx_ptr buffer_metadata = {0};
   if (cs->so->info.uses_buffer_textures) {
      buffer_metadata = agx_pool_alloc_aligned(
         &batch->pool, nr_textures * AGX_G17P_EMRT_DESCRIPTOR_SIZE, 32);
      memset(buffer_metadata.cpu, 0, nr_textures * AGX_G17P_EMRT_DESCRIPTOR_SIZE);
   }
   batch->stage_uniforms[stage].buffer_texture_base = buffer_metadata.gpu;

   for (unsigned i = 0; i < MIN2(nr_textures, nr_active_textures); ++i) {
      struct agx_sampler_view *tex = ctx->stage[stage].textures[i];
      struct agx_gallium_texture_descriptor *texture =
         agx_texture_descriptor_at(T_tex.cpu, descriptor_size, i);

      if (tex == NULL) {
         if (!agx_gallium_set_null_native_texture(descriptor_generation,
                                                   texture))
            {
               agx_native_descriptor_reject_batch(batch, "null texture");
               return;
            }
         continue;
      }

      struct agx_resource *rsrc = tex->rsrc;
      agx_batch_reads(batch, tex->rsrc);
      if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_COMPUTE)
         agx_batch_note_vtx_read(batch, rsrc);

      /* Admit only resources covered by the native-backed stored path. */
      if (dev->chip == AGX_CHIP_G18G && rsrc->base.b.nr_samples > 1 &&
          !agx_g18_msaa4_storage(dev, rsrc, tex->format)) {
         agx_native_descriptor_reject_batch(batch, "G18G multisample texture");
         return;
      }

      /* Re-emit state because the layout might have changed from under us.
       * TODO: optimize this somehow?
       */
      if (!agx_pack_texture(descriptor_generation, &tex->desc, rsrc,
                            tex->format, &tex->base))
         {
            agx_native_descriptor_reject_batch(batch, "sampled texture");
            return;
         }

      memcpy(texture, &tex->desc, descriptor_size);
      if (buffer_metadata.cpu && tex->base.target == PIPE_BUFFER) {
         uint8_t *entry = (uint8_t *)buffer_metadata.cpu +
                          i * AGX_G17P_EMRT_DESCRIPTOR_SIZE;
         const uint32_t elements =
            agx_texture_buffer_size_el(tex->format, tex->base.u.buf.size);
         const uint64_t address = agx_map_gpu(rsrc) + tex->base.u.buf.offset;
         const uint32_t rgb32 = tex->format == PIPE_FORMAT_R32G32B32_FLOAT ||
                                tex->format == PIPE_FORMAT_R32G32B32_UINT ||
                                tex->format == PIPE_FORMAT_R32G32B32_SINT;
         memcpy(entry, &tex->desc, descriptor_size);
         memcpy(entry + AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET, &elements, 4);
         memcpy(entry + AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET, &address, 8);
         memcpy(entry + AGX_G17P_BUFFER_TEXTURE_RGB32_OFFSET, &rgb32, 4);
      }
   }

   for (unsigned i = nr_active_textures; i < nr_textures; ++i) {
      if (!agx_gallium_set_null_native_texture(
             descriptor_generation,
             agx_texture_descriptor_at(T_tex.cpu, descriptor_size, i)))
         {
            agx_native_descriptor_reject_batch(batch, "null texture");
            return;
         }
   }

   const unsigned image_records = cs->so->info.image_records;
   for (unsigned i = 0; i < nr_images; ++i) {
      batch->stage_uniforms[stage].image_buffer_size[i] = 0;
      /* Image descriptors come in pairs after the textures, each followed on
       * G17P by its software tail record. */
      const unsigned image = nr_textures + (image_records * i);
      struct agx_gallium_texture_descriptor *texture =
         agx_texture_descriptor_at(T_tex.cpu, descriptor_size, image);
      struct agx_gallium_pbe_descriptor *pbe =
         agx_pbe_descriptor_at(T_tex.cpu, descriptor_size, image + 1);
      uint8_t *tail = image_records > 2
                         ? (uint8_t *)agx_texture_descriptor_at(
                              T_tex.cpu, descriptor_size, image + 2)
                         : NULL;
      if (tail)
         memset(tail, 0, descriptor_size);

      if (!(ctx->stage[stage].image_mask & BITFIELD64_BIT(i))) {
         if (!agx_gallium_set_null_native_texture(descriptor_generation,
                                                   texture))
            {
               agx_native_descriptor_reject_batch(batch, "null storage texture");
               return;
            }
         if (!agx_gallium_set_null_native_pbe(descriptor_generation, pbe))
            {
               agx_native_descriptor_reject_batch(batch, "null storage PBE");
               return;
            }
         continue;
      }

      struct pipe_image_view *view = &ctx->stage[stage].images[i];
      if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_COMPUTE) {
         if (view->shader_access & PIPE_IMAGE_ACCESS_WRITE)
            batch->vtx_needs_render_barrier = true;
         else
            agx_batch_note_vtx_read(batch, agx_resource(view->resource));
      }
      agx_batch_track_image(batch, view);
      if (agx_chip_native_or_g18g(dev->chip) &&
          view->resource->target == PIPE_BUFFER)
         batch->stage_uniforms[stage].image_buffer_size[i] =
            agx_texture_buffer_size_el(view->format, view->u.buf.size);

      /* A store to another binding does not consume this image's PBE.
       * Shader/API access flags can conservatively include writes even for
       * a size query; use the actual descriptor consumers recorded by NIR. */
      const bool needs_pbe =
         cs->so->info.image_pbe_mask & BITFIELD64_BIT(i);

      if (cs->so->info.image_address_mask & BITFIELD64_BIT(i)) {
         uint32_t metadata[10];
         if (!agx_image_address_metadata(metadata, view) ||
             (view->resource->nr_samples > 1 && dev->chip != AGX_CHIP_G18G)) {
            agx_native_descriptor_reject_batch(batch, "G18G raw image layout");
            return;
         }
         memcpy(unif->image_address_metadata[i], metadata, sizeof(metadata));
      }

      /* Reads use the native MS texture/fetch path. Bound global atomics and
       * raw MS stores use the validated software metadata above. Neither
       * path consumes a PBE; ordinary image writes and sideband users still
       * require their own native encoding proof. */
      if (dev->chip == AGX_CHIP_G18G && view->resource &&
          view->resource->nr_samples > 1 &&
          (needs_pbe || cs->so->info.reads_image_sideband ||
           !agx_g18_msaa4_storage(dev, agx_resource(view->resource),
                                 view->format))) {
         if (dev->debug & AGX_DBG_FAULTTRACE)
            fprintf(stderr,
                    "G18_IMAGE_REJECT index=%u stage=%u format=%s samples=%u "
                    "shader_access=%u access=%u shader_writes=%u needs_pbe=%u "
                    "sideband=%u storage=%u\n",
                    i, stage, util_format_short_name(view->format),
                    view->resource->nr_samples, view->shader_access, view->access,
                    cs->so->info.writes_images, needs_pbe,
                    cs->so->info.reads_image_sideband,
                    agx_g18_msaa4_storage(dev, agx_resource(view->resource),
                                         view->format));
         agx_native_descriptor_reject_batch(batch, "G18G multisample image");
         return;
      }

      struct pipe_sampler_view sampler_view = util_image_to_sampler_view(view);

      /* For the texture descriptor, lower cubes to 2D arrays. This matches the
       * transform done in the compiler. Also, force 2D arrays for internal
       * blitter images, this helps reduce shader variants.
       */
      bool internal = (view->access & PIPE_IMAGE_ACCESS_DRIVER_INTERNAL);

      if (target_is_cube(sampler_view.target) ||
          (sampler_view.target == PIPE_TEXTURE_3D && internal))
         sampler_view.target = PIPE_TEXTURE_2D_ARRAY;

      if (!agx_pack_texture(descriptor_generation, texture,
                            agx_resource(view->resource), view->format,
                            &sampler_view))
         {
            agx_native_descriptor_reject_batch(batch, "storage texture");
            return;
         }

      /* A G18G shader that never consumes this image's PBE has no
       * consumer for this descriptor: native run112 kick 19's read-only
       * launch binds the texture and a sampler record and no PBE at all.
       * Bind the null PBE instead of the image's, whose AGX2 software
       * sideband (filled for every writeable image) has no Gen10 encoding
       * and would drop the batch. */
      if (dev->chip == AGX_CHIP_G18G && !needs_pbe) {
         if (!agx_gallium_set_null_native_pbe(descriptor_generation, pbe)) {
            agx_native_descriptor_reject_batch(batch, "null storage PBE");
            return;
         }
         continue;
      }

      /* A shared compressed Interchange image cannot be reallocated with
       * PIPE_BIND_SHADER_IMAGE in agx_set_shader_images(), leaving
       * layout.writeable_image false and layout.compressed true. Fail closed
       * on writable storage-image bindings instead of writing through a
       * compressed Interchange PBE. */
      if (!internal && (view->shader_access & PIPE_IMAGE_ACCESS_WRITE) &&
          view->resource->target != PIPE_BUFFER &&
          agx_resource_is_shared_interchange(agx_resource(view->resource))) {
         agx_native_descriptor_reject_batch(
            batch, "writable interchange storage image");
         return;
      }

      /* The lowered state-0 write, and every G18G image written by native
       * stores only (agx3_emit_storage_image_write_g18g: imgwci addresses
       * the texel from the PBE alone), leave the AGX2 software sideband
       * out; only an image atomic or texel address would read it. */
      const bool native_storage_write =
         (cs->g17p_storage_write && i == 0) ||
         (dev->chip == AGX_CHIP_G18G && !cs->so->info.reads_image_sideband);
      if (!agx_batch_upload_pbe_internal(batch, pbe, view, false, false, false,
                                         false, native_storage_write, NULL))
         {
            agx_native_descriptor_reject_batch(batch, "storage PBE");
            return;
         }

      if (tail)
         agx_g17p_fill_storage_tail(tail, descriptor_size, view, image, false);
   }

   if (stage == MESA_SHADER_FRAGMENT &&
       agx_tilebuffer_spills(&batch->tilebuffer_layout)) {

      void *out =
         (uint8_t *)T_tex.cpu +
         (descriptor_size * agx_nr_tex_descriptors_without_spilled_rts(cs));

      agx_upload_spilled_rt_descriptors(out, batch, descriptor_generation,
         agx_nr_tex_descriptors_without_spilled_rts(cs),
         agx_g18_native_ms_spills(batch));
      if (agx_chip_native_agx3(dev->chip)) {
         batch->g17p_emrt_heap = agx_upload_g17p_spilled_rt_table(
            batch, out, agx_nr_tex_descriptors_without_spilled_rts(cs));
         if (!batch->g17p_emrt_heap)
            agx_native_descriptor_rejected(dev, "native spilled-RT table");
      }
   }

   batch->texture_count[stage] = nr_tex_descriptors;
   batch->stage_uniforms[stage].texture_base = T_tex.gpu;

   if (agx_chip_agx3_path(dev->chip)) {
      for (unsigned i = 0; i < PIPE_MAX_SAMPLERS; ++i)
         batch->stage_uniforms[stage].texture_handle[i] = i;
   }
}

uint16_t
agx_sampler_heap_add(struct agx_device *dev, struct agx_sampler_heap *heap,
                     struct agx_sampler_packed *sampler)
{
   /* Allocate (maximally sized) BO if we haven't already */
   if (!heap->bo) {
      heap->bo = agx_bo_create(dev, AGX_SAMPLER_HEAP_SIZE * AGX_SAMPLER_LENGTH,
                               0, AGX_BO_WRITEBACK, "Sampler heap");

      assert(heap->count == 0);
   }

   /* TODO search */

   /* Precondition: there is room in the heap */
   assert(heap->count < AGX_SAMPLER_HEAP_SIZE);
   struct agx_sampler_packed *samplers = agx_bo_map(heap->bo);
   memcpy(samplers + heap->count, sampler, sizeof(*sampler));

   return heap->count++;
}

/* API sampler state in the native heap records. G17P is the HAL200 record
 * with its own compare enable. G17G's Gen5 record overlaps the compare
 * function with the reduction mode and selects the compare interpretation
 * with compare_mode; the comparison itself is requested by the sample
 * instruction. This mirrors hk_pack_native_sampler(), including LEQUAL for a
 * sampler without a comparison. */
static void
agx_pack_g17_api_sampler(enum agx_chip chip, void *out,
                         const struct pipe_sampler_state *state,
                         enum agx_border_colour border_colour,
                         bool bias_in_descriptor)
{
   const bool compare = state->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;
   const enum agx_compare_func compare_func =
      compare ? agx_compare_funcs[state->compare_func]
              : AGX_COMPARE_FUNC_LEQUAL;

   if (agx_chip_hal300_path(chip)) {
      agx_pack(out, SAMPLER_GEN5, cfg) {
         cfg.minimum_lod = state->min_lod;
         cfg.maximum_lod = state->max_lod;
         cfg.maximum_anisotropy =
            util_next_power_of_two(MAX2(state->max_anisotropy, 1));
         cfg.magnify = agx_filters[state->mag_img_filter];
         cfg.minify = agx_filters[state->min_img_filter];
         cfg.mip_filter = agx_mip_filter_from_pipe(state->min_mip_filter);
         cfg.wrap_s = agx_wrap_from_pipe(state->wrap_s);
         cfg.wrap_t = agx_wrap_from_pipe(state->wrap_t);
         cfg.wrap_r = agx_wrap_from_pipe(state->wrap_r);
         cfg.pixel_coordinates = state->unnormalized_coords;
         /* The API bias is applied exactly once. When nir_lower_tex sums it
          * into the shader's own bias -- which it does for every chip but
          * G17P -- carrying it here as well applies it twice, and the
          * descriptor field quantises to 1/64 on top of the hardware's own
          * quantisation of the level it samples at. Measured on G17G with a
          * level-valued mip pyramid, a sampler bias of one selected level
          * two and a bias of two selected level four. */
         if (bias_in_descriptor)
            cfg.lod_bias = CLAMP(state->lod_bias, -16.0, 16.0);
         cfg.seamful_cube_maps = !state->seamless_cube_map;
         cfg.border_colour = border_colour;
         cfg.compare_mode = true;
         cfg.compare_func = compare_func;
      }
      return;
   }

   agx_pack(out, SAMPLER_HAL200, cfg) {
      cfg.minimum_lod = state->min_lod;
      cfg.maximum_lod = state->max_lod;
      cfg.maximum_anisotropy =
         util_next_power_of_two(MAX2(state->max_anisotropy, 1));
      cfg.magnify = agx_filters[state->mag_img_filter];
      cfg.minify = agx_filters[state->min_img_filter];
      cfg.mip_filter = agx_mip_filter_from_pipe(state->min_mip_filter);
      cfg.wrap_s = agx_wrap_from_pipe(state->wrap_s);
      cfg.wrap_t = agx_wrap_from_pipe(state->wrap_t);
      cfg.wrap_r = agx_wrap_from_pipe(state->wrap_r);
      cfg.pixel_coordinates = state->unnormalized_coords;
      cfg.compare_func = agx_compare_funcs[state->compare_func];
      cfg.compare_enable = compare;
      if (bias_in_descriptor)
         cfg.lod_bias = CLAMP(state->lod_bias, -16.0, 16.0);
      cfg.seamful_cube_maps = !state->seamless_cube_map;
      cfg.border_colour = border_colour;
   }
}

/* Native table entry for an API sampler, with the given border colour. */
static struct agx_gallium_native_sampler
agx_native_sampler_state(const struct pipe_sampler_state *state,
                         enum agx_border_colour border_colour)
{
   const bool compare_enable =
      state->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;

   return (struct agx_gallium_native_sampler){
      .minimum_lod = state->min_lod,
      .maximum_lod = state->max_lod,
      .maximum_anisotropy =
         util_next_power_of_two(MAX2(state->max_anisotropy, 1)),
      .magnify = agx_filters[state->mag_img_filter],
      .minify = agx_filters[state->min_img_filter],
      .mip_filter = agx_mip_filter_from_pipe(state->min_mip_filter),
      .wrap_s = agx_wrap_from_pipe(state->wrap_s),
      .wrap_t = agx_wrap_from_pipe(state->wrap_t),
      .wrap_r = agx_wrap_from_pipe(state->wrap_r),
      .pixel_coordinates = state->unnormalized_coords,
      .compare_func = agx_compare_funcs[state->compare_func],
      .compare_enable = compare_enable,
      /* An ordinary sample takes the bias as a shader operand, summed with
       * the shader's own in floating point; only a depth comparison, whose
       * level has no per-lane form here, keeps it in the descriptor where the
       * sampler has a bias field. See agx_create_context(). */
      .lod_bias = compare_enable ? CLAMP(state->lod_bias, -16.0, 16.0) : 0.0f,
      .seamful_cube_maps = !state->seamless_cube_map,
      .border_colour = border_colour,
   };
}


/* API sampler state in the native heap records. G17P is the HAL200 record
 * with its own compare enable. G17G's Gen5 record overlaps the compare
 * function with the reduction mode and selects the compare interpretation
 * with compare_mode; the comparison itself is requested by the sample
 * instruction. This mirrors hk_pack_native_sampler(), including LEQUAL for a
 * sampler without a comparison. */

static void
agx_upload_samplers(struct agx_batch *batch, struct agx_compiled_shader *cs,
                    mesa_shader_stage stage)
{
   struct agx_context *ctx = batch->ctx;

   unsigned nr_samplers = sampler_count(ctx, stage);
   bool custom_borders = ctx->stage[stage].custom_borders;

   struct agx_device *dev = agx_device(ctx->base.screen);
   const bool native = agx_chip_native_agx3(dev->chip);
   /* G18G uses the same native table, packed by agx_pack_g17_api_sampler(). */
   const bool g18 = dev->chip == AGX_CHIP_G18G;
   size_t sampler_length = native || g18 || agx_chip_g17_m5(dev->chip)
      ? agx_chip_sampler_heap_element_size(dev->chip)
      : AGX_SAMPLER_LENGTH + (custom_borders ? AGX_BORDER_LENGTH : 0);

   if (native || g18) {
      /* Keep ordinary sampler handles stable at i+1.  If any sampler needs a
       * custom border, reserve a disjoint clamp-to-zero bank at slot 33, as
       * Honeykrisp does with its second sampler plane. */
      const bool query_lod = cs->so && cs->so->info.uses_query_lod;
      unsigned count = nr_samplers;
      if (custom_borders)
         count = (agx_shadow_gather_custom_border(dev->chip)
                     ? AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE
                     : AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE) +
                 ctx->stage[stage].sampler_count;
      if (query_lod)
         count = MAX2(count, AGX_GALLIUM_G17P_QUERY_LOD_BASE +
                             ctx->stage[stage].sampler_count);
      struct agx_ptr T = agx_pool_alloc_aligned(
         &batch->pool, sampler_length * count, 64);
      memset(T.cpu, 0, sampler_length * count);

      /* Native indexed read() and sample() share this table. Match
       * Honeykrisp's ABI by keeping the exact texel-fetch sampler at zero and
       * placing API samplers after it. */
      if (!agx_gallium_pack_g17p_txf_sampler(dev->chip, T.cpu))
         {
            agx_native_descriptor_reject_batch(batch, "texel-fetch sampler");
            return;
         }

      /* Keep API sampler zero at its existing handle one when present.
       * Image-only MS reads still consume handle one, so populate it with
       * the same texel-fetch state rather than an absent table entry. */
      if (!ctx->stage[stage].sampler_count)
         memcpy((uint8_t *)T.cpu + sampler_length, T.cpu, sampler_length);

      for (unsigned i = 0; i < ctx->stage[stage].sampler_count; ++i) {
         struct agx_sampler_state *sampler = ctx->stage[stage].samplers[i];
         uint8_t *out = (uint8_t *)T.cpu + sampler_length * (i + 1);
         if (sampler) {
            const struct pipe_sampler_state *state = &sampler->base;
            const struct agx_gallium_native_sampler native =
               agx_native_sampler_state(
                  state, sampler->uses_custom_border
                            ? AGX_BORDER_COLOUR_OPAQUE_WHITE
                            : sampler->border_colour);
            if (g18)
               agx_pack_g17_api_sampler(dev->chip, out, state,
                                        native.border_colour,
                                        !ctx->support_lod_bias);
            else if (!agx_gallium_pack_native_sampler(dev->chip, (void *)out,
                                                      &native))
               agx_native_descriptor_rejected(dev, "sampler");

            if (sampler->uses_custom_border) {
               uint16_t clamp_handle;
               if (!agx_gallium_g17p_custom_border_handle(dev->chip, i,
                                                            &clamp_handle)) {
                  agx_native_descriptor_reject_batch(batch,
                                                       "custom-border sampler");
                  return;
               }

               uint8_t *clamp = (uint8_t *)T.cpu +
                                sampler_length * clamp_handle;
               struct agx_gallium_native_sampler clamp_state =
                  agx_native_sampler_state(
                     state, AGX_BORDER_COLOUR_TRANSPARENT_BLACK);
               if (g18) {
                  agx_pack_g17_api_sampler(dev->chip, clamp, state,
                                           AGX_BORDER_COLOUR_TRANSPARENT_BLACK,
                                           !ctx->support_lod_bias);
                  continue;
               }
               if (agx_shadow_gather_custom_border(dev->chip))
                  clamp_state.compare_enable = false;
               if (!agx_gallium_pack_native_sampler(dev->chip, (void *)clamp,
                                                    &clamp_state))
                  agx_native_descriptor_rejected(dev,
                                                 "custom-border sampler");

               if (agx_shadow_gather_custom_border(dev->chip)) {
                  clamp_state.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
                  uint8_t *probe = (uint8_t *)T.cpu + sampler_length *
                     (AGX_GALLIUM_G17P_BORDER_NONCOMPARE_BASE + i);
                  if (!agx_gallium_pack_native_sampler(dev->chip, (void *)probe,
                                                       &clamp_state))
                     agx_native_descriptor_rejected(dev,
                                                    "custom-border probe sampler");
               }
            }
         } else {
            if (g18) {
               /* An unbound slot keeps the texel-fetch sampler's state. */
               agx_gallium_pack_g17p_txf_sampler(
                  dev->chip, (struct agx_gallium_sampler_descriptor *)out);
               continue;
            }
            const struct agx_gallium_native_sampler null_state = {
               .maximum_lod = INFINITY,
               .maximum_anisotropy = 1,
               .mip_filter = AGX_MIP_FILTER_NEAREST,
               .wrap_s = AGX_WRAP_CLAMP_TO_BORDER,
               .wrap_t = AGX_WRAP_CLAMP_TO_BORDER,
               .wrap_r = AGX_WRAP_CLAMP_TO_BORDER,
               .compare_func = AGX_COMPARE_FUNC_LEQUAL,
               .border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK,
            };
            if (!agx_gallium_pack_native_sampler(dev->chip, (void *)out,
                                                 &null_state))
               agx_native_descriptor_rejected(dev, "null sampler");
         }
      }

      if (query_lod) {
         for (unsigned i = 0; i < ctx->stage[stage].sampler_count; ++i) {
            const struct agx_sampler_state *sampler =
               ctx->stage[stage].samplers[i];
            const uint8_t *ordinary =
               (const uint8_t *)T.cpu + sampler_length * (i + 1);
            uint8_t *query = (uint8_t *)T.cpu + sampler_length *
               (AGX_GALLIUM_G17P_QUERY_LOD_BASE + i);
            /* A Gen4 sampler has no bias field to carry the API bias into
             * the LOD query, so the query reads the ordinary entry. */
            if (!agx_chip_sampler_has_lod_bias(dev->chip)) {
               memcpy(query, ordinary, sampler_length);
               continue;
            }
            agx_unpack(NULL, ordinary, SAMPLER_HAL200, cfg);
            cfg.lod_bias = sampler ? CLAMP(sampler->base.lod_bias, -16.0, 16.0)
                                   : 0.0;
            AGX_SAMPLER_HAL200_pack((uint32_t *)query, &cfg);
         }
      }

      batch->sampler_count[stage] = count;
      batch->samplers[stage] = T.gpu;
      return;
   }

   if (agx_chip_g17_m5(dev->chip)) {
      /* Keep ordinary sampler handles stable at i+1.  If any sampler needs a
       * custom border, reserve a disjoint clamp-to-zero bank at slot 33, as
       * Honeykrisp does with its second sampler plane. */
      const unsigned count = custom_borders
                                ? AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE +
                                     ctx->stage[stage].sampler_count
                                : nr_samplers;
      struct agx_ptr T = agx_pool_alloc_aligned(
         &batch->pool, sampler_length * count, 64);
      memset(T.cpu, 0, sampler_length * count);

      /* Native indexed read() and sample() share this table. Match
       * Honeykrisp's ABI by keeping the exact texel-fetch sampler at zero and
       * placing API samplers after it. */
      if (!agx_gallium_pack_g17p_txf_sampler(dev->chip, T.cpu)) {
         agx_native_descriptor_reject_batch(batch, "texel-fetch sampler");
         return;
      }

      for (unsigned i = 0; i < ctx->stage[stage].sampler_count; ++i) {
         struct agx_sampler_state *sampler = ctx->stage[stage].samplers[i];
         uint8_t *out = (uint8_t *)T.cpu + sampler_length * (i + 1);
         if (sampler) {
            const struct pipe_sampler_state *state = &sampler->base;
            agx_pack_g17_api_sampler(dev->chip, out, state,
                                     sampler->uses_custom_border
                                        ? AGX_BORDER_COLOUR_OPAQUE_WHITE
                                        : sampler->border_colour,
                                     !ctx->support_lod_bias);

            if (sampler->uses_custom_border) {
               uint16_t clamp_handle;
               if (!agx_gallium_g17p_custom_border_handle(dev->chip, i,
                                                           &clamp_handle)) {
                  agx_native_descriptor_reject_batch(batch,
                                                     "custom-border sampler");
                  return;
               }

               uint8_t *clamp = (uint8_t *)T.cpu +
                                sampler_length * clamp_handle;
               agx_pack_g17_api_sampler(dev->chip, clamp, state,
                                        AGX_BORDER_COLOUR_TRANSPARENT_BLACK,
                                        !ctx->support_lod_bias);
            }
         } else {
            /* An unbound slot keeps the texel-fetch sampler's state. */
            agx_gallium_pack_g17p_txf_sampler(
               dev->chip, (struct agx_gallium_sampler_descriptor *)out);
         }
      }

      batch->sampler_count[stage] = count;
      batch->samplers[stage] = T.gpu;
      return;
   }

   struct agx_ptr T =
      agx_pool_alloc_aligned(&batch->pool, sampler_length * nr_samplers, 64);

   /* Sampler #0 is reserved for txf */
   agx_pack_txf_sampler(T.cpu);

   /* Remaining samplers are API samplers */
   uint8_t *out_sampler = (uint8_t *)T.cpu + sampler_length;
   for (unsigned i = 0; i < ctx->stage[stage].sampler_count; ++i) {
      struct agx_sampler_state *sampler = ctx->stage[stage].samplers[i];
      struct agx_sampler_packed *out = (struct agx_sampler_packed *)out_sampler;

      if (sampler) {
         *out = sampler->desc;

         if (custom_borders) {
            STATIC_ASSERT(sizeof(sampler->border) == AGX_BORDER_LENGTH);

            memcpy(out_sampler + AGX_SAMPLER_LENGTH, &sampler->border,
                   AGX_BORDER_LENGTH);
         } else {
            assert(!sampler->uses_custom_border && "invalid combination");
         }
      } else {
         memset(out, 0, sampler_length);
      }

      out_sampler += sampler_length;
   }

   batch->sampler_count[stage] = nr_samplers;
   batch->samplers[stage] = T.gpu;
}

static void
agx_update_descriptors(struct agx_batch *batch, struct agx_compiled_shader *cs)
{
   struct agx_context *ctx = batch->ctx;
   if (!cs)
      return;

   mesa_shader_stage stage = cs->stage;
   if (!ctx->stage[stage].dirty)
      return;

   if (ctx->stage[stage].dirty & AGX_STAGE_DIRTY_CONST)
      agx_set_cbuf_uniforms(batch, stage);

   if (ctx->stage[stage].dirty & AGX_STAGE_DIRTY_SSBO)
      agx_set_ssbo_uniforms(batch, stage);

   if (ctx->stage[stage].dirty & AGX_STAGE_DIRTY_IMAGE)
      agx_upload_textures(batch, cs, stage);

   if (ctx->stage[stage].dirty & AGX_STAGE_DIRTY_SAMPLER)
      agx_set_sampler_uniforms(batch, stage);

   if (ctx->stage[stage].dirty & AGX_STAGE_DIRTY_SAMPLER)
      agx_upload_samplers(batch, cs, stage);

   struct agx_stage_uniforms *unif = &batch->stage_uniforms[stage];
   const size_t full_size =
      agx_stage_uniforms_size(agx_device(ctx->base.screen)->chip);
   size_t size = full_size;

   /* A vertex, fragment, or compute shader reads its stage table only through
    * the ranges it pushes, unless it indexes the table dynamically (marked by
    * UINT16_MAX in stage_uniforms_upload_size). Geometry and tessellation
    * draws keep the whole table.
    */
   if (!ctx->gs && !ctx->in_tess &&
       (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_FRAGMENT ||
        stage == MESA_SHADER_COMPUTE)) {
      if (cs->stage_uniforms_upload_size != UINT16_MAX)
         size = MIN2((size_t)cs->stage_uniforms_upload_size, full_size);
   }

   batch->uniforms.tables[AGX_SYSVAL_STAGE(stage)] =
      size ? agx_pool_upload_aligned(&batch->pool, unif, size, 16) : 0;
}

static bool
agx_bind_g17p_preamble(const struct agx_shader_info *info, const void *binary,
                       struct agx_bo *original_bo,
                       struct agx_ioto_preshader_desc *preamble)
{
   *preamble = (struct agx_ioto_preshader_desc){
      .external_va = AGX_IOTO_PRESHADER_NO_VA,
   };
   if (!info->has_preamble)
      return !info->preamble_size && !info->nr_preamble_gprs &&
             !info->preamble_scratch_size &&
             !info->agx3_preamble_staging_known &&
             !info->agx3_preamble_staging_halfs;

   if (!info->nr_preamble_gprs || !info->agx3_preamble_staging_known ||
       info->preamble_scratch_size ||
       !original_bo || !original_bo->va ||
       info->main_offset > info->binary_size ||
       info->main_size > info->binary_size - info->main_offset)
      return false;

   /* The compiler gives preambles uniform inputs, with no incoming GPR ABI.
    * Do not request Apple's optional entry-register prelude. Reuse the
    * original binary's upload: linking relocates only the main program. */
   if (!agx_ioto_preshader_bind_range(
          binary, info->binary_size, original_bo->va->addr, original_bo->size,
          info->preamble_offset, info->preamble_size, preamble))
      return false;

   return info->preamble_offset + info->preamble_size <= info->main_offset ||
          info->main_offset + info->main_size <= info->preamble_offset;
}

struct agx_g17p_graphics_template {
   struct agx_g17p_graphics_template *next;
   struct agx3_g17p_graphics_program_desc desc;
   struct agx3_ioto_uniform_range ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   struct agx3_g17p_graphics_program_template program;
   bool has_plan;
   struct hk_g17p_patch_plan plan;
};

#define AGX_G17P_GRAPHICS_TEMPLATE_LIMIT 8u
#define AGX_G17P_GRAPHICS_TEMPLATE_MAX_SHAPES 2u

static inline bool
agx_g17p_graphics_desc_supports_plan(
   const struct agx3_g17p_graphics_program_desc *desc)
{
   return !desc->preshader && !desc->preloaded_root &&
          !desc->constant_program.code_size &&
          !(desc->cprog_inputs && desc->cprog_inputs->count);
}

static inline bool
agx_g17p_graphics_template_matches_shape(
   const struct agx_g17p_graphics_template *entry,
   const struct agx3_g17p_graphics_program_desc *desc)
{
   const struct agx3_g17p_graphics_program_desc *a = &entry->desc;
   if (a->uniform_range_count != desc->uniform_range_count ||
       a->shader_va != desc->shader_va ||
       a->shader_binary_size != desc->shader_binary_size ||
       a->scratch_size != desc->scratch_size || a->nr_gprs != desc->nr_gprs ||
       a->compiler_d_valid != desc->compiler_d_valid ||
       a->compiler_d != desc->compiler_d || a->fragment != desc->fragment ||
       a->writes_sample_mask != desc->writes_sample_mask ||
       a->colour_classes != desc->colour_classes ||
       a->imageblock.sample_size_B != desc->imageblock.sample_size_B ||
       a->imageblock.nr_samples != desc->imageblock.nr_samples ||
       a->imageblock.tile_size_px != desc->imageblock.tile_size_px)
      return false;

   for (unsigned i = 0; i < desc->uniform_range_count; ++i) {
      if (entry->ranges[i].start_halfs != desc->uniform_ranges[i].start_halfs ||
          entry->ranges[i].size_halfs != desc->uniform_ranges[i].size_halfs)
         return false;
   }
   return true;
}

static inline bool
agx_g17p_graphics_template_ranges_valid(
   const struct agx3_ioto_uniform_range *ranges, size_t count)
{
   for (size_t i = 0; i < count; ++i) {
      const uint64_t bytes = (uint64_t)ranges[i].size_halfs * 2;
      if (!ranges[i].size_halfs || (ranges[i].size_halfs & 1) ||
          (ranges[i].start_halfs & 1) || (ranges[i].address & 3) ||
          ranges[i].address > UINT64_MAX - (bytes - 4))
         return false;
   }
   return true;
}

static inline bool
agx_g17p_graphics_template_emit_plan(
   const struct agx_g17p_graphics_template *entry,
   const struct agx3_ioto_uniform_range *ranges, void *output, size_t capacity,
   size_t *written)
{
   const uint16_t len = entry->program.program.length;
   if (len > capacity)
      return false;

   /* Patch in cached stack memory before one sequential write into the
    * write-combined pipeline pool for ordinary program sizes. */
   if (likely(len <= 512)) {
      uint8_t buf[512];
      memcpy(buf, entry->program.program.bytes, len);
      hk_g17p_patch_plan_apply(&entry->plan, ranges, buf);
      memcpy(output, buf, len);
   } else {
      memcpy(output, entry->program.program.bytes, len);
      hk_g17p_patch_plan_apply(&entry->plan, ranges, output);
   }
   *written = len;
   return true;
}

static struct agx_g17p_graphics_template *
agx_g17p_graphics_program_find_plan(
   struct agx_linked_shader *linked,
   const struct agx3_g17p_graphics_program_desc *desc)
{
   if (!agx_g17p_graphics_desc_supports_plan(desc) ||
       !agx_g17p_graphics_template_ranges_valid(desc->uniform_ranges,
                                                desc->uniform_range_count))
      return NULL;

   struct agx_g17p_graphics_template **slot =
      &linked->g17p_graphics_templates;
   while (*slot) {
      struct agx_g17p_graphics_template *entry = *slot;
      if (entry->has_plan &&
          agx_g17p_graphics_template_matches_shape(entry, desc) &&
          hk_g17p_patch_plan_fits(&entry->plan, desc->uniform_ranges)) {
         if (*slot != linked->g17p_graphics_templates) {
            *slot = entry->next;
            entry->next = linked->g17p_graphics_templates;
            linked->g17p_graphics_templates = entry;
         }
         return entry;
      }
      slot = &entry->next;
   }
   return NULL;
}

static bool
agx_g17p_graphics_program_relocate_cached(
   enum agx_chip chip, struct agx_linked_shader *linked,
   const struct agx3_g17p_graphics_program_desc *desc, void *output,
   size_t capacity, size_t *written)
{
   const struct agx_g17p_graphics_template *plan =
      agx_g17p_graphics_program_find_plan(linked, desc);
   if (plan)
      return agx_g17p_graphics_template_emit_plan(plan, desc->uniform_ranges,
                                                  output, capacity, written);

   struct agx_g17p_graphics_template **slot =
      &linked->g17p_graphics_templates;
   while (*slot) {
      struct agx_g17p_graphics_template *entry = *slot;
      if (entry->has_plan) {
         slot = &entry->next;
         continue;
      }
      uint16_t length;
      /* The composer checks the complete shader/imageblock state, range
       * layout, address-cache topology and immediate instruction widths.
       * Only the draw-varying addresses may be patched; a different shape
       * goes through full generation instead. */
      if (agx3_g17p_graphics_program_template_relocate(
             chip, &entry->desc, desc, &entry->program, output, capacity,
             &length)) {
         if (*slot != linked->g17p_graphics_templates) {
            *slot = entry->next;
            entry->next = linked->g17p_graphics_templates;
            linked->g17p_graphics_templates = entry;
         }
         *written = length;
         return true;
      }
      slot = &entry->next;
   }
   return false;
}

static bool
agx_g17p_graphics_program_generate_cached(
   enum agx_chip chip, struct agx_linked_shader *linked,
   const struct agx3_g17p_graphics_program_desc *desc, void *output,
   size_t capacity, size_t *written)
{
   const bool can_plan = agx_g17p_graphics_desc_supports_plan(desc);
   unsigned count = 0;
   unsigned shapes = 0;
   struct agx_g17p_graphics_template **slot =
      &linked->g17p_graphics_templates;
   while (*slot) {
      if (can_plan && (*slot)->has_plan &&
          agx_g17p_graphics_template_matches_shape(*slot, desc))
         ++shapes;
      slot = &(*slot)->next;
      ++count;
   }

   struct agx_g17p_graphics_template *entry =
      rzalloc(linked, struct agx_g17p_graphics_template);
   if (!entry)
      return agx3_g17p_graphics_program_write_unpublished(
         chip, desc, output, capacity, written);

   entry->desc = *desc;
   memcpy(entry->ranges, desc->uniform_ranges,
          desc->uniform_range_count * sizeof(entry->ranges[0]));
   entry->desc.uniform_ranges = entry->ranges;

   /* A layout whose ranges keep moving relative to each other gets one
    * address-independent template (each range its own 8-byte LDIMM base),
    * matching Honeykrisp's loader cache. */
   const bool try_independent =
      can_plan && shapes >= AGX_G17P_GRAPHICS_TEMPLATE_MAX_SHAPES &&
      desc->uniform_range_count <= HK_G17P_PATCH_MAX_RANGES;
   bool generated = false;
   if (try_independent) {
      entry->desc.patchable = true;
      generated = agx3_g17p_graphics_program_template_generate(
                     chip, &entry->desc, &entry->program) &&
                  entry->program.program.length <= capacity;
      if (!generated)
         entry->desc.patchable = desc->patchable;
   }
   if (!generated) {
      if (!agx3_g17p_graphics_program_template_generate(
             chip, &entry->desc, &entry->program) ||
          entry->program.program.length > capacity) {
         ralloc_free(entry);
         return false;
      }
   }

   if (can_plan) {
      entry->has_plan = hk_g17p_patch_plan_init(
         &entry->plan, entry->ranges, entry->desc.uniform_range_count,
         entry->program.relocations, entry->program.relocation_count,
         entry->program.program.length, entry->desc.patchable);
   }

   /* Bound memory and lookup cost even when an application keeps changing
    * its uniform allocation layout. The coldest entry is evicted. */
   if (count == AGX_G17P_GRAPHICS_TEMPLATE_LIMIT) {
      slot = &linked->g17p_graphics_templates;
      while ((*slot)->next)
         slot = &(*slot)->next;
      ralloc_free(*slot);
      *slot = NULL;
   }
   entry->next = linked->g17p_graphics_templates;
   linked->g17p_graphics_templates = entry;
   linked->g17p_graphics_templates_generated++;
   *written = entry->program.program.length;
   memcpy(output, entry->program.program.bytes, *written);
   return true;
}

static bool
agx_g17p_graphics_program_cached(enum agx_chip chip,
                                 struct agx_linked_shader *linked,
                                 const struct agx3_g17p_graphics_program_desc *desc,
                                 void *output, size_t capacity,
                                 size_t *written)
{
   if (agx_g17p_graphics_program_relocate_cached(chip, linked, desc, output,
                                                 capacity, written))
      return true;
   return agx_g17p_graphics_program_generate_cached(chip, linked, desc, output,
                                                    capacity, written);
}
/* The static TPS frame a graphics program declares. G18 programs it in both
 * SPECTPR words as a 16-byte-aligned byte count divided by four (native1054:
 * 528 and 2064 bytes -> 132 and 516); other chips keep the compiler value. */
static uint32_t
agx_graphics_tps_frame(const struct agx_device *dev,
                       const struct agx_linked_shader *linked)
{
   const uint32_t frame =
      MAX2(linked->scratch_size, linked->agx3_spectpr_scratch_size);
   return dev->chip == AGX_CHIP_G18G ? ALIGN_POT(frame, 16) : frame;
}

struct agx_g17p_uniform_range_locs {
   int draw_id_range;
   uint16_t draw_id_offset;
   uint16_t draw_id_size;
   int params_range;
   bool uses_indirect_root;
};

static bool
agx_g17p_collect_graphics_uniform_ranges(
   struct agx_batch *batch, struct agx_compiled_shader *cs,
   struct agx_linked_shader *linked,
   struct agx3_ioto_uniform_range ranges[AGX3_G17P_GRAPHICS_MAX_RANGES],
   size_t *count_out, struct agx_g17p_uniform_range_locs *locs)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   size_t count = 0;
   const mesa_shader_stage stage = cs->stage;
   if (locs) {
      *locs = (struct agx_g17p_uniform_range_locs){
         .draw_id_range = -1,
         .params_range = -1,
      };
   }

   /* A hardware vertex program -- the vertex shader, a tessellation
    * evaluation shader run as one, or a geometry shader's copy program --
    * takes the pair exactly when it was compiled to read it. The copy
    * program samples nothing even when its geometry shader, whose texture
    * table this stage's count describes, does. */
   const bool resource_pair =
      batch->texture_count[stage] != 0 &&
      (stage == MESA_SHADER_FRAGMENT || cs->g17p_texture_resources);

   if (batch->texture_count[stage] && !resource_pair &&
       stage == MESA_SHADER_FRAGMENT)
      return false;
   if (resource_pair) {
      if (!batch->stage_uniforms[stage].texture_base ||
          !batch->samplers[stage] ||
          (batch->ctx->stage[stage].custom_borders &&
           !cs->g17p_custom_border_emulation))
         return false;
      const uint64_t pointers[2] = {
         batch->stage_uniforms[stage].texture_base,
         batch->samplers[stage],
      };
      const uint64_t pointer_va = agx_pool_upload_aligned(
         &batch->pool, pointers, sizeof(pointers), 8);
      if (!pointer_va)
         return false;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 8,
         .address = pointer_va,
      };
   }

   const uint32_t draw_id_off = offsetof(struct agx_draw_uniforms, draw_id);
   for (unsigned i = 0; i < cs->push_range_count; ++i) {
      const struct agx_push_range *push = &cs->push[i];
      const uint64_t table = batch->uniforms.tables[push->table];

      /* Fragment uniform zero is deliberately replaced by the resource pair.
       * A textured vertex/TES program instead shifts its complete fixed ABI;
       * trimming a VBO or draw-parameter range would corrupt vertex fetches. */
      if (resource_pair &&
          (stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL) &&
          push->uniform < AGX_ABI_G17P_TEXTURE_POINTER_HALFS)
         return false;

      /* Draw parameters are intentionally absent for draws whose shader does
       * not consume them. This is the same omission as the legacy builder. */
      if (push->table == AGX_SYSVAL_TABLE_PARAMS && !table)
         continue;
      unsigned start = push->uniform;
      unsigned length = push->length;
      uint64_t source = table + push->offset;
      const bool blend_constants =
         resource_pair && stage == MESA_SHADER_FRAGMENT &&
         push->table == AGX_SYSVAL_TABLE_ROOT &&
         push->offset == offsetof(struct agx_draw_uniforms, blend_constant) &&
         push->uniform == AGX_ABI_FUNI_BLEND_R && push->length == 8;

      if (blend_constants) {
         if (!linked->g17p_blend_constant_base)
            return false;
         start = linked->g17p_blend_constant_base;
      } else if (resource_pair && start < 8) {
         const unsigned end = start + length;
         if (end <= 8)
            continue;
         source += (8 - start) * 2;
         length = end - 8;
         start = 8;
      }

      /* Native Ioto moves uniform dwords. A single live 16-bit stage sysval
       * (for example the custom-border companion sentinel) therefore needs
       * its adjacent half uploaded as harmless padding. The stage table is a
       * fixed driver allocation, so prove that adjacent half is in bounds. */
      if ((length & 1) && push->table == AGX_SYSVAL_STAGE(stage)) {
         if ((uint64_t)push->offset + ((uint64_t)length + 1) * 2 >
             agx_stage_uniforms_size(dev->chip))
            return false;
         ++length;
      }
      if (!table || (start & 1) || (length & 1) ||
          count == AGX3_G17P_GRAPHICS_MAX_RANGES)
         return false;

      if (locs) {
         if (push->table == AGX_SYSVAL_TABLE_ROOT) {
            const uint32_t off = (uint32_t)(source - table);
            const uint32_t size_B = length * 2;
            if (off < sizeof(batch->uniforms.tables[AGX_SYSVAL_TABLE_ROOT]))
               locs->uses_indirect_root = true;
            if (off <= draw_id_off &&
                off + size_B >= draw_id_off + sizeof(batch->uniforms.draw_id)) {
               if (locs->draw_id_range >= 0)
                  locs->uses_indirect_root = true;
               locs->draw_id_range = (int)count;
               locs->draw_id_offset = (uint16_t)off;
               locs->draw_id_size = (uint16_t)size_B;
            }
         } else if (push->table == AGX_SYSVAL_TABLE_PARAMS) {
            if (locs->params_range >= 0)
               return false;
            locs->params_range = (int)count;
         }
      }

      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = start,
         .size_halfs = length,
         .address = source,
      };
   }

   if (cs->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &cs->b.info.rodata;
      const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
      if (!cs->bo || !cs->bo->va || (rodata->base_uniform & 1) ||
          (rodata->size_16 & 1) || count == AGX3_G17P_GRAPHICS_MAX_RANGES ||
          rodata->offset > cs->bo->size ||
          size_B > cs->bo->size - rodata->offset)
         return false;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = rodata->base_uniform,
         .size_halfs = rodata->size_16,
         .address = cs->bo->va->addr + rodata->offset,
      };
   }

   if (stage == MESA_SHADER_FRAGMENT && linked->g17p_emrt_heap_base) {
      if (!batch->g17p_emrt_heap || count == AGX3_G17P_GRAPHICS_MAX_RANGES)
         return false;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = linked->g17p_emrt_heap_base,
         .size_halfs = 4,
         .address = batch->g17p_emrt_heap,
      };
   }

   *count_out = count;
   return true;
}

static uint32_t
agx_build_g17p_graphics_program(struct agx_batch *batch,
                                struct agx_compiled_shader *cs,
                                struct agx_linked_shader *linked,
                                uint64_t *full_va)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   /* Only a fragment constant program the native protocol launches */
   const bool fs_cprog = cs->stage == MESA_SHADER_FRAGMENT &&
                         cs->b.info.has_preamble &&
                         agx_g17p_fs_cprog_qualified(&cs->b.info);
   if (!linked || !linked->bo || !linked->bo->va || !linked->size ||
       ((cs->b.info.has_preamble || cs->b.info.preamble_scratch_size) &&
        !fs_cprog))
      return 0;
   if (fs_cprog && (!cs->bo || !cs->bo->va ||
                    cs->b.info.preamble_offset > cs->b.info.binary_size ||
                    cs->b.info.preamble_size >
                       cs->b.info.binary_size - cs->b.info.preamble_offset ||
                    (cs->b.info.agx3_preamble_tps_inputs &&
                     cs->b.info.preamble_scratch_size <
                        cs->b.info.agx3_preamble_inputs.storage_bytes)))
      return 0;

   /* Indirect records change their vertex parameters, not their fragment
    * inputs. Keep the immutable fragment setup in this batch instead of
    * rebuilding and uploading it for every record. Be conservative about
    * all fixed-function and fragment state; shader changes are checked too.
    * The cache is cleared whenever the batch's executable pool is replaced. */
   const unsigned vertex_dirty =
      AGX_DIRTY_VS | AGX_DIRTY_VS_PROG | AGX_DIRTY_VERTEX;
   if (cs->stage == MESA_SHADER_FRAGMENT &&
       !(ctx->dirty & ~vertex_dirty) && batch->g17p_fragment_program &&
       batch->g17p_fragment_shader == cs &&
       batch->g17p_fragment_linked == linked) {
      if (full_va)
         *full_va = batch->g17p_fragment_program;
      return agx_usc_addr(dev, batch->g17p_fragment_program);
   }

   struct agx3_ioto_uniform_range
      ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t count = 0;
   if (!agx_g17p_collect_graphics_uniform_ranges(batch, cs, linked, ranges,
                                                 &count, NULL))
      return 0;

   if (!agx_va_in_usc_window(dev, linked->bo->va->addr + linked->entry_offset))
      return 0;

   const uint32_t pass_type =
      cs->stage == MESA_SHADER_FRAGMENT
         ? linked->fragment_control.opaque[0] >> 29
         : AGX_PASS_TYPE_OPAQUE;
   struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .program_va = 0,
      .shader_va = linked->bo->va->addr + linked->entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = count,
      .shader_binary_size = linked->size,
      .scratch_size =
         MAX2(linked->scratch_size, linked->agx3_spectpr_scratch_size),
      .nr_gprs = linked->nr_gprs,
      .compiler_d_valid = linked->g17p_compiler_d_valid,
      .compiler_d = linked->g17p_compiler_d,
      .fragment = cs->stage == MESA_SHADER_FRAGMENT,
      .writes_sample_mask = pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
                            pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,
      .colour_classes = linked->g17p_colour_classes,
      .imageblock = {
         .sample_size_B = batch->tilebuffer_layout.sample_size_B,
         .nr_samples = batch->tilebuffer_layout.nr_samples,
         .tile_size_px = batch->tilebuffer_layout.tile_size,
      },
   };

   struct agx_bo *program_bo = NULL;
   struct agx_ptr t;
   size_t program_length = 0;

   /* Without a constant program suffix, relocation does not depend on
    * program_va. Try the cached patch plan before computing a conservative size
    * bound so plan hits allocate the exact program size with no 6 KiB staging
    * buffer. */
   bool relocated = false;
   if (!fs_cprog && linked->g17p_graphics_templates) {
      const struct agx_g17p_graphics_template *plan =
         agx_g17p_graphics_program_find_plan(linked, &desc);
      if (plan) {
         const size_t len = plan->program.program.length;
         t = agx_pool_alloc_aligned_with_bo(
            &batch->pipeline_pool, len, AGX_IOTO_PROGRAM_ALIGNMENT,
            &program_bo);
         if (!t.cpu || !program_bo || !agx_va_in_usc_window(dev, t.gpu) ||
             !agx_g17p_graphics_template_emit_plan(plan, ranges, t.cpu, len,
                                                   &program_length))
            return 0;
         relocated = true;
      }
   }

   if (!relocated) {
      size_t capacity;
      if (!agx3_g17p_graphics_program_bound(dev->chip, ranges, count,
                                             &capacity))
         return 0;

      /* The constant program branches to its code in the shader's own upload;
       * the loader grows by that branch and its prelude only. */
      if (fs_cprog)
         capacity += 512;

      t = agx_pool_alloc_aligned_with_bo(
         &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT,
         &program_bo);
      if (!t.cpu || !program_bo || !agx_va_in_usc_window(dev, t.gpu))
         return 0;

      desc.program_va = t.gpu;
      if (fs_cprog) {
         /* As hk_upload_g17p_graphics_program(): the main shader's own upload
          * holds the constant program; linking relocates only the main part. */
         const struct agx_shader_info *pi = &cs->b.info;
         desc.constant_program = (struct agx_ioto_constant_program){
            .code = (const uint8_t *)agx_bo_map(cs->bo) + pi->preamble_offset,
            .code_size = pi->preamble_size,
            .code_va = cs->bo->va->addr + pi->preamble_offset,
            .absolute_branch = true,
            .request_prelude = pi->agx3_preamble_uniform_inputs,
         };
         if (pi->agx3_preamble_tps_inputs)
            desc.cprog_inputs = &pi->agx3_preamble_inputs;
      }

      if (!agx_g17p_graphics_program_cached(dev->chip, linked, &desc, t.cpu,
                                            capacity, &program_length))
         return 0;
   }

   agx_bo_assert_range(program_bo, t.gpu - program_bo->va->addr,
                       program_length);
   if (full_va)
      *full_va = t.gpu;
   if (cs->stage == MESA_SHADER_FRAGMENT) {
      batch->g17p_fragment_program = t.gpu;
      batch->g17p_fragment_shader = cs;
      batch->g17p_fragment_linked = linked;
   }
   return agx_usc_addr(dev, t.gpu);
}

static bool
agx_m5_graphics_program_cached(struct agx_device *dev,
                               struct agx_linked_shader *linked,
                               const struct agx3_g17p_graphics_program_desc *desc,
                               struct agx3_g17p_graphics_program *program);

/* agx3d templates made, for ASAHI_MESA_DEBUG=verify */
static uint32_t agx_g18g_templates_made;

/* ASAHI_MESA_DEBUG=verify: check a G18G program from a template against a
 * fresh composition and full re-encoding, as the M5 builder does. */
static void
agx_g18g_graphics_template_verify(const struct agx_linked_shader *linked,
                                  const struct agx3_g17p_graphics_program_desc *desc,
                                  const struct agx3_g17p_graphics_program *program)
{
   struct agx3_g17p_graphics_program_desc d = *desc;
   struct agx3_g17p_graphics_program fresh;
   const struct agx3_transcode_ioto ioto = {
      .api_program = true,
      .shader_gprs = linked->nr_gprs,
   };
   const char *why = NULL;
   unsigned at = 0;
   d.patchable = true;
   const bool same =
      (d.scratch_size ? agx3_g18g_graphics_program_generate_tps(&d, &fresh)
                      : agx3_g18g_graphics_program_generate(&d, &fresh)) &&
      agx3_reencode_program(&fresh, &ioto, &why, &at) &&
      fresh.length == program->length &&
      !memcmp(fresh.bytes, program->bytes, program->length);
   static uint32_t checked, differed;
   const uint32_t n = p_atomic_inc_return(&checked);
   if (!same && p_atomic_inc_return(&differed) <= 8)
      fprintf(stderr, "G18G_PROGRAM_CACHE differs from a fresh composition "
                      "(fragment %u, %zu ranges)\n", d.fragment,
              d.uniform_range_count);
   if (!(n & (n - 1)))
      fprintf(stderr, "G18G_PROGRAM_CACHE verified %u programs, %u differed, "
                      "%u templates made\n",
              n, p_atomic_read(&differed),
              p_atomic_read(&agx_g18g_templates_made));
}

/* G18G (j873): the M5-profile graphics program with agx3d re-encoding and the
 * G18 TPS-frame envelope. Every other native chip takes
 * agx_build_g17p_graphics_program() above. */
static uint32_t
agx_build_agx3_graphics_program(struct agx_batch *batch,
                                struct agx_compiled_shader *cs,
                                struct agx_linked_shader *linked,
                                uint64_t *full_va, const char **why)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   /* Every refusal below returns zero, and the caller can only say which
    * stage failed. Name the condition under the fault trace so a refused
    * program is one run away from a diagnosis. */
#define AGX3_PROGRAM_REFUSE(reason)                                            \
   do {                                                                        \
      if (dev->debug & AGX_DBG_FAULTTRACE) {                                   \
         fprintf(stderr, "M5_PROGRAM_REJECT stage=%u %s\n", cs->stage,         \
                 (reason));                                                    \
      }                                                                        \
      if (why) *why = (reason); \
      return 0;                                                                \
   } while (0)

   if (!linked || !linked->bo || !linked->bo->va || !linked->size)
      AGX3_PROGRAM_REFUSE("unlinked shader");
   struct agx_ioto_preshader_desc preamble;
   const struct agx_ioto_preshader_desc *preshader = NULL;
   if (agx_chip_native_preamble(dev->chip)) {
      if (!agx_bind_g17p_preamble(&cs->b.info, cs->b.binary, cs->bo, &preamble))
         AGX3_PROGRAM_REFUSE("the preamble cannot be bound");
      if (cs->b.info.has_preamble)
         preshader = &preamble;
   } else if (cs->b.info.has_preamble || cs->b.info.preamble_scratch_size) {
      AGX3_PROGRAM_REFUSE("preamble");
   }

   /* Indirect records change their vertex parameters, not their fragment
    * inputs. Keep the immutable fragment setup in this batch instead of
    * rebuilding and uploading it for every record. Be conservative about
    * all fixed-function and fragment state; shader changes are checked too.
    * The cache is cleared whenever the batch's executable pool is replaced. */
   const unsigned vertex_dirty =
      AGX_DIRTY_VS | AGX_DIRTY_VS_PROG | AGX_DIRTY_VERTEX;
   if (agx_chip_native_hal200(dev->chip) && cs->stage == MESA_SHADER_FRAGMENT &&
       !(ctx->dirty & ~vertex_dirty) && batch->g17p_fragment_program &&
       batch->g17p_fragment_shader == cs &&
       batch->g17p_fragment_linked == linked) {
      if (full_va)
         *full_va = batch->g17p_fragment_program;
      return agx_usc_addr(dev, batch->g17p_fragment_program);
   }

   struct agx3_ioto_uniform_range
      ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t count = 0;
   const mesa_shader_stage stage = cs->stage;
   const bool vertex_program = cs->b.info.stage == MESA_SHADER_VERTEX;
   /* G17G fragments reserve the same eight-half prefix for their texture
    * and sampler table pointers: Apple's own applegpu_g17g fragment with one
    * bindful texture moves every other uniform up by eight halves relative
    * to the texture-free control, and a two-texture fragment samples with
    * immediate indices 0 and 1 into that one prefix (gpu/
    * m5-texture-abi-reference.metal, artifacts/gpu/native/
    * m5-texture-abi-witness-20260913.txt). The resource pair below is that
    * prefix. */

   /* A hardware vertex program -- the vertex shader, a tessellation
    * evaluation shader run as one, or a geometry shader's copy program --
    * takes the pair exactly when it was compiled to read it. The copy
    * program samples nothing even when its geometry shader, whose texture
    * table this stage's count describes, does. */
   const bool resource_pair =
      batch->texture_count[stage] != 0 &&
      (stage == MESA_SHADER_FRAGMENT || cs->g17p_texture_resources);

   if (batch->texture_count[stage] && !resource_pair &&
       stage == MESA_SHADER_FRAGMENT)
      return 0;
   if (resource_pair) {
      if (!batch->stage_uniforms[stage].texture_base)
         AGX3_PROGRAM_REFUSE("no texture table is uploaded for the stage");
      if (!batch->samplers[stage])
         AGX3_PROGRAM_REFUSE("no sampler table is uploaded for the stage");
      if (batch->ctx->stage[stage].custom_borders &&
          !cs->g17p_custom_border_emulation)
         AGX3_PROGRAM_REFUSE("custom border colours are bound but the program "
                     "has no border emulation");
      const uint64_t pointers[2] = {
         batch->stage_uniforms[stage].texture_base,
         batch->samplers[stage],
      };
      const uint64_t pointer_va = agx_pool_upload_aligned(
         &batch->pool, pointers, sizeof(pointers), 8);
      if (!pointer_va)
         AGX3_PROGRAM_REFUSE("the texture/sampler pointer pair could not be uploaded");
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 8,
         .address = pointer_va,
      };
   }

   for (unsigned i = 0; i < cs->push_range_count; ++i) {
      const struct agx_push_range *push = &cs->push[i];
      const uint64_t table = batch->uniforms.tables[push->table];

      /* Fragment uniform zero is deliberately replaced by the resource pair.
       * A textured vertex/TES/copy program shifts its complete fixed ABI;
       * trimming a VBO or draw-parameter range would corrupt vertex fetches. */
      if (resource_pair && vertex_program &&
          push->uniform < AGX_ABI_G17P_TEXTURE_POINTER_HALFS)
         AGX3_PROGRAM_REFUSE("a textured vertex program pushes a uniform range "
                     "below the texture/sampler pointer pair");

      /* Draw parameters are intentionally absent for draws whose shader does
       * not consume them. This is the same omission as the legacy builder. */
      if (push->table == AGX_SYSVAL_TABLE_PARAMS && !table)
         continue;
      unsigned start = push->uniform;
      unsigned length = push->length;
      uint64_t source = table + push->offset;
      const bool blend_constants =
         resource_pair && stage == MESA_SHADER_FRAGMENT &&
         push->table == AGX_SYSVAL_TABLE_ROOT &&
         push->offset == offsetof(struct agx_draw_uniforms, blend_constant) &&
         push->uniform == AGX_ABI_FUNI_BLEND_R && push->length == 8;

      if (blend_constants) {
         uint16_t blend_base;
         /* As j873: the chip-independent G17P uniform-file rule. */
         if (!agx_g17p_fragment_blend_constant_base(AGX_CHIP_G17P, cs,
                                                    &blend_base))
            AGX3_PROGRAM_REFUSE("the fragment blend constants have no native base");
         start = blend_base;
      } else if (resource_pair && start < 8) {
         const unsigned end = start + length;
         if (end <= 8)
            continue;
         source += (8 - start) * 2;
         length = end - 8;
         start = 8;
      }

      /* Native Ioto moves uniform dwords. A single live 16-bit root or stage
       * sysval needs its adjacent half uploaded as padding. Both tables are
       * fixed driver allocations; prove the padding stays inside that table.
       * In particular, masked fragments consume the root no_epilog_discard
       * halfword even when no other root value is adjacent in the push range. */
      if ((length & 1) &&
          (push->table == AGX_SYSVAL_STAGE(stage) ||
           push->table == AGX_SYSVAL_TABLE_ROOT)) {
         const size_t table_size = push->table == AGX_SYSVAL_TABLE_ROOT
                                      ? sizeof(struct agx_draw_uniforms)
                                      : agx_stage_uniforms_size(dev->chip);
         if ((uint64_t)push->offset + ((uint64_t)length + 1) * 2 >
             table_size)
            AGX3_PROGRAM_REFUSE("a 16-bit sysval sits at the end of its "
                        "table and cannot be padded to a dword");
         ++length;
      }
      if (!table)
         AGX3_PROGRAM_REFUSE("a pushed sysval table is not uploaded");
      if ((start & 1) || (length & 1))
         AGX3_PROGRAM_REFUSE("a pushed uniform range is not dword aligned");
      if (count == ARRAY_SIZE(ranges))
         AGX3_PROGRAM_REFUSE("the program pushes more uniform ranges than the Ioto "
                     "composer accepts (AGX3_G17P_GRAPHICS_MAX_RANGES)");

      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = start,
         .size_halfs = length,
         .address = source,
      };
   }

   if (cs->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &cs->b.info.rodata;
      const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
      if (!cs->bo || !cs->bo->va || (rodata->base_uniform & 1) ||
          (rodata->size_16 & 1) || rodata->offset > cs->bo->size ||
          size_B > cs->bo->size - rodata->offset)
         AGX3_PROGRAM_REFUSE("the promoted constant range is malformed");
      if (count == ARRAY_SIZE(ranges))
         AGX3_PROGRAM_REFUSE("the promoted constants do not fit beside the pushed "
                     "uniform ranges (AGX3_G17P_GRAPHICS_MAX_RANGES)");
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = rodata->base_uniform,
         .size_halfs = rodata->size_16,
         .address = cs->bo->va->addr + rodata->offset,
      };
   }

   if (stage == MESA_SHADER_FRAGMENT && linked->g17p_emrt_heap_base) {
      if (!batch->g17p_emrt_heap || count == ARRAY_SIZE(ranges))
         return 0;
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = linked->g17p_emrt_heap_base,
         .size_halfs = 4,
         .address = batch->g17p_emrt_heap,
      };
   }

   const uint32_t pass_type =
      cs->stage == MESA_SHADER_FRAGMENT
         ? linked->fragment_control.opaque[0] >> 29
         : AGX_PASS_TYPE_OPAQUE;
   struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .shader_va = linked->bo->va->addr + linked->entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = count,
      .preshader = preshader,
      .shader_binary_size = linked->size,
      .scratch_size = agx_graphics_tps_frame(dev, linked),
      .nr_gprs = linked->nr_gprs,
      .compiler_d_valid = linked->g17p_compiler_d_valid,
      .compiler_d = linked->g17p_compiler_d,
      .fragment = cs->stage == MESA_SHADER_FRAGMENT,
      .writes_sample_mask = pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
                            pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,
      .colour_classes = linked->g17p_colour_classes,
      .imageblock = {
         .sample_size_B = batch->tilebuffer_layout.sample_size_B,
         .nr_samples = batch->tilebuffer_layout.nr_samples,
         .tile_size_px = batch->tilebuffer_layout.tile_size,
      },
   };

   /* G18 vertex/fragment TPS frames are admitted only when the kernel sizes
    * the shared TA/3D pool from this render's resource suffix, and only
    * inside the envelope that sizing is evidenced for. Without the feature
    * the draw keeps its original refusal. */
   const bool g18_tps = dev->chip == AGX_CHIP_G18G && desc.scratch_size;
   if (g18_tps) {
      if (!(dev->params.features & DRM_ASAHI_FEATURE_G18_GRAPHICS_SCRATCH))
         AGX3_PROGRAM_REFUSE("Ioto program generation");
      if (desc.imageblock.nr_samples != 1 ||
          desc.imageblock.tile_size_px != 32 * 32 ||
          !desc.imageblock.sample_size_B ||
          desc.imageblock.sample_size_B > 8 ||
          util_framebuffer_get_num_layers(&batch->key) != 1)
         AGX3_PROGRAM_REFUSE("G18 graphics scratch outside the single-sample, "
                             "single-layer, one-page tile envelope");
   }

   /* G18G: from the linked shader's template for this shape */
   struct agx3_g17p_graphics_program program;
   const bool cached = agx_reencodes_agx3d(dev) &&
                       agx_m5_graphics_program_cached(dev, linked, &desc,
                                                      &program);
   size_t capacity;
   if (cached) {
      capacity = program.length;
   } else {
      const bool bounded = agx_reencodes_agx3d(dev)
         ? agx3_g17g_graphics_program_bound(agx_ioto_chip(dev), ranges, count,
                                            &capacity)
         : agx3_g17p_graphics_program_bound_with_preshader(
              dev->chip, ranges, count, preshader, &capacity);
      if (!bounded)
         AGX3_PROGRAM_REFUSE("program size bound");
      capacity = agx_ioto_capacity(dev, capacity);
   }

   struct agx_bo *program_bo = NULL;
   struct agx_ptr t = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT,
      &program_bo);
   if (!t.cpu || !program_bo)
      AGX3_PROGRAM_REFUSE("the program could not be allocated in the pipeline pool");
   if (!agx_va_in_usc_window(dev, t.gpu))
      AGX3_PROGRAM_REFUSE("the program lies outside the USC window");
   if (!agx_va_in_usc_window(dev,
                             linked->bo->va->addr + linked->entry_offset))
      AGX3_PROGRAM_REFUSE("the shader binary lies outside the USC window");
   desc.program_va = t.gpu;

   size_t program_length;
   if (agx_chip_native_hal200(dev->chip)) {
      if (!agx_g17p_graphics_program_cached(dev->chip, linked, &desc, t.cpu,
                                            capacity,
                                            &program_length))
         AGX3_PROGRAM_REFUSE("Ioto program generation");
   } else {
      if (cached && (dev->debug & AGX_DBG_VERIFY))
         agx_g18g_graphics_template_verify(linked, &desc, &program);
      const bool generated = cached ||
         (agx_reencodes_agx3d(dev)
             ? (g18_tps ? agx3_g18g_graphics_program_generate_tps(&desc, &program)
                        : agx3_g17g_graphics_program_generate(dev->chip, &desc,
                                                              &program)) &&
                  agx_g18g_reencode_program(dev, &program, true, linked->nr_gprs)
             : agx3_g17p_graphics_program_generate(dev->chip, &desc, &program));
      if (!generated || program.length > capacity) {
         if (dev->debug & AGX_DBG_FAULTTRACE) {
            fprintf(stderr,
                    "M5_PROGRAM_DESC size=%u gprs=%u scratch=%u d=%u,%u "
                    "fragment=%u coverage=%u partitions=%u ranges=%zu "
                    "imageblock=%u,%u,%u length=%u capacity=%zu\n",
                    desc.shader_binary_size, desc.nr_gprs, desc.scratch_size,
                    desc.compiler_d_valid, desc.compiler_d, desc.fragment,
                    desc.writes_sample_mask, desc.colour_classes, count,
                    desc.imageblock.sample_size_B, desc.imageblock.nr_samples,
                    desc.imageblock.tile_size_px,
                    generated ? program.length : 0u, capacity);
         }
         AGX3_PROGRAM_REFUSE("Ioto program generation");
      }
      memcpy(t.cpu, program.bytes, program.length);
      program_length = program.length;
      if (dev->chip == AGX_CHIP_G18G) {
         agx_g18_note_program(batch, desc.fragment, linked->nr_gprs,
                              desc.scratch_size, desc.compiler_d / 2);
      }
   }

   if (agx_bo_sync_for_device(program_bo,
                               t.gpu - program_bo->va->addr,
                               program_length))
      AGX3_PROGRAM_REFUSE("program upload");
   if (full_va)
      *full_va = t.gpu;
   if (agx_chip_native_hal200(dev->chip) && cs->stage == MESA_SHADER_FRAGMENT) {
      batch->g17p_fragment_program = t.gpu;
      batch->g17p_fragment_shader = cs;
      batch->g17p_fragment_linked = linked;
   }
   return agx_usc_addr(dev, t.gpu);
}
#undef AGX3_PROGRAM_REFUSE


/* M5-family native program cache. Composing a G17G Ioto program costs tens of
 * microseconds of CPU, and Gallium needs one per stage per draw because the
 * uniform tables it names are uploaded per draw: on the M5 Pro that composer
 * took 86% of Minecraft's GL CPU time (~1,800 draws per frame). Honeykrisp
 * keeps the composed bytes per linked shader and patches only the range
 * addresses (hk_g17p_gfx_loader_create()); do the same. The template is
 * composed in the address-independent form (every range loads its own base
 * with full-width LDIMMs), so one template fits every address the draw's
 * tables get, and only the shape (imageblock, range starts and sizes) keys it.
 * The bytes don't depend on the program's own address (no constant program),
 * which Honeykrisp relies on as well; ASAHI_MESA_DEBUG=verify checks every
 * patched program against a fresh composition.
 *
 * G18G re-encodes every program as agx3d, which on the M6 cost as much again
 * as composing it: a Chromium Google Maps drag spent 86% of its GPU process's
 * CPU time building programs. Its templates also keep the agx3d form
 * (agx3_reencode_template), where only the LDIMMs of the range bases change:
 * the plan's relocations, each a low and a high half. */
struct agx_m5_graphics_template {
   struct agx_m5_graphics_template *next;
   struct agx_ioto_g17p_imageblock imageblock;
   uint16_t range_start[HK_G17P_PATCH_MAX_RANGES];
   uint16_t range_size[HK_G17P_PATCH_MAX_RANGES];
   uint8_t range_count;
   uint16_t length;
   struct hk_g17p_patch_plan plan;
   /* G18G: the agx3d form of bytes, NULL if none could be made: the shape
    * is then composed in full without trying again. */
   struct agx3_reencode_template *agx3d;
   /* When agx3d is kept, replaced or given up (agx_g18g_template_policy.h) */
   struct agx_g18g_template_state agx3d_state;
   uint8_t bytes[];
};

#define AGX_M5_GRAPHICS_TEMPLATE_LIMIT 8u

static simple_mtx_t agx_m5_template_lock = SIMPLE_MTX_INITIALIZER;

static bool
agx_m5_graphics_template_matches(const struct agx_m5_graphics_template *t,
                                 const struct agx3_g17p_graphics_program_desc *d)
{
   if (t->range_count != d->uniform_range_count ||
       t->imageblock.sample_size_B != d->imageblock.sample_size_B ||
       t->imageblock.nr_samples != d->imageblock.nr_samples ||
       t->imageblock.tile_size_px != d->imageblock.tile_size_px)
      return false;
   for (unsigned i = 0; i < t->range_count; ++i) {
      if (t->range_start[i] != d->uniform_ranges[i].start_halfs ||
          t->range_size[i] != d->uniform_ranges[i].size_halfs)
         return false;
   }
   return hk_g17p_patch_plan_fits(&t->plan, d->uniform_ranges);
}

/* False if a G18G LDIMM takes another size for desc's addresses */
static bool
agx_m5_graphics_template_emit(const struct agx_device *dev,
                              const struct agx_m5_graphics_template *t,
                              const struct agx3_g17p_graphics_program_desc *desc,
                              struct agx3_g17p_graphics_program *program)
{
   if (agx_reencodes_agx3d(dev)) {
      uint32_t immediates[2 * HK_G17P_PATCH_MAX_RELOCS];
      hk_g17p_patch_plan_ldimms(&t->plan, desc->uniform_ranges, NULL,
                                immediates);
      return agx3_reencode_template_emit(t->agx3d, immediates, program);
   }
   program->length = t->length;
   memcpy(program->bytes, t->bytes, t->length);
   hk_g17p_patch_plan_apply(&t->plan, desc->uniform_ranges, program->bytes);
   return true;
}

/* G18G: make t's agx3d form from its agx3c template, if it can be */
static void
agx_g18g_graphics_template_reencode(struct agx_m5_graphics_template *t,
                                    const struct agx_linked_shader *linked,
                                    const struct agx3_g17p_graphics_program *agx3c)
{
   const struct agx3_transcode_ioto ioto = {
      .api_program = true,
      .shader_gprs = linked->nr_gprs,
   };
   unsigned offsets[2 * HK_G17P_PATCH_MAX_RELOCS];
   const unsigned count =
      hk_g17p_patch_plan_ldimms(&t->plan, NULL, offsets, NULL);
   const char *why = NULL;
   unsigned at = 0;
   t->agx3d = agx3_reencode_template_create(t, agx3c, &ioto, offsets, count,
                                            &why, &at);
}

/* Write the program for desc (whose program_va isn't needed) from the linked
 * shader's template for its shape, composing the template on first use.
 * False when the shape can't be patched: the caller then composes in full.
 * Everything in the descriptor but the shape is the linked shader's own. */
static bool
agx_m5_graphics_program_cached(struct agx_device *dev,
                               struct agx_linked_shader *linked,
                               const struct agx3_g17p_graphics_program_desc *desc,
                               struct agx3_g17p_graphics_program *program)
{
   const bool g18g = agx_reencodes_agx3d(dev);
   if ((!agx_chip_g17_m5(dev->chip) && !g18g) ||
       desc->uniform_range_count > HK_G17P_PATCH_MAX_RANGES)
      return false;

   simple_mtx_lock(&agx_m5_template_lock);
   struct agx_m5_graphics_template **slot = &linked->m5_graphics_templates;
   unsigned count = 0;
   struct agx_g18g_template_state replaced = {0};
   while (*slot) {
      struct agx_m5_graphics_template *t = *slot;
      if (agx_m5_graphics_template_matches(t, desc)) {
         *slot = t->next;
         /* A G18G shape without an agx3d form is composed in full */
         const bool unpatchable = g18g && !t->agx3d;
         const bool emitted =
            !unpatchable && agx_m5_graphics_template_emit(dev, t, desc, program);
         /* A draw the agx3d form refused (an LDIMM takes another size for
          * these addresses) is composed in full; the policy decides when the
          * template is replaced or given up. */
         if (g18g && t->agx3d) {
            switch (agx_g18g_template_after_draw(&t->agx3d_state, emitted)) {
            case AGX_G18G_TEMPLATE_KEEP:
               break;
            case AGX_G18G_TEMPLATE_REPLACE:
               replaced = agx_g18g_template_replacement(&t->agx3d_state);
               ralloc_free(t);
               t = NULL;
               break;
            case AGX_G18G_TEMPLATE_DROP:
               ralloc_free(t->agx3d);
               t->agx3d = NULL;
               break;
            }
            if (!t)
               break;
         }
         t->next = linked->m5_graphics_templates;
         linked->m5_graphics_templates = t;
         simple_mtx_unlock(&agx_m5_template_lock);
         return emitted;
      }
      slot = &t->next;
      ++count;
   }

   struct agx_m5_graphics_template *created = NULL;
   struct agx3_g17p_graphics_program_template *composed =
      malloc(sizeof(*composed));
   struct agx3_g17p_graphics_program_desc d = *desc;
   d.program_va = dev->shader_base + 0x10000;
   d.patchable = true;
   if (composed &&
       (g18g ? agx3_g18g_graphics_program_template_generate(&d, composed)
             : agx3_g17g_graphics_program_template_generate(dev->chip, &d,
                                                            composed))) {
      created = rzalloc_size(linked, sizeof(*created) +
                                        composed->program.length);
      if (created &&
          hk_g17p_patch_plan_init(&created->plan, d.uniform_ranges,
                                  d.uniform_range_count, composed->relocations,
                                  composed->relocation_count,
                                  composed->program.length, true)) {
         if (g18g) {
            agx_g18g_graphics_template_reencode(created, linked,
                                                &composed->program);
            p_atomic_inc(&agx_g18g_templates_made);
         }
         created->agx3d_state = replaced;
         created->imageblock = d.imageblock;
         created->range_count = d.uniform_range_count;
         for (unsigned i = 0; i < created->range_count; ++i) {
            created->range_start[i] = d.uniform_ranges[i].start_halfs;
            created->range_size[i] = d.uniform_ranges[i].size_halfs;
         }
         created->length = composed->program.length;
         memcpy(created->bytes, composed->program.bytes, created->length);

         /* Bound memory and lookup cost: the coldest entry is evicted. */
         if (count == AGX_M5_GRAPHICS_TEMPLATE_LIMIT) {
            slot = &linked->m5_graphics_templates;
            while ((*slot)->next)
               slot = &(*slot)->next;
            ralloc_free(*slot);
            *slot = NULL;
         }
         created->next = linked->m5_graphics_templates;
         linked->m5_graphics_templates = created;
         /* The template has desc's addresses: it emits as made */
         if (g18g && !created->agx3d)
            created = NULL;
         else
            agx_m5_graphics_template_emit(dev, created, desc, program);
      } else {
         ralloc_free(created);
         created = NULL;
      }
   }
   free(composed);
   simple_mtx_unlock(&agx_m5_template_lock);
   return created != NULL;
}

static uint32_t
agx_build_native_g17g_graphics_program(struct agx_batch *batch,
                                struct agx_compiled_shader *cs,
                                struct agx_linked_shader *linked,
                                uint64_t *full_va)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   /* Every refusal below returns zero, and the caller can only say which
    * stage failed. Name the condition under the fault trace so a refused
    * program is one run away from a diagnosis. */
#define AGX3_PROGRAM_REFUSE(reason)                                            \
   do {                                                                        \
      if (dev->debug & AGX_DBG_FAULTTRACE) {                                   \
         fprintf(stderr, "M5_PROGRAM_REJECT stage=%u %s\n", cs->stage,         \
                 (reason));                                                    \
      }                                                                        \
      return 0;                                                                \
   } while (0)

   if (!linked || !linked->bo || !linked->bo->va || !linked->size)
      AGX3_PROGRAM_REFUSE("unlinked shader");
   if (cs->b.info.has_preamble || cs->b.info.preamble_scratch_size)
      AGX3_PROGRAM_REFUSE("preamble");

   struct agx3_ioto_uniform_range
      ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t count = 0;
   const mesa_shader_stage stage = cs->stage;
   /* A tessellation evaluation shader run as the vertex shader takes the
    * pair like the vertex shader. */
   const bool vertex_stage =
      stage == MESA_SHADER_VERTEX || stage == MESA_SHADER_TESS_EVAL;
   const bool resource_pair =
      batch->texture_count[stage] != 0 &&
      (stage == MESA_SHADER_FRAGMENT ||
       (vertex_stage && cs->g17p_texture_resources));

   /* G17G fragments reserve the same eight-half prefix for their texture
    * and sampler table pointers: Apple's own applegpu_g17g fragment with one
    * bindful texture moves every other uniform up by eight halves relative
    * to the texture-free control, and a two-texture fragment samples with
    * immediate indices 0 and 1 into that one prefix (gpu/
    * m5-texture-abi-reference.metal, artifacts/gpu/native/
    * m5-texture-abi-witness-20260913.txt). The resource pair below is that
    * prefix. */

   if (batch->texture_count[stage] && !resource_pair)
      AGX3_PROGRAM_REFUSE("textures without the resource prefix");
   if (resource_pair) {
      if (!batch->stage_uniforms[stage].texture_base ||
          !batch->samplers[stage] ||
          (batch->ctx->stage[stage].custom_borders &&
           !cs->g17p_custom_border_emulation))
         AGX3_PROGRAM_REFUSE("texture or sampler table");
      const uint64_t pointers[2] = {
         batch->stage_uniforms[stage].texture_base,
         batch->samplers[stage],
      };
      const uint64_t pointer_va = agx_pool_upload_aligned(
         &batch->pool, pointers, sizeof(pointers), 8);
      if (!pointer_va)
         AGX3_PROGRAM_REFUSE("resource prefix upload");
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = 0,
         .size_halfs = 8,
         .address = pointer_va,
      };
   }

   for (unsigned i = 0; i < cs->push_range_count; ++i) {
      const struct agx_push_range *push = &cs->push[i];
      const uint64_t table = batch->uniforms.tables[push->table];

      /* Fragment uniform zero is deliberately replaced by the resource pair.
       * A textured VS instead shifts its complete fixed ABI; trimming a VBO
       * or draw-parameter range here would corrupt vertex fetches. */
      if (resource_pair && vertex_stage &&
          push->uniform < AGX_ABI_G17P_TEXTURE_POINTER_HALFS)
         AGX3_PROGRAM_REFUSE("textured vertex ABI shift");

      /* Draw parameters are intentionally absent for draws whose shader does
       * not consume them. This is the same omission as the legacy builder. */
      if (push->table == AGX_SYSVAL_TABLE_PARAMS && !table)
         continue;
      unsigned start = push->uniform;
      unsigned length = push->length;
      uint64_t source = table + push->offset;
      const bool blend_constants =
         resource_pair && stage == MESA_SHADER_FRAGMENT &&
         push->table == AGX_SYSVAL_TABLE_ROOT &&
         push->offset == offsetof(struct agx_draw_uniforms, blend_constant) &&
         push->uniform == AGX_ABI_FUNI_BLEND_R && push->length == 8;

      if (blend_constants) {
         uint16_t blend_base;
         if (!agx_g17p_fragment_blend_constant_base(dev->chip, cs,
                                                    &blend_base))
            AGX3_PROGRAM_REFUSE("blend constant base");
         start = blend_base;
      } else if (resource_pair && start < 8) {
         const unsigned end = start + length;
         if (end <= 8)
            continue;
         source += (8 - start) * 2;
         length = end - 8;
         start = 8;
      }

      /* Native Ioto moves uniform dwords. A single live 16-bit stage sysval
       * (for example the custom-border companion sentinel) therefore needs
       * its adjacent half uploaded as harmless padding. The stage table is a
       * fixed driver allocation, so prove that adjacent half is in bounds.
       * The root table likewise: a discard reads no_epilog_discard alone. */
      if ((length & 1) && (push->table == AGX_SYSVAL_STAGE(stage) ||
                           push->table == AGX_SYSVAL_TABLE_ROOT)) {
         const size_t table_size = push->table == AGX_SYSVAL_TABLE_ROOT
                                      ? sizeof(struct agx_draw_uniforms)
                                      : agx_stage_uniforms_size(dev->chip);
         if ((uint64_t)push->offset + ((uint64_t)length + 1) * 2 > table_size)
            AGX3_PROGRAM_REFUSE("odd stage uniform at the table end");
         ++length;
      }
      if (!table || (start & 1) || (length & 1) ||
          count == ARRAY_SIZE(ranges))
         AGX3_PROGRAM_REFUSE("push range shape");

      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = start,
         .size_halfs = length,
         .address = source,
      };
   }

   if (cs->b.info.rodata.size_16) {
      const struct agx_rodata *rodata = &cs->b.info.rodata;
      const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
      if (!cs->bo || !cs->bo->va || (rodata->base_uniform & 1) ||
          (rodata->size_16 & 1) || count == ARRAY_SIZE(ranges) ||
          rodata->offset > cs->bo->size ||
          size_B > cs->bo->size - rodata->offset)
         AGX3_PROGRAM_REFUSE("rodata range");
      ranges[count++] = (struct agx3_ioto_uniform_range){
         .start_halfs = rodata->base_uniform,
         .size_halfs = rodata->size_16,
         .address = cs->bo->va->addr + rodata->offset,
      };
   }

   const uint32_t pass_type =
      cs->stage == MESA_SHADER_FRAGMENT
         ? linked->fragment_control.opaque[0] >> 29
         : AGX_PASS_TYPE_OPAQUE;
   struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .shader_va = linked->bo->va->addr + linked->entry_offset,
      .uniform_ranges = ranges,
      .uniform_range_count = count,
      .shader_binary_size = linked->size,
      .scratch_size = linked->scratch_size,
      .nr_gprs = linked->nr_gprs,
      .compiler_d_valid = linked->g17p_compiler_d_valid,
      .compiler_d = linked->g17p_compiler_d,
      .fragment = cs->stage == MESA_SHADER_FRAGMENT,
      .writes_sample_mask = pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
                            pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,
      .colour_classes = linked->g17p_colour_classes,
      .imageblock = {
         .sample_size_B = batch->tilebuffer_layout.sample_size_B,
         .nr_samples = batch->tilebuffer_layout.nr_samples,
         .tile_size_px = batch->tilebuffer_layout.tile_size,
      },
   };

   struct agx3_g17p_graphics_program program;
   const bool cached =
      agx_m5_graphics_program_cached(dev, linked, &desc, &program);
   size_t capacity;
   if (cached) {
      capacity = program.length;
   } else {
      const bool bounded = agx_chip_g17_m5(dev->chip)
         ? agx3_g17g_graphics_program_bound(dev->chip, ranges, count, &capacity)
         : agx3_g17p_graphics_program_bound(dev->chip, ranges, count, &capacity);
      if (!bounded)
         AGX3_PROGRAM_REFUSE("program size bound");
   }

   struct agx_bo *program_bo = NULL;
   struct agx_ptr t = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT,
      &program_bo);
   if (!t.cpu || !program_bo || !agx_va_in_usc_window(dev, t.gpu) ||
       !agx_va_in_usc_window(dev,
                            linked->bo->va->addr + linked->entry_offset))
      AGX3_PROGRAM_REFUSE("program allocation");
   desc.program_va = t.gpu;

   if (cached && (dev->debug & AGX_DBG_VERIFY)) {
      struct agx3_g17p_graphics_program fresh;
      desc.patchable = true;
      const bool same =
         agx3_g17g_graphics_program_generate(dev->chip, &desc, &fresh) &&
         fresh.length == program.length &&
         !memcmp(fresh.bytes, program.bytes, program.length);
      desc.patchable = false;
      static uint32_t checked, differed;
      const uint32_t c = p_atomic_inc_return(&checked);
      if (!same && p_atomic_fetch_add(&differed, 1) < 8)
         fprintf(stderr, "M5_PROGRAM_CACHE differs from a fresh composition "
                         "(stage %u, %zu ranges)\n", stage, count);
      if (!(c & (c - 1)))
         fprintf(stderr, "M5_PROGRAM_CACHE verified %u programs, %u differed\n",
                 c, p_atomic_read(&differed));
   }
   const bool generated = cached ||
      (agx_chip_g17_m5(dev->chip)
          ? agx3_g17g_graphics_program_generate(dev->chip, &desc, &program)
          : agx3_g17p_graphics_program_generate(dev->chip, &desc, &program));
   if (!generated || program.length > capacity) {
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         fprintf(stderr,
                 "M5_PROGRAM_DESC size=%u gprs=%u scratch=%u d=%u,%u "
                 "fragment=%u coverage=%u partitions=%u ranges=%zu "
                 "imageblock=%u,%u,%u length=%u capacity=%zu\n",
                 desc.shader_binary_size, desc.nr_gprs, desc.scratch_size,
                 desc.compiler_d_valid, desc.compiler_d, desc.fragment,
                 desc.writes_sample_mask, desc.colour_classes, count,
                 desc.imageblock.sample_size_B, desc.imageblock.nr_samples,
                 desc.imageblock.tile_size_px,
                 generated ? program.length : 0u, capacity);
      }
      AGX3_PROGRAM_REFUSE("Ioto program generation");
   }

   memcpy(t.cpu, program.bytes, program.length);
   if (agx_bo_sync_for_device(program_bo,
                               t.gpu - program_bo->va->addr,
                               program.length))
      AGX3_PROGRAM_REFUSE("program upload");
   if (agx_m5_render_resources(dev)) {
      /* The kernel sizes the render's pools from this (hk_cmd_draw.c,
       * hk_m5_note_render_resources): primary storage is the larger of the
       * spill frame and the SPECTPR frame; a vertex program's secondary
       * storage is twice its staging extent; a fragment program carries the
       * render's imageblock. */
      const bool fragment = cs->stage == MESA_SHADER_FRAGMENT;
      const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;
      const uint32_t samples = MAX2(tib->nr_samples, 1);
      const uint32_t sample_bytes = tib->tile_size * tib->sample_size_B;
      const uint32_t staging_B = linked->g17p_compiler_d_valid
                                    ? linked->g17p_compiler_d
                                    : linked->nr_gprs;
      const struct drm_asahi_render_program_v2 entry = {
         .legacy = {
            .stage = fragment,
            .primary_bytes =
               MAX2(linked->scratch_size, linked->agx3_spectpr_scratch_size),
            .secondary_bytes = fragment ? 0 : staging_B * 2,
            .imageblock_bytes = fragment ? sample_bytes * samples : 0,
            .multisample_bytes = !fragment     ? 0
                                 : samples == 2 ? sample_bytes / 4
                                 : samples == 4 ? sample_bytes
                                 : samples == 8 ? 4 * sample_bytes
                                                : 0,
         },
      };
      agx_batch_note_m5_render_program(batch, &entry);
      batch->m5_shader_writes |= cs->so && cs->so->info.writes_memory;
   }

   if (full_va)
      *full_va = t.gpu;
   return agx_usc_addr(dev, t.gpu);
}
#undef AGX3_PROGRAM_REFUSE

static uint32_t
agx_build_pipeline(struct agx_batch *batch, struct agx_compiled_shader *cs,
                   struct agx_linked_shader *linked,
                   mesa_shader_stage phys_stage, unsigned variable_shared_mem,
                   size_t max_subgroups, uint64_t *full_va, const char **why)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (dev->chip == AGX_CHIP_G18G && phys_stage != MESA_SHADER_COMPUTE)
      return agx_build_agx3_graphics_program(batch, cs, linked, full_va, why);
   if (agx_chip_g17_m5(dev->chip) && phys_stage != MESA_SHADER_COMPUTE)
      return agx_build_native_g17g_graphics_program(batch, cs, linked, full_va);
   if (agx_chip_native_agx3(dev->chip) && phys_stage != MESA_SHADER_COMPUTE)
      return agx_build_g17p_graphics_program(batch, cs, linked, full_va);
   unsigned constant_push_ranges = DIV_ROUND_UP(cs->b.info.rodata.size_16, 64);

   size_t usc_size =
      agx_usc_size(constant_push_ranges + cs->push_range_count + 2);

   struct agx_ptr t =
      agx_pool_alloc_aligned(&batch->pipeline_pool, usc_size, 64);

   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   mesa_shader_stage stage = cs->stage;

   if (batch->texture_count[stage]) {
      agx_usc_pack(&b, TEXTURE, cfg) {
         cfg.start = 0;
         cfg.count =
            MIN2(batch->texture_count[stage], AGX_NUM_TEXTURE_STATE_REGS);
         cfg.buffer = batch->stage_uniforms[stage].texture_base;
      }
   }

   if (batch->sampler_count[stage]) {
      agx_usc_pack(&b, SAMPLER, cfg) {
         cfg.start = 0;
         cfg.count = batch->sampler_count[stage];
         cfg.buffer = batch->samplers[stage];
      }
   }

   for (unsigned i = 0; i < cs->push_range_count; ++i) {
      unsigned table = cs->push[i].table;
      uint64_t table_ptr = batch->uniforms.tables[table];

      /* Params may be omitted if the VS prolog does not read them, but the
       * reservation is always there in the API shader just in case.
       */
      if (table == AGX_SYSVAL_TABLE_PARAMS && !table_ptr)
         continue;

      assert(table_ptr);

      agx_usc_uniform(&b, cs->push[i].uniform, cs->push[i].length,
                      table_ptr + cs->push[i].offset);
   }

   if (cs->bo) {
      /* HAL200 reconstructs executable-window uniform pointers relative to
       * USC_EXEC_BASE, just like shader pointers.  Passing the full G17P VA
       * to the legacy 40-bit USC Uniform packer either trips its bounds check
       * or relies on accidental truncation in release builds. */
      const uint64_t rodata_base =
         agx_chip_native_agx3(dev->chip) ? agx_usc_addr(dev, cs->bo->va->addr)
                                    : cs->bo->va->addr;
      agx_usc_immediates(&b, &cs->b.info.rodata, rodata_base);
   }

   uint32_t max_scratch_size =
      MAX2(cs->b.info.scratch_size, cs->b.info.preamble_scratch_size);

   if (max_scratch_size > 0) {
      unsigned preamble_size = (cs->b.info.preamble_scratch_size > 0) ? 1 : 0;

      switch (phys_stage) {
      case MESA_SHADER_FRAGMENT:
         agx_scratch_alloc(&ctx->scratch_fs, max_scratch_size, max_subgroups);
         batch->fs_scratch = true;
         batch->fs_preamble_scratch =
            MAX2(batch->fs_preamble_scratch, preamble_size);
         break;
      case MESA_SHADER_VERTEX:
         agx_scratch_alloc(&ctx->scratch_vs, max_scratch_size, max_subgroups);
         batch->vs_scratch = true;
         batch->vs_preamble_scratch =
            MAX2(batch->vs_preamble_scratch, preamble_size);
         break;
      default:
         agx_scratch_alloc(&ctx->scratch_cs, max_scratch_size, max_subgroups);
         batch->cs_scratch = true;
         batch->cs_preamble_scratch =
            MAX2(batch->cs_preamble_scratch, preamble_size);
         break;
      }
   }

   if (stage == MESA_SHADER_FRAGMENT) {
      agx_usc_push_packed(&b, SHARED, &batch->tilebuffer_layout.usc);
   } else {
      agx_usc_shared_non_fragment(&b, &cs->b.info, variable_shared_mem);
   }

   if (linked) {
      agx_usc_push_packed(&b, SHADER, linked->shader);
      agx_usc_push_packed(&b, REGISTERS, linked->regs);

      if (stage == MESA_SHADER_FRAGMENT)
         agx_usc_push_packed(&b, FRAGMENT_PROPERTIES, linked->fragment_props);
   } else {
      agx_usc_pack(&b, SHADER, cfg) {
         cfg.code =
            agx_usc_addr(dev, cs->bo->va->addr + cs->b.info.main_offset);
         cfg.unk_2 = 3;
      }

      agx_usc_pack(&b, REGISTERS, cfg) {
         cfg.register_count = cs->b.info.nr_gprs;
         cfg.spill_size = cs->b.info.scratch_size
                             ? agx_scratch_get_bucket(cs->b.info.scratch_size)
                             : 0;
      }
   }

   if (cs->b.info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(dev, cs->bo->va->addr + cs->b.info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   if (full_va)
      *full_va = t.gpu;

   return agx_usc_addr(dev, t.gpu);
}

static bool
agx_g17p_compute_failed(const char *step)
{
   mesa_loge("asahi: G17P compute dispatch failed at: %s", step);
   return false;
}

/* Record every native dispatch; a tessellated draw can exceed the number of
 * launches in a geometry prepass. Allocation failure rejects the batch via
 * the caller, so an unrecorded command never reaches the hardware. */
static bool
agx_record_cdm_dispatch(struct agx_batch *batch, uint32_t start, uint32_t end,
                        struct agx_grid grid)
{
   struct agx_cdm_dispatch *dispatch =
      util_dynarray_grow(&batch->cdm_dispatch, struct agx_cdm_dispatch, 1);
   if (!dispatch)
      return false;

   *dispatch = (struct agx_cdm_dispatch){
      .start = start, .end = end, .trace_mode = grid.mode,
      .trace_indirect = agx_is_indirect(grid) ? grid.ptr : 0,
   };
   return true;
}

/*
 * A fast-linked program (a software vertex shader behind the prolog that
 * fetches its attributes and publishes the vertex and instance IDs from the
 * invocation index) already lives in its own AGX3 code container, so it is
 * dispatched from there; a bare main program is wrapped in a container here.
 */
#define AGX_G17P_COMPUTE_TEMPLATE_LIMIT 8u
#define AGX_G17P_COMPUTE_TEMPLATE_MAX_SHAPES 2u

struct agx_g17p_compute_template {
   struct agx_g17p_compute_template *next;
   uint64_t shader_va;
   uint64_t constant_program_va;
   const struct agx3_cprog_inputs *cprog_inputs;
   struct agx3_cdm_direct_template_sources launch_sources;
   uint32_t shader_binary_size;
   uint32_t scratch_size;
   uint32_t compiler_d;
   uint32_t threadgroup_memory_size_B;
   uint32_t imageblock_stride_B;
   uint32_t constant_program_size;
   uint16_t nr_gprs;
   uint16_t length;
   bool compiler_d_valid;
   bool constant_program_prelude;
   bool independent;
   uint8_t ldshdr_f1;
   uint8_t range_count;
   uint16_t range_start[HK_G17P_PATCH_MAX_RANGES];
   uint16_t range_size[HK_G17P_PATCH_MAX_RANGES];
   struct agx3_compute_program launch;
   struct hk_g17p_patch_plan plan;
   uint8_t bytes[];
};

void
agx_g17p_compute_templates_free(struct agx_g17p_compute_template **head)
{
   struct agx_g17p_compute_template *cur = *head;
   *head = NULL;
   while (cur) {
      struct agx_g17p_compute_template *next = cur->next;
      free(cur);
      cur = next;
   }
}

static inline bool
agx_g17p_compute_template_matches(
   const struct agx_g17p_compute_template *t,
   const struct agx3_g17p_compute_program_desc *d)
{
   if (t->shader_va != d->shader_va || t->ldshdr_f1 != d->ldshdr_f1 ||
       t->range_count != d->uniform_range_count ||
       t->shader_binary_size != d->shader_binary_size ||
       t->scratch_size != d->scratch_size || t->nr_gprs != d->nr_gprs ||
       t->compiler_d_valid != d->compiler_d_valid ||
       t->compiler_d != d->compiler_d ||
       t->threadgroup_memory_size_B != d->threadgroup_memory_size_B ||
       t->imageblock_stride_B != d->imageblock_stride_B ||
       t->launch_sources.ti_spill_bytes != d->launch_sources.ti_spill_bytes ||
       t->launch_sources.ti_largest_ipr_bytes !=
          d->launch_sources.ti_largest_ipr_bytes ||
       t->launch_sources.ray_traversal_depth_nonzero !=
          d->launch_sources.ray_traversal_depth_nonzero ||
       t->constant_program_va != d->constant_program.code_va ||
       t->constant_program_size != d->constant_program.code_size ||
       t->constant_program_prelude != d->constant_program.request_prelude ||
       t->cprog_inputs != d->cprog_inputs)
      return false;

   for (unsigned i = 0; i < t->range_count; ++i) {
      if (t->range_start[i] != d->uniform_ranges[i].start_halfs ||
          t->range_size[i] != d->uniform_ranges[i].size_halfs)
         return false;
   }
   return true;
}

static struct agx_g17p_compute_template *
agx_g17p_compute_template_get(struct agx_device *dev,
                              struct agx_g17p_compute_template **head,
                              unsigned *generated_count,
                              const struct agx3_g17p_compute_program_desc *desc)
{
   if (!head || desc->preshader ||
       desc->uniform_range_count > HK_G17P_PATCH_MAX_RANGES ||
       (desc->constant_program.code_size &&
        (!desc->constant_program.force_external ||
         !desc->constant_program.absolute_branch)) ||
       !agx_g17p_graphics_template_ranges_valid(desc->uniform_ranges,
                                                desc->uniform_range_count))
      return NULL;

   unsigned count = 0;
   unsigned shapes = 0;
   struct agx_g17p_compute_template **slot = head;
   while (*slot) {
      struct agx_g17p_compute_template *entry = *slot;
      if (agx_g17p_compute_template_matches(entry, desc)) {
         if (hk_g17p_patch_plan_fits(&entry->plan, desc->uniform_ranges)) {
            if (*slot != *head) {
               *slot = entry->next;
               entry->next = *head;
               *head = entry;
            }
            return entry;
         }
         ++shapes;
      }
      slot = &entry->next;
      ++count;
   }

   const uint64_t dummy_va = dev->shader_base + 0x10000;
   if (!agx_va_in_usc_window(dev, dummy_va) ||
       (dummy_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)) ||
       dummy_va >= (UINT64_C(1) << 48))
      return NULL;

   struct agx_ioto_load_relocation rels[HK_G17P_PATCH_MAX_RELOCS];
   size_t rel_count = 0;
   const bool independent = shapes >= AGX_G17P_COMPUTE_TEMPLATE_MAX_SHAPES;
   struct agx3_g17p_compute_program_desc composed = *desc;
   composed.program_va = dummy_va;
   composed.relocations = rels;
   composed.relocation_capacity = ARRAY_SIZE(rels);
   composed.relocation_count = &rel_count;
   composed.patchable = independent;

   struct agx3_g17p_compute_program *generated = malloc(sizeof(*generated));
   if (!generated)
      return NULL;

   struct agx_g17p_compute_template *tpl = NULL;
   if (!agx3_g17p_compute_program_generate(dev->chip, &composed, generated))
      goto out;

   tpl = calloc(1, sizeof(*tpl) + generated->length);
   if (!tpl)
      goto out;

   *tpl = (struct agx_g17p_compute_template){
      .shader_va = desc->shader_va,
      .constant_program_va = desc->constant_program.code_va,
      .cprog_inputs = desc->cprog_inputs,
      .launch_sources = desc->launch_sources,
      .shader_binary_size = desc->shader_binary_size,
      .scratch_size = desc->scratch_size,
      .compiler_d = desc->compiler_d,
      .threadgroup_memory_size_B = desc->threadgroup_memory_size_B,
      .imageblock_stride_B = desc->imageblock_stride_B,
      .constant_program_size = desc->constant_program.code_size,
      .nr_gprs = desc->nr_gprs,
      .length = generated->length,
      .compiler_d_valid = desc->compiler_d_valid,
      .constant_program_prelude = desc->constant_program.request_prelude,
      .independent = independent,
      .ldshdr_f1 = desc->ldshdr_f1,
      .range_count = desc->uniform_range_count,
      .launch = generated->launch,
   };
   memcpy(tpl->bytes, generated->bytes, generated->length);
   for (unsigned i = 0; i < tpl->range_count; ++i) {
      tpl->range_start[i] = desc->uniform_ranges[i].start_halfs;
      tpl->range_size[i] = desc->uniform_ranges[i].size_halfs;
   }

   if (!hk_g17p_patch_plan_init(&tpl->plan, desc->uniform_ranges,
                                desc->uniform_range_count, rels, rel_count,
                                generated->length, independent)) {
      free(tpl);
      tpl = NULL;
      goto out;
   }

   if (count == AGX_G17P_COMPUTE_TEMPLATE_LIMIT) {
      slot = head;
      while ((*slot)->next)
         slot = &(*slot)->next;
      free(*slot);
      *slot = NULL;
   }
   tpl->next = *head;
   *head = tpl;
   if (generated_count)
      (*generated_count)++;

out:
   free(generated);
   return tpl;
}

static bool
agx_g17p_compute_program_emit_cached(
   struct agx_batch *batch, struct agx_g17p_compute_template **head,
   unsigned *generated_count,
   const struct agx3_g17p_compute_program_desc *desc,
   struct agx3_compute_program *launch_out)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   if (!agx_va_in_usc_window(dev, desc->shader_va))
      return false;

   const struct agx_g17p_compute_template *tpl =
      agx_g17p_compute_template_get(dev, head, generated_count, desc);
   if (tpl) {
      struct agx_bo *program_bo = NULL;
      struct agx_ptr program = agx_pool_alloc_aligned_with_bo(
         &batch->pipeline_pool, tpl->length, AGX_IOTO_PROGRAM_ALIGNMENT,
         &program_bo);
      if (!program.cpu || !program_bo || !agx_va_in_usc_window(dev, program.gpu))
         return false;

      if (likely(tpl->length <= 512)) {
         uint8_t buf[512];
         memcpy(buf, tpl->bytes, tpl->length);
         hk_g17p_patch_plan_apply(&tpl->plan, desc->uniform_ranges, buf);
         memcpy(program.cpu, buf, tpl->length);
      } else {
         memcpy(program.cpu, tpl->bytes, tpl->length);
         hk_g17p_patch_plan_apply(&tpl->plan, desc->uniform_ranges,
                                  program.cpu);
      }

      if (unlikely(dev->debug & AGX_DBG_VERIFY)) {
         struct agx3_g17p_compute_program *fresh = malloc(sizeof(*fresh));
         if (fresh) {
            struct agx3_g17p_compute_program_desc verify_desc = *desc;
            verify_desc.program_va = program.gpu;
            verify_desc.patchable = tpl->independent;
            const bool ok =
               agx3_g17p_compute_program_generate(dev->chip, &verify_desc,
                                                  fresh) &&
               fresh->length == tpl->length &&
               memcmp(fresh->bytes, program.cpu, tpl->length) == 0;
            if (!ok) {
               mesa_loge("asahi: G17P compute template mismatch (ranges=%zu)",
                         desc->uniform_range_count);
            }
            free(fresh);
         }
      }

      agx_bo_assert_range(program_bo, program.gpu - program_bo->va->addr,
                          tpl->length);
      *launch_out = tpl->launch;
      launch_out->program_va = program.gpu;
      return true;
   }

   struct agx_bo *program_bo = NULL;
   struct agx_ptr program = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
      AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   if (!program.cpu || !program_bo || !agx_va_in_usc_window(dev, program.gpu))
      return false;

   struct agx3_g17p_compute_program_desc direct = *desc;
   direct.program_va = program.gpu;
   struct agx3_g17p_compute_program *native = malloc(sizeof(*native));
   if (!native)
      return false;
   const bool ok =
      agx3_g17p_compute_program_generate(dev->chip, &direct, native);
   if (ok) {
      memcpy(program.cpu, native->bytes, native->length);
      agx_bo_assert_range(program_bo, program.gpu - program_bo->va->addr,
                          native->length);
      *launch_out = native->launch;
   }
   free(native);
   return ok;
}

static bool
agx_g17p_append_compute(struct agx_batch *batch,
                        struct agx_compiled_shader *cs,
                        const struct agx_shader_info *info,
                        const void *binary, struct agx_bo *constant_bo,
                        const struct agx_linked_shader *linked,
                        struct agx_grid grid, struct agx_workgroup wg,
                        const struct agx3_ioto_uniform_range *ranges,
                        size_t range_count, unsigned variable_shared_mem)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   /* Mode 2 reads the whole grid -- three global thread counts then three
    * local sizes -- from six u32 in GPU memory, so the workgroup this call
    * was handed is not the one that launches and must not be validated. */
   const bool indirect_local = grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL;
   const bool g17g = agx_chip_g17_m5(dev->chip);

   /* G17G takes the captured M5 direct record and its own program
    * generator; the G17P path below cannot encode it. */
   if (g17g) {
      /* An indirect-local grid comes only from the geometry prepass of a
       * tessellated draw (agx_launch_gs_prerast), as in hk. */
      if ((!agx_chip_native_hal200(dev->chip) && !g17g) ||
          (g17g && grid.mode != AGX_CDM_MODE_DIRECT &&
           !(indirect_local && agx_m5_gpu_tess(batch->ctx) &&
             batch->ctx->in_tess)) ||
          !info || !binary ||
          (grid.mode != AGX_CDM_MODE_DIRECT && !indirect_local) ||
          (indirect_local &&
           (!grid.ptr || grid.ptr >= (UINT64_C(1) << 48) || (grid.ptr & 3))) ||
          !info->main_size || !info->nr_gprs ||
          info->has_preamble || info->preamble_scratch_size || info->scratch_size ||
          !info->agx3_staging_known ||
          (info->rodata.size_16
              ? (!constant_bo || info->rodata.offset != 0 ||
                 info->main_offset !=
                    ALIGN_POT((uint32_t)info->rodata.size_16 * 2, AGX_CODE_ALIGN))
              : (info->main_offset != 0 &&
                 info->main_offset != AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE)) ||
          info->main_offset > info->binary_size ||
          info->main_size > info->binary_size - info->main_offset ||
          (range_count && !ranges) ||
          (!indirect_local && (!wg.x || !wg.y || !wg.z)) ||
          (linked && (!linked->bo || !linked->bo->va || !linked->size ||
                      !linked->nr_gprs || !linked->g17p_compiler_d_valid ||
                      linked->scratch_size))) {
         if (info) {
            mesa_loge("asahi: rejected G17P compute metadata: grid=%u "
                      "preamble=%u scratch=%u/%u main=%u+%u/%u "
                      "gprs=%u staging_known=%u workgroup=%ux%ux%u",
                      (unsigned)grid.mode, (unsigned)info->has_preamble,
                      info->preamble_scratch_size, info->scratch_size,
                      info->main_offset, info->main_size, info->binary_size,
                      info->nr_gprs, (unsigned)info->agx3_staging_known,
                      wg.x, wg.y, wg.z);
         }
         return false;
      }

      uint64_t shader_va;
      uint32_t shader_binary_size, nr_gprs, compiler_d;
      struct agx_bo *shader_bo = NULL;
      size_t shader_sync_offset = 0, shader_sync_size = 0;

      if (linked) {
         shader_va = linked->bo->va->addr + linked->entry_offset;
         shader_binary_size = linked->size;
         nr_gprs = linked->nr_gprs;
         compiler_d = linked->g17p_compiler_d;
      } else {
         const size_t block_size = agx3_code_block_size(info->main_size);
         if (!block_size || block_size > SIZE_MAX - AGX3_CONTAINER_LEAD_SIZE)
            return false;

         struct agx_ptr shader = agx_pool_alloc_aligned_with_bo(
            &batch->pipeline_pool, AGX3_CONTAINER_LEAD_SIZE + block_size, 128,
            &shader_bo);
         if (!shader.cpu || !shader_bo)
            return false;

         agx3_build_container_lead(shader.cpu, dev->chip,
                                   dev->params.num_cores_per_cluster);
          const uint8_t *bytes = binary;
          /* Compiler rodata precedes the aligned main program and is bound
           * through the uniform ranges below. It is data, not an executable
           * constant program; synthesize the container's program in that case. */
          const uint8_t *constant_program =
             info->main_offset && !info->rodata.size_16 ? bytes : NULL;
          if (!agx3_build_code_block(
                (uint8_t *)shader.cpu + AGX3_CONTAINER_LEAD_SIZE, block_size,
                bytes + info->main_offset, info->main_size, info->main_size,
                 constant_program, false))
            return false;
         shader_va =
            shader.gpu + AGX3_CONTAINER_LEAD_SIZE + AGX3_CODE_BLOCK_ENTRY_OFFSET;
         shader_binary_size = info->main_size;
         nr_gprs = info->nr_gprs;
         compiler_d = (uint32_t)info->agx3_staging_halfs * 2;
         shader_sync_offset = shader.gpu - shader_bo->va->addr;
         shader_sync_size = AGX3_CONTAINER_LEAD_SIZE + block_size;
      }

      struct agx_bo *program_bo = NULL;
      struct agx_ptr program = agx_pool_alloc_aligned_with_bo(
         &batch->pipeline_pool, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
         AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
      if (!program.cpu || !program_bo || !agx_va_in_usc_window(dev, shader_va) ||
          !agx_va_in_usc_window(dev, program.gpu))
         return false;

      struct agx3_cdm_direct_template_sources launch_sources;
      if (!agx3_cdm_direct_template_sources_compute(
             info->uses_resource_info_query, info->uses_ray_traversal,
             info->ray_traversal_depth_nonzero, &launch_sources))
         return false;

      const uint64_t shared = (uint64_t)info->local_size + variable_shared_mem;
      if (shared > UINT32_MAX)
         return false;
      const struct agx3_g17p_compute_program_desc desc = {
         .program_va = program.gpu,
         .shader_va = shader_va,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = shader_binary_size,
         .scratch_size = info->scratch_size,
         .nr_gprs = nr_gprs,
         .compiler_d_valid = true,
         .compiler_d = compiler_d,
         .threadgroup_memory_size_B =
            shared ? MAX2(256u, ALIGN_POT((uint32_t)shared, 64u)) : 0,
         .imageblock_stride_B = info->imageblock_stride,
         .launch_sources = launch_sources,
         /* The program announces how it will be launched.  An indirectly
          * launched one takes its dimensions from the geometry block; a record
          * that selects indirect mode while its program still announces
          * DIRECT_ADD3 runs no invocations at all.  hk_cmd_dispatch.c picks the
          * kind the same way. */
         .ldshdr_f1 = indirect_local ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                                     : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      };
      /* G17G takes the captured M5 direct record: a four-word CDM Launch AGX3
       * header carrying a 48-bit pipeline VA, where agx_cdm_launch()'s AGX2
       * packet has only a two-word header and a 26-bit pipeline field. The
       * program is the graphics generator's non-fragment form, exactly as
       * hk_cmd_dispatch.c drives it. */
      struct agx3_g17p_compute_program native;
      struct agx3_compute_program g17g_launch;
      size_t program_length;
      if (g17g) {
         const struct agx3_g17p_graphics_program_desc g17g_desc = {
            .usc_clusters = dev->params.num_clusters_total,
            .program_va = program.gpu,
            .shader_va = shader_va,
            .uniform_ranges = ranges,
            .uniform_range_count = range_count,
            .shader_binary_size = info->main_size,
            .nr_gprs = info->nr_gprs,
            .compiler_d_valid = info->agx3_staging_known,
            .compiler_d = (uint32_t)info->agx3_staging_halfs * 2,
            .threadgroup_memory_size_B =
               shared ? MAX2(256u, ALIGN_POT((uint32_t)shared, 64u)) : 0,
         };
         struct agx3_g17p_graphics_program g17g_program;
         const struct agx3_g17p_native_compute_program_desc bind_desc = {
            .program_va = program.gpu,
            .shader_va = shader_va,
            .launch_sources = launch_sources,
         };
         if (!agx3_g17g_compute_program_generate(dev->chip, &g17g_desc,
                                                 &g17g_program) ||
             !agx3_m5_native_compute_program_bind(dev->chip, &bind_desc, &g17g_launch))
            return false;
         memcpy(program.cpu, g17g_program.bytes, g17g_program.length);
         program_length = g17g_program.length;
      } else {
         if (!agx3_g17p_compute_program_generate(dev->chip, &desc, &native))
            return false;
         memcpy(program.cpu, native.bytes, native.length);
         program_length = native.length;
      }

      if ((shader_bo && agx_bo_sync_for_device(shader_bo, shader_sync_offset,
                                               shader_sync_size)) ||
          agx_bo_sync_for_device(program_bo, program.gpu - program_bo->va->addr,
                                 program_length))
         return false;

      /* Every caller allocates the encoder before launching, but a missing one
       * here would be a null map rather than a refusal, and on G17G a refusal
       * simply falls back to the AGX2 encoder. */
      if (!batch->cdm.bo)
         return false;
      const uint8_t *cdm_base_map = agx_bo_map(batch->cdm.bo);
      if (!cdm_base_map)
         return false;
      const uint32_t dispatch_start = batch->cdm.current - cdm_base_map;
      struct agx3_compute_dispatch_sink sink = {
         .data = batch->cdm.current,
         .capacity = batch->cdm.end - batch->cdm.current,
      };
      if (indirect_local) {
         /* The kernel that writes this grid is already its own dispatch earlier
          * in the stream, behind a full barrier, so the launch stands alone.
          * M5 reads the local size from the grid too (hk_dispatch_m5_native). */
         if (!(g17g ? agx3_m5_native_indirect_dispatch_append(
                         &g17g_launch, grid.ptr, NULL, 5, &sink)
                    : agx3_g17p_native_indirect_compute_launch_append(
                         &native.launch, grid.ptr, 0, &sink)))
            return false;
      } else {
         const struct agx3_compute_direct_grid direct = {
            .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
            .size = {grid.count[0], grid.count[1], grid.count[2]},
            .local_size = {wg.x, wg.y, wg.z},
            .dimensions = grid.count[2] != 1 || wg.z != 1
                             ? 3
                          : grid.count[1] != 1 || wg.y != 1 ? 2 : 1,
         };
         if (!(g17g ? agx3_m5_native_compute_dispatch_append(&g17g_launch, &direct, 5, &sink)
                     : agx3_g17p_native_compute_dispatch_append(&native.launch, &direct, &sink)))
            return false;
      }
      batch->cdm.current += sink.length;

      /* This append bypasses agx_launch_internal(), so terminate and record the
       * dispatch here too: the M5 profile covers one dispatch per command, and
       * agx_batch_submit() sends each recorded extent as its own submission. A
       * dispatch left unrecorded is a batch with no submittable compute. */
      if (g17g) {
         uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
         if (!agx3_cdm_stream_terminate_pack(dev->chip, terminate) ||
             (size_t)(batch->cdm.end - batch->cdm.current) < sizeof(terminate))
            return false;

         memcpy(batch->cdm.current, terminate, sizeof(terminate));
         batch->cdm.current += sizeof(terminate);

         if (!agx_record_cdm_dispatch(batch, dispatch_start,
                                     batch->cdm.current - cdm_base_map,
                                     grid))
            return false;

         /* The dispatch's storage, as hk_m5_compute_resources() lists it.
          * This path refuses spills, so only the workgroup's shared memory
          * and size vary. */
         if (agx_m5_render_resources(dev)) {
            const struct drm_asahi_compute_program profile = {
               .primary_bytes = info->scratch_size,
               .threadgroup_bytes =
                  shared ? MAX2(256u, ALIGN_POT((uint32_t)shared, 64u)) : 0,
               .threads = wg.x * wg.y * wg.z,
            };
            agx_batch_note_m5_compute_program(batch, &profile);
         }
      }

      if (constant_bo)
         agx_batch_add_bo(batch, constant_bo);
      if (linked)
         agx_batch_add_bo(batch, linked->bo);
      return true;
   }

   /* The compiler places promoted constants ahead of main and pads main to
    * 0x80, so a shader with rodata has main_offset >= 0x80 and only rodata
    * before it.  Those constants are published as a uniform range by the
    * caller; the code block gets the generated compute constant program, as
    * the D3D12 native compute path already does. */
   const bool rodata_prefix =
      info && info->rodata.size_16 && info->main_offset &&
      !(info->main_offset % 0x80) &&
      info->rodata.offset + (uint64_t)info->rodata.size_16 * 2 <=
         info->main_offset;
   const bool cs_cprog =
      !linked && agx_chip_native_agx3(dev->chip) &&
      agx_chip_native_preamble(dev->chip) && info && info->has_preamble &&
      agx_g17p_cs_cprog_qualified(info);
   /* The thread-private frame SPECTPR declares: register spills, and the
    * 16-byte word a device atomic returns its old value through. Declaring
    * less than the code uses is not refused by anything; the dispatch just
    * never completes. Same frame hk_cmd_dispatch.c declares. */
   const uint32_t tps_size =
      !info ? 0
      : linked ? MAX2(linked->scratch_size, linked->agx3_spectpr_scratch_size)
               : MAX2(info->scratch_size, info->agx3_spectpr_scratch_size);
   if (!agx_chip_native_agx3(dev->chip) || !info || !binary ||
       (grid.mode != AGX_CDM_MODE_DIRECT && !indirect_local) ||
       (indirect_local &&
        (!grid.ptr || grid.ptr >= (UINT64_C(1) << 48) || (grid.ptr & 3))) ||
       !info->main_size || !info->nr_gprs ||
       ((info->has_preamble || info->preamble_scratch_size) && !cs_cprog) ||
       tps_size > AGX3_G17P_MAX_SPILL_B || (tps_size & 3) ||
       !info->agx3_staging_known ||
       (info->main_offset != 0 && !rodata_prefix && !cs_cprog &&
        info->main_offset != AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE) ||
       info->main_offset > info->binary_size ||
       info->main_size > info->binary_size - info->main_offset ||
       (range_count && !ranges) ||
       (!indirect_local && (!wg.x || !wg.y || !wg.z)) ||
       (cs_cprog &&
        (!constant_bo || !constant_bo->va || !constant_bo->va->addr ||
         constant_bo->va->addr >= (UINT64_C(1) << 48) ||
         (constant_bo->va->addr & 1) || info->binary_size > constant_bo->size ||
         info->binary_size > (UINT64_C(1) << 48) - constant_bo->va->addr ||
         !agx_bo_map(constant_bo) ||
         (info->agx3_preamble_tps_inputs && !range_count))) ||
       (linked && (!linked->bo || !linked->bo->va || !linked->size ||
                   !linked->nr_gprs || !linked->g17p_compiler_d_valid))) {
      if (info) {
         mesa_loge("asahi: rejected G17P compute metadata: grid=%u "
                   "preamble=%u scratch=%u/%u main=%u+%u/%u "
                   "gprs=%u staging_known=%u workgroup=%ux%ux%u",
                   (unsigned)grid.mode, (unsigned)info->has_preamble,
                   info->preamble_scratch_size, info->scratch_size,
                   info->main_offset, info->main_size, info->binary_size,
                   info->nr_gprs, (unsigned)info->agx3_staging_known,
                   wg.x, wg.y, wg.z);
      }
      return false;
   }

   /* An empty direct grid -- a geometry or transform feedback setup kernel
    * for a draw with nothing to process -- launches no invocations. The
    * native launch record cannot encode a zero size, so emit nothing, as
    * the hardware would have done with the legacy record. */
   if (!indirect_local &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return true;

   uint64_t shader_va;
   uint32_t shader_binary_size, nr_gprs, compiler_d;
   struct agx_bo *shader_bo = NULL;
   size_t shader_sync_offset = 0, shader_sync_size = 0;

   if (linked) {
      shader_va = linked->bo->va->addr + linked->entry_offset;
      shader_binary_size = linked->size;
      nr_gprs = linked->nr_gprs;
      compiler_d = linked->g17p_compiler_d;
   } else {
      const size_t block_size = agx3_code_block_size(info->main_size);
      if (!block_size || block_size > SIZE_MAX - AGX3_CONTAINER_LEAD_SIZE)
         return agx_g17p_compute_failed("code block size");

      const size_t container_size = AGX3_CONTAINER_LEAD_SIZE + block_size;
      const uint8_t *bytes = binary;
      const uint8_t *container_cprog =
         info->main_offset && !rodata_prefix && !cs_cprog ? bytes : NULL;

      if (cs && !cs->agx3_compute_bo) {
         struct agx_bo *bo = agx_bo_create(
            dev, container_size, 0, AGX_BO_EXEC | AGX_BO_LOW_VA,
            "Compute container");
         uint8_t *map = bo && bo->va ? agx_bo_map(bo) : NULL;
         if (map) {
            agx3_build_container_lead(map, dev->chip,
                                      dev->params.num_cores_per_cluster);
            const uint64_t va = bo->va->addr + AGX3_CONTAINER_LEAD_SIZE +
                                AGX3_CODE_BLOCK_ENTRY_OFFSET;
            if (agx3_build_code_block(map + AGX3_CONTAINER_LEAD_SIZE,
                                      block_size, bytes + info->main_offset,
                                      info->main_size, info->main_size,
                                      container_cprog, false) &&
                agx_va_in_usc_window(dev, va)) {
               agx_bo_assert_range(bo, 0, container_size);
               cs->agx3_compute_bo = bo;
               cs->agx3_compute_va = va;
            } else {
               agx_bo_unreference(dev, bo);
            }
         } else {
            agx_bo_unreference(dev, bo);
         }
      }

      if (cs && cs->agx3_compute_bo && cs->agx3_compute_va) {
         shader_va = cs->agx3_compute_va;
         agx_batch_add_bo(batch, cs->agx3_compute_bo);
      } else {
         struct agx_ptr shader = agx_pool_alloc_aligned_with_bo(
            &batch->pipeline_pool, container_size, 128, &shader_bo);
         if (!shader.cpu || !shader_bo)
            return agx_g17p_compute_failed("shader allocation");

         agx3_build_container_lead(shader.cpu, dev->chip,
                                   dev->params.num_cores_per_cluster);
         if (!agx3_build_code_block(
                (uint8_t *)shader.cpu + AGX3_CONTAINER_LEAD_SIZE, block_size,
                bytes + info->main_offset, info->main_size, info->main_size,
                container_cprog, false))
            return agx_g17p_compute_failed("code block build");
         shader_va = shader.gpu + AGX3_CONTAINER_LEAD_SIZE +
                     AGX3_CODE_BLOCK_ENTRY_OFFSET;
         shader_sync_offset = shader.gpu - shader_bo->va->addr;
         shader_sync_size = container_size;
      }

      shader_binary_size = info->main_size;
      nr_gprs = info->nr_gprs;
      compiler_d = (uint32_t)info->agx3_staging_halfs * 2;
   }

   if (!agx_va_in_usc_window(dev, shader_va))
      return agx_g17p_compute_failed("program allocation or USC window");

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(
          info->uses_resource_info_query &&
             agx_chip_query_ti_spill(dev->chip),
          info->uses_ray_traversal, info->ray_traversal_depth_nonzero,
          &launch_sources))
      return agx_g17p_compute_failed("launch template");

   struct agx_ioto_constant_program constant_program = {0};
   const struct agx3_cprog_inputs *cprog_inputs = NULL;
   if (cs_cprog) {
      constant_program = (struct agx_ioto_constant_program){
         .code = (const uint8_t *)agx_bo_map(constant_bo) +
                 info->preamble_offset,
         .code_size = info->preamble_size,
         .code_va = constant_bo->va->addr + info->preamble_offset,
         .force_external = true,
         .absolute_branch = true,
         .request_prelude = info->agx3_preamble_uniform_inputs,
      };
      if (info->agx3_preamble_tps_inputs)
         cprog_inputs = &info->agx3_preamble_inputs;
      const unsigned cprog_frame =
         ALIGN_POT(MAX2((unsigned)info->agx3_preamble_inputs.storage_bytes,
                        info->preamble_scratch_size),
                   16);
      launch_sources.ti_spill_bytes =
         MAX2(launch_sources.ti_spill_bytes, cprog_frame);
      launch_sources.ti_largest_ipr_bytes =
         MAX2(launch_sources.ti_largest_ipr_bytes,
              (uint32_t)info->agx3_preamble_staging_halfs * 2u);
      nr_gprs = MAX2(nr_gprs, info->nr_preamble_gprs);
      compiler_d =
         MAX2(compiler_d, (uint32_t)info->agx3_preamble_staging_halfs * 2u);
   }

   const uint64_t shared = (uint64_t)info->local_size + variable_shared_mem;
   if (shared > UINT32_MAX)
      return agx_g17p_compute_failed("shared memory size");
   const struct agx3_g17p_compute_program_desc desc = {
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = shader_binary_size,
      .scratch_size = tps_size,
      .nr_gprs = nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = compiler_d,
      .threadgroup_memory_size_B =
         shared ? MAX2(256u, ALIGN_POT((uint32_t)shared, 64u)) : 0,
      .imageblock_stride_B = info->imageblock_stride,
      .launch_sources = launch_sources,
      /* The program announces how it will be launched.  An indirectly
       * launched one takes its dimensions from the geometry block; a record
       * that selects indirect mode while its program still announces
       * DIRECT_ADD3 runs no invocations at all.  hk_cmd_dispatch.c picks the
       * kind the same way. */
      .ldshdr_f1 = indirect_local ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                                  : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      .constant_program = constant_program,
      .cprog_inputs = cprog_inputs,
   };
   struct agx3_compute_program launch;
   if (!agx_g17p_compute_program_emit_cached(
          batch, cs ? &cs->g17p_compute_templates : NULL,
          cs ? &cs->g17p_compute_templates_generated : NULL, &desc, &launch))
      return agx_g17p_compute_failed("Ioto program generation");

   if (shader_bo)
      agx_bo_assert_range(shader_bo, shader_sync_offset, shader_sync_size);

   struct agx3_compute_dispatch_sink sink = {
      .data = batch->cdm.current,
      .capacity = batch->cdm.end - batch->cdm.current,
   };
   /* The software graphics stages (vertex, tessellation and geometry
    * programs run as compute) hand their results to the next dispatch of the
    * stream through memory, with no barrier of their own in between. The
    * launch's weak tail orders them on HAL200. On G15, a dependency inside
    * the stream is published only by the wide flush, so there every launch
    * ends in it, as Honeykrisp's software stages do. */
   /* A dispatch of the driver's compute blitter feeds later reads through the
    * texture unit as well as memory -- each compressed mip level is sampled by
    * the dispatch that generates the next one -- so it ends in the strong
    * flush everywhere, as the precompiled meta kernels' AGX_BARRIER_ALL and
    * every Borealis launch do. */
   const uint8_t barrier_flags =
      agx_chip_cdm_wide_visibility(dev->chip) ||
            batch->ctx->compute_blitter.active
         ? 0x5
         : 0;
   if (indirect_local) {
      /* The kernel that writes this grid is already its own dispatch earlier
       * in the stream, behind a full barrier, so the launch stands alone. */
      if (!agx3_g17p_native_indirect_compute_launch_append(
             &launch, grid.ptr, barrier_flags, &sink))
         return agx_g17p_compute_failed("indirect launch append");
   } else {
      const struct agx3_compute_direct_grid direct = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {grid.count[0], grid.count[1], grid.count[2]},
         .local_size = {wg.x, wg.y, wg.z},
         .dimensions = grid.count[2] != 1 || wg.z != 1
                          ? 3
                       : grid.count[1] != 1 || wg.y != 1 ? 2 : 1,
      };
      if (!agx3_g17p_native_compute_dispatch_append_with_barrier(
             &launch, &direct, barrier_flags, &sink))
         return agx_g17p_compute_failed("direct launch append");
   }
   batch->cdm.current += sink.length;
   if (constant_bo)
      agx_batch_add_bo(batch, constant_bo);
   if (linked)
      agx_batch_add_bo(batch, linked->bo);
   return true;
}

/* G18G (j873): the same dispatch on the M5 compute profile, with the G18G
 * indirect records and agx3d re-encoding. Every other chip takes
 * agx_g17p_append_compute() above. */
static bool
agx_g18_append_compute(struct agx_batch *batch,
                       const struct agx_shader_info *info,
                       const void *binary, struct agx_bo *constant_bo,
                       const struct agx_linked_shader *linked,
                       struct agx_grid grid, struct agx_workgroup wg,
                       const struct agx3_ioto_uniform_range *ranges,
                       size_t range_count, unsigned variable_shared_mem)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   const bool g18 = dev->chip == AGX_CHIP_G18G;
   /* Mode 2 reads the whole grid -- three global thread counts then three
    * local sizes -- from six u32 in GPU memory, so the workgroup this call
    * was handed is not the one that launches and must not be validated. */
   const bool indirect_local = grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL;
   /* The compiler places promoted constants ahead of main and pads main to
    * 0x80, so a shader with rodata has main_offset >= 0x80 and only rodata
    * before it.  Those constants are published as a uniform range by the
    * caller; the code block gets the generated compute constant program, as
    * the D3D12 native compute path already does. */
   const bool rodata_prefix =
      info && info->rodata.size_16 && info->main_offset &&
      !(info->main_offset % 0x80) &&
      info->rodata.offset + (uint64_t)info->rodata.size_16 * 2 <=
         info->main_offset;
   /* The thread-private frame SPECTPR declares: register spills, and the
    * 16-byte word a device atomic returns its old value through. Declaring
    * less than the code uses is not refused by anything; the dispatch just
    * never completes. Same frame hk_cmd_dispatch.c declares. */
   const uint32_t tps_size =
      !info ? 0
      : linked ? MAX2(linked->scratch_size, linked->agx3_spectpr_scratch_size)
               : MAX2(info->scratch_size, info->agx3_spectpr_scratch_size);
   /* G18G takes both of Apple's indirect records. */
   const bool g18_indirect = g18 && agx_is_indirect(grid);
   if ((!agx_chip_native_hal200(dev->chip) && !g18) ||
       !info || !binary ||
       (grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL && !g18_indirect) ||
       (agx_is_indirect(grid) &&
        (!grid.ptr || grid.ptr >= (UINT64_C(1) << 48) || (grid.ptr & 3))) ||
       !info->main_size || !info->nr_gprs ||
       info->preamble_scratch_size ||
       (agx_chip_native_hal200(dev->chip) &&
        (info->has_preamble || tps_size > AGX3_G17P_MAX_SPILL_B ||
         (tps_size & 3))) ||
       !info->agx3_staging_known ||
       (g18
           ? (info->rodata.size_16
                 ? (!constant_bo || info->rodata.offset != 0 ||
                    info->main_offset !=
                       ALIGN_POT((uint32_t)info->rodata.size_16 * 2, AGX_CODE_ALIGN))
                 : (info->main_offset != 0 &&
                    info->main_offset != AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE))
           : (info->main_offset != 0 && !rodata_prefix &&
              info->main_offset != AGX3_CONTAINER_CONSTANT_PROGRAM_SIZE)) ||
       info->main_offset > info->binary_size ||
       info->main_size > info->binary_size - info->main_offset ||
       (range_count && !ranges) ||
       (!indirect_local && (!wg.x || !wg.y || !wg.z)) ||
       (linked && (!linked->bo || !linked->bo->va || !linked->size ||
                   !linked->nr_gprs || !linked->g17p_compiler_d_valid ||
                   (g18 && (linked->g17p_compiler_d & 1))))) {
      if (info) {
         mesa_loge("asahi: rejected G17P compute metadata: grid=%u "
                   "preamble=%u scratch=%u/%u main=%u+%u/%u "
                   "gprs=%u staging_known=%u workgroup=%ux%ux%u",
                   (unsigned)grid.mode, (unsigned)info->has_preamble,
                   info->preamble_scratch_size, info->scratch_size,
                   info->main_offset, info->main_size, info->binary_size,
                   info->nr_gprs, (unsigned)info->agx3_staging_known,
                   wg.x, wg.y, wg.z);
      }
      return false;
   }

   /* An empty direct grid -- a geometry or transform feedback setup kernel
    * for a draw with nothing to process -- launches no invocations. The
    * native launch record cannot encode a zero size, so emit nothing, as
    * the hardware would have done with the legacy record. */
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return true;

   struct agx_ioto_preshader_desc preamble;
   if (!agx_bind_g17p_preamble(info, binary, constant_bo, &preamble))
      return false;

   uint64_t shader_va;
   uint32_t shader_binary_size, nr_gprs, compiler_d;
   struct agx_bo *shader_bo = NULL;
   size_t shader_sync_offset = 0, shader_sync_size = 0;

   if (linked) {
      shader_va = linked->bo->va->addr + linked->entry_offset;
      shader_binary_size = linked->size;
      nr_gprs = linked->nr_gprs;
      compiler_d = linked->g17p_compiler_d;
   } else {
      /* G18G takes the container its graphics, background and end-of-tile
       * programs already use (agx_linker.c agx3_build_container,
       * agx_bg_eot.c agx_g17p_upload_bg_eot): the 0x640 lead with four agx3d
       * helper tables and an agx3d constant program. The G17P lead and its
       * AGX3B vertex_compute program are not agx3d code. */
      const size_t lead_size = g18 ? AGX3_G18G_CONTAINER_LEAD_SIZE
                                   : AGX3_CONTAINER_LEAD_SIZE;
      const size_t block_size = agx3_code_block_size(info->main_size);
      if (!block_size || block_size > SIZE_MAX - lead_size)
         return agx_g17p_compute_failed("code block size");

      struct agx_ptr shader = agx_pool_alloc_aligned_with_bo(
         &batch->pipeline_pool, lead_size + block_size, 128, &shader_bo);
      if (!shader.cpu || !shader_bo)
         return agx_g17p_compute_failed("shader allocation");

      if (g18) {
         if (!agx3_build_g18g_container_lead(shader.cpu, lead_size))
            return false;
      } else {
         agx3_build_container_lead(shader.cpu, dev->chip,
                                   dev->params.num_cores_per_cluster);
      }
       const uint8_t *bytes = binary;
       /* Compiler rodata precedes the aligned main program and is bound
        * through the uniform ranges below. It is data, not an executable
        * constant program; synthesize the container's program in that case. */
       const uint8_t *constant_program =
          info->main_offset && !info->rodata.size_16 ? bytes : NULL;
      uint8_t *block = (uint8_t *)shader.cpu + lead_size;
       if (!agx3_build_code_block(
             block, block_size,
             bytes + info->main_offset, info->main_size, info->main_size,
              constant_program, false))
         return agx_g17p_compute_failed("code block build");
      /* As agx_bg_eot.c: the compiled text is agx3d already; only the
       * synthesized constant program is not. */
      if (g18 && !constant_program)
         agx3_build_container_stop_program_agx3d(
            block + AGX3_CODE_BLOCK_HEADER_SIZE);
      shader_va = shader.gpu + lead_size + AGX3_CODE_BLOCK_ENTRY_OFFSET;
      shader_binary_size = info->main_size;
      nr_gprs = info->nr_gprs;
      compiler_d = (uint32_t)info->agx3_staging_halfs * 2;
      shader_sync_offset = shader.gpu - shader_bo->va->addr;
      shader_sync_size = lead_size + block_size;
   }

   struct agx_bo *program_bo = NULL;
   struct agx_ptr program = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
      AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   if (!program.cpu || !program_bo || !agx_va_in_usc_window(dev, shader_va) ||
       !agx_va_in_usc_window(dev, program.gpu))
      return agx_g17p_compute_failed("program allocation or USC window");

   struct agx3_cdm_direct_template_sources launch_sources;
   if (!agx3_cdm_direct_template_sources_compute(
          info->uses_resource_info_query, info->uses_ray_traversal,
          info->ray_traversal_depth_nonzero, &launch_sources))
      return agx_g17p_compute_failed("launch template");

   const uint32_t scratch_bytes = linked
      ? MAX2(linked->scratch_size, linked->agx3_spectpr_scratch_size)
      : MAX2(info->scratch_size, info->agx3_spectpr_scratch_size);
   const uint64_t shared = (uint64_t)info->local_size + variable_shared_mem;
   /* G17P's SPECLM carries a threadgroup-memory size, and so does G18G's:
    * Apple's G18 reduction kernel publishes f18 = bytes, f24 = 1 (native
    * kick 18). G17G itself has no such witness, so keep declining it there
    * rather than running a kernel without its threadgroup memory. */
   if (shared > UINT32_MAX || (g18 && shared > 32768))
      return agx_g17p_compute_failed("shared memory size");
   const struct agx3_g17p_compute_program_desc desc = {
      .program_va = program.gpu,
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .preshader = info->has_preamble ? &preamble : NULL,
      .shader_binary_size = shader_binary_size,
      .scratch_size = tps_size,
      .nr_gprs = nr_gprs,
      .compiler_d_valid = true,
      .compiler_d = compiler_d,
      .threadgroup_memory_size_B =
         shared ? MAX2(256u, ALIGN_POT((uint32_t)shared, 64u)) : 0,
      .imageblock_stride_B = info->imageblock_stride,
      .launch_sources = launch_sources,
      /* The program announces how it will be launched.  An indirectly
       * launched one takes its dimensions from the geometry block; a record
       * that selects indirect mode while its program still announces
       * DIRECT_ADD3 runs no invocations at all.  hk_cmd_dispatch.c picks the
       * kind the same way. */
      .ldshdr_f1 = indirect_local ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                                  : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
   };
   /* G17G takes the captured M5 direct record: a four-word CDM Launch AGX3
    * header carrying a 48-bit pipeline VA, where agx_cdm_launch()'s AGX2
    * packet has only a two-word header and a 26-bit pipeline field. The
    * program is the graphics generator's non-fragment form, exactly as
    * hk_cmd_dispatch.c drives it. */
   struct agx3_g17p_compute_program native;
   struct agx3_compute_program g17g_launch;
   size_t program_length;
   if (g18) {
      const struct agx3_g17p_graphics_program_desc g17g_desc = {
         .usc_clusters = dev->params.num_clusters_total,
         .program_va = program.gpu,
         .shader_va = shader_va,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = shader_binary_size,
         .scratch_size = scratch_bytes,
         .nr_gprs = nr_gprs,
         .compiler_d_valid = true,
         .compiler_d = compiler_d,
         /* G18 load/store indices now count scalar elements, so SPECLM
          * needs the actual byte footprint, as in the native reduction. */
         .threadgroup_memory_size_B =
            shared ? MAX2(256u, ALIGN_POT((uint32_t)shared, 64u)) : 0,
      };
      struct agx3_g17p_graphics_program g17g_program;
      const struct agx3_g17p_native_compute_program_desc bind_desc = {
         .program_va = program.gpu,
         .shader_va = shader_va,
         .launch_sources = launch_sources,
      };
      if (!agx3_g18g_compute_program_generate(&g17g_desc, &g17g_program) ||
          !agx_g18g_reencode_program(dev, &g17g_program, true, nr_gprs) ||
          g17g_program.length > AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE ||
          !agx3_g17g_native_compute_program_bind(&bind_desc, &g17g_launch))
         return false;
      memcpy(program.cpu, g17g_program.bytes, g17g_program.length);
      program_length = g17g_program.length;
   } else {
      if (!agx3_g17p_compute_program_generate(dev->chip, &desc, &native))
         return agx_g17p_compute_failed("Ioto program generation");
      memcpy(program.cpu, native.bytes, native.length);
      program_length = native.length;
   }

   if ((shader_bo && agx_bo_sync_for_device(shader_bo, shader_sync_offset,
                                            shader_sync_size)) ||
       agx_bo_sync_for_device(program_bo, program.gpu - program_bo->va->addr,
                              program_length))
      return agx_g17p_compute_failed("shader/program sync");

   /* Every caller allocates the encoder before launching, but a missing one
    * here would be a null map rather than a refusal, and on G17G a refusal
    * simply falls back to the AGX2 encoder. */
   if (!batch->cdm.bo)
      return false;
   const uint8_t *cdm_base_map = agx_bo_map(batch->cdm.bo);
   if (!cdm_base_map)
      return false;
   const uint32_t dispatch_start = batch->cdm.current - cdm_base_map;
   struct agx3_compute_dispatch_sink sink = {
      .data = batch->cdm.current,
      .capacity = batch->cdm.end - batch->cdm.current,
   };
   if (g18_indirect) {
      /* Apple never 1D-masks an indirect template; bit 23 stays a template
       * bit, set for threadgroup memory as in the direct record below. */
      if (shared)
         g17g_launch.launch_template.word0 |= UINT32_C(1) << 23;
      const uint32_t local[3] = {wg.x, wg.y, wg.z};
      if (!(indirect_local
               ? agx3_g17g_native_indirect_compute_launch_append(
                    &g17g_launch, grid.ptr,
                    agx_m5_dispatch_barrier_flags(dev), &sink)
               : agx3_g17g_native_indirect_groups_compute_launch_append(
                    &g17g_launch, grid.ptr, local,
                    agx_m5_dispatch_barrier_flags(dev), &sink)))
         return false;
   } else if (indirect_local) {
      /* The kernel that writes this grid is already its own dispatch earlier
       * in the stream, behind a full barrier, so the launch stands alone. */
      if (!agx3_g17p_native_indirect_compute_launch_append(&native.launch,
                                                           grid.ptr, 0, &sink))
         return agx_g17p_compute_failed("indirect launch append");
   } else {
      const struct agx3_compute_direct_grid direct = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {grid.count[0], grid.count[1], grid.count[2]},
         .local_size = {wg.x, wg.y, wg.z},
         .dimensions = grid.count[2] != 1 || wg.z != 1
                          ? 3
                       : grid.count[1] != 1 || wg.y != 1 ? 2 : 1,
      };
      /* A G18 launch normalizes the template for a 1D grid. The raw native
       * append helper does not apply the semantic packer's bits17..19 mask.
       * Keep bit23 and higher-dimensional launches intact. */
      if (dev->chip == AGX_CHIP_G18G &&
          direct.size[1] < 2 && direct.size[2] < 2)
         g17g_launch.launch_template.word0 &= UINT32_C(0xfff1ffff);
      /* Apple's G18 threadgroup-memory kernel (native kick 18, reduction)
       * launches with word0 bit 23 set. Without it, the 64-thread reduction
       * summed only 16 lanes on hardware (Mesa165, run164). */
      if (dev->chip == AGX_CHIP_G18G && shared)
         g17g_launch.launch_template.word0 |= UINT32_C(1) << 23;

      if (!(g18 ? agx3_g17g_native_compute_dispatch_append(
                      &g17g_launch, &direct, agx_m5_dispatch_barrier_flags(dev),
                      &sink)
                  : agx3_g17p_native_compute_dispatch_append(&native.launch, &direct, &sink)))
         return agx_g17p_compute_failed("direct launch append");
   }
   batch->cdm.current += sink.length;

   /* This append bypasses agx_launch_internal(), so terminate and record the
    * dispatch here too: the M5 profile covers one dispatch per command, and
    * agx_batch_submit() sends each recorded extent as its own submission. A
    * dispatch left unrecorded is a batch with no submittable compute. */
   if (g18) {
      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      if (!agx3_cdm_stream_terminate_pack(dev->chip, terminate) ||
          (size_t)(batch->cdm.end - batch->cdm.current) < sizeof(terminate))
         return false;

      memcpy(batch->cdm.current, terminate, sizeof(terminate));
      batch->cdm.current += sizeof(terminate);

      if (!agx_record_cdm_dispatch(batch, dispatch_start,
                                   batch->cdm.current - cdm_base_map,
                                   grid))
         return false;
      if (g18) {
         struct agx_cdm_dispatch *dispatch = util_dynarray_element(
            &batch->cdm_dispatch, struct agx_cdm_dispatch,
            util_dynarray_num_elements(&batch->cdm_dispatch, struct agx_cdm_dispatch) - 1);
         /* Match the effective linked program encoded above. G18 API shared
          * indices count elements, so v2 reports its actual byte footprint. */
         dispatch->resources = (struct drm_asahi_g18_compute_resources){
            .version = 2,
            .flags = 1u | (info->has_preamble ? 2u : 0u) |
                     (info->has_preamble && info->agx3_preamble_staging_known
                         ? 4u : 0u) | (indirect_local ? 8u : 0u),
            .registers_16 = nr_gprs,
            .scratch_bytes = linked ? linked->scratch_size : info->scratch_size,
            .spectpr_scratch_bytes = linked
               ? linked->agx3_spectpr_scratch_size : info->agx3_spectpr_scratch_size,
            .staging_16 = compiler_d / 2,
            .constant_registers_16 = info->has_preamble ? info->nr_preamble_gprs : 0,
            .constant_scratch_bytes = info->has_preamble ? info->preamble_scratch_size : 0,
            .constant_staging_16 = info->has_preamble ? info->agx3_preamble_staging_halfs : 0,
            .local_bytes = info->local_size,
            .imageblock_stride = info->imageblock_stride,
            .variable_shared_bytes = variable_shared_mem,
            .local_size = {indirect_local ? 0 : wg.x,
                           indirect_local ? 0 : wg.y,
                           indirect_local ? 0 : wg.z},
         };
      }
   }

   if (constant_bo)
      agx_batch_add_bo(batch, constant_bo);
   if (linked)
      agx_batch_add_bo(batch, linked->bo);
   return true;
}

static void
agx_launch_internal(struct agx_batch *batch, struct agx_grid grid,
                    struct agx_workgroup wg,
                    struct agx_cdm_launch_word_0_packed launch,
                    mesa_shader_stage stage, uint32_t usc)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);

   /* TODO: Ensure space if we allow multiple kernels in a batch */
   uint32_t *out = (uint32_t *)batch->cdm.current;
   const uint8_t *base = agx_bo_map(batch->cdm.bo);
   const uint32_t start = (uint8_t *)out - base;

   out = agx_cdm_launch(out, dev->chip, grid, wg, launch, usc);
   out = agx_cdm_barrier(out, dev->chip);

   /* One dispatch per command on AGX3c: terminate this one where it ends and
    * record its extent, so agx_batch_submit() can send each separately. The
    * terminator costs eight bytes per dispatch in the same encoder; the
    * stream is never submitted as a whole on this part, so the interleaved
    * terminators are only ever the end of their own command. */
   if (agx_chip_isa_encoding(dev->chip) == AGX_ISA_ENCODING_AGX3C) {
      uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
      if (!agx3_cdm_stream_terminate_pack(dev->chip, terminate))
         UNREACHABLE("AGX3c CDM terminate must be grounded");
      memcpy(out, terminate, sizeof(terminate));
      out += sizeof(terminate) / 4;

      if (!agx_record_cdm_dispatch(batch, start,
                                  (uint32_t)((uint8_t *)out - base),
                                  grid)) {
         agx_native_descriptor_reject_batch(batch, "compute dispatch allocation");
         return;
      }
   }

   batch->cdm.current = (void *)out;
   assert(batch->cdm.current <= batch->cdm.end &&
          "Failed to reserve sufficient space in encoder");
}

/* Upload an AGX3 precompiled kernel's arguments and publish them, with its
 * promoted constants, as Ioto uniform ranges. Shared by the G17P and M5
 * launches below; refuses the batch and returns false on failure. */
static bool
agx_precomp_agx3_uniform_ranges(struct agx_batch *batch,
                                const struct agx_precompiled_shader *cs,
                                const void *args, size_t arg_size,
                                struct agx3_ioto_uniform_range ranges[2],
                                unsigned *range_count)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);

   /* Match the precompiler's rebased arguments and optional runtime printf
    * address. Pad packed argument structs for Ioto's whole-dword loads. */
   const size_t printf_offset = cs->info.printf_uniform
      ? (size_t)(cs->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2
      : 0;
   const size_t upload_size = ALIGN_POT(
      cs->info.printf_uniform ? MAX2(arg_size, printf_offset + 8) : arg_size, 4);
   struct agx_ptr uploaded = agx_pool_alloc_aligned(&batch->pool, upload_size, 4);
   if (!uploaded.cpu || upload_size / 2 > UINT16_MAX)
      {
         agx_native_descriptor_reject_batch(batch, "precompiled argument upload");
         return false;
      }
   memcpy(uploaded.cpu, args, arg_size);
   memset((uint8_t *)uploaded.cpu + arg_size, 0, upload_size - arg_size);
   if (cs->info.printf_uniform)
      memcpy((uint8_t *)uploaded.cpu + printf_offset,
             &dev->printf_buffer_address, sizeof(uint64_t));
   if (!uploaded.gpu) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute resources");
      return false;
   }

   /* Above the halves the AGX3 hardware ABI owns, matching where the
    * precompiler now places these loads. */
   ranges[0] = (struct agx3_ioto_uniform_range){
      .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
      .size_halfs = upload_size / 2,
      .address = uploaded.gpu,
   };
   *range_count = 1;

   /* Constants the compiler promoted out of the code are published as a
    * second range, exactly as the legacy USC path maps them with
    * agx_usc_immediates(). Their base is already in this ABI's numbering:
    * agx_opt_promote_constants() allocates above push_count, which starts
    * at the reserved preamble. Without this the kernel reads whatever the
    * uniform file happened to hold -- which is why the launch-shape check
    * refused a kernel carrying rodata rather than dispatching one. */
   if (cs->info.rodata.size_16) {
      ranges[(*range_count)++] = (struct agx3_ioto_uniform_range){
         .start_halfs = cs->info.rodata.base_uniform,
         .size_halfs = cs->info.rodata.size_16,
         .address = cs->ptr + cs->rodata_offset,
      };
   }
   return true;
}

static bool
agx_precomp_shader_launchable(const struct agx_device *dev,
                              const struct agx_precompiled_shader *cs)
{
   if (!cs)
      return false;

   const bool g18 = dev->chip == AGX_CHIP_G18G;
   const bool native_agx3 = agx_chip_native_agx3(dev->chip);
   const bool g17_m5 = agx_chip_g17_m5(dev->chip);
   if (!g18 && !native_agx3 && !g17_m5)
      return true;

   if (!cs->bo || !cs->info.main_size || !cs->info.nr_gprs ||
       cs->info.nr_preamble_gprs ||
       (cs->info.rodata.size_16 && !cs->rodata_offset) ||
       (cs->info.printf_uniform &&
        cs->info.printf_uniform < AGX_ABI_AGX3_PRECOMP_ARG_HALFS) ||
       !cs->info.agx3_staging_known ||
       cs->entry_offset > UINT64_MAX - cs->ptr ||
       !cs->b.workgroup.x || !cs->b.workgroup.y || !cs->b.workgroup.z)
      return false;

   if (g18)
      return !cs->info.local_size ||
             (uint64_t)cs->info.local_size * 4 <= 32768;

   if (native_agx3)
      return true;

   return (!cs->info.agx3_tps_size ||
           (dev->params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES)) &&
          cs->info.local_size <= 32768;
}

/* G17G's and G18G's precompiled kernels, through the M5 compute profile the
 * API compute path uses (agx_g17p_append_compute): the graphics generator's
 * compute form, re-encoded as agx3d on G18G, and the captured M5 direct
 * launch record, one terminated dispatch per command.
 *
 * G18G carries its static TPS frame and a threadgroup-memory
 * size exactly as the API compute path declares it (Apple's G18 reduction
 * kernel, native kick 18), so a kernel that keeps cross-subgroup scratch in
 * threadgroup memory -- the prefix sums, the restart unroll -- launches the
 * same way there. G17G has no such witness and still refuses one, since it
 * would read and write memory it was never given. */
static void
agx_launch_precomp_m5(struct agx_batch *batch, struct agx_grid grid,
                      const struct agx_precompiled_shader *cs,
                      const void *args, size_t arg_size)
{
   /* Driver work, not an application dispatch (agx_batch_sibling_queue). */
   batch->compute_internal = true;

   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   const bool g18 = dev->chip == AGX_CHIP_G18G;

   if ((dev->debug & AGX_DBG_FAULTTRACE) &&
       (cs->info.nr_preamble_gprs || cs->info.agx3_tps_size ||
        cs->info.local_size || agx_is_indirect(grid)))
      fprintf(stderr,
              "M5_PRECOMP preamble_gprs=%u tps=%u local=%u indirect=%u "
              "wg=%ux%ux%u gprs=%u\n",
              cs->info.nr_preamble_gprs, cs->info.agx3_tps_size,
              cs->info.local_size, agx_is_indirect(grid), cs->b.workgroup.x,
              cs->b.workgroup.y, cs->b.workgroup.z, cs->info.nr_gprs);

   if (!agx_precomp_shader_launchable(dev, cs) ||
       !args || !arg_size || arg_size > UINT16_MAX * 2u ||
       /* G18G launches an indirect-local grid through Apple's six-word
        * record; the setup kernels write the six u32 it reads. */
       (agx_is_indirect(grid) &&
        (!g18 || grid.mode != AGX_CDM_MODE_INDIRECT_LOCAL || !grid.ptr ||
         grid.ptr >= (UINT64_C(1) << 48) || (grid.ptr & 3)))) {
      agx_native_descriptor_reject_batch(batch, "precompiled M5 compute program");
      return;
   }

   /* grid.count shares storage with grid.ptr. */
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   struct agx3_ioto_uniform_range ranges[2];
   unsigned range_count;
   if (!agx_precomp_agx3_uniform_ranges(batch, cs, args, arg_size, ranges,
                                        &range_count))
      return;

   struct agx_bo *program_bo = NULL;
   struct agx_ptr program = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
      AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   const uint64_t shader_va = cs->ptr + cs->entry_offset;
   const uint32_t scratch_bytes = MAX3(cs->info.scratch_size,
      cs->info.agx3_spectpr_scratch_size, cs->info.agx3_tps_size);
   const struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .program_va = program.gpu,
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = cs->info.main_size,
      .scratch_size = g18 ? scratch_bytes : 0,
      .nr_gprs = cs->info.nr_gprs,
      .compiler_d_valid = cs->info.agx3_staging_known,
      .compiler_d = (uint32_t)cs->info.agx3_staging_halfs * 2,
      /* The API compute path's footprint: AGX3C shared indices are still
       * over-strided 4x, so allocate the strided size. */
      .threadgroup_memory_size_B =
         cs->info.local_size
            ? MAX2(256u, ALIGN_POT(cs->info.local_size * 4, 64u))
            : 0,
   };
   struct agx3_g17p_native_compute_program_desc bind_desc = {
      .program_va = program.gpu,
      .shader_va = shader_va,
   };
   struct agx3_g17p_graphics_program generated;
   struct agx3_compute_program launch;
   if (!program.cpu || !program_bo ||
       !agx_va_in_usc_window(dev, program.gpu) ||
       !agx_va_in_usc_window(dev, shader_va) ||
       !agx3_cdm_direct_template_sources_compute(false, false, false,
                                                 &bind_desc.launch_sources) ||
       !(g18 ? agx3_g18g_compute_program_generate(&desc, &generated)
              : agx3_g17g_compute_program_generate(agx_ioto_chip(dev), &desc,
                                                    &generated)) ||
       !agx_g18g_reencode_program(dev, &generated, true, cs->info.nr_gprs) ||
       generated.length > AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE ||
       !agx3_g17g_native_compute_program_bind(&bind_desc, &launch)) {
      agx_native_descriptor_reject_batch(batch, "precompiled M5 compute Ioto");
      return;
   }
   memcpy(program.cpu, generated.bytes, generated.length);
   if (agx_bo_sync_for_device(program_bo, program.gpu - program_bo->va->addr,
                              generated.length)) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute upload");
      return;
   }

   const uint8_t *cdm_base_map =
      batch->cdm.bo ? agx_bo_map(batch->cdm.bo) : NULL;
   if (!cdm_base_map) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute launch");
      return;
   }
   const uint32_t dispatch_start = batch->cdm.current - cdm_base_map;
   struct agx3_compute_dispatch_sink sink = {
      .data = batch->cdm.current,
      .capacity = batch->cdm.end - batch->cdm.current,
   };
   bool launched;
   if (agx_is_indirect(grid)) {
      /* Apple never 1D-masks an indirect template; bit 23 marks threadgroup
       * memory as in the direct record. The six u32 the setup kernels write
       * carry this kernel's own workgroup as the local size. */
      if (cs->info.local_size)
         launch.launch_template.word0 |= UINT32_C(1) << 23;
      launched = agx3_g17g_native_indirect_compute_launch_append(
         &launch, grid.ptr, agx_m5_dispatch_barrier_flags(dev), &sink);
   } else {
      const struct agx3_compute_direct_grid direct = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {grid.count[0], grid.count[1], grid.count[2]},
         .local_size = {cs->b.workgroup.x, cs->b.workgroup.y,
                        cs->b.workgroup.z},
         .dimensions = grid.count[2] != 1 || cs->b.workgroup.z != 1 ? 3
                       : grid.count[1] != 1 || cs->b.workgroup.y != 1 ? 2 : 1,
      };
      /* A threadgroup-memory kernel launches as the API compute path
       * launches one on G18G: the 1D template normalization and word0 bit 23
       * of Apple's reduction kernel (native kick 18; without the bit the
       * reduction summed 16 of 64 lanes on hardware, Mesa165/run164). */
      if (g18 && cs->info.local_size) {
         if (direct.size[1] < 2 && direct.size[2] < 2)
            launch.launch_template.word0 &= UINT32_C(0xfff1ffff);
         launch.launch_template.word0 |= UINT32_C(1) << 23;
      }
      launched = agx3_g17g_native_compute_dispatch_append(
         &launch, &direct, agx_m5_dispatch_barrier_flags(dev), &sink);
   }
   /* Every M5-profile dispatch is its own command, so the barrier the
    * caller asked for is already implied; end it as the API path does. */
   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   if (!launched ||
       !agx3_cdm_stream_terminate_pack(dev->chip, terminate) ||
       sink.capacity - sink.length < sizeof(terminate)) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute launch");
      return;
   }
   memcpy(sink.data + sink.length, terminate, sizeof(terminate));
   sink.length += sizeof(terminate);
   batch->cdm.current += sink.length;

   /* As agx_g17p_append_compute: the M5 profile covers one dispatch per
    * command, and agx_batch_submit() sends each recorded extent alone. */
   if (!agx_record_cdm_dispatch(batch, dispatch_start,
                                batch->cdm.current - cdm_base_map,
                                grid)) {
      agx_native_descriptor_reject_batch(batch, "compute dispatch allocation");
      return;
   }
   if (g18) {
      struct agx_cdm_dispatch *dispatch = util_dynarray_element(
         &batch->cdm_dispatch, struct agx_cdm_dispatch,
         util_dynarray_num_elements(&batch->cdm_dispatch, struct agx_cdm_dispatch) - 1);
      dispatch->resources = (struct drm_asahi_g18_compute_resources){
         .version = 1,
         .flags = 1u | (agx_is_indirect(grid) ? 8u : 0u),
         .registers_16 = cs->info.nr_gprs,
         .scratch_bytes = MAX2(cs->info.scratch_size, cs->info.agx3_tps_size),
         .spectpr_scratch_bytes = cs->info.agx3_spectpr_scratch_size,
         .staging_16 = cs->info.agx3_staging_halfs,
         .local_bytes = cs->info.local_size,
         .imageblock_stride = cs->info.imageblock_stride,
         .local_size = {agx_is_indirect(grid) ? 0 : cs->b.workgroup.x,
                        agx_is_indirect(grid) ? 0 : cs->b.workgroup.y,
                        agx_is_indirect(grid) ? 0 : cs->b.workgroup.z},
      };
   }
   agx_batch_add_bo(batch, cs->bo);
}

/* The kernels of the GPU tessellation prepass, and of the geometry prepass
 * a tessellated draw feeds. Indirect patch draws are read back on the CPU,
 * so the indirect tessellation setup is never needed. */
static bool
agx_m5_precomp_admitted(enum libagx_program program, bool in_tess)
{
   switch (program) {
   case LIBAGX_TESS_ISOLINE_0:
   case LIBAGX_TESS_ISOLINE_1:
   case LIBAGX_TESS_TRI_0:
   case LIBAGX_TESS_TRI_1:
   case LIBAGX_TESS_QUAD_0:
   case LIBAGX_TESS_QUAD_1:
   case LIBAGX_PREFIX_SUM_TESS_0:
   case LIBAGX_PREFIX_SUM_TESS_1:
      return true;
   case LIBAGX_GS_SETUP_INDIRECT:
   case LIBAGX_PREFIX_SUM_GEOM:
      return in_tess;
   default:
      return false;
   }
}

/* A precompiled kernel on G17G, G17S and G17C, as hk_dispatch_precomp_m5()
 * launches it: the M5 compute program with the kernel's spill frame and
 * threadgroup memory, and the M5 launch record. Unlike hk, each dispatch is
 * its own command (agx_batch_submit()), as on the API compute path. */
static void
agx_launch_precomp_g17_m5(struct agx_batch *batch, struct agx_grid grid,
                          const struct agx_precompiled_shader *cs,
                          const void *args, size_t arg_size)
{
   /* Driver work, not an application dispatch (agx_batch_sibling_queue). */
   batch->compute_internal = true;

   struct agx_device *dev = agx_device(batch->ctx->base.screen);

   if (!agx_precomp_shader_launchable(dev, cs) ||
       !args || !arg_size || arg_size > UINT16_MAX * 2u ||
       (agx_is_indirect(grid) &&
        (!grid.ptr || grid.ptr >= (UINT64_C(1) << 48) || (grid.ptr & 3)))) {
      agx_native_descriptor_reject_batch(batch, "precompiled M5 compute program");
      return;
   }

   /* grid.count shares storage with grid.ptr. */
   if (!agx_is_indirect(grid) &&
       (!grid.count[0] || !grid.count[1] || !grid.count[2]))
      return;

   struct agx3_ioto_uniform_range ranges[2];
   unsigned range_count;
   if (!agx_precomp_agx3_uniform_ranges(batch, cs, args, arg_size, ranges,
                                        &range_count))
      return;

   struct agx_bo *program_bo = NULL;
   struct agx_ptr program = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE,
      AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   const uint64_t shader_va = cs->ptr + cs->entry_offset;
   const struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .program_va = program.gpu,
      .shader_va = shader_va,
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
      .shader_binary_size = cs->info.main_size,
      /* The tessellator spills (agx_launch_precomp's G17P path). */
      .scratch_size = cs->info.agx3_tps_size,
      .nr_gprs = cs->info.nr_gprs,
      .compiler_d_valid = cs->info.agx3_staging_known,
      .compiler_d = (uint32_t)cs->info.agx3_staging_halfs * 2,
      /* The prefix sums keep cross-subgroup scratch here; allocate the 4x
       * strided footprint, as the API and G18G paths do. */
      .threadgroup_memory_size_B =
         cs->info.local_size
            ? MAX2(256u, ALIGN_POT(cs->info.local_size * 4, 64u))
            : 0,
   };
   struct agx3_g17p_native_compute_program_desc bind_desc = {
      .program_va = program.gpu,
      .shader_va = shader_va,
   };
   struct agx3_g17p_graphics_program generated;
   struct agx3_compute_program launch;
   if (!program.cpu || !program_bo ||
       !agx_va_in_usc_window(dev, program.gpu) ||
       !agx_va_in_usc_window(dev, shader_va) ||
       !agx3_cdm_direct_template_sources_compute(false, false, false,
                                                 &bind_desc.launch_sources) ||
       !agx3_g17g_compute_program_generate(dev->chip, &desc, &generated) ||
       generated.length > AGX3_G17P_COMPUTE_PROGRAM_MAX_SIZE ||
       !agx3_m5_native_compute_program_bind(dev->chip, &bind_desc, &launch)) {
      agx_native_descriptor_reject_batch(batch, "precompiled M5 compute Ioto");
      return;
   }
   memcpy(program.cpu, generated.bytes, generated.length);
   if (agx_bo_sync_for_device(program_bo, program.gpu - program_bo->va->addr,
                              generated.length)) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute upload");
      return;
   }

   const uint8_t *cdm_base_map =
      batch->cdm.bo ? agx_bo_map(batch->cdm.bo) : NULL;
   if (!cdm_base_map) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute launch");
      return;
   }
   const uint32_t dispatch_start = batch->cdm.current - cdm_base_map;
   struct agx3_compute_dispatch_sink sink = {
      .data = batch->cdm.current,
      .capacity = batch->cdm.end - batch->cdm.current,
   };
   const uint32_t local[3] = {cs->b.workgroup.x, cs->b.workgroup.y,
                              cs->b.workgroup.z};
   const uint8_t flags = agx_m5_dispatch_barrier_flags(dev);
   bool launched;
   if (agx_is_indirect(grid)) {
      launched = agx3_m5_native_indirect_dispatch_append(
         &launch, grid.ptr,
         grid.mode == AGX_CDM_MODE_INDIRECT_GLOBAL ? local : NULL, flags,
         &sink);
   } else {
      const struct agx3_compute_direct_grid direct = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {grid.count[0], grid.count[1], grid.count[2]},
         .local_size = {local[0], local[1], local[2]},
         .dimensions = grid.count[2] != 1 || local[2] != 1 ? 3
                       : grid.count[1] != 1 || local[1] != 1 ? 2 : 1,
      };
      launched = agx3_m5_native_compute_dispatch_append(&launch, &direct,
                                                        flags, &sink);
   }

   uint8_t terminate[AGX3_CDM_STREAM_TERMINATE_SIZE];
   if (!launched ||
       !agx3_cdm_stream_terminate_pack(dev->chip, terminate) ||
       sink.capacity - sink.length < sizeof(terminate)) {
      agx_native_descriptor_reject_batch(batch, "precompiled compute launch");
      return;
   }
   memcpy(sink.data + sink.length, terminate, sizeof(terminate));
   sink.length += sizeof(terminate);
   batch->cdm.current += sink.length;

   if (!agx_record_cdm_dispatch(batch, dispatch_start,
                                batch->cdm.current - cdm_base_map,
                                grid)) {
      agx_native_descriptor_reject_batch(batch, "compute dispatch allocation");
      return;
   }

   /* The spill frame and threadgroup memory, as hk_m5_compute_resources()
    * lists them. The check above admits a spill frame on this feature. */
   if (dev->params.features & DRM_ASAHI_FEATURE_COMPUTE_RESOURCES) {
      const struct drm_asahi_compute_program profile = {
         .primary_bytes = desc.scratch_size,
         .threadgroup_bytes = desc.threadgroup_memory_size_B,
         .threads = local[0] * local[1] * local[2],
      };
      agx_batch_note_m5_compute_program(batch, &profile);
   }
   agx_batch_add_bo(batch, cs->bo);
}

bool
agx_precomp_launchable(struct agx_context *ctx, enum libagx_program program)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (!dev->libagx_programs || !dev->libagx_programs[program])
      return false;

   if (agx_chip_g17_m5(dev->chip) &&
       !(agx_m5_gpu_tess(ctx) &&
         agx_m5_precomp_admitted(program, ctx->in_tess)))
      return false;

   const struct agx_precompiled_shader *cs =
      agx_get_precompiled(&ctx->bg_eot, program);
   return agx_precomp_shader_launchable(dev, cs);
}

void
agx_launch_precomp(struct agx_batch *batch, struct agx_grid grid,
                   enum agx_barrier barrier, enum libagx_program program,
                   void *args, size_t arg_size)
{
   /* Driver work, not an application dispatch (agx_batch_sibling_queue). */
   batch->compute_internal = true;

   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   struct agx_precompiled_shader *cs =
      agx_get_precompiled(&batch->ctx->bg_eot, program);

   if (dev->chip == AGX_CHIP_G18G) {
      /* G18G's libagx table leaves out the kernels its encoders cannot take
       * yet (asahi_clc), so a precompiled kernel -- the tessellator, the
       * geometry and restart prefix sums, query copies -- can be absent
       * there. Refuse the batch rather than launch through a null program. */
      if (!cs) {
         agx_native_descriptor_reject_batch(batch,
                                              "a precompiled libagx program");
         return;
      }

      agx_launch_precomp_m5(batch, grid, cs, args, arg_size);
      return;
   }

   if (agx_chip_native_agx3(dev->chip)) {
      if (!agx_precomp_shader_launchable(dev, cs) ||
          !args || !arg_size || arg_size > UINT16_MAX * 2u ||
          (agx_is_indirect(grid) &&
           (grid.mode != AGX_CDM_MODE_INDIRECT_LOCAL || !grid.ptr ||
            grid.ptr >= (UINT64_C(1) << 48) || (grid.ptr & 3)))) {
         mesa_loge("asahi: refusing precompiled program %u on G17P: "
                   "grid=%u preamble_gprs=%u rodata=%u/%u printf=%u "
                   "staging_known=%u args=%zu",
                   (unsigned)program, (unsigned)grid.mode,
                   cs ? cs->info.nr_preamble_gprs : 0,
                   cs ? cs->info.rodata.size_16 : 0,
                   cs ? (unsigned)cs->rodata_offset : 0,
                   cs ? cs->info.printf_uniform : 0,
                   cs ? (unsigned)cs->info.agx3_staging_known : 0, arg_size);
         agx_native_descriptor_rejected(dev, "precompiled compute program");
      }

      /* Mode 2 reads the whole grid -- three global thread counts, then
       * three local sizes -- from six u32 in GPU memory, as
       * libagx_tess_setup_indirect() writes them for the tessellation
       * kernels. */
      const bool indirect_local = grid.mode == AGX_CDM_MODE_INDIRECT_LOCAL;
      if (!indirect_local &&
          (!grid.count[0] || !grid.count[1] || !grid.count[2]))
         return;

      /* Match the precompiler's rebased arguments and optional runtime printf
       * address. Pad packed argument structs for Ioto's whole-dword loads. */
      const size_t printf_offset = cs->info.printf_uniform
         ? (size_t)(cs->info.printf_uniform - AGX_ABI_AGX3_PRECOMP_ARG_HALFS) * 2
         : 0;
      const size_t upload_size = ALIGN_POT(
         cs->info.printf_uniform ? MAX2(arg_size, printf_offset + 8) : arg_size, 4);
      struct agx_ptr uploaded = agx_pool_alloc_aligned(&batch->pool, upload_size, 4);
      if (!uploaded.cpu || upload_size / 2 > UINT16_MAX)
         agx_native_descriptor_rejected(dev, "precompiled argument upload");
      memcpy(uploaded.cpu, args, arg_size);
      memset((uint8_t *)uploaded.cpu + arg_size, 0, upload_size - arg_size);
      if (cs->info.printf_uniform)
         memcpy((uint8_t *)uploaded.cpu + printf_offset,
                &dev->printf_buffer_address, sizeof(uint64_t));
      const uint64_t uploaded_data = uploaded.gpu;
      /* Above the halves the AGX3 hardware ABI owns, matching where the
       * precompiler now places these loads. */
      const struct agx3_ioto_uniform_range range = {
         .start_halfs = AGX_ABI_AGX3_PRECOMP_ARG_HALFS,
         .size_halfs = upload_size / 2,
         .address = uploaded_data,
      };

      /* Constants the compiler promoted out of the code are published as a
       * second range, exactly as the legacy USC path maps them with
       * agx_usc_immediates(). Their base is already in this ABI's numbering:
       * agx_opt_promote_constants() allocates above push_count, which starts
       * at the reserved preamble. Without this the kernel reads whatever the
       * uniform file happened to hold -- which is why the launch-shape check
       * refused a kernel carrying rodata rather than dispatching one. */
      struct agx3_ioto_uniform_range ranges[2] = {range};
      unsigned range_count = 1;
      if (cs->info.rodata.size_16) {
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = cs->info.rodata.base_uniform,
            .size_halfs = cs->info.rodata.size_16,
            .address = cs->ptr + cs->rodata_offset,
         };
      }
      struct agx3_cdm_direct_template_sources launch_sources;
      if (!uploaded_data ||
          !agx3_cdm_direct_template_sources_compute(false, false, false,
                                                     &launch_sources))
         agx_native_descriptor_rejected(dev, "precompiled compute resources");

      const struct agx3_g17p_compute_program_desc desc = {
         .shader_va = cs->ptr + cs->entry_offset,
         .uniform_ranges = ranges,
         .uniform_range_count = range_count,
         .shader_binary_size = cs->info.main_size,
         /* The kernel's thread-private spill frame, as hk declares it. The
          * software tessellator's emitting pass spills (68 bytes for
          * triangles); launched with no frame it never completes, which is
          * how every tessellated draw with a level above one used to time
          * out the compute queue. */
         .scratch_size = cs->info.agx3_tps_size,
         .nr_gprs = cs->info.nr_gprs,
         .compiler_d_valid = cs->info.agx3_staging_known,
         .compiler_d = (uint32_t)cs->info.agx3_staging_halfs * 2,
         .launch_sources = launch_sources,
         /* Precompiled kernels use threadgroup memory too -- the restart
          * unroll and the prefix sums keep their cross-subgroup scratch
          * there -- and the API compute path above declares it while this
          * one did not.  Leaving it zero does not fail the dispatch, so the
          * shortfall is silent.  Same rounding as that path. */
         .threadgroup_memory_size_B =
            cs->info.local_size
               ? MAX2(256u, ALIGN_POT(cs->info.local_size, 64u))
               : 0,
         .imageblock_stride_B = cs->info.imageblock_stride,
         /* The program announces how it is launched, as the API compute
          * path's does (agx_g17p_append_compute()). */
         .ldshdr_f1 = indirect_local
                         ? AGX3_G17P_INDIRECT_MAIN_LDSHDR_F1
                         : AGX_IOTO_G17P_COMPUTE_LDSHDR_F1_DIRECT_ADD3,
      };
      struct agx3_compute_program launch;
      if (!agx_g17p_compute_program_emit_cached(
             batch, &batch->ctx->g17p_precomp_templates[program],
             &batch->ctx->g17p_precomp_templates_generated, &desc, &launch))
         agx_native_descriptor_rejected(dev, "precompiled compute Ioto");

      const struct agx3_compute_direct_grid direct = {
         .source = AGX3_CDM_DIRECT_GRID_GLOBAL_THREADS,
         .size = {grid.count[0], grid.count[1], grid.count[2]},
         .local_size = {cs->b.workgroup.x, cs->b.workgroup.y,
                        cs->b.workgroup.z},
         .dimensions = grid.count[2] != 1 || cs->b.workgroup.z != 1
                          ? 3
                       : grid.count[1] != 1 || cs->b.workgroup.y != 1 ? 2 : 1,
      };
      struct agx3_compute_dispatch_sink sink = {
         .data = batch->cdm.current,
         .capacity = batch->cdm.end - batch->cdm.current,
      };
      /* Meta copies and fills feed later graphics and CPU transfers. Preserve
       * AGX_BARRIER_ALL's cache-visibility contract with Apple's grounded
       * strong post-dispatch barrier. Direct API compute remains on the
       * existing minimal flags-0 barrier. */
      const uint8_t barrier_flags =
         (barrier & AGX_BARRIER_ALL) ? 0x5 : 0;
      if (indirect_local
             ? !agx3_g17p_native_indirect_compute_launch_append(
                  &launch, grid.ptr, barrier_flags, &sink)
             : !agx3_g17p_native_compute_dispatch_append_with_barrier(
                  &launch, &direct, barrier_flags, &sink))
         agx_native_descriptor_rejected(dev, "precompiled compute launch");
      batch->cdm.current += sink.length;
      agx_batch_add_bo(batch, cs->bo);
      return;
   }

   /* M5 launches the tessellation and geometry kernels (as hk does) and
    * refuses the batch for every other one, instead of dereferencing a
    * missing program on the AGX2 path below. */
   if (agx_chip_g17_m5(dev->chip)) {
      if (cs && agx_m5_gpu_tess(batch->ctx) &&
          agx_m5_precomp_admitted(program, batch->ctx->in_tess))
         agx_launch_precomp_g17_m5(batch, grid, cs, args, arg_size);
      else
         agx_native_descriptor_reject_batch(batch, "precompiled compute program");
      return;
   }

   struct agx_ptr t =
      agx_pool_alloc_aligned(&batch->pipeline_pool, agx_usc_size(15), 64);

   uint64_t uploaded_data =
      agx_pool_upload_aligned(&batch->pool, args, arg_size, 4);

   uint32_t usc = agx_usc_addr(dev, t.gpu);
   agx_usc_words_precomp(t.cpu, &cs->b, uploaded_data, arg_size);

   agx_batch_add_bo(batch, cs->bo);
   agx_launch_internal(batch, grid, cs->b.workgroup, cs->b.launch,
                       MESA_SHADER_COMPUTE, usc);
}

/* G18G: a single-sample 32x32 tile of sixteen-byte samples, which every
 * RGBA32 and RGB32 (RGBX32 physical) colour attachment needs. Nothing new is
 * encoded for it:
 *  - SPECLM byte 7 is the sample size in bytes (M5 renders publish ...1808
 *    for one RGBA8 attachment and ...1820 for eight, nothing else moving;
 *    see g17g_graphics_program_generate), and the agx3d re-encoding carries
 *    the field through unchanged;
 *  - the kernel derives the +0x1fb0 sample size from the utile bytes
 *    (tib_blocks << 1 = 16 at 32x32);
 *  - the end-of-tile program stores the tile through the PBE, so it compiles
 *    to exactly the witnessed eight-byte shape (68 bytes, 8 GPRs, 8 staging
 *    halves), which the ABI check below still enforces. */
static bool
agx_g18g_wide_sample_tile(const struct agx_device *dev,
                          const struct agx_tilebuffer_layout *tib)
{
   return dev->chip == AGX_CHIP_G18G && tib->sample_size_B == 16 &&
          ((tib->nr_samples == 1 && tib->tile_size == 32 * 32) ||
           agx_g18_msaa4_tile_layout(dev, tib));
}

static struct asahi_bg_eot
agx_build_g17g_bg_eot(struct agx_batch *batch, struct agx_bg_eot_key *key,
                      bool store)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   /* A render with no colour attachment still runs a background and an
    * end-of-tile program per tile; both simply have nothing to move, because
    * the tilebuffer holds no colour. Apple's own colour-free render carries
    * exactly that shape -- a real program pair whose resource specification
    * is zero in both slots, against 0xffff800000000000 for a colour
    * background (registers 0x15369/0x15379 of the depth-only capture in
    * logs/m5-state-matrix-c-native-hv-20260914-055541).
    */
   const bool no_color = batch->key.nr_cbufs == 0;
   const unsigned word_mask = dev->chip == AGX_CHIP_G18G
                                ? agx_g18g_word_mrt_mask(&key->tib) |
                                     agx_g18g_raw32_mrt_mask(&key->tib) |
                                     agx_g18g_native_mixed_mask(&key->tib) : 0;
   const unsigned ms_mask = dev->chip == AGX_CHIP_G18G
                              ? agx_g18g_ms_mrt_mask(&key->tib) : 0;
   const unsigned mrt_mask = word_mask | ms_mask;
   const unsigned target_mask = no_color ? 0 : mrt_mask ? mrt_mask : 1;
   const unsigned targets = util_bitcount(target_mask);
   const unsigned slot_count = util_last_bit(target_mask);
   const unsigned first_rt = target_mask ? ffs(target_mask) - 1 : 0;
   const struct pipe_surface *surf = &batch->key.cbufs[first_rt];
   /* An attachment without valid contents may be initialized to zero. Keep
    * LOAD distinct: partial renders must preserve the previous tile contents.
    */
   bool undefined[PIPE_MAX_COLOR_BUFS] = {0};
   bool has_clear = false, has_load = false;
   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (!(target_mask & BITFIELD_BIT(rt))) {
         if (key->op[rt] != AGX_BG_EOT_NONE || batch->key.cbufs[rt].texture)
            return agx_reject_g17g_tile(batch, "M5 tile inactive attachment");
         continue;
      }
      if (!batch->key.cbufs[rt].texture)
         return agx_reject_g17g_tile(batch, "M5 tile missing attachment");
      undefined[rt] = !store && key->op[rt] == AGX_BG_EOT_NONE;
      if (undefined[rt])
         key->op[rt] = AGX_BG_CLEAR;
      if (store ? key->op[rt] != AGX_EOT_STORE
                : (key->op[rt] != AGX_BG_CLEAR && key->op[rt] != AGX_BG_LOAD))
         return agx_reject_g17g_tile(batch, "M5 tile attachment operation");
      has_clear |= key->op[rt] == AGX_BG_CLEAR;
      has_load |= key->op[rt] == AGX_BG_LOAD;
   }
   /* Native1130 (run 1139): every one-batch workload of capture 1130 and
    * every clear-only batch of capture 1121 clears all of its attachments,
    * and every load workload of capture 1130 reloads all of them; a
    * clear/reload mix of a mixed-format layout has no witness. */
   if (dev->chip == AGX_CHIP_G18G && !store &&
       agx_g18g_native_mixed_mask(&key->tib) && has_clear == has_load)
      return agx_reject_g17g_tile(batch, "M5 mixed-format background operation");
   const enum agx_bg_eot_op op = key->op[first_rt];
   /* The final direct resolve writes its single-sample destination. All
    * other stores preserve the actual multisample attachment, including
    * partial renders; their backgrounds reload each stored sample. */
   struct pipe_surface resolve_surf = {0};
   const bool has_resolve =
      agx_g17p_direct_resolve_surface(batch, &resolve_surf);
   const bool msaa4 =
      agx_g18_msaa4_tile_layout(dev, &key->tib) &&
      agx_g18_msaa4_zs_surface(dev, &batch->key.zsbuf) &&
      (no_color || agx_g18_msaa_color_surfaces(batch)) &&
      (!batch->key.resolve || has_resolve) &&
      (no_color || !store || key->multisample_store || has_resolve);
   const bool resolve_store =
      msaa4 && !no_color && store && !key->multisample_store;
   if ((no_color ? surf->texture || op != AGX_BG_EOT_NONE ||
                      key->tib.sample_size_B != (msaa4 ? 8 : 0)
                 : !surf->texture ||
                      (key->tib.sample_size_B != 8 && !mrt_mask &&
                       !agx_g18g_wide_sample_tile(dev, &key->tib)) ||
                      (store ? op != AGX_EOT_STORE
                             : (op != AGX_BG_CLEAR && op != AGX_BG_LOAD))) ||
       (key->tib.layered &&
         (dev->chip != AGX_CHIP_G18G ||
          (targets != 1 && !no_color && !word_mask) ||
         (key->tib.nr_samples != 1 && !msaa4) || batch->key.resolve)) ||
       (key->tib.nr_samples != 1 && !msaa4) ||
       key->tib.tile_size != (msaa4 && key->tib.nr_samples == 4 ? 32 * 16 : 32 * 32) ||
       agx_tilebuffer_spills(&key->tib) ||
       agx_tilebuffer_offset_B(&key->tib, first_rt) ||
       (batch->key.resolve && !msaa4)) {
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         fprintf(stderr,
                 "M5_TILE_REJECT store=%u op=%u texture=%u layered=%u "
                 "samples=%u sample_size=%u tile=%u spills=%u offset=%u "
                 "resolve=%u format=%s\n",
                 store, op, !!surf->texture, key->tib.layered,
                 key->tib.nr_samples, key->tib.sample_size_B,
                 key->tib.tile_size, agx_tilebuffer_spills(&key->tib),
                 agx_tilebuffer_offset_B(&key->tib, first_rt), !!batch->key.resolve,
                 util_format_name(surf->format));
      }
      return agx_reject_g17g_tile(batch, "M5 tile layout or operation");
   }
   /* Reload pointers use u0..7. Mixed clear values must not overlap them. */
   const unsigned clear_base = has_load ? 8 : 4;
   if (has_clear)
      key->reserved_preamble = clear_base + 8 * slot_count;

   struct agx_bg_eot_key shader_key = *key;
   if (store && mrt_mask) {
      /* The native specialized EOT addresses dense PBEs. Compact its key
       * and the table below together, retaining each logical tile offset. */
      if (agx_tilebuffer_compact(&key->tib, &shader_key.tib, NULL) != targets)
         return agx_reject_g17g_tile(batch, "M5 EOT target compaction");
      for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt)
         shader_key.op[rt] = rt < targets ? AGX_EOT_STORE : AGX_BG_EOT_NONE;
   }
   struct agx_bg_eot_shader *shader =
      agx_get_bg_eot_shader(&ctx->bg_eot, &shader_key);
   if (!shader || !shader->bo || !shader->bo->va || !shader->ptr ||
       !shader->info.binary_size || shader->info.has_preamble ||
       shader->info.rodata.size_16 || !shader->info.agx3_staging_known) {
      mesa_loge("asahi: native tile shader refused: op=%u format=%s "
                "(%s%s%s%s)",
                op, util_format_name(surf->format),
                !shader || !shader->info.binary_size ? "no binary " : "",
                shader && shader->info.has_preamble ? "preamble " : "",
                shader && shader->info.rodata.size_16 ? "rodata " : "",
                shader && !shader->info.agx3_staging_known ? "staging-unknown" : "");
      return agx_reject_g17g_tile(batch, "M5 tile shader");
   }

   struct agx3_ioto_uniform_range ranges[3];
   unsigned range_count = 0;
   if (no_color) {
      /* Nothing is loaded, cleared or stored, so the program binds nothing.
       * The shader still has to be a real, complete one -- the hardware runs
       * it for every tile -- so hold it to the same no-preamble, no-texture
       * shape the other two operations are held to. */
      if (shader->info.push_count || shader->info.uses_txf)
         return agx_reject_g17g_tile(batch, "M5 colour-free tile shader ABI");
      range_count = 0;
   } else if (store) {
      /* Match the M5 native EOT ABI already used by Honeykrisp: the PBE
       * table pointer occupies u0..3 and the tile offset occupies u4..5.
       */
      /* Complete native G18 programs, including coordinates and all stores. */
      static const uint16_t g18_sizes[] = {0, 68, 106, 136, 166, 200, 234, 268, 302};
      static const uint16_t g18_array_sizes[] =
         {0, 88, 126, 156, 186, 220, 254, 288, 322};
      const unsigned eot_size =
         dev->chip == AGX_CHIP_G18G
            ? (key->tib.layered ? g18_array_sizes[targets] : g18_sizes[targets])
            : 78;
      if (shader->info.binary_size != eot_size || shader->info.nr_gprs != 8 ||
          shader->info.agx3_staging_halfs != 8 * targets) {
         if (dev->debug & AGX_DBG_FAULTTRACE)
            fprintf(stderr, "M5_EOT_ABI size=%u gprs=%u staging=%u\n",
                    shader->info.binary_size, shader->info.nr_gprs,
                    shader->info.agx3_staging_halfs);
         return agx_reject_g17g_tile(batch, "M5 EOT shader ABI");
      }
      struct agx_ptr pbe = agx_pool_alloc_aligned(
         &batch->pool, targets * sizeof(struct agx_gallium_pbe_descriptor), 32);
      /* The fragment and background shaders already encode sRGB into the
       * packed imageblock. Store those bytes without a second conversion. */
      unsigned pbe_index = 0;
      uint64_t arguments[1 + PIPE_MAX_COLOR_BUFS] = {pbe.gpu};
      /* Native1130 (run 1139): a mixed-format end of tile stores a packed
       * R11G11B10F word with IMGWBLK f29 = 3, the class of tiles held in the
       * attachment's own encoding (RGBA8 Unorm, RGB10A2 Unorm), so its PBE
       * names the attachment format, as RGB10A2's does. Other G18 layouts keep
       * the raw-word R32_UINT PBE that pairs with their W32 store. */
      const bool native_mixed =
         dev->chip == AGX_CHIP_G18G && agx_g18g_native_mixed_mask(&key->tib);
      u_foreach_bit(rt, target_mask) {
         struct pipe_image_view view = image_view_for_surface(
            resolve_store ? &resolve_surf : &batch->key.cbufs[rt]);
         view.format = dev->chip == AGX_CHIP_G18G && !native_mixed &&
                             view.format == PIPE_FORMAT_R11G11B10_FLOAT
                          ? PIPE_FORMAT_R32_UINT
                          : util_format_linear(view.format);
         if (!agx_batch_upload_pbe(
                batch, (struct agx_gallium_pbe_descriptor *)pbe.cpu + pbe_index,
                &view, true, true, false,
                batch->feedback & (PIPE_CLEAR_COLOR0 << rt)))
            return agx_reject_g17g_tile(batch, "M5 EOT PBE");
         arguments[1 + pbe_index++] = agx_tilebuffer_offset_B(&key->tib, rt);
      }
      /* Native offset arguments are eight bytes apart: RT0 is u4 and RT1
       * is u8, with unused padding at u6. The BTNF metadata's argument
       * offsets 0/2 name dwords, not adjacent CProg half pairs. */
      const uint64_t table = agx_pool_upload_aligned(
         &batch->pool, arguments, 8 + 8 * targets, 16);
      ranges[range_count++] =
         (struct agx3_ioto_uniform_range){0, 2 + 4 * targets, table};
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         fprintf(stderr, "M5_EOT_PBE address=%#llx descriptor=",
                 (unsigned long long)pbe.gpu);
         for (unsigned i = 0; i < sizeof(struct agx_gallium_pbe_descriptor); ++i)
            fprintf(stderr, "%02x", ((const uint8_t *)pbe.cpu)[i]);
         fputc('\n', stderr);
      }
   } else {
      if (shader->info.push_count != key->reserved_preamble ||
          !!shader->info.uses_txf != has_load)
         return agx_reject_g17g_tile(batch, "M5 background shader ABI");
      if (has_load) {
         /* Background NIR names texture indices 0, 2, ... for RT0, RT1, ... */
         const unsigned texture_count = 2 * slot_count - 1;
         struct agx_ptr texture = agx_pool_alloc_aligned(
            &batch->pool,
            texture_count * sizeof(struct agx_gallium_texture_descriptor), 32);
         memset(texture.cpu, 0,
                texture_count * sizeof(struct agx_gallium_texture_descriptor));
         for (unsigned rt = 0; rt < slot_count; ++rt) {
            if (key->op[rt] != AGX_BG_LOAD)
               continue;
            const struct pipe_surface *target = &batch->key.cbufs[rt];
            struct pipe_sampler_view sampler_view =
               sampler_view_for_surface(target);
            if (!agx_pack_texture(
                   AGX_DESCRIPTOR_GENERATION_10,
                   (struct agx_gallium_texture_descriptor *)texture.cpu +
                      2 * rt,
                   agx_resource(target->texture), target->format,
                   &sampler_view))
               return agx_reject_g17g_tile(batch, "M5 reload texture");
         }
         struct agx_gallium_sampler_descriptor samplers[2] = {0};
         /* Same Gen5 texel-fetch sampler as hk_pack_native_txf_sampler(G17G).
          * Gen5's comparison mode bit is required even for a non-shadow fetch.
          */
         agx_pack(samplers[0].opaque, SAMPLER_GEN5, cfg) {
            cfg.minimum_lod = 0.0f;
            cfg.maximum_lod = INFINITY;
            cfg.maximum_anisotropy = 1;
            cfg.mip_filter = AGX_MIP_FILTER_NEAREST;
            cfg.wrap_s = cfg.wrap_t = cfg.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
            cfg.compare_mode = true;
            cfg.compare_func = AGX_COMPARE_FUNC_LEQUAL;
            cfg.border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
         }
         /* Native 2D-MS reads name sampler1. Ordinary fetches name sampler0.
          * The background owns a separate heap from application samplers;
          * provide both entries before its four per-sample reloads execute. */
         samplers[1] = samplers[0];
         const unsigned sampler_count = msaa4 ? 2 : 1;
         const uint64_t sampler_va = agx_pool_upload_aligned(
            &batch->pool, samplers, sampler_count * sizeof(samplers[0]), 32);
         const uint64_t pointers[2] = {texture.gpu, sampler_va};
         const uint64_t table = agx_pool_upload_aligned(&batch->pool, pointers,
                                                        sizeof(pointers), 8);
         ranges[range_count++] = (struct agx3_ioto_uniform_range){0, 4, table};
         ranges[range_count++] =
            (struct agx3_ioto_uniform_range){4, 4, table + 8};
      }
      if (has_clear) {
         union pipe_color_union colors[PIPE_MAX_COLOR_BUFS] = {0};
         for (unsigned rt = 0; rt < slot_count; ++rt) {
            if (key->op[rt] == AGX_BG_CLEAR && !undefined[rt])
               colors[rt] = batch->clear_color[rt];
         }
         const uint64_t clear = agx_pool_upload_aligned(
            &batch->pool, colors, slot_count * sizeof(colors[0]), 16);
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            clear_base, 8 * slot_count, clear};
         if (dev->debug & AGX_DBG_FAULTTRACE) {
            u_foreach_bit(rt, target_mask) {
               if (key->op[rt] != AGX_BG_CLEAR)
                  continue;
               const union pipe_color_union *color = &colors[rt];
               fprintf(stderr,
                       "M5_CLEAR rt=%u undefined=%u rgba=%g,%g,%g,%g format=%s\n",
                       rt, undefined[rt], color->f[0], color->f[1], color->f[2],
                       color->f[3], util_format_name(batch->key.cbufs[rt].format));
            }
         }
      }
   }

   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(agx_ioto_chip(dev), ranges,
                                         range_count, &capacity))
      return agx_reject_g17g_tile(batch, "M5 tile program size");
   capacity = agx_ioto_capacity(dev, capacity);
   struct agx_bo *program_bo = NULL;
   struct agx_ptr ptr = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   if (!ptr.cpu || !program_bo || !agx_va_in_usc_window(dev, ptr.gpu) ||
       !agx_va_in_usc_window(dev, shader->ptr + shader->info.main_offset))
      return agx_reject_g17g_tile(batch, "M5 tile program allocation");
   const struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .program_va = ptr.gpu,
      .shader_va = shader->ptr + shader->info.main_offset,
      .shader_binary_size = shader->info.binary_size,
      .scratch_size = shader->info.scratch_size,
      .nr_gprs = shader->info.nr_gprs,
      .compiler_d_valid = shader->info.agx3_staging_known,
      .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
      .fragment = true,
      .imageblock = {key->tib.sample_size_B, key->tib.nr_samples, key->tib.tile_size},
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
   };
   struct agx3_g17p_graphics_program program;
   if (!agx3_g17g_tile_program_generate(dev->chip, &desc, store,
                                        &program) ||
       !agx_g18g_reencode_program(dev, &program, false,
                                  shader->info.nr_gprs) ||
       program.length > capacity)
      return agx_reject_g17g_tile(batch, "M5 tile Ioto program");
   if (dev->chip == AGX_CHIP_G18G) {
      agx_g18_note_tile(batch, store, shader->info.nr_gprs,
                        shader->info.agx3_staging_halfs);
   }
   memcpy(ptr.cpu, program.bytes, program.length);
   if (agx_bo_sync_for_device(program_bo, ptr.gpu - program_bo->va->addr,
                              program.length))
      return agx_reject_g17g_tile(batch, "M5 tile program upload");
   agx_batch_add_bo(batch, shader->bo);
   if (dev->debug & AGX_DBG_FAULTTRACE) {
      fprintf(stderr, "M5_TILE op=%u program=%#llx shader=%#llx size=%u gprs=%u staging=%u ranges=%u\n",
              op, (unsigned long long)ptr.gpu,
              (unsigned long long)desc.shader_va, shader->info.binary_size,
              shader->info.nr_gprs, shader->info.agx3_staging_halfs, range_count);
      for (unsigned i = 0; i < range_count; ++i)
         fprintf(stderr, "M5_TILE_RANGE start=%u halfs=%u address=%#llx\n",
                 ranges[i].start_halfs, ranges[i].size_halfs,
                 (unsigned long long)ranges[i].address);
      fprintf(stderr, "M5_TILE_IOTO data=");
      for (unsigned i = 0; i < program.length; ++i)
         fprintf(stderr, "%02x", program.bytes[i]);
      fputc('\n', stderr);
   }
   return (struct asahi_bg_eot){
      .usc = agx_usc_addr(dev, ptr.gpu),
      .rsrc_spec = store || no_color ? 0 : UINT64_C(0xffff800000000040),
   };
}

/*
 * Background or end-of-tile program for a G17P multi-target layout that the
 * dedicated RT0/dual/wide programs cannot express: mixed formats, or targets
 * that do not share one operation. Only the targets the operation touches are
 * numbered, each keeping its tile offset; reloads come before clears, because
 * a reload reads its texture and sampler pointers from uniform halves 0..7,
 * which a target-zero clear value would otherwise overlap. This is the
 * Gallium form of hk_build_g17p_mixed_bg_eot().
 */
static struct asahi_bg_eot
agx_g17p_build_mixed_bg_eot(struct agx_batch *batch,
                            const struct agx_bg_eot_key *framebuffer_key,
                            bool store)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);
   const unsigned descriptor_size =
      agx_gallium_texture_descriptor_size(descriptor_generation);
   const unsigned samples = MAX2(framebuffer_key->tib.nr_samples, 1);

   /* Multisample targets are stored unresolved; a resolve or a
    * single-sample WSI reload needs the single-target programs. */
   if (samples > 1 && (batch->key.resolve || framebuffer_key->single_sample_reload))
      agx_native_descriptor_rejected(dev, "multisample mixed MRT resolve");
   if (descriptor_size != AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE ||
       agx_gallium_pbe_descriptor_size(descriptor_generation) !=
          AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE)
      agx_native_descriptor_rejected(dev, "mixed MRT descriptor geometry");

   struct agx_bg_eot_key key = *framebuffer_key;
   memset(key.tib.logical_format, 0, sizeof(key.tib.logical_format));
   memset(key.tib.spilled, 0, sizeof(key.tib.spilled));
   memset(key.tib._offset_B, 0, sizeof(key.tib._offset_B));
   memset(key.op, 0, sizeof(key.op));
   key.mixed_mrt = true;
   key.single_sample_reload = false;
   key.multisample_store = store && samples > 1;
   key.reserved_preamble = 0;

   static const enum agx_bg_eot_op order[] = {AGX_BG_LOAD, AGX_BG_CLEAR,
                                              AGX_EOT_STORE};
   const struct pipe_surface *surfs[PIPE_MAX_COLOR_BUFS];
   unsigned rts[PIPE_MAX_COLOR_BUFS];
   unsigned count = 0, reloads = 0;
   for (unsigned i = 0; i < ARRAY_SIZE(order); ++i) {
      for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
         if (framebuffer_key->op[rt] != order[i])
            continue;

         const struct pipe_surface *surf = &batch->key.cbufs[rt];
         if (!surf->texture ||
             util_res_sample_count(surf->texture) != samples)
            agx_native_descriptor_rejected(dev, "mixed MRT attachment");

         key.tib.logical_format[count] =
            framebuffer_key->tib.logical_format[rt];
         key.tib._offset_B[count] = framebuffer_key->tib._offset_B[rt];
         key.op[count] = order[i];
         rts[count] = rt;
         surfs[count++] = surf;
         reloads += order[i] == AGX_BG_LOAD;
      }
   }
   assert(count > 0 && "uniform layouts take the dedicated programs");

   const bool clears = !store && count > reloads;
   if (clears)
      key.reserved_preamble = 4 + 8 * count;

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&ctx->bg_eot, &key);
   if (!shader || !shader->bo || !shader->bo->va || !shader->ptr ||
       !shader->info.binary_size ||
       shader->info.main_offset >= shader->info.binary_size ||
       shader->info.main_size >
          shader->info.binary_size - shader->info.main_offset ||
       shader->info.has_preamble || shader->info.rodata.size_16 ||
       !shader->info.nr_gprs ||
       (!store && (shader->info.push_count != key.reserved_preamble ||
                   shader->info.uses_txf != (reloads > 0)))) {
      mesa_loge("asahi: G17P mixed MRT %s shader is unproven: push=%u "
                "want=%u txf=%u count=%u reloads=%u samples=%u",
                store ? "end-of-tile" : "background",
                shader ? shader->info.push_count : 0, key.reserved_preamble,
                shader ? shader->info.uses_txf : 0, count, reloads, samples);
      agx_native_descriptor_rejected(dev, "mixed MRT shader");
   }
   agx_batch_add_bo(batch, shader->bo);

   const enum agx3_g17p_rt0_bg_op bg_op =
      reloads == count ? AGX3_G17P_RT0_BG_RELOAD
      : reloads        ? AGX3_G17P_RT0_BG_CLEAR_RELOAD
                       : AGX3_G17P_RT0_BG_CLEAR;
   const size_t capacity =
      store       ? (count > 4 ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                               : AGX_IOTO_G17P_QUAD_EOT_PROGRAM_MAX_SIZE)
      : count > 4 ? AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE
      : bg_op == AGX3_G17P_RT0_BG_CLEAR_RELOAD
         ? AGX_IOTO_G17P_CLEAR_RELOAD_PROGRAM_MAX_SIZE
         : AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
   struct agx_bo *program_bo = NULL;
   struct agx_ptr program_ptr = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   if (!program_ptr.cpu || !program_bo ||
       !agx_va_in_usc_window(dev, program_ptr.gpu) ||
       !agx_va_in_usc_window(dev, shader->ptr + shader->info.main_offset))
      agx_native_descriptor_rejected(dev, "mixed MRT program allocation");

   const struct agx_ioto_g17p_imageblock imageblock = {
      .sample_size_B = key.tib.sample_size_B,
      .nr_samples = key.tib.nr_samples,
      .tile_size_px = key.tib.tile_size,
   };
   size_t program_length;
   uint64_t resource_spec;
   if (store) {
      struct agx_ptr pbe =
         agx_pool_alloc_aligned(&batch->pool, descriptor_size * count, 32);
      for (unsigned i = 0; i < count; ++i) {
         struct pipe_image_view view = image_view_for_surface(surfs[i]);
         view.format = agx_g17p_rt0_pbe_format(view.format);
         const bool no_compress =
            batch->feedback & (PIPE_CLEAR_COLOR0 << rts[i]);
         if (!agx_batch_upload_pbe(
                batch, agx_pbe_descriptor_at(pbe.cpu, descriptor_size, i),
                &view, true, true, false, no_compress))
            return agx_reject_g17g_tile(batch, "mixed MRT end-of-tile PBE");
      }

      const struct agx3_g17p_rt0_eot_program_desc desc = {
         .program_va = program_ptr.gpu,
         .shader_va = shader->ptr + shader->info.main_offset,
         .pbe_argument_va = pbe.gpu,
         .imageblock_offset_B = key.tib._offset_B[0],
         .attachment_count = count,
         .imageblock_offset1_B = key.tib._offset_B[1],
         .imageblock_offset2_B = key.tib._offset_B[2],
         .imageblock_offset3_B = key.tib._offset_B[3],
         .imageblock_offset4_B = key.tib._offset_B[4],
         .imageblock_offset5_B = key.tib._offset_B[5],
         .imageblock_offset6_B = key.tib._offset_B[6],
         .imageblock_offset7_B = key.tib._offset_B[7],
         .mixed_mrt = count > 1,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_known
                             ? shader->info.agx3_staging_halfs
                             : 0,
         .imageblock = imageblock,
      };
      struct agx3_g17p_rt0_eot_program native;
      if (!agx3_g17p_rt0_eot_program_generate(dev->chip, &desc, &native) ||
          native.length > capacity)
         agx_native_descriptor_rejected(dev, "mixed MRT end-of-tile program");
      memcpy(program_ptr.cpu, native.bytes, native.length);
      program_length = native.length;
      resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
   } else {
      struct agx3_g17p_rt0_bg_program_desc desc = {
         .program_va = program_ptr.gpu,
         .shader_va = shader->ptr + shader->info.main_offset,
         .shader_binary_size = shader->info.binary_size,
         .scratch_size = shader->info.scratch_size,
         .nr_gprs = shader->info.nr_gprs,
         .staging_halfs = shader->info.agx3_staging_halfs,
         .clear_push_halfs = shader->info.push_count,
         .clear_start_halfs = 4 + 8 * reloads,
         .staging_known = shader->info.agx3_staging_known,
         .op = bg_op,
         .imageblock = imageblock,
      };

      if (clears) {
         union pipe_color_union clear_values[PIPE_MAX_COLOR_BUFS];
         for (unsigned i = reloads; i < count; ++i) {
            if (!batch->uploaded_clear_color[rts[i]])
               agx_native_descriptor_rejected(dev, "mixed MRT clear value");
            clear_values[i - reloads] = batch->clear_color[rts[i]];
         }
         desc.clear_value_va = agx_pool_upload_aligned(
            &batch->pool, clear_values,
            sizeof(clear_values[0]) * (count - reloads), 16);
         if (!desc.clear_value_va)
            agx_native_descriptor_rejected(dev, "mixed MRT clear upload");
      }

      if (reloads) {
         /* Reload n reads native texture index 2n; the odd entries stay
          * zero. */
         const size_t texture_size = descriptor_size * (2 * reloads - 1);
         struct agx_ptr textures =
            agx_pool_alloc_aligned(&batch->pool, texture_size, 64);
         memset(textures.cpu, 0, texture_size);
         for (unsigned i = 0; i < reloads; ++i) {
            struct pipe_sampler_view sampler_view =
               sampler_view_for_surface(surfs[i]);
            if (!agx_pack_texture(
                   descriptor_generation,
                   agx_texture_descriptor_at(textures.cpu, descriptor_size,
                                             2 * i),
                   agx_resource(surfs[i]->texture), surfs[i]->format,
                   &sampler_view))
               return agx_reject_g17g_tile(batch, "mixed MRT texture");
         }

         const unsigned sampler_size =
            agx_chip_sampler_heap_element_size(dev->chip);
         if (sampler_size != AGX_GALLIUM_SAMPLER_AGX3_SIZE)
            agx_native_descriptor_rejected(dev, "mixed MRT sampler geometry");
         struct agx_ptr sampler =
            agx_pool_alloc_aligned(&batch->pool, sampler_size, 64);
         if (!agx_gallium_pack_g17p_txf_sampler(dev->chip, sampler.cpu))
            agx_native_descriptor_rejected(dev, "mixed MRT sampler");

         const uint64_t pointers[2] = {textures.gpu, sampler.gpu};
         const uint64_t table = agx_pool_upload_aligned(
            &batch->pool, pointers, sizeof(pointers), 8);
         desc.texture_pointer_va = table;
         desc.sampler_pointer_va = table + sizeof(uint64_t);
      }

      struct agx3_g17p_rt0_bg_program native;
      if (!agx3_g17p_rt0_bg_program_generate(dev->chip, &desc, &native) ||
          native.length > capacity ||
          !agx3_g17p_rt0_bg_resource_spec(dev->chip, bg_op, &resource_spec))
         agx_native_descriptor_rejected(dev, "mixed MRT background program");
      memcpy(program_ptr.cpu, native.bytes, native.length);
      program_length = native.length;
   }

   agx_bo_assert_range(program_bo, program_ptr.gpu - program_bo->va->addr,
                       program_length);

   return (struct asahi_bg_eot){
      .usc = agx_usc_addr(dev, program_ptr.gpu),
      .rsrc_spec = resource_spec,
   };
}

/* Publish an M5 background or end-of-tile program: the Ioto program that
 * launches shader with the given uniform ranges over the batch's tile. */
static struct asahi_bg_eot
agx_m5_publish_tile_program(struct agx_batch *batch,
                            const struct agx_bg_eot_key *key,
                            struct agx_bg_eot_shader *shader,
                            const struct agx3_ioto_uniform_range *ranges,
                            unsigned range_count, bool store,
                            uint64_t rsrc_spec)
{
   struct agx_device *dev = agx_device(batch->ctx->base.screen);

   size_t capacity;
   if (!agx3_g17g_graphics_program_bound(dev->chip, ranges, range_count, &capacity))
      return agx_reject_g17g_tile(batch, "M5 tile program size");
   struct agx_bo *program_bo = NULL;
   struct agx_ptr ptr = agx_pool_alloc_aligned_with_bo(
      &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
   if (!ptr.cpu || !program_bo || !agx_va_in_usc_window(dev, ptr.gpu) ||
       !agx_va_in_usc_window(dev, shader->ptr + shader->info.main_offset))
      return agx_reject_g17g_tile(batch, "M5 tile program allocation");
   const struct agx3_g17p_graphics_program_desc desc = {
      .usc_clusters = dev->params.num_clusters_total,
      .program_va = ptr.gpu,
      .shader_va = shader->ptr + shader->info.main_offset,
      .shader_binary_size = shader->info.binary_size,
      .scratch_size = shader->info.scratch_size,
      .nr_gprs = shader->info.nr_gprs,
      .compiler_d_valid = shader->info.agx3_staging_known,
      .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
      .fragment = true,
      .imageblock = {key->tib.sample_size_B, key->tib.nr_samples, key->tib.tile_size},
      .uniform_ranges = ranges,
      .uniform_range_count = range_count,
   };
   struct agx3_g17p_graphics_program program;
   if (!agx3_g17g_tile_program_generate(dev->chip, &desc, store, &program) ||
       program.length > capacity)
      return agx_reject_g17g_tile(batch, "M5 tile Ioto program");
   memcpy(ptr.cpu, program.bytes, program.length);
   if (agx_bo_sync_for_device(program_bo, ptr.gpu - program_bo->va->addr,
                              program.length))
      return agx_reject_g17g_tile(batch, "M5 tile program upload");
   agx_batch_add_bo(batch, shader->bo);
   if (dev->debug & AGX_DBG_FAULTTRACE) {
      fprintf(stderr, "M5_TILE op=%u program=%#llx shader=%#llx size=%u gprs=%u staging=%u ranges=%u\n",
              key->op[0], (unsigned long long)ptr.gpu,
              (unsigned long long)desc.shader_va, shader->info.binary_size,
              shader->info.nr_gprs, shader->info.agx3_staging_halfs, range_count);
      for (unsigned i = 0; i < range_count; ++i)
         fprintf(stderr, "M5_TILE_RANGE start=%u halfs=%u address=%#llx\n",
                 ranges[i].start_halfs, ranges[i].size_halfs,
                 (unsigned long long)ranges[i].address);
      fprintf(stderr, "M5_TILE_IOTO data=");
      for (unsigned i = 0; i < program.length; ++i)
         fprintf(stderr, "%02x", program.bytes[i]);
      fputc('\n', stderr);
   }
   return (struct asahi_bg_eot){
      .usc = agx_usc_addr(dev, ptr.gpu),
      .rsrc_spec = rsrc_spec,
      .primary_storage_B = desc.scratch_size,
      .staging_storage_B = desc.compiler_d,
   };
}

/* The texel-fetch sampler an M5 background program reloads attachments
 * with: the same Gen5 sampler as hk_pack_native_txf_sampler(G17G). Gen5's
 * comparison mode bit is required even for a non-shadow fetch. */
static uint64_t
agx_m5_upload_txf_sampler(struct agx_batch *batch)
{
   struct agx_gallium_sampler_descriptor sampler = {0};
   agx_pack(sampler.opaque, SAMPLER_GEN5, cfg) {
      cfg.minimum_lod = 0.0f;
      cfg.maximum_lod = INFINITY;
      cfg.maximum_anisotropy = 1;
      cfg.mip_filter = AGX_MIP_FILTER_NEAREST;
      cfg.wrap_s = cfg.wrap_t = cfg.wrap_r = AGX_WRAP_CLAMP_TO_BORDER;
      cfg.compare_mode = true;
      cfg.compare_func = AGX_COMPARE_FUNC_LEQUAL;
      cfg.border_colour = AGX_BORDER_COLOUR_TRANSPARENT_BLACK;
   }
   return agx_pool_upload_aligned(&batch->pool, &sampler, sizeof(sampler), 32);
}

/*
 * The tile programs of a batch that keeps two or more colour attachments in
 * the tilebuffer (agx_batch::g17g_mrt_resident), in the argument layout of
 * Borealis's M5 multi-target tile programs (hk_build_m5_mrt_bg_eot), whose
 * shaders both drivers take from agx_bg_eot.c:
 *
 *  - the end of tile stores the stored attachments in attachment order. u0
 *    holds the dense table of their PBEs and u4 + 4i the tile offset of the
 *    i-th; the program addresses the PBE of each by its position;
 *  - the background reads its clear colours from 16 bytes per attachment at
 *    u4 (u8 when it also reloads), and reloads attachment i through texture
 *    descriptor 2i of the table at u0, with the texel-fetch sampler at u4.
 *
 * An attachment without defined contents is cleared to zero, as the
 * single-attachment programs do.
 */
static struct asahi_bg_eot
agx_build_native_g17g_mrt_bg_eot(struct agx_batch *batch,
                                 struct agx_bg_eot_key *key, bool store)
{
   struct agx_context *ctx = batch->ctx;
   const unsigned count = batch->key.nr_cbufs;
   unsigned active[PIPE_MAX_COLOR_BUFS], active_count = 0;
   bool undefined[PIPE_MAX_COLOR_BUFS] = {0};
   bool loads = false, clears = false;

   if (agx_tilebuffer_spills(&key->tib) || key->tib.layered ||
       key->tib.nr_samples != 1 || key->tib.tile_size != 32 * 32 ||
       batch->key.resolve)
      return agx_reject_g17g_tile(batch, "M5 multi-target tile layout");

   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (rt >= count || !batch->key.cbufs[rt].texture) {
         if (key->op[rt] != AGX_BG_EOT_NONE)
            return agx_reject_g17g_tile(batch, "M5 tile inactive attachment");
         continue;
      }

      if (store) {
         if (key->op[rt] != AGX_EOT_STORE)
            return agx_reject_g17g_tile(batch, "M5 tile attachment operation");
      } else {
         undefined[rt] = key->op[rt] == AGX_BG_EOT_NONE;
         if (undefined[rt])
            key->op[rt] = AGX_BG_CLEAR;
         if (key->op[rt] != AGX_BG_CLEAR && key->op[rt] != AGX_BG_LOAD)
            return agx_reject_g17g_tile(batch, "M5 tile attachment operation");
         loads |= key->op[rt] == AGX_BG_LOAD;
         clears |= key->op[rt] == AGX_BG_CLEAR;
      }
      active[active_count++] = rt;
   }
   if (active_count < 2)
      return agx_reject_g17g_tile(batch, "M5 tile attachment count");

   key->mixed_mrt = true;
   const unsigned clear_base = loads ? 8 : 4;
   if (clears)
      key->reserved_preamble = clear_base + 8 * count;

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&ctx->bg_eot, key);
   if (!shader || !shader->bo || !shader->bo->va || !shader->ptr ||
       !shader->info.binary_size || !shader->info.nr_gprs ||
       shader->info.has_preamble || shader->info.rodata.size_16 ||
       shader->info.scratch_size || !shader->info.agx3_staging_known ||
       shader->info.push_count != key->reserved_preamble ||
       shader->info.uses_txf != loads)
      return agx_reject_g17g_tile(batch, "M5 multi-target tile shader");

   struct agx3_ioto_uniform_range ranges[3];
   unsigned range_count = 0;
   if (store) {
      struct agx_gallium_pbe_descriptor pbes[PIPE_MAX_COLOR_BUFS];
      uint64_t arguments[1 + PIPE_MAX_COLOR_BUFS] = {0};
      for (unsigned i = 0; i < active_count; ++i) {
         const unsigned rt = active[i];
         struct pipe_image_view view =
            image_view_for_surface(&batch->key.cbufs[rt]);
         /* The fragment and background shaders already encode sRGB into the
          * packed imageblock. Store those bytes without a second
          * conversion. */
         view.format = util_format_linear(view.format);
         if (!agx_batch_upload_pbe(batch, &pbes[i], &view, true, true, false,
                                   batch->feedback & (PIPE_CLEAR_COLOR0 << rt)))
            return agx_reject_g17g_tile(batch, "M5 EOT PBE");
         arguments[1 + i] = agx_tilebuffer_offset_B(&key->tib, rt);
      }
      arguments[0] = agx_pool_upload_aligned(
         &batch->pool, pbes, active_count * sizeof(pbes[0]), 32);
      const uint64_t table = agx_pool_upload_aligned(
         &batch->pool, arguments, 8 * (1 + active_count), 16);
      ranges[range_count++] = (struct agx3_ioto_uniform_range){
         0, 4 * (1 + active_count), table};
   } else {
      if (loads) {
         struct agx_gallium_texture_descriptor textures[2 * PIPE_MAX_COLOR_BUFS];
         memset(textures, 0, 2 * count * sizeof(textures[0]));
         for (unsigned rt = 0; rt < count; ++rt) {
            if (key->op[rt] != AGX_BG_LOAD)
               continue;

            const struct pipe_surface *surf = &batch->key.cbufs[rt];
            struct pipe_sampler_view sampler_view =
               sampler_view_for_surface(surf);
            if (!agx_pack_texture(AGX_DESCRIPTOR_GENERATION_10,
                                  &textures[2 * rt],
                                  agx_resource(surf->texture), surf->format,
                                  &sampler_view))
               return agx_reject_g17g_tile(batch, "M5 reload texture");
         }
         const uint64_t pointers[2] = {
            agx_pool_upload_aligned(&batch->pool, textures,
                                    2 * count * sizeof(textures[0]), 32),
            agx_m5_upload_txf_sampler(batch),
         };
         const uint64_t table = agx_pool_upload_aligned(
            &batch->pool, pointers, sizeof(pointers), 16);
         ranges[range_count++] = (struct agx3_ioto_uniform_range){0, 4, table};
         ranges[range_count++] =
            (struct agx3_ioto_uniform_range){4, 4, table + 8};
      }
      if (clears) {
         union pipe_color_union colors[PIPE_MAX_COLOR_BUFS];
         memset(colors, 0, count * sizeof(colors[0]));
         for (unsigned rt = 0; rt < count; ++rt) {
            if (key->op[rt] == AGX_BG_CLEAR && !undefined[rt])
               colors[rt] = batch->clear_color[rt];
         }
         const uint64_t table = agx_pool_upload_aligned(
            &batch->pool, colors, count * sizeof(colors[0]), 16);
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            clear_base, 8 * count, table};
      }
   }

   return agx_m5_publish_tile_program(
      batch, key, shader, ranges, range_count, store,
      store ? 0 : UINT64_C(0xffff800000000040));
}

static struct asahi_bg_eot
agx_build_native_g17g_bg_eot(struct agx_batch *batch, struct agx_bg_eot_key *key,
                      bool store, bool partial_render)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   const struct pipe_surface *surf = &batch->key.cbufs[0];
   if (batch->g17g_mrt_resident)
      return agx_build_native_g17g_mrt_bg_eot(batch, key, store);

   /* A render with no colour attachment still runs a background and an
    * end-of-tile program per tile; both simply have nothing to move, because
    * the tilebuffer holds no colour. Apple's own colour-free render carries
    * exactly that shape -- a real program pair whose resource specification
    * is zero in both slots, against 0xffff800000000000 for a colour
    * background (registers 0x15369/0x15379 of the depth-only capture in
    * logs/m5-state-matrix-c-native-hv-20260914-055541).
    */
   const bool no_color = batch->key.nr_cbufs == 0;
   /* Spilled attachments (agx_build_tilebuffer_layout_rt0_resident) live in
    * memory, where the fragments write them: neither tile program moves
    * them. A clear of one is refused below; clears of more than one
    * attachment are split (agx_g17g_split_mrt_clear). */
   for (unsigned rt = 1; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (key->tib.spilled[rt] && key->op[rt] != AGX_BG_CLEAR)
         key->op[rt] = AGX_BG_EOT_NONE;
   }
   /* An attachment without valid contents may be initialized to zero. Keep
    * LOAD distinct: partial renders must preserve the previous tile contents.
    */
   const bool undefined = !store && !no_color && key->op[0] == AGX_BG_EOT_NONE;
   if (undefined)
      key->op[0] = AGX_BG_CLEAR;
   const enum agx_bg_eot_op op = key->op[0];
   /* A colour-free multisample layout still reserves one empty 8-byte
    * sample (agx_tilebuffer.c); a depth-only 2x/4x pass has that. */
   if ((no_color ? surf->texture || op != AGX_BG_EOT_NONE ||
                      key->tib.sample_size_B !=
                         (key->tib.nr_samples > 1 ? 8 : 0)
                 : !surf->texture || !key->tib.sample_size_B ||
                      key->tib.sample_size_B % 4 ||
                      key->tib.sample_size_B > 32 ||
                      (store ? op != AGX_EOT_STORE
                             : (op != AGX_BG_CLEAR && op != AGX_BG_LOAD))) ||
       key->tib.layered ||
       (key->tib.nr_samples != 1 && key->tib.nr_samples != 2 &&
        key->tib.nr_samples != 4) ||
       key->tib.tile_size != (key->tib.nr_samples == 4 ? 32 * 16 : 32 * 32) ||
       key->tib.spilled[0] || agx_tilebuffer_offset_B(&key->tib, 0) ||
       (batch->key.resolve && key->tib.nr_samples == 1)) {
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         fprintf(stderr,
                 "M5_TILE_REJECT store=%u op=%u texture=%u layered=%u "
                 "samples=%u sample_size=%u tile=%u spills=%u offset=%u "
                 "resolve=%u format=%s\n",
                 store, op, !!surf->texture, key->tib.layered,
                 key->tib.nr_samples, key->tib.sample_size_B,
                 key->tib.tile_size, agx_tilebuffer_spills(&key->tib),
                 agx_tilebuffer_offset_B(&key->tib, 0), !!batch->key.resolve,
                 util_format_name(surf->format));
      }
      return agx_reject_g17g_tile(batch, "M5 tile layout or operation");
   }
   for (unsigned rt = 1; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (key->op[rt] != AGX_BG_EOT_NONE)
         return agx_reject_g17g_tile(batch, "M5 tile attachment count");
   }
   if (op == AGX_BG_CLEAR)
      key->reserved_preamble = AGX3_G17P_RT0_CLEAR_PUSH_HALFS;

   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&ctx->bg_eot, key);
   if (!shader || !shader->bo || !shader->bo->va || !shader->ptr ||
       !shader->info.binary_size || shader->info.has_preamble ||
       shader->info.rodata.size_16 || !shader->info.agx3_staging_known) {
      mesa_loge("asahi: native tile shader refused: op=%u format=%s "
                "(%s%s%s%s)",
                op, util_format_name(surf->format),
                !shader || !shader->info.binary_size ? "no binary " : "",
                shader && shader->info.has_preamble ? "preamble " : "",
                shader && shader->info.rodata.size_16 ? "rodata " : "",
                shader && !shader->info.agx3_staging_known ? "staging-unknown" : "");
      return agx_reject_g17g_tile(batch, "M5 tile shader");
   }

   struct agx3_ioto_uniform_range ranges[2];
   unsigned range_count = 1;
   if (no_color) {
      /* Nothing is loaded, cleared or stored, so the program binds nothing.
       * The shader still has to be a real, complete one -- the hardware runs
       * it for every tile -- so hold it to the same no-preamble, no-texture
       * shape the other two operations are held to. */
      if (shader->info.push_count || shader->info.uses_txf)
         return agx_reject_g17g_tile(batch, "M5 colour-free tile shader ABI");
      range_count = 0;
   } else if (store) {
      /* Match the M5 native EOT ABI already used by Honeykrisp: the PBE
       * table pointer occupies u0..3 and the tile offset occupies u4..5.
       */
      /* The ABI is the uniform layout above (eight staged halves); the
       * store itself follows the format, so its size and register count
       * vary (78 bytes and eight registers for RGBA8). */
      if (shader->info.nr_gprs > 16 || shader->info.agx3_staging_halfs != 8)
         return agx_reject_g17g_tile(batch, "M5 EOT shader ABI");
      struct agx_ptr pbe = agx_pool_alloc_aligned(
         &batch->pool, sizeof(struct agx_gallium_pbe_descriptor), 32);
      /* A multisampled tile stores through its resolve target's
       * single-sample PBE when one is attached (the PBE resolves, as on
       * G17P), else unresolved through the multisample PBE. */
      struct pipe_surface resolve_surf;
      const struct pipe_surface *store_surf = surf;
      if (key->tib.nr_samples > 1 && batch->key.resolve) {
         if (!agx_g17p_direct_resolve_surface(batch, &resolve_surf))
            return agx_reject_g17g_tile(batch, "M5 resolve target");
         store_surf = &resolve_surf;
      }
      struct pipe_image_view view = image_view_for_surface(store_surf);
      /* The fragment and background shaders already encode sRGB into the
       * packed imageblock. Store those bytes without a second conversion. */
      view.format = util_format_linear(view.format);
      if (!agx_batch_upload_pbe(batch, pbe.cpu, &view, true, true, false,
                                batch->feedback & PIPE_CLEAR_COLOR0))
         return agx_reject_g17g_tile(batch, "M5 EOT PBE");
      const uint64_t arguments[2] = {pbe.gpu, 0};
      const uint64_t table = agx_pool_upload_aligned(
         &batch->pool, arguments, sizeof(arguments), 16);
      ranges[0] = (struct agx3_ioto_uniform_range){0, 6, table};
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         fprintf(stderr, "M5_EOT_PBE address=%#llx descriptor=",
                 (unsigned long long)pbe.gpu);
         for (unsigned i = 0; i < sizeof(struct agx_gallium_pbe_descriptor); ++i)
            fprintf(stderr, "%02x", ((const uint8_t *)pbe.cpu)[i]);
         fputc('\n', stderr);
      }
   } else if (op == AGX_BG_CLEAR) {
      if (shader->info.push_count != AGX3_G17P_RT0_CLEAR_PUSH_HALFS ||
          shader->info.uses_txf)
         return agx_reject_g17g_tile(batch, "M5 clear shader ABI");
      const union pipe_color_union zero = {0};
      const uint64_t clear = agx_pool_upload_aligned(
         &batch->pool, undefined ? &zero : &batch->clear_color[0], 16, 16);
      ranges[0] = (struct agx3_ioto_uniform_range){4, 8, clear};
      if (dev->debug & AGX_DBG_FAULTTRACE) {
         const union pipe_color_union *color =
            undefined ? &zero : &batch->clear_color[0];
         fprintf(stderr, "M5_CLEAR undefined=%u rgba=%g,%g,%g,%g format=%s\n",
                 undefined, color->f[0], color->f[1], color->f[2], color->f[3],
                 util_format_name(surf->format));
      }
   } else {
      if (shader->info.push_count || !shader->info.uses_txf)
         return agx_reject_g17g_tile(batch, "M5 reload shader ABI");
      struct agx_ptr texture = agx_pool_alloc_aligned(
         &batch->pool, sizeof(struct agx_gallium_texture_descriptor), 32);
      struct pipe_sampler_view sampler_view = sampler_view_for_surface(surf);
      if (!agx_pack_texture(AGX_DESCRIPTOR_GENERATION_10, texture.cpu,
                            agx_resource(surf->texture), surf->format,
                            &sampler_view))
         return agx_reject_g17g_tile(batch, "M5 reload texture");
      const uint64_t pointers[2] = {texture.gpu,
                                    agx_m5_upload_txf_sampler(batch)};
      const uint64_t table = agx_pool_upload_aligned(
         &batch->pool, pointers, sizeof(pointers), 8);
      ranges[0] = (struct agx3_ioto_uniform_range){0, 4, table};
      ranges[1] = (struct agx3_ioto_uniform_range){4, 4, table + 8};
      range_count = 2;
   }

   return agx_m5_publish_tile_program(
      batch, key, shader, ranges, range_count, store,
      store || no_color ? 0 : UINT64_C(0xffff800000000040));
}

static bool agx_g18_layered_d16_native(const struct agx_device *dev,
                                       const struct agx_batch *batch);

/* Layered1083: a batch whose layered D16 draws were all refused still
 * submitted its clear-only render, so page/cacheline-strided D16 arrays
 * (64x64x18, 117x117x6, 256x256x6, 173x173x18) reached the kernel without a
 * native witness. Every render -- drawn, clear-only or partial -- builds its
 * background and end-of-tile programs below, so refuse the whole batch there
 * unless its D16 array is one of the native1060 shapes that the draw profile
 * admits. Single-layer D16 and layered Z32/Z32S8 renders are unaffected. */
static bool
agx_g18_layered_d16_outside_native(const struct agx_device *dev,
                                   const struct agx_batch *batch)
{
   const struct pipe_surface *zs = &batch->key.zsbuf;

   if (dev->chip != AGX_CHIP_G18G || !zs->texture ||
       agx_resource(zs->texture)->layout.format != PIPE_FORMAT_Z16_UNORM)
      return false;

   const bool layered = batch->tilebuffer_layout.layered ||
                        batch->key.layers > 1 ||
                        util_framebuffer_get_num_layers(&batch->key) > 1;
   return layered && !agx_g18_layered_d16_native(dev, batch);
}

struct asahi_bg_eot
agx_build_bg_eot(struct agx_batch *batch, bool store, bool partial_render)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);
   const unsigned descriptor_size =
      agx_gallium_texture_descriptor_size(descriptor_generation);
   if (!descriptor_size)
      return agx_reject_g17g_tile(batch, "background heap geometry");

   if (batch->native_rejected)
      return (struct asahi_bg_eot){0};

   for (unsigned rt = 0; rt < batch->key.nr_cbufs; ++rt) {
      const struct pipe_surface *surf = &batch->key.cbufs[rt];
      if (surf->texture &&
          !ail_is_view_compatible(&agx_resource(surf->texture)->layout,
                                  surf->format))
         return agx_reject_g17g_tile(batch,
                                     "incompatible compressed render target");
   }

   if (agx_g18_layered_d16_outside_native(dev, batch)) {
      if ((dev->debug & AGX_DBG_FAULTTRACE) && !batch->native_rejected) {
         const struct pipe_surface *zs = &batch->key.zsbuf;
         const struct ail_layout *layout = &agx_resource(zs->texture)->layout;
         fprintf(stderr,
                 "M5_LAYERED_D16_REJECT width=%u height=%u layers=%u "
                 "samples=%u cbufs=%u layer_stride=%u compressed=%u levels=%u "
                 "level=%u first_layer=%u draws=%u refused_draws=%u "
                 "clear=%#x\n",
                 (unsigned)batch->key.width, (unsigned)batch->key.height,
                 util_framebuffer_get_num_layers(&batch->key),
                 (unsigned)batch->tilebuffer_layout.nr_samples,
                 (unsigned)batch->key.nr_cbufs,
                 (unsigned)layout->layer_stride_B, (unsigned)layout->compressed,
                 (unsigned)layout->levels, (unsigned)zs->level,
                 (unsigned)zs->first_layer, (unsigned)batch->draws,
                 batch->draws_refused, (unsigned)batch->clear);
      }
      return agx_reject_g17g_tile(batch,
                                  "G18 layered D16 render outside native1060 shapes");
   }

   /* Construct the key */
   struct agx_bg_eot_key key = {
      .tib = batch->tilebuffer_layout,
      /* An API resolve target contains the output, not necessarily the
       * previous multisample contents. Only the loader's explicit seed uses
       * that image as a background source. */
      .single_sample_reload =
         (agx_chip_native_agx3(dev->chip) ? batch->g17p_resolve_reload
                                   : agx_chip_g17_m5(dev->chip)) &&
         !store && !partial_render &&
         batch->tilebuffer_layout.nr_samples > 1 && batch->key.resolve,
      /* A persistent attachment or partial render must keep each sample.
       * Resolving it here would discard data needed by the next reload. */
      .multisample_store =
         (agx_chip_native_or_m5(dev->chip) ||
          agx_g18_msaa4_tile_layout(dev, &batch->tilebuffer_layout)) && store &&
         batch->tilebuffer_layout.nr_samples > 1 &&
         (partial_render || !batch->key.resolve ||
          (agx_chip_native_agx3(dev->chip) && batch->g17p_resolve_preserve)),
      .store_and_resolve =
         agx_chip_native_agx3(dev->chip) && store && !partial_render &&
         batch->tilebuffer_layout.nr_samples > 1 && batch->key.resolve &&
         batch->g17p_resolve_preserve,
   };

   bool needs_textures_for_spilled_rts =
      agx_tilebuffer_spills(&batch->tilebuffer_layout) && !partial_render &&
      !store;

   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      const struct pipe_surface *surf = &batch->key.cbufs[rt];

      if (surf->texture == NULL)
         continue;

      if (agx_chip_native_agx3(dev->chip) &&
          batch->tilebuffer_layout.spilled[rt]) {
         key.op[rt] = AGX_BG_EOT_NONE;
      } else if (store) {
         /* TODO: Suppress stores to discarded render targets */
         key.op[rt] = AGX_EOT_STORE;
      } else if (batch->tilebuffer_layout.spilled[rt] && partial_render) {
         /* Partial render programs exist only to store/load the tilebuffer to
          * main memory. When render targets are already spilled to main memory,
          * there's nothing to do.
          */
         key.op[rt] = AGX_BG_EOT_NONE;
      } else {
         bool valid = (batch->load & (PIPE_CLEAR_COLOR0 << rt));
         bool clear = (batch->clear & (PIPE_CLEAR_COLOR0 << rt));
         bool load = valid && !clear;

         /* Don't read back spilled render targets, they're already in memory */
         load &= !batch->tilebuffer_layout.spilled[rt];

         /* The background program used for partial renders must always load
          * whatever was stored in the mid-frame end-of-tile program.
          */
         load |= partial_render;

         key.op[rt] = load    ? AGX_BG_LOAD
                      : clear ? AGX_BG_CLEAR
                              : AGX_BG_EOT_NONE;
      }
   }

   if (dev->chip == AGX_CHIP_G18G)
      return agx_build_g17g_bg_eot(batch, &key, store);
   if (agx_chip_g17_m5(dev->chip))
      return agx_build_native_g17g_bg_eot(batch, &key, store, partial_render);

   if (agx_chip_native_agx3(dev->chip)) {
      if (agx_tilebuffer_spills(&key.tib)) {
         bool active = false;
         for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
            active |= key.op[rt] != AGX_BG_EOT_NONE;
            if (key.tib.spilled[rt]) {
               key.tib.logical_format[rt] = PIPE_FORMAT_NONE;
               key.tib.spilled[rt] = false;
               key.tib._offset_B[rt] = 0;
            }
         }
         if (active)
            return agx_g17p_build_mixed_bg_eot(batch, &key, store);
      }
      /* Match each admitted tile layout to its complete background/EOT
       * program and argument layout. */
      if (!agx_gallium_g17p_tile_layout_supported(&key.tib)) {
         const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;
         mesa_loge("asahi: G17P tile layout is unproven: %u samples, "
                   "%u B/sample, %u px tile, layered=%u",
                   tib->nr_samples, tib->sample_size_B, tib->tile_size,
                   tib->layered);
         for (unsigned rt = 0; rt < ARRAY_SIZE(tib->logical_format); ++rt) {
            if (tib->logical_format[rt] != PIPE_FORMAT_NONE)
               mesa_loge("asahi:   target %u is %s at byte %u%s", rt,
                         util_format_short_name(tib->logical_format[rt]),
                         tib->_offset_B[rt],
                         tib->spilled[rt] ? " (spilled)" : "");
         }
         agx_native_descriptor_rejected(dev, "background tile layout");
      }

      const enum agx_g17p_mrt_layout mrt =
         agx_g17p_classify_mrt(&key.tib);
      if (key.tib.layered && mrt == AGX_G17P_MRT_MIXED) {
         for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
            if (key.op[rt] != AGX_BG_EOT_NONE)
               return agx_g17p_build_mixed_bg_eot(batch, &key, store);
         }
      }
      const bool wboit_mrt =
         !key.tib.layered && mrt == AGX_G17P_MRT_WBOIT;
      const bool rgba8_mrt =
         !key.tib.layered && mrt == AGX_G17P_MRT_RGBA8;
      const bool dual_mrt = wboit_mrt || rgba8_mrt;
      const unsigned rgba8_mask = agx_g17p_rgba8_mrt_mask(&key.tib);
      const unsigned wide_count = util_bitcount(rgba8_mask);
      const unsigned first_rt = rgba8_mask ? ffs(rgba8_mask) - 1 : 0;
      const enum agx_bg_eot_op op = key.op[first_rt];

      for (unsigned rt = 1; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
         const enum agx_bg_eot_op rt_op = key.op[rt];
         const bool rt1_supported =
            ((rt == 1 && dual_mrt) || (rgba8_mask & BITFIELD_BIT(rt))) &&
            (store ? rt_op == AGX_EOT_STORE
                   : (rt_op == AGX_BG_CLEAR || rt_op == AGX_BG_LOAD));
         /* Anything the dedicated programs cannot express -- mixed
          * formats, or targets that do not share one operation -- takes
          * the general mixed-format programs. */
         if ((rt_op != AGX_BG_EOT_NONE && !rt1_supported) ||
             ((rgba8_mask & BITFIELD_BIT(rt)) && rt_op != op))
            return agx_g17p_build_mixed_bg_eot(batch, &key, store);
      }

      const bool dual_mrt_active = dual_mrt &&
                                   key.op[1] != AGX_BG_EOT_NONE;
      const unsigned attachment_count = key.store_and_resolve ? 2 :
         wide_count && op != AGX_BG_EOT_NONE ? wide_count : dual_mrt_active ? 2 : 1;
      const unsigned attachment_mask =
         key.store_and_resolve ? BITFIELD_BIT(first_rt) :
         rgba8_mask ? rgba8_mask : BITFIELD_MASK(attachment_count);
      const unsigned slot_count =
         rgba8_mask ? util_last_bit(rgba8_mask) : attachment_count;
      if (dual_mrt_active && key.op[1] != op)
         return agx_g17p_build_mixed_bg_eot(batch, &key, store);
      if (store ? (op != AGX_EOT_STORE && op != AGX_BG_EOT_NONE)
                : (op != AGX_BG_CLEAR && op != AGX_BG_LOAD &&
                   op != AGX_BG_EOT_NONE))
         agx_native_descriptor_rejected(dev, "background operation");

      if (!store && op == AGX_BG_CLEAR)
         key.reserved_preamble =
            wide_count ? 4 + 8 * slot_count
            : rgba8_mrt && dual_mrt_active
               ? AGX3_G17P_RGBA8_MRT_CLEAR_PUSH_HALFS
            : wboit_mrt && dual_mrt_active
               ? AGX3_G17P_WBOIT_MRT_CLEAR_PUSH_HALFS
               : AGX3_G17P_RT0_CLEAR_PUSH_HALFS;

      struct agx_bg_eot_shader *shader =
         agx_get_bg_eot_shader(&ctx->bg_eot, &key);
      const bool shader_supported =
         shader && shader->bo && shader->bo->va && shader->ptr &&
         shader->info.binary_size &&
         /* The G17P per-sample lowering repeats the body once per sample, so
          * this size guard scales with the sample count: a four-sample reload
          * is 990 bytes where the single-sample one is well under 512. */
         shader->info.binary_size <=
            (wide_count
                ? 4096u
                : 512u * MAX2(batch->tilebuffer_layout.nr_samples, 1u)) &&
         shader->info.main_offset < shader->info.binary_size &&
         shader->info.main_size <=
            shader->info.binary_size - shader->info.main_offset &&
         !shader->info.has_preamble && !shader->info.rodata.size_16 &&
         shader->info.nr_gprs;
      const bool resources_match =
         shader &&
         (store ||
          (op == AGX_BG_EOT_NONE
              ? shader->info.push_count == 0 && !shader->info.uses_txf
              : agx3_g17p_rt0_bg_shader_resources_match(
                   op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                      : AGX3_G17P_RT0_BG_RELOAD,
                   shader->info.push_count, shader->info.uses_txf)));
      if (!shader_supported || !resources_match) {
         mesa_loge("asahi: G17P background shader is unproven: op %u, "
                   "%u pushed halves, txf=%u, preamble=%u, rodata=%u, "
                   "gprs=%u, %u samples, binary %u B (bound %u), main %u@%u",
                   op, shader ? shader->info.push_count : 0,
                   shader ? shader->info.uses_txf : 0,
                   shader ? shader->info.has_preamble : 0,
                   shader ? shader->info.rodata.size_16 : 0,
                   shader ? shader->info.nr_gprs : 0,
                   batch->tilebuffer_layout.nr_samples,
                   shader ? shader->info.binary_size : 0,
                   wide_count ? 4096u
                              : 512u * MAX2(batch->tilebuffer_layout.nr_samples,
                                            1u),
                   shader ? shader->info.main_size : 0,
                   shader ? shader->info.main_offset : 0);
         agx_native_descriptor_rejected(dev, "background shader");
      }
      agx_batch_add_bo(batch, shader->bo);

      size_t capacity;
      if (op == AGX_BG_EOT_NONE)
         capacity = AGX_IOTO_G17P_FRAGMENT_NO_RESOURCE_PROGRAM_SIZE;
      else if (wide_count || key.store_and_resolve)
         /* A wide background is not an RT0 reload: eight clear values need
          * eight load chunks, and the generated program runs past the RT0
          * bound (344 bytes against 246 for eight RGBA8 targets, which is
          * what KHR-GL30.draw_buffers clears). */
         capacity = store ? AGX_IOTO_G17P_OCTA_EOT_PROGRAM_MAX_SIZE
                          : AGX_IOTO_G17P_OCTA_BACKGROUND_PROGRAM_MAX_SIZE;
      else if (store)
         capacity = AGX_IOTO_G17P_RT0_EOT_PROGRAM_SIZE;
      else if (op == AGX_BG_CLEAR)
         capacity = rgba8_mrt && dual_mrt_active
                       ? AGX_IOTO_G17P_RGBA8_MRT_CLEAR_PROGRAM_SIZE
                    : wboit_mrt && dual_mrt_active
                       ? AGX_IOTO_G17P_WBOIT_MRT_CLEAR_PROGRAM_SIZE
                       : AGX_IOTO_G17P_RT0_CLEAR_PROGRAM_SIZE;
      else
         capacity = AGX_IOTO_G17P_RT0_RELOAD_PROGRAM_MAX_SIZE;
      struct agx_bo *program_bo = NULL;
      struct agx_ptr program_ptr = agx_pool_alloc_aligned_with_bo(
         &batch->pipeline_pool, capacity, AGX_IOTO_PROGRAM_ALIGNMENT,
         &program_bo);
      if (!program_ptr.cpu || !program_bo ||
          !agx_va_in_usc_window(dev, program_ptr.gpu) ||
          !agx_va_in_usc_window(dev,
                                shader->ptr + shader->info.main_offset))
         agx_native_descriptor_rejected(dev, "background program allocation");

      size_t program_length = 0;
      uint64_t resource_spec = 0;
      if (store && op == AGX_EOT_STORE) {
         const struct pipe_surface *surf = &batch->key.cbufs[first_rt];
         struct pipe_surface resolve_surf;
         if (key.tib.nr_samples > 1 && !key.multisample_store) {
            if (!agx_g17p_direct_resolve_surface(batch, &resolve_surf))
               agx_native_descriptor_rejected(dev, "direct colour resolve");
            surf = &resolve_surf;
         }
         const struct pipe_surface *surf1 =
            dual_mrt_active ? &batch->key.cbufs[1] : NULL;
         if (!surf->texture || (dual_mrt_active && !surf1->texture) ||
             descriptor_size != AGX_IOTO_G17P_RT0_PBE_ARGUMENT_SIZE)
            agx_native_descriptor_rejected(dev, "end-of-tile attachment");
         if (key.multisample_store &&
             (dual_mrt_active ||
              util_res_sample_count(surf->texture) != key.tib.nr_samples))
            agx_native_descriptor_rejected(dev, "multisample store attachment");

         const size_t pbe_size = descriptor_size * attachment_count;
         struct agx_ptr pbe =
            agx_pool_alloc_aligned(&batch->pool, pbe_size, 32);
         unsigned pbe_index = 0;
         u_foreach_bit(rt, attachment_mask) {
            const struct pipe_surface *attachment =
               rt == first_rt ? surf : &batch->key.cbufs[rt];
            if (!attachment->texture)
               agx_native_descriptor_rejected(dev, "end-of-tile MRT attachment");
            struct pipe_image_view view = image_view_for_surface(attachment);
            view.format = agx_g17p_rt0_pbe_format(view.format);
            const bool no_compress =
               batch->feedback & (PIPE_CLEAR_COLOR0 << rt);
            struct agx_gallium_pbe_descriptor *argument =
               agx_pbe_descriptor_at(pbe.cpu, descriptor_size, pbe_index++);
            if (!agx_batch_upload_pbe(batch, argument, &view, true, true,
                                      false, no_compress))
               return agx_reject_g17g_tile(batch, "end-of-tile PBE");
         }

         if (key.store_and_resolve) {
            if (!agx_g17p_direct_resolve_surface(batch, &resolve_surf))
               agx_native_descriptor_rejected(dev, "source-preserving resolve");
            struct pipe_image_view view = image_view_for_surface(&resolve_surf);
            view.format = agx_g17p_rt0_pbe_format(view.format);
            struct agx_gallium_pbe_descriptor *argument =
               agx_pbe_descriptor_at(pbe.cpu, descriptor_size, pbe_index++);
            if (!agx_batch_upload_pbe(batch, argument, &view, true, true,
                                      false, false))
               return agx_reject_g17g_tile(batch, "resolve destination PBE");
         }

         struct agx3_g17p_rt0_eot_program_desc desc = {
            .program_va = program_ptr.gpu,
            .shader_va = shader->ptr + shader->info.main_offset,
            .pbe_argument_va = pbe.gpu,
            .imageblock_offset_B =
               agx_tilebuffer_offset_B(&key.tib, first_rt),
            .attachment_count = attachment_count,
            /* Several targets stored at once is the mixed-format builtin,
             * whichever API named the layout: the selectors already come
             * from each target's own logical format (agx_bg_eot.c). */
            .mixed_mrt = attachment_count >= 2,
            .imageblock_offset1_B = key.store_and_resolve
               ? agx_tilebuffer_offset_B(&key.tib, first_rt)
               : dual_mrt_active ? agx_tilebuffer_offset_B(&key.tib, 1) : 0,
            .shader_binary_size = shader->info.binary_size,
            .scratch_size = shader->info.scratch_size,
            .nr_gprs = shader->info.nr_gprs,
            .staging_halfs = shader->info.agx3_staging_known
                                ? shader->info.agx3_staging_halfs
                                : 0,
            .imageblock = {
               .sample_size_B = key.tib.sample_size_B,
               .nr_samples = key.tib.nr_samples,
               .tile_size_px = key.tib.tile_size,
            },
         };
         uint8_t *const offsets[] = {
            &desc.imageblock_offset1_B, &desc.imageblock_offset2_B,
            &desc.imageblock_offset3_B, &desc.imageblock_offset4_B,
            &desc.imageblock_offset5_B, &desc.imageblock_offset6_B,
            &desc.imageblock_offset7_B,
         };
         unsigned offset_index = 0;
         u_foreach_bit(rt, attachment_mask & ~BITFIELD_BIT(first_rt)) {
            if (offset_index >= ARRAY_SIZE(offsets))
               break;
            *offsets[offset_index++] = agx_tilebuffer_offset_B(&key.tib, rt);
         }
         struct agx3_g17p_rt0_eot_program native;
         if (!agx3_g17p_rt0_eot_program_generate(dev->chip, &desc, &native) ||
             native.length > capacity)
            agx_native_descriptor_rejected(dev, "end-of-tile Ioto program");
         memcpy(program_ptr.cpu, native.bytes, native.length);
         program_length = native.length;
         resource_spec = AGX_IOTO_G17P_EOT_RESOURCE_SPEC;
      } else if (op == AGX_BG_EOT_NONE) {
         const struct agx3_g17p_graphics_program_desc desc = {
            .usc_clusters = dev->params.num_clusters_total,
            .program_va = program_ptr.gpu,
            .shader_va = shader->ptr + shader->info.main_offset,
            .shader_binary_size = shader->info.binary_size,
            .scratch_size = shader->info.scratch_size,
            .nr_gprs = shader->info.nr_gprs,
            .compiler_d_valid = shader->info.agx3_staging_known,
            .compiler_d = (uint32_t)shader->info.agx3_staging_halfs * 2,
            .fragment = true,
            .imageblock = {
               .sample_size_B = key.tib.sample_size_B,
               .nr_samples = key.tib.nr_samples,
               .tile_size_px = key.tib.tile_size,
            },
         };
         struct agx3_g17p_graphics_program native;
         if (!agx3_g17p_graphics_program_generate(dev->chip, &desc, &native) ||
             native.length > capacity ||
             !agx3_g17p_rt0_bg_resource_spec(
                dev->chip, AGX3_G17P_RT0_BG_NONE, &resource_spec))
            agx_native_descriptor_rejected(dev, "empty background Ioto program");
         memcpy(program_ptr.cpu, native.bytes, native.length);
         program_length = native.length;
      } else {
         const struct pipe_surface *surf = &batch->key.cbufs[first_rt];
         struct pipe_surface resolve_surf;
         if (key.single_sample_reload) {
            if (!agx_g17p_direct_resolve_surface(batch, &resolve_surf))
               agx_native_descriptor_rejected(dev,
                                              "single-sample WSI reload");
            surf = &resolve_surf;
         }
         if (!surf->texture)
            agx_native_descriptor_rejected(dev, "background attachment");

         struct agx3_g17p_rt0_bg_program_desc desc = {
            .program_va = program_ptr.gpu,
            .shader_va = shader->ptr + shader->info.main_offset,
            .shader_binary_size = shader->info.binary_size,
            .scratch_size = shader->info.scratch_size,
            .nr_gprs = shader->info.nr_gprs,
            .staging_halfs = shader->info.agx3_staging_halfs,
            .clear_push_halfs = shader->info.push_count,
            .staging_known = shader->info.agx3_staging_known,
            .op = op == AGX_BG_CLEAR ? AGX3_G17P_RT0_BG_CLEAR
                                     : AGX3_G17P_RT0_BG_RELOAD,
            .imageblock = {
               .sample_size_B = key.tib.sample_size_B,
               .nr_samples = key.tib.nr_samples,
               .tile_size_px = key.tib.tile_size,
            },
         };

         if (op == AGX_BG_CLEAR) {
            if (!batch->uploaded_clear_color[first_rt])
               agx_native_descriptor_rejected(dev, "background clear value");
            if (attachment_count > 1 || first_rt != 0) {
               union pipe_color_union clear_values[PIPE_MAX_COLOR_BUFS] = {0};
               u_foreach_bit(rt, attachment_mask) {
                  if (!batch->uploaded_clear_color[rt])
                     agx_native_descriptor_rejected(dev, "background MRT clear value");
                  clear_values[rt] = batch->clear_color[rt];
               }
               const size_t clear_size =
                  wboit_mrt ? 20 : slot_count * sizeof(clear_values[0]);
               desc.clear_value_va = agx_pool_upload_aligned(
                  &batch->pool, clear_values, clear_size, 16);
               if (!desc.clear_value_va)
                  agx_native_descriptor_rejected(dev,
                                                 "background MRT clear upload");
            } else {
               desc.clear_value_va = batch->uploaded_clear_color[0];
            }
         } else {
            /* Each RT occupies an even texture index in the reload shader. */
            const unsigned texture_count = 2 * slot_count - 1;
            struct agx_ptr textures = agx_pool_alloc_aligned(
               &batch->pool, descriptor_size * texture_count, 64);
            memset(textures.cpu, 0, descriptor_size * texture_count);
            u_foreach_bit(rt, attachment_mask) {
               const struct pipe_surface *attachment =
                  rt == first_rt ? surf : &batch->key.cbufs[rt];
               if (!attachment->texture)
                  agx_native_descriptor_rejected(dev, "background MRT attachment");
               struct pipe_sampler_view sampler_view =
                  sampler_view_for_surface(attachment);
               struct agx_gallium_texture_descriptor *texture =
                  agx_texture_descriptor_at(textures.cpu, descriptor_size,
                                            rt * 2);
               if (!agx_pack_texture(descriptor_generation, texture,
                                     agx_resource(attachment->texture),
                                     attachment->format, &sampler_view))
                  return agx_reject_g17g_tile(batch, "background texture");
            }

            const unsigned sampler_size =
               agx_chip_sampler_heap_element_size(dev->chip);
            if (sampler_size != AGX_GALLIUM_SAMPLER_AGX3_SIZE)
               agx_native_descriptor_rejected(dev, "background sampler geometry");
            /* Staged multisample texel fetches select sampler one. Ordinary
             * reloads use zero; both entries have the native txf state. */
            const unsigned sampler_count =
               key.tib.nr_samples > 1 && !key.single_sample_reload ? 2 : 1;
            struct agx_ptr sampler = agx_pool_alloc_aligned(
               &batch->pool, sampler_count * sampler_size, 64);
            if (!agx_gallium_pack_g17p_txf_sampler(dev->chip, sampler.cpu))
               agx_native_descriptor_rejected(dev, "background texel-fetch sampler");

            if (sampler_count > 1)
               memcpy((uint8_t *)sampler.cpu + sampler_size, sampler.cpu,
                      sampler_size);

            const uint64_t pointers[2] = {textures.gpu, sampler.gpu};
            const uint64_t table = agx_pool_upload_aligned(
               &batch->pool, pointers, sizeof(pointers), 8);
            desc.texture_pointer_va = table;
            desc.sampler_pointer_va = table + sizeof(uint64_t);
         }

         struct agx3_g17p_rt0_bg_program native;
         if (!agx3_g17p_rt0_bg_program_generate(dev->chip, &desc, &native) ||
             native.length > capacity ||
             !agx3_g17p_rt0_bg_resource_spec(dev->chip, desc.op,
                                              &resource_spec))
            agx_native_descriptor_rejected(dev, "background Ioto program");
         memcpy(program_ptr.cpu, native.bytes, native.length);
         program_length = native.length;
      }

      agx_bo_assert_range(program_bo, program_ptr.gpu - program_bo->va->addr,
                          program_length);

      return (struct asahi_bg_eot){
         .usc = agx_usc_addr(dev, program_ptr.gpu),
         .rsrc_spec = resource_spec,
      };
   }

   /* Begin building the legacy pipeline. */
   size_t usc_size = agx_usc_size(3 + PIPE_MAX_COLOR_BUFS);
   struct agx_ptr t =
      agx_pool_alloc_aligned(&batch->pipeline_pool, usc_size, 64);
   struct agx_usc_builder b = agx_usc_builder(t.cpu, usc_size);

   bool needs_sampler = false;
   unsigned uniforms = 0;
   unsigned nr_tex = 0;

   for (unsigned rt = 0; rt < PIPE_MAX_COLOR_BUFS; ++rt) {
      if (key.op[rt] == AGX_BG_LOAD) {
         /* Each reloaded render target is textured */
         needs_sampler = true;

         /* Will be uploaded later, this would be clobbered */
         if (needs_textures_for_spilled_rts)
            continue;

         struct agx_ptr texture =
            agx_pool_alloc_aligned(&batch->pool, descriptor_size, 64);
         const struct pipe_surface *surf = &batch->key.cbufs[rt];
         assert(surf->texture != NULL && "cannot load nonexistent attachment");

         struct agx_resource *rsrc = agx_resource(surf->texture);
         struct pipe_sampler_view sampler_view = sampler_view_for_surface(surf);

         if (!agx_pack_texture(descriptor_generation, texture.cpu, rsrc,
                               surf->format, &sampler_view))
            return agx_reject_g17g_tile(batch, "background texture");

         agx_usc_pack(&b, TEXTURE, cfg) {
            /* Shifted to match eMRT indexing, could be optimized */
            cfg.start = rt * 2;
            cfg.count = 1;
            cfg.buffer = texture.gpu;
         }

         nr_tex = (rt * 2) + 1;
      } else if (key.op[rt] == AGX_BG_CLEAR) {
         assert(batch->uploaded_clear_color[rt] && "set when cleared");
         agx_usc_uniform(&b, 4 + (8 * rt), 8, batch->uploaded_clear_color[rt]);
         uniforms = MAX2(uniforms, 4 + (8 * rt) + 8);
      } else if (key.op[rt] == AGX_EOT_STORE) {
         struct pipe_image_view view =
            image_view_for_surface(&batch->key.cbufs[rt]);
         struct agx_ptr pbe =
            agx_pool_alloc_aligned(&batch->pool, descriptor_size, 256);

         /* The tilebuffer is already in sRGB space if needed. Do not convert */
         view.format = util_format_linear(view.format);

         bool no_compress = batch->feedback & (PIPE_CLEAR_COLOR0 << rt);
         if (!agx_batch_upload_pbe(batch, pbe.cpu, &view, true, true, false,
                                   no_compress))
            return agx_reject_g17g_tile(batch, "end-of-tile PBE");

         agx_usc_pack(&b, TEXTURE, cfg) {
            cfg.start = rt;
            cfg.count = 1;
            cfg.buffer = pbe.gpu;
         }

         nr_tex = rt + 1;
      }
   }

   if (needs_textures_for_spilled_rts) {
      /* Upload texture/PBE descriptors for each render target so we can clear
       * spilled render targets.
       */
      struct agx_ptr descs = agx_pool_alloc_aligned(
         &batch->pool, descriptor_size * 2 * batch->key.nr_cbufs, 64);
      agx_upload_spilled_rt_descriptors(descs.cpu, batch,
                                        descriptor_generation, 0, false);

      agx_usc_pack(&b, TEXTURE, cfg) {
         cfg.start = 0;
         cfg.count = 2 * batch->key.nr_cbufs;
         cfg.buffer = descs.gpu;
      }

      nr_tex = MAX2(nr_tex, 2 * batch->key.nr_cbufs);

      /* Bind the base as u0_u1 for bindless access */
      agx_usc_uniform(&b, 0, 4,
                      agx_pool_upload_aligned(&batch->pool, &descs.gpu, 8, 8));
      uniforms = MAX2(uniforms, 4);
   }

   /* All render targets share a sampler */
   if (needs_sampler) {
      struct agx_ptr sampler =
         agx_pool_alloc_aligned(&batch->pool, AGX_SAMPLER_LENGTH, 64);

      agx_pack(sampler.cpu, SAMPLER, cfg) {
         cfg.minimum_lod = 0.0f;
         cfg.maximum_lod = INFINITY;
         cfg.magnify = AGX_FILTER_LINEAR;
         cfg.minify = AGX_FILTER_NEAREST;
         cfg.mip_filter = AGX_MIP_FILTER_NONE;
         cfg.wrap_s = AGX_WRAP_CLAMP_TO_EDGE;
         cfg.wrap_t = AGX_WRAP_CLAMP_TO_EDGE;
         cfg.wrap_r = AGX_WRAP_CLAMP_TO_EDGE;
         cfg.pixel_coordinates = true;
         cfg.compare_func = AGX_COMPARE_FUNC_ALWAYS;
      }

      agx_usc_pack(&b, SAMPLER, cfg) {
         cfg.start = 0;
         cfg.count = 1;
         cfg.buffer = sampler.gpu;
      }
   }

   agx_usc_push_packed(&b, SHARED, &batch->tilebuffer_layout.usc);

   /* Get the shader */
   key.reserved_preamble = uniforms;
   struct agx_bg_eot_shader *shader = agx_get_bg_eot_shader(&ctx->bg_eot, &key);
   if (!shader)
      agx_native_descriptor_rejected(dev, "background shader");

   agx_batch_add_bo(batch, shader->bo);
   assert(shader->info.rodata.size_16 == 0);

   agx_usc_pack(&b, SHADER, cfg) {
      cfg.code = agx_usc_addr(dev, shader->ptr + shader->info.main_offset);
      cfg.unk_2 = 0;
   }

   agx_usc_pack(&b, REGISTERS, cfg)
      cfg.register_count = shader->info.nr_gprs;

   if (shader->info.has_preamble) {
      agx_usc_pack(&b, PRESHADER, cfg) {
         cfg.code =
            agx_usc_addr(dev, shader->ptr + shader->info.preamble_offset);
      }
   } else {
      agx_usc_pack(&b, NO_PRESHADER, cfg)
         ;
   }

   struct asahi_bg_eot ret = {.usc = t.gpu};

   struct agx_counts_packed counts;
   agx_pack(&counts, COUNTS, cfg) {
      cfg.uniform_register_count = shader->info.push_count;
      cfg.preshader_register_count = shader->info.nr_preamble_gprs;
      cfg.texture_state_register_count = nr_tex;
      cfg.sampler_state_register_count =
         agx_translate_sampler_state_count(needs_sampler ? 1 : 0, false);

      if (!store)
         cfg.unknown = 0xFFFF;
   }

   memcpy(&ret.rsrc_spec, &counts, sizeof(counts));

   return ret;
}

/*
 * Return the standard sample positions, packed into a 32-bit word with fixed
 * point nibbles for each x/y component of the (at most 4) samples. This is
 * suitable for programming the PPP_MULTISAMPLECTL control register.
 */
static uint32_t
agx_default_sample_positions(unsigned nr_samples)
{
   switch (nr_samples) {
   case 1:
      return 0x88;
   case 2:
      return 0x44cc;
   case 4:
      return 0xeaa26e26;
   default:
      UNREACHABLE("Invalid sample count");
   }
}

void
agx_batch_init_state(struct agx_batch *batch)
{
   if (batch->initialized)
      return;

   if (agx_batch_is_compute(batch)) {
      batch->initialized = true;

      struct agx_context *ctx = batch->ctx;
      struct agx_device *dev = agx_device(ctx->base.screen);
      uint8_t *out = batch->cdm.current;

      /* See below */
      agx_push(out, CDM_BARRIER, cfg) {
         cfg.usc_cache_inval = true;
         cfg.unk_5 = true;
         cfg.unk_6 = true;
         cfg.unk_8 = true;
         // cfg.unk_11 = true;
         // cfg.unk_20 = true;
         if (dev->params.num_clusters_total > 1) {
            // cfg.unk_24 = true;
            if (dev->params.gpu_generation == 13) {
               cfg.unk_4 = true;
               // cfg.unk_26 = true;
            }
         }
      }

      return;
   }

   struct agx_device *dev = agx_device(batch->ctx->base.screen);

   /* Emit state on the batch that we don't change and so don't dirty track */
   uint8_t *out = batch->vdm.current;

   /* Barrier to enforce GPU-CPU coherency, in case this batch is back to back
    * with another that caused stale data to be cached and the CPU wrote to it
    * in the meantime.
    */
   if (dev->chip == AGX_CHIP_G18G) {
      /* None of the J873 native render or compute streams (sessions-20260925
       * a, b, g1, g2) contains the barrier word 0x21000960, so its G18G
       * encoding has no witness. Emit none, as Apple's streams do. */
   } else if (agx_chip_native_or_m5(dev->chip)) {
      struct agx_vdm_barrier_core barrier;
      const bool packed = agx_vdm_barrier_core_pack(
         dev->chip,
         dev->params.features & DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE,
         &barrier);
      assert(packed);
      if (!packed)
         return;

      memcpy(out, barrier.bytes, barrier.length);
      out += barrier.length;
   } else {
      agx_push(out, VDM_BARRIER, cfg) {
         cfg.usc_cache_inval = true;
      }
   }

   if (agx_chip_hal300_path(dev->chip)) {
      /* Same fixed-layout initial state as Honeykrisp's M5 path. These
       * records establish the processor's initial state even for clear-only
       * batches; the AGX2 selector graph cannot initialize HAL300.
       */
      const struct agx_ppp_w_clamp_cull2_fields fields = {
         .w_clamp_raw = 0, .cull2_raw = UINT32_C(0x80),
      };
      const struct agx_ppp_hal300_depth_stencil_fields ds_fields = {
         .fragment_control_raw = UINT32_C(0x000007aa),
         .fragment_front_face_raw = UINT32_C(0xe2e00000),
         .fragment_back_face_raw = UINT32_C(0xf0c00000),
         .fragment_front_stencil_raw = UINT32_C(0x00023020),
         .fragment_back_stencil_raw = UINT32_C(0x0001ffff),
      };
      struct agx_ppp_w_clamp_cull2_core initial;
      struct agx_ppp_hal300_depth_stencil_core ds;
      if (!agx_ppp_hal300_depth_stencil_core_pack(dev->chip, &ds_fields, &ds) ||
          !agx_ppp_w_clamp_cull2_core_pack(dev->chip, &fields, &initial))
         {
            agx_native_descriptor_reject_batch(batch, "M5 initial PPP state");
            return;
         }
      const uint64_t ds_gpu = agx_pool_upload_aligned(
         &batch->pool, ds.bytes, ds.length, AGX_PPP_HEADER_ALIGN);
      const uint64_t initial_gpu = agx_pool_upload_aligned(
         &batch->pool, initial.bytes, initial.length, AGX_PPP_HEADER_ALIGN);
      struct agx_vdm_ppp_state_core records[2];
      if (!ds_gpu || !initial_gpu ||
          !agx_gallium_ppp_record(dev->chip, ds_gpu, ds.length / 4, &records[0]) ||
          !agx_gallium_ppp_record(dev->chip, initial_gpu, initial.length / 4, &records[1]))
         {
            agx_native_descriptor_reject_batch(batch, "M5 initial PPP pointers");
            return;
         }
      for (unsigned i = 0; i < ARRAY_SIZE(records); ++i) {
         memcpy(out, records[i].bytes, records[i].length);
         out += records[i].length;
      }
   } else if (agx_chip_native_agx3(dev->chip)) {
      const struct agx_ppp_w_clamp_cull2_fields fields = {
         .w_clamp_raw = 0,
         .cull2_raw = UINT32_C(0x80),
      };
      struct agx_ppp_w_clamp_cull2_core initial;
      const bool packed =
         agx_ppp_w_clamp_cull2_core_pack(dev->chip, &fields, &initial);
      assert(packed);
      if (!packed)
         return;

      struct agx_ptr T = agx_pool_alloc_aligned(
         &batch->pool, initial.length, AGX_PPP_HEADER_ALIGN);
      memcpy(T.cpu, initial.bytes, initial.length);

      struct agx_ppp_update ppp = {
         .head = (uint8_t *)T.cpu + initial.length,
         .gpu_base = T.gpu,
         .total_size = initial.length,
         .cpu_base = T.cpu,
      };
      const bool emitted = agx_gallium_ppp_fini(batch, &out, &ppp);
      assert(emitted);
      if (!emitted)
         return;
   } else {
      struct AGX_PPP_HEADER present = {
         .w_clamp = true,
         .occlusion_query_2 = true,
         .output_unknown = true,
         .varying_word_2 = true,
         .viewport_count = 1, /* irrelevant */
      };

      size_t size = agx_ppp_update_size(&present);
      struct agx_ptr T =
         agx_pool_alloc_aligned(&batch->pool, size, AGX_PPP_HEADER_ALIGN);
      struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &present);

      /* clang-format off */
      agx_ppp_push(&ppp, W_CLAMP, cfg) cfg.w_clamp = 1e-10;
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY_2, cfg);
      agx_ppp_push(&ppp, OUTPUT_UNKNOWN, cfg);
      agx_ppp_push(&ppp, VARYING_2, cfg);
      /* clang-format on */

      const bool emitted = agx_gallium_ppp_fini(batch, &out, &ppp);
      assert(emitted);
      if (!emitted)
         return;
   }
   batch->vdm.current = out;

   /* Mark it as initialized now, since agx_batch_writes() will check this. */
   batch->initialized = true;

   /* Choose a tilebuffer layout given the framebuffer key */
   enum pipe_format formats[PIPE_MAX_COLOR_BUFS] = {0};
   for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
      formats[i] = batch->key.cbufs[i].format;
   }

   /* G17G's tile budget is a full 32 KiB with at most 2048 samples per tile
    * (mesa-aurora agx3c's M5 limits): a 4x 16-byte sample keeps the 32x16
    * tile its tile programs take. The default 32 KiB - 1 halved it to 16x16,
    * which they refuse (dEQP-GLES3 fbo.msaa.4_samples.rgba32f). The epilog
    * key carries the same limits (agx_fs_epilog_apply_device_policy). */
   /* A G17G render of more than one colour attachment keeps them all in the
    * tilebuffer where the multi-target tile programs move them
    * (agx_g17g_mrt_resident). Otherwise (agx_g17g_mrt_one_pass) it keeps
    * attachment 0 there and spills the others, which its fragments load and
    * store as images. */
   batch->g17g_mrt_resident = agx_g17g_fb_mrt_resident(
      agx_device(batch->ctx->base.screen), &batch->key);
   if (batch->g17g_mrt_resident) {
      batch->tilebuffer_layout = agx_build_tilebuffer_layout_with_limits(
         formats, batch->key.nr_cbufs, 1, false, 0,
         AGX_G17G_MAX_BYTES_PER_TILE, AGX_G17G_MAX_SAMPLES_PER_TILE);
   } else if (agx_chip_g17_m5(agx_device(batch->ctx->base.screen)->chip) &&
              batch->key.nr_cbufs > 1) {
      batch->tilebuffer_layout = agx_build_tilebuffer_layout_rt0_resident(
         formats, batch->key.nr_cbufs,
         util_framebuffer_get_num_samples(&batch->key),
         AGX_G17G_MAX_BYTES_PER_TILE, AGX_G17G_MAX_SAMPLES_PER_TILE);
   } else if (agx_chip_g17_m5(agx_device(batch->ctx->base.screen)->chip)) {
      batch->tilebuffer_layout = agx_build_tilebuffer_layout_with_limits(
         formats, batch->key.nr_cbufs,
         util_framebuffer_get_num_samples(&batch->key),
         util_framebuffer_get_num_layers(&batch->key) > 1, 0,
         AGX_G17G_MAX_BYTES_PER_TILE, AGX_G17G_MAX_SAMPLES_PER_TILE);
   } else {
      batch->tilebuffer_layout = agx_gallium_tilebuffer_layout(
         agx_device(batch->ctx->base.screen), formats, batch->key.nr_cbufs,
         util_framebuffer_get_num_samples(&batch->key),
         util_framebuffer_get_num_layers(&batch->key) > 1);
   }

   if (agx_device(batch->ctx->base.screen)->debug & AGX_DBG_SMALLTILE)
      batch->tilebuffer_layout.tile_size = 16 * 16;

   /* If the layout spilled render targets, we need to decompress those render
    * targets to ensure we can write to them.
    */
   if (agx_tilebuffer_spills(&batch->tilebuffer_layout)) {
      for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
         if (!batch->tilebuffer_layout.spilled[i])
            continue;

         struct pipe_surface *surf = &batch->key.cbufs[i];
         if (!surf->texture)
            continue;

         struct agx_resource *rsrc = agx_resource(surf->texture);
         struct ail_layout *layout = &rsrc->layout;
         unsigned level = surf->level;

         if (!ail_is_level_logically_compressed(layout, level))
            continue;

         if (agx_gallium_shared_compression_supported(
                agx_chip_descriptor_generation(dev->chip)) ||
             agx_resource_is_shared(rsrc)) {
            agx_decompress_inplace(batch, surf, "Render target spilled");
         } else {
            /* agx_set_framebuffer_state() pre-decompresses spilling private
             * compressed targets on AGX3 before copying ctx->framebuffer so
             * blitter restore cannot re-enter batch setup. */
            assert(!"Spilled compressed render target was not pre-decompressed "
                    "in agx_set_framebuffer_state()");
            agx_decompress(batch->ctx, rsrc, "Render target spilled");
         }
      }
   }

   if (batch->key.zsbuf.texture) {
      unsigned level = batch->key.zsbuf.level;
      struct agx_resource *rsrc = agx_resource(batch->key.zsbuf.texture);

      agx_batch_writes(batch, rsrc, level);

      if (rsrc->separate_stencil)
         agx_batch_writes(batch, rsrc->separate_stencil, level);
   }

   for (unsigned i = 0; i < batch->key.nr_cbufs; ++i) {
      if (batch->key.cbufs[i].texture) {
         struct agx_resource *rsrc = agx_resource(batch->key.cbufs[i].texture);
         unsigned level = batch->key.cbufs[i].level;

         if (!ail_is_view_compatible(&rsrc->layout, batch->key.cbufs[i].format))
            agx_native_descriptor_reject_batch(
               batch, "incompatible compressed render target");

         if (agx_resource_valid(rsrc, level))
            batch->load |= PIPE_CLEAR_COLOR0 << i;

         agx_batch_writes(batch, rsrc, level);
         assert(agx_resource_valid(rsrc, level));
      }
   }

   if (batch->key.resolve) {
      struct agx_resource *resolve = agx_resource(batch->key.resolve);
      agx_batch_writes(batch, resolve, 0);
      assert(agx_resource_valid(resolve, 0));
   }

   /* Set up standard sample positions */
   batch->uniforms.ppp_multisamplectl =
      agx_default_sample_positions(batch->tilebuffer_layout.nr_samples);
}

static enum agx_object_type
agx_point_object_type(struct agx_rasterizer *rast)
{
   return (rast->base.sprite_coord_mode == PIPE_SPRITE_COORD_UPPER_LEFT)
             ? AGX_OBJECT_TYPE_POINT_SPRITE_UV01
             : AGX_OBJECT_TYPE_POINT_SPRITE_UV10;
}

#define MAX_PPP_UPDATES 2
#define IS_DIRTY(ST)    !!(ctx->dirty & AGX_DIRTY_##ST)

/* HAL300 uses full Ioto program pointers and separate viewport/base/graphics
 * PPP records. Use the same captured codecs as Honeykrisp; the legacy USC and
 * HAL200 bind-tail layouts are not compatible with this command processor.
 */
static bool
agx_native_g17g_viewport_scissor_records(struct agx_batch *batch,
                                 struct agx_vdm_ppp_state_core records[2])
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   const struct agx_rasterizer *rast = ctx->rast;
   unsigned minx, miny, maxx, maxy;
   const struct pipe_viewport_state *vp = &ctx->viewport[0];
   agx_get_scissor_extents(vp, rast->base.scissor ? &ctx->scissor[0] : NULL,
                           &batch->key, &minx, &miny, &maxx, &maxy);
   float minz, maxz;
   util_viewport_zmin_zmax(vp, rast->base.clip_halfz, &minz, &maxz);
   const struct agx_ppp_hal300_scissor_record_fields scissor_fields = {
      .min_x = minx, .max_x = maxx, .min_y = miny, .max_y = maxy,
      .min_z_raw = fui(minz), .max_z_raw = fui(maxz),
   };
   struct agx_ppp_hal300_scissor_record scissor;
   if (maxx > UINT16_MAX || maxy > UINT16_MAX ||
       !agx_ppp_hal300_scissor_record_pack(dev->chip, &scissor_fields, &scissor))
      return false;
   /* As on AGX2: twice the units, the slope scale and the clamp. The
    * graphics record enables it only for triangles with a bias. */
   const struct agx_ppp_hal300_depth_bias_record_fields bias_fields = {
      .depth_bias_raw = fui(rast->depth_bias ? rast->base.offset_units * 2.0f : 0.0f),
      .slope_scale_raw = fui(rast->depth_bias ? rast->base.offset_scale : 0.0f),
      .clamp_raw = fui(rast->depth_bias ? rast->base.offset_clamp : 0.0f),
   };
   struct agx_ppp_hal300_depth_bias_record bias;
   if (!agx_ppp_hal300_depth_bias_record_pack(dev->chip, &bias_fields, &bias))
      return false;
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields base_fields = {
      .scissor_base = agx_pool_upload_aligned(&batch->pool, scissor.bytes,
                                               scissor.length, 64),
      .depth_bias_base = agx_pool_upload_aligned(&batch->pool, bias.bytes,
                                                  bias.length, 64),
   };
   struct agx_ppp_hal300_depth_bias_scissor_bases bases;
   if (!base_fields.scissor_base || !base_fields.depth_bias_base ||
       !agx_ppp_hal300_depth_bias_scissor_bases_pack(dev->chip, &base_fields, &bases))
      return false;

   uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
   agx_pack(region_clip, REGION_CLIP, cfg) {
      cfg.enable = true;
      cfg.min_x = minx / 32; cfg.min_y = miny / 32;
      cfg.max_x = DIV_ROUND_UP(MAX2(maxx, 1), 32);
      cfg.max_y = DIV_ROUND_UP(MAX2(maxy, 1), 32);
   }
   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = 1,
      .viewports[0] = {
         .translate_x = fui(vp->translate[0]), .scale_x = fui(vp->scale[0]),
         .translate_y = fui(vp->translate[1]), .scale_y = fui(vp->scale[1]),
         .translate_z = fui(rast->base.clip_halfz ? vp->translate[2]
                                : vp->translate[2] - vp->scale[2]),
         .scale_z = fui(rast->base.clip_halfz ? vp->scale[2] : vp->scale[2] * 2.0f),
      },
   };
   memcpy(&viewport_fields.region_clip_raw[0], region_clip, sizeof(region_clip));
   uint8_t control[AGX_VIEWPORT_CONTROL_LENGTH];
   agx_pack(control, VIEWPORT_CONTROL, cfg) ;
   memcpy(&viewport_fields.viewport_control_raw, control, sizeof(control));
   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(dev->chip, &viewport_fields, &viewport))
      return false;

   const uint64_t vp_gpu = agx_pool_upload_aligned(&batch->pool, viewport.bytes,
                                                   viewport.length, 64);
   const uint64_t bases_gpu = agx_pool_upload_aligned(&batch->pool, bases.bytes,
                                                      bases.length, 64);
   return vp_gpu && bases_gpu &&
          agx_gallium_ppp_record(dev->chip, vp_gpu, viewport.length / 4, &records[0]) &&
          agx_gallium_ppp_record(dev->chip, bases_gpu, bases.length / 4, &records[1]);
}


static uint32_t
agx_g17p_stencil_word(const struct pipe_stencil_state *stencil);

static uint8_t *
agx_encode_native_g17g_state(struct agx_batch *batch, uint8_t *out)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_compiled_shader *hw_vs = ctx->gs ? ctx->gs->gs_copy : ctx->vs;
   struct agx_linked_shader *hw_vs_linked =
      ctx->gs ? ctx->linked.gs_copy : ctx->linked.vs;
   const struct agx_compiled_shader *vs = hw_vs;
   struct agx_linked_shader *fs = ctx->linked.fs;
   const struct agx_rasterizer *rast = ctx->rast;
   struct agx_varyings_fs *cf = fs ? &fs->cf : NULL;
   /* The reference binding whose interpolation the record carries: the last
    * one, or with flat bindings beside interpolated ones (alacritty), the
    * last interpolated one. */
   const struct agx_cf_binding *color = NULL;
   bool cf_flat = false, cf_interp = false, cf_persp = false, cf_linear = false;
   if (cf && cf->nr_bindings >= 1 && cf->nr_bindings <= 16) {
      for (unsigned i = 0; i < cf->nr_bindings; ++i) {
         if (cf->bindings[i].slot == VARYING_SLOT_POS)
            continue;
         if (cf->bindings[i].smooth) {
            cf_interp = true;
            cf_persp |= cf->bindings[i].perspective;
            cf_linear |= !cf->bindings[i].perspective;
            /* With both interpolated modes, the perspective one is the
             * reference: it needs the position's W binding. */
            if (!color || !color->perspective || cf->bindings[i].perspective)
               color = &cf->bindings[i];
         } else {
            cf_flat = true;
         }
      }
      if (!(cf_flat && cf_interp) && !(cf_persp && cf_linear))
         color = &cf->bindings[cf->nr_bindings - 1];
   }
   /* Perspective and noperspective varyings side by side: the record counts
    * them together (each binding carries its own mode), as it counts flat
    * ones beside an interpolated mode. */
   const bool mixed_interp = cf_persp && cf_linear;
   const bool mixed_cf = (cf_flat && cf_interp) || mixed_interp;
   /* The UVS groups are laid out in enum order: position, varyings, point
    * size, layer/viewport, clip distance. The profile admits a point-size
    * group and a clip-distance group and keeps layer/viewport empty, so the
    * clip-distance group is whatever runs from its offset to the end. */
   const bool psiz = vs && vs->uvs.group_offs[UVS_LAYER_VIEWPORT] ==
                              vs->uvs.group_offs[UVS_PSIZ] + 1;
   const unsigned clip_dist_words =
      vs && vs->uvs.size > vs->uvs.group_offs[UVS_CLIP_DIST]
         ? vs->uvs.size - vs->uvs.group_offs[UVS_CLIP_DIST]
         : 0;
   const bool vertex_color_output =
      vs && vs->uvs.user_size >= 1 &&
      vs->uvs.user_size <= 64 &&
      vs->uvs.size == 4 + vs->uvs.user_size + psiz + clip_dist_words;
   bool color_varying = vertex_color_output && color;
   unsigned coefficient_count = 0;
   if (color_varying) {
      for (unsigned i = 0; i < cf->nr_bindings; ++i) {
         const struct agx_cf_binding *b = &cf->bindings[i];
         coefficient_count += b->count;
         if (b->slot == VARYING_SLOT_POS) {
            /* W for perspective-correct varyings, Z (and W) when the
             * fragment reads gl_FragCoord.z (fragment_reads_z). */
            color_varying &= (i == 0 ||
                              (i == 1 &&
                               cf->bindings[0].slot == VARYING_SLOT_POS)) &&
                             (b->offset == 2 || b->offset == 3) &&
                             b->count >= 1 && b->offset + b->count <= 4 &&
                             b->smooth && !b->perspective;
         } else {
            /* Every binding must interpolate the same way as the colour
             * binding; smooth and perspective are independent, and a
             * smooth binding that is not perspective-correct is a
             * noperspective varying. */
            color_varying &= b->slot < ARRAY_SIZE(vs->uvs.components) &&
                             b->count > 0 &&
                             b->offset + b->count <= vs->uvs.components[b->slot] &&
                             ((mixed_cf && !b->smooth) || mixed_interp ||
                              (b->smooth == color->smooth &&
                               b->perspective == color->perspective));
         }
      }
      /* Only a perspective-correct varying needs the position binding that
       * supplies its W; a flat or noperspective one does not, which is why
       * a noperspective blit arrives with a single binding. All-flat and
       * all-noperspective bindings take the mixed shape's limit: each
       * binding carries its mode, as the reference binding does (the
       * linker interpolates convergent inputs, a GS's uniform mat3, as
       * noperspective). */
      color_varying &= coefficient_count == cf->nr_cf &&
         (color->perspective ? (cf->nr_bindings >= 2 &&
                               cf->bindings[0].slot == VARYING_SLOT_POS)
                             : cf->nr_bindings <= 16);
      /* Record 3's word total follows 4 + components past eight (Apple's
       * stream, ten components); alacritty's text shader has a position
       * binding and three smooth ones, 4 + 4 + 2 components. */
   }
   /* Shared utility vertex shaders can export a color that a constant-color
    * fragment shader ignores. Keep the vertex UVS layout intact while emitting
    * zero fragment interpolation bindings for that independently linked stage. */
   const bool no_varyings = vs && cf && !cf->nr_bindings && !cf->nr_cf &&
      ((vs->uvs.user_size == 0 &&
        vs->uvs.size == 4 + psiz + clip_dist_words) || vertex_color_output);
   /* A fragment reading only position Z and/or W (gl_FragCoord.zw, no
    * varyings). Apple's position-only state exports nothing but the
    * position and flags the coefficient read in OUTPUT_SELECT
    * (gpu/g17g-witness/ppp/state-plain, position_coefficients). */
   bool position_only = vs && cf && cf->nr_bindings >= 1 &&
      cf->nr_bindings <= 2 && vs->uvs.user_size == 0 &&
      vs->uvs.size == 4 + psiz + clip_dist_words;
   unsigned position_components = 0;
   for (unsigned i = 0; position_only && i < cf->nr_bindings; ++i) {
      const struct agx_cf_binding *b = &cf->bindings[i];
      /* gl_PointCoord is a coefficient the rasterizer generates (the
       * linker's point-coord source), so it reads no vertex export either. */
      position_only &= b->slot == VARYING_SLOT_PNTC
         ? b->offset == 0 && b->count >= 1 && b->count <= 2 && b->smooth &&
              batch->reduced_prim == MESA_PRIM_POINTS
         : b->slot == VARYING_SLOT_POS &&
              (b->offset == 2 || b->offset == 3) && b->count >= 1 &&
              b->offset + b->count <= 4 && b->smooth && !b->perspective;
      position_components += b->count;
   }
   position_only &= position_components == cf->nr_cf;
   const bool no_bias = !rast->depth_bias ||
      (rast->base.offset_units == 0 && rast->base.offset_scale == 0 &&
       rast->base.offset_clamp == 0);
   const unsigned fragment_pass = fs ? fs->fragment_control.opaque[0] >> 29 : 0;
   /*
    * A D32 depth attachment is admitted. Its per-draw state is the graphics
    * record's depth group, decoded from one controlled render per comparison
    * on 25G76 (gpu/m5-state-matrix-reference.m); the attachment itself, its
    * clear value and its load/store are the ZLS registers, which this
    * driver already fills generically and the kernel already forwards.
    * Depth testing against an attachment that was never bound is refused
    * rather than silently ignored. Stencil storage still needs its own
    * witnesses for the operation encodings and stays out.
    */
   /*
    * A render with no colour attachment at all is admitted, which is how
    * the depth-format conformance tests draw. The state that moves with it
    * is the graphics record's varying-count field and the ISP record's last
    * word, both taken from a controlled colour-free capture; everything
    * else, including the colour-free tile programs, is untouched because
    * there is no colour to load or store.
    */
   /* A render of more attachments keeps them all in the tile, where the
    * multi-target tile programs move them (agx_g17g_mrt_resident), or keeps
    * attachment 0 alone there and spills the others
    * (agx_g17g_mrt_one_pass). The records below only see whether there is
    * colour (no_color_attachment). */
   bool others_spilled = true;
   for (unsigned rt = 1; rt < batch->key.nr_cbufs; ++rt) {
      others_spilled &= !batch->key.cbufs[rt].texture ||
                        batch->tilebuffer_layout.spilled[rt];
   }
   const bool color_attachment_ok =
      batch->key.nr_cbufs == 0 || batch->g17g_mrt_resident ||
      (batch->key.cbufs[0].texture && others_spilled);
   const struct pipe_surface *zsbuf = &batch->key.zsbuf;
   /* With the comparison disabled the depth unit performs no test and, per
    * the API, no write either, and a disabled stencil unit does nothing at
    * all. The captured record for that state is the inactive depth group
    * regardless of what is attached, and the attachment itself is described
    * only by the ZLS registers, which Mesa packs from the layout for every
    * depth format and the kernel forwards unchanged. So an attachment this
    * render never reads or writes is admitted whatever its format -- which
    * is how a conformance surface with a depth-stencil buffer draws for
    * every case that does not ask for depth. Actually testing against an
    * attachment still needs the witnessed D32 depth group.
    */
   const bool zs_units_idle = !ctx->zs->base.depth_enabled &&
                              !ctx->zs->base.stencil[0].enabled &&
                              !ctx->zs->base.stencil[1].enabled;
   /* Which depth formats may be tested against. Two controlled renders on
    * 25G76 answered this (gpu/m5-state-matrix-reference.m modes depth16 and
    * depth32s8, captured 2026-09-14): a 16-bit depth plane moves exactly the
    * ZLS control's format field, the clear register from a float to a 0..
    * 65535 integer, and one ISP flag -- all three of which the driver and
    * the kernel already derive from the layout -- while no PPP record moves
    * at all. A combined depth-stencil plane used for depth alone is
    * byte-identical to a plain D32 one. Stencil *operations* are a separate
    * axis and still refused above.
    */
   /* Judge the storage the hardware sees: Z24 views are Z32F underneath
    * (u_transfer_helper, see fixup_border_zs), which is how WebRender's
    * DEPTH_COMPONENT24 opaque pass reaches the witnessed D32 group. */
   const enum pipe_format zs_format =
      zsbuf->texture ? agx_resource(zsbuf->texture)->layout.format
                     : zsbuf->format;
   const bool depth_format_ok =
      zs_format == PIPE_FORMAT_Z32_FLOAT ||
      zs_format == PIPE_FORMAT_Z16_UNORM ||
      zs_format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
      (zs_format == PIPE_FORMAT_S8_UINT && !ctx->zs->base.depth_enabled);
   /* A render with no attachment at all (GL_FRAMEBUFFER_DEFAULT_WIDTH,
    * observed through occlusion queries or image stores) is the colour-free
    * render without a depth plane: nothing to load or store. */
   const bool depth_attachment_ok =
      zsbuf->texture ? (zs_units_idle || depth_format_ok)
                     : !ctx->zs->base.depth_enabled;
   /* A fragment that writes gl_FragDepth is carried by the record encoder's
    * fragment_writes_depth; one that writes stencil moves fields with no
    * witness, and encoding it as though it did not would be silently wrong
    * rendering rather than a refusal. */
   /* Twenty-odd independent conditions; a draw that misses the profile is
    * worth naming the one it missed, exactly as the program path does. */
   const char *outside_profile = NULL;
#define M5_DRAW_OUTSIDE(cond, reason)                                          \
   do {                                                                        \
      if (!outside_profile && (cond))                                          \
         outside_profile = (reason);                                           \
   } while (0)

   M5_DRAW_OUTSIDE(!vs, "no vertex shader");
   M5_DRAW_OUTSIDE(!fs, "no fragment shader");
   if (vs && fs && cf) {
      M5_DRAW_OUTSIDE(!no_varyings && !color_varying && !position_only,
                      "varyings outside the colour shape");
      M5_DRAW_OUTSIDE(ctx->fs->epilog_key.write_s, "fragment writes stencil");
      M5_DRAW_OUTSIDE(vs->b.info.has_preamble || vs->b.info.nr_preamble_gprs ||
                         vs->b.info.preamble_scratch_size,
                      "vertex preamble");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_POSITION] != 0,
                      "UVS position group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_VARYINGS] != 4,
                      "UVS varying group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_PSIZ] != 4 + vs->uvs.user_size,
                      "UVS point-size group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_LAYER_VIEWPORT] !=
                         4 + vs->uvs.user_size + psiz,
                      "UVS layer/viewport group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_CLIP_DIST] !=
                         vs->uvs.group_offs[UVS_LAYER_VIEWPORT],
                      "UVS clip-distance group");
      /* The output selector's low sixteen bits are the sixteen named
       * clip-distance planes; bit 18 is point size. Each enabled plane
       * contributes one UVS word, which is the same count the graphics
       * record carries, so the two have to agree. Everything above the
       * plane bits must still be exactly the point-size bit -- notably the
       * varyings bit stays clear, as it does on every shape measured so
       * far. */
      M5_DRAW_OUTSIDE((vs->uvs.osel.opaque[0] & ~UINT32_C(0xffff)) !=
                         (psiz ? 0x40000 : 0),
                      "UVS output selector");
      M5_DRAW_OUTSIDE(util_bitcount(vs->uvs.osel.opaque[0] & 0xffff) !=
                         clip_dist_words,
                      "UVS clip-distance planes");
      M5_DRAW_OUTSIDE(vs->b.info.nonzero_viewport, "nonzero viewport index");
      M5_DRAW_OUTSIDE(fragment_pass != AGX_PASS_TYPE_OPAQUE &&
                         fragment_pass != AGX_PASS_TYPE_TRANSLUCENT &&
                         fragment_pass != AGX_PASS_TYPE_PUNCH_THROUGH &&
                         fragment_pass != AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,
                      "fragment pass type");
   }
   /* Points join triangles: the object type is witnessed end to end (see
    * agx_ppp_hal300_graphics_fields::object_type) and Apple's own point
    * draw exports a point size, which the profile's UVS terms above already
    * require to agree with the output selector. */
   M5_DRAW_OUTSIDE(batch->reduced_prim != MESA_PRIM_TRIANGLES &&
                      batch->reduced_prim != MESA_PRIM_POINTS &&
                      batch->reduced_prim != MESA_PRIM_LINES,
                   "primitive is not triangles, lines or points");
   M5_DRAW_OUTSIDE(batch->reduced_prim == MESA_PRIM_POINTS && !psiz,
                   "point draw without a point size");
   M5_DRAW_OUTSIDE(!color_attachment_ok, "colour attachment shape");
   M5_DRAW_OUTSIDE(!depth_attachment_ok, "depth attachment shape");
   M5_DRAW_OUTSIDE(batch->key.layers > 1, "layered render");
   M5_DRAW_OUTSIDE(batch->tilebuffer_layout.nr_samples != 1 &&
                      batch->tilebuffer_layout.nr_samples != 2 &&
                      batch->tilebuffer_layout.nr_samples != 4,
                   "multisampled");
   M5_DRAW_OUTSIDE(rast->base.fill_front != PIPE_POLYGON_MODE_FILL ||
                      rast->base.fill_back != PIPE_POLYGON_MODE_FILL,
                   "polygon fill mode");
   M5_DRAW_OUTSIDE(rast->base.depth_clamp, "depth clamp");
   M5_DRAW_OUTSIDE(!rast->base.depth_clip_near || !rast->base.depth_clip_far,
                   "depth clip disabled");
#undef M5_DRAW_OUTSIDE

   if (outside_profile) {
      mesa_loge("G17G Gallium draw exceeds the native triangle profile: %s",
                outside_profile);
      if ((dev->debug & AGX_DBG_FAULTTRACE) && vs && fs) {
         fprintf(stderr, "M5_DRAW_REJECT varyings=%u/%u psiz=%u uvs=%u user=%u groups=%u,%u,%u,%u,%u osel=%#x cf=%u,%u reads_z=%u preamble=%u,%u,%u viewport=%u\n",
                 no_varyings, color_varying, psiz, vs->uvs.size, vs->uvs.user_size,
                 vs->uvs.group_offs[UVS_POSITION], vs->uvs.group_offs[UVS_VARYINGS],
                 vs->uvs.group_offs[UVS_PSIZ], vs->uvs.group_offs[UVS_LAYER_VIEWPORT],
                 vs->uvs.group_offs[UVS_CLIP_DIST], vs->uvs.osel.opaque[0],
                 cf->nr_bindings, cf->nr_cf, cf->reads_z, vs->b.info.has_preamble,
                 vs->b.info.nr_preamble_gprs, vs->b.info.preamble_scratch_size,
                 vs->b.info.nonzero_viewport);
         if (color)
            fprintf(stderr, "M5_DRAW_REJECT_CF offset=%u count=%u smooth=%u perspective=%u\n",
                    color->offset, color->count, color->smooth, color->perspective);
         for (unsigned i = 0; cf && i < cf->nr_bindings; ++i)
            fprintf(stderr, "M5_DRAW_REJECT_BINDING %u slot=%u offset=%u count=%u smooth=%u perspective=%u\n",
                    i, cf->bindings[i].slot, cf->bindings[i].offset,
                    cf->bindings[i].count, cf->bindings[i].smooth,
                    cf->bindings[i].perspective);
         fprintf(stderr, "M5_DRAW_REJECT_STATE prim=%u cbufs=%u zs=%s layers=%u samples=%u pass=%u discard=%u fill=%u,%u bias=%u clamp=%u clip=%u,%u depth=%u stencil=%u,%u query=%u\n",
                 batch->reduced_prim, batch->key.nr_cbufs,
                 batch->key.zsbuf.texture
                    ? util_format_name(batch->key.zsbuf.format)
                    : "none",
                 batch->key.layers, batch->tilebuffer_layout.nr_samples,
                 fs->fragment_control.opaque[0] >> 29, rast->base.rasterizer_discard,
                 rast->base.fill_front, rast->base.fill_back, !no_bias,
                 rast->base.depth_clamp, rast->base.depth_clip_near,
                 rast->base.depth_clip_far, ctx->zs->base.depth_enabled,
                 ctx->zs->base.stencil[0].enabled, ctx->zs->base.stencil[1].enabled,
                 !!(ctx->active_queries && ctx->occlusion_query));
      } else if (dev->debug & AGX_DBG_FAULTTRACE) {
         /* A draw without a fragment shader still has attachment state worth
          * seeing; the varying report above simply has nothing to say. */
         fprintf(stderr,
                 "M5_DRAW_REJECT_STATE prim=%u cbufs=%u zs=%s layers=%u "
                 "samples=%u vs=%u fs=%u\n",
                 batch->reduced_prim, batch->key.nr_cbufs,
                 util_format_name(batch->key.zsbuf.format), batch->key.layers,
                 batch->tilebuffer_layout.nr_samples, !!vs, !!fs);
      }
      return NULL;
   }

   batch->varyings = 0;
   if (cf->nr_bindings) {
      const size_t size = AGX_CF_BINDING_HEADER_LENGTH +
                          cf->nr_bindings * AGX_CF_BINDING_LENGTH;
      struct agx_ptr t = agx_pool_alloc_aligned(&batch->pipeline_pool, size, 64);
      if (!t.cpu)
         return NULL;
      agx_link_varyings_vs_fs(t.cpu, &batch->linked_varyings, vs->uvs.user_size,
                              cf, rast->base.flatshade_first ? 0 : 2, 0,
                              &batch->generate_primitive_id);
      if (agx_chip_uses_shade_model_5(dev->chip))
         agx_cf_bindings_select_perspective(t.cpu, cf);
      /* Varying Counts packs smooth at bit 0, flat at bit 8 and linear
       * (noperspective) at bit 16; the record below uses the same layout. */
      const unsigned interpolation_shift =
         color->perspective ? 0 : color->smooth ? 16 : 8;
      const uint32_t counts = batch->linked_varyings.counts_32.opaque[0];
      /* Flat beside one interpolated mode: smooth or linear, matching the
       * reference binding, together covering every exported component. */
      const bool counts_ok = mixed_cf
         ? !(counts >> 24) &&
           (mixed_interp ||
            (((counts >> 8) & 0xff) &&
             ((counts >> (color->perspective ? 16 : 0)) & 0xff) == 0)) &&
           (counts & 0xff) + ((counts >> 8) & 0xff) +
                 ((counts >> 16) & 0xff) == vs->uvs.user_size
         : counts == (vs->uvs.user_size << interpolation_shift);
      if (batch->generate_primitive_id ||
          batch->linked_varyings.counts_16.opaque[0] != 0 || !counts_ok) {
         /* Every other refusal in this profile is named; this one used to
          * drop the draw silently (glxgears' flat-shaded faces). */
         static bool reported;
         if (!reported) {
            reported = true;
            mesa_loge("G17G Gallium draw refused: varying counts %#x (16-bit "
                      "%#x) do not fit the %s colour binding over %u "
                      "components%s",
                      counts, batch->linked_varyings.counts_16.opaque[0],
                      color->perspective ? "perspective"
                      : color->smooth    ? "noperspective"
                                         : "flat",
                      vs->uvs.user_size,
                      batch->generate_primitive_id ? ", primitive ID" : "");
         }
         return NULL;
      }
      batch->varyings = t.gpu;
   }

   uint64_t vertex_program = 0, fragment_program = 0;
   agx_build_pipeline(batch, hw_vs, hw_vs_linked,
                      MESA_SHADER_VERTEX, 0, 0, &vertex_program, NULL);
   agx_build_pipeline(batch, agx_fs_launch_shader(ctx), ctx->linked.fs,
                      MESA_SHADER_FRAGMENT, 0, 0, &fragment_program, NULL);
   if (!vertex_program || !fragment_program) {
      mesa_loge("G17G Gallium could not build the native %s program",
                !vertex_program ? (!fragment_program ? "vertex and fragment"
                                                     : "vertex")
                                : "fragment");
      return NULL;
   }

   /* Match Borealis (hk_native_vertex_template_enabled): keep the vertex
    * template enabled when the hardware vertex/GS-copy program writes memory
    * or transform feedback, and rely on the PPP rasterizer_discard bit below
    * to suppress fragment rasterization. */
   const struct agx_uncompiled_shader *hw_vs_uncompiled =
      hw_vs->so ? hw_vs->so
                : (ctx->gs ? ctx->stage[MESA_SHADER_GEOMETRY].shader
                           : ctx->stage[MESA_SHADER_VERTEX].shader);
   const bool writes_memory =
      (hw_vs_uncompiled && hw_vs_uncompiled->info.writes_memory) ||
      ctx->streamout.num_targets > 0;
   const struct agx_vdm_state_update_draw_info draw_info = {
      .source_layout = AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled = !rast->base.rasterizer_discard || writes_memory,
      .raw_provoking_selector = rast->base.flatshade_first ? 0 : 2,
      .template_info = {
         .pipeline = vertex_program,
         .raw_vs_output_count_0 = vs->uvs.size,
         .raw_active_context_selector = UINT32_MAX,
      },
   };
   struct agx_vdm_state_update vertex;
   if (!agx_vdm_state_update_draw_pack(dev->chip, &draw_info, &vertex))
      return NULL;

   uint8_t visibility_mode = AGX_VISIBILITY_MODE_NONE;
   uint16_t query_index = 0;
   if (ctx->active_queries && ctx->occlusion_query) {
      visibility_mode =
         ctx->occlusion_query->type == PIPE_QUERY_OCCLUSION_COUNTER
            ? AGX_VISIBILITY_MODE_COUNTING
            : AGX_VISIBILITY_MODE_BOOLEAN;
      query_index = agx_get_oq_index(batch, ctx->occlusion_query);
   }

   /* The native field model counts interpolation through either the single
    * mode below or, for layouts mixing modes, the linker's varying counts.
    * G17G's linear counts live at bits 16..23 and its position-only reads
    * carry no varying components, so those shapes name their counts
    * explicitly rather than falling into the single-mode shift. The
    * single-mode shape also stops at eight varying components and three
    * coefficient bindings (agx_ppp_hal300_graphics_core_pack); a colour
    * layout past either limit takes the linker's counts instead. */
   const bool general_varyings =
      position_only ||
      (color_varying &&
       (mixed_cf || (!color->perspective && color->smooth) ||
        vs->uvs.user_size > 8 || cf->nr_bindings > 3));
   const uint32_t counts_32 = batch->linked_varyings.counts_32.opaque[0];

   const struct agx_ppp_hal300_graphics_fields graphics_fields = {
      .fragment_program = fragment_program,
      .fragment_reads_depth = cf && cf->reads_z,
      .visibility_mode = visibility_mode,
      .visibility_index = query_index,
      .no_color_attachment = batch->key.nr_cbufs == 0,
      .fragment_pass = fragment_pass,
      /* Bit 26 of the shader's own fragment control; a discarding shader
       * carries it beside its punch-through pass type. */
      .disable_tri_merging =
         (fs->fragment_control.opaque[0] & BITFIELD_BIT(26)) != 0,
      .fragment_writes_depth = ctx->fs->epilog_key.write_z,
      .varying_table = batch->varyings, .cf_binding_count = cf->nr_bindings,
      .general_varyings = general_varyings,
      .varying_counts_32 = counts_32,
      .cf_register_count = cf->nr_cf,
      .uvs_words = vs->uvs.size,
      /* OUTPUT_SELECT: point size (bit 18), varying/coefficient production
       * (bit 16), a fragment reading depth (bit 21), then the clip-distance
       * plane mask. Layer/viewport stays empty in the admitted profile. */
      .output_select =
         (psiz ? 0x40000 : 0) |
         ((vs->uvs.user_size || cf->nr_bindings) ? 0x10000 : 0) |
         ((cf && cf->reads_z) ? 0x200000 : 0) |
         ((UINT32_C(1) << clip_dist_words) - 1),
      .cull_mode = rast->base.cull_face,
      .front_face_ccw = agx_front_face_ccw(ctx), .point_size_output = psiz,
      /* The same provoking vertex as the VDM selector above. Honeykrisp
       * programs both on M5 (46478b5cfdc: with the PPP selector left at 0,
       * clipped last-vertex triangles took their flat inputs from vertex 0),
       * and the G18G captures carry 2 here for GL's last-vertex default. */
      .flat_shading_vertex = rast->base.flatshade_first ? 0 : 2,
      .clip_distance_words = clip_dist_words,
      .object_type = batch->reduced_prim == MESA_PRIM_POINTS
                        ? AGX_PPP_HAL300_POINTS
                     : batch->reduced_prim == MESA_PRIM_LINES
                        ? AGX_PPP_HAL300_LINES
                        : AGX_PPP_HAL300_TRIANGLES,
      .line_width = batch->reduced_prim == MESA_PRIM_LINES ? rast->line_width : 0,
      .vertex_varying_components = vs->uvs.user_size,
      .fragment_varying_components =
         !color_varying ? 0
         : mixed_cf ? (counts_32 & 0xff) + ((counts_32 >> 16) & 0xff)
                    : vs->uvs.user_size,
      .interpolation = !color_varying || color->perspective
                          ? AGX_PPP_HAL300_SMOOTH
                          : color->smooth ? AGX_PPP_HAL300_LINEAR
                                          : AGX_PPP_HAL300_FLAT,
      /* PIPE_FUNC_* and the record share the never..always order. A depth
       * attachment with the test disabled encodes exactly like no
       * attachment at all, which the M5 capture matrix established. */
      .depth_active = ctx->zs->base.depth_enabled,
      .depth_compare =
         ctx->zs->base.depth_enabled ? ctx->zs->base.depth_func : 0,
      .depth_write =
         ctx->zs->base.depth_enabled && ctx->zs->base.depth_writemask,
      /* As on AGX2, always scissor: the record emitted with the viewport
       * holds the viewport clipped by the scissor, and the hardware does
       * not clip to the viewport otherwise. */
      .scissor_enable = true,
      .depth_bias_enable = !no_bias && batch->reduced_prim == MESA_PRIM_TRIANGLES,
      .stencil_active = ctx->zs->base.stencil[0].enabled,
      .front_stencil = agx_g17p_stencil_word(&ctx->zs->base.stencil[0]),
      .back_stencil = agx_g17p_stencil_word(ctx->zs->base.stencil[1].enabled
                                               ? &ctx->zs->base.stencil[1]
                                               : &ctx->zs->base.stencil[0]),
      .front_stencil_reference = ctx->stencil_ref.ref_value[0],
      .back_stencil_reference =
         ctx->stencil_ref.ref_value[ctx->zs->base.stencil[1].enabled ? 1 : 0],
      .rasterizer_discard = rast->base.rasterizer_discard,
   };
   struct agx_ppp_hal300_graphics_core graphics;
   if (!agx_ppp_hal300_graphics_core_pack(dev->chip, &graphics_fields, &graphics))
      return NULL;

   struct agx_vdm_ppp_state_core records[2 + AGX_PPP_HAL300_GRAPHICS_RECORDS];
   unsigned record_count = 0;
   /* These records depend only on rasterizer, viewport and scissor state.
    * The admitted HAL300 path has one viewport, so shader rebinding cannot
    * change the viewport count. As on AGX2, the pool and encoded state live
    * for the whole batch; switching batches dirties all state.
    */
   if (ctx->dirty & (AGX_DIRTY_VIEWPORT | AGX_DIRTY_SCISSOR_ZBIAS | AGX_DIRTY_RS)) {
      if (!agx_native_g17g_viewport_scissor_records(batch, records))
         return NULL;
      record_count = 2;
   }
   const uint64_t graphics_gpu = agx_pool_upload_aligned(&batch->pool, graphics.bytes,
                                                         sizeof(graphics.bytes), 64);
   if (!graphics_gpu)
      return NULL;
   for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
      if (!agx_gallium_ppp_record(dev->chip, graphics_gpu + graphics.offsets[i],
                                      graphics.lengths[i] / 4, &records[record_count]))
         return NULL;
      record_count++;
   }

   memcpy(out, vertex.bytes, vertex.length);
   out += vertex.length;
   for (unsigned i = 0; i < record_count; ++i) {
      memcpy(out, records[i].bytes, records[i].length);
      out += records[i].length;
   }
   return out;
}

static uint32_t
agx_g17p_stencil_word(const struct pipe_stencil_state *stencil)
{
   return agx_ppp_hal200_stencil_pack(
      stencil->enabled, stencil->func,
      agx_stencil_ops[stencil->fail_op],
      agx_stencil_ops[stencil->zfail_op],
      agx_stencil_ops[stencil->zpass_op], stencil->valuemask,
      stencil->writemask);
}

static uint8_t *
agx_g17p_emit_bind_state(struct agx_batch *batch, uint8_t *out,
                         struct agx_compiled_shader *vs,
                         uint32_t *draw_state_offset)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_rasterizer *rast = ctx->rast;
   if (!agx_chip_native_agx3(dev->chip) || !draw_state_offset)
      return NULL;

   /* Gallium does not track the highest initialized viewport. When a shader
    * can select a nonzero viewport, preserve the established AGX path and
    * upload the complete API-visible array. Keep ordinary shaders on the
    * byte-identical one-viewport fast path. */
   const unsigned viewport_count =
      vs->b.info.nonzero_viewport ? AGX_MAX_VIEWPORTS : 1;
   static_assert(AGX_MAX_VIEWPORTS ==
                 AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS);

   const bool is_points = batch->reduced_prim == MESA_PRIM_POINTS;
   const bool is_lines = batch->reduced_prim == MESA_PRIM_LINES;
   const enum agx_object_type object_type =
      is_points  ? agx_point_object_type(rast)
      : is_lines ? AGX_OBJECT_TYPE_LINE
                 : AGX_OBJECT_TYPE_TRIANGLE;

   const bool depth_bias_enable =
      rast->depth_bias && object_type == AGX_OBJECT_TYPE_TRIANGLE;
   unsigned minx[AGX_MAX_VIEWPORTS], miny[AGX_MAX_VIEWPORTS];
   unsigned maxx[AGX_MAX_VIEWPORTS], maxy[AGX_MAX_VIEWPORTS];
   struct agx_ppp_hal200_raster_state raster_states[AGX_MAX_VIEWPORTS];
   for (unsigned i = 0; i < viewport_count; ++i) {
      agx_get_scissor_extents(
         &ctx->viewport[i], rast->base.scissor ? &ctx->scissor[i] : NULL,
         &batch->key, &minx[i], &miny[i], &maxx[i], &maxy[i]);
      float minz, maxz;
      util_viewport_zmin_zmax(&ctx->viewport[i], rast->base.clip_halfz, &minz,
                              &maxz);
      const struct agx_ppp_hal200_raster_fields raster_fields = {
         .scissor_enable = true,
         .depth_bias_enable = depth_bias_enable,
         .max_x = MIN2(maxx[i], UINT16_MAX),
         .min_x = MIN2(minx[i], UINT16_MAX),
         .max_y = MIN2(maxy[i], UINT16_MAX),
         .min_y = MIN2(miny[i], UINT16_MAX),
         .min_z_raw = fui(minz),
         .max_z_raw = fui(maxz),
         .depth_bias_raw =
            fui(depth_bias_enable ? rast->base.offset_units * 2.0f : 0.0f),
         .slope_scale_raw =
            fui(depth_bias_enable ? rast->base.offset_scale : 0.0f),
         .clamp_raw = fui(depth_bias_enable ? rast->base.offset_clamp : 0.0f),
      };
      if (!agx_ppp_hal200_raster_state_build(
             dev->chip, &raster_fields, &raster_states[i]))
         return NULL;
   }

   uint8_t *scissors = util_dynarray_grow_bytes(
      &batch->scissor, viewport_count, sizeof(raster_states[0].scissor));
   uint8_t *depth_bias_records = util_dynarray_grow_bytes(
      &batch->depth_bias, viewport_count,
      sizeof(raster_states[0].depth_bias));
   for (unsigned i = 0; i < viewport_count; ++i) {
      memcpy(scissors + i * sizeof(raster_states[i].scissor),
             raster_states[i].scissor, sizeof(raster_states[i].scissor));
      memcpy(depth_bias_records + i * sizeof(raster_states[i].depth_bias),
             raster_states[i].depth_bias,
             sizeof(raster_states[i].depth_bias));
   }
   const struct agx_ppp_hal200_raster_state *raster_state = &raster_states[0];

   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = viewport_count,
   };
   for (unsigned i = 0; i < viewport_count; ++i) {
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx[i] / 32;
         cfg.min_y = miny[i] / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx[i], 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy[i], 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip,
             sizeof(region_clip));
      const struct pipe_viewport_state *vp = &ctx->viewport[i];
      viewport_fields.viewports[i] =
         (struct agx_ppp_viewport_transform_raw){
            .translate_x = fui(vp->translate[0]),
            .scale_x = fui(vp->scale[0]),
            .translate_y = fui(vp->translate[1]),
            .scale_y = fui(vp->scale[1]),
            .translate_z = fui(rast->base.clip_halfz
                                  ? vp->translate[2]
                                  : vp->translate[2] - vp->scale[2]),
            .scale_z = fui(rast->base.clip_halfz ? vp->scale[2]
                                                 : vp->scale[2] * 2.0f),
         };
   }
   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(dev->chip, &viewport_fields,
                                            &viewport))
      return NULL;

   uint64_t fragment_program = 0;
   agx_build_pipeline(batch, agx_fs_launch_shader(ctx), ctx->linked.fs,
                      MESA_SHADER_FRAGMENT, 0, 0, &fragment_program, NULL);
   if (agx_drop_for_null_g17p_program(ctx, ctx->fs, fragment_program,
                                      "fragment", NULL))
      return NULL;

   const struct agx_ppp_hal200_fragment_fields fs_fields = {
      .resource_counts = ctx->linked.fs->cf.nr_bindings << 16,
      .varying_table = batch->varyings,
      .stage_program = fragment_program,
   };
   struct agx_ppp_hal200_graphics_core fs_record;
   if (!agx_ppp_hal200_fragment_core_pack(dev->chip, &fs_fields, &fs_record))
      return NULL;

   uint32_t output_size;
   if (!agx_ppp_hal200_output_size_pack(vs->uvs.size, 0, &output_size))
      return NULL;
   uint32_t viewport_count_word;
   if (!agx_ppp_hal200_viewport_count_pack(dev->chip, viewport_count,
                                            &viewport_count_word))
      return NULL;

   const bool stencil_test = ctx->zs->base.stencil[0].enabled;
   enum agx_visibility_mode visibility = AGX_VISIBILITY_MODE_NONE;
   uint16_t query_index = 0;
   if (ctx->active_queries && ctx->occlusion_query) {
      visibility = ctx->occlusion_query->type == PIPE_QUERY_OCCLUSION_COUNTER
                      ? AGX_VISIBILITY_MODE_COUNTING
                      : AGX_VISIBILITY_MODE_BOOLEAN;
      query_index = agx_get_oq_index(batch, ctx->occlusion_query);
   }
   const uint32_t linked_fragment_control =
      ctx->linked.fs->fragment_control.opaque[0];
   const uint32_t pass_type = linked_fragment_control >> 29;
   if (pass_type != AGX_PASS_TYPE_OPAQUE &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT &&
       pass_type != AGX_PASS_TYPE_PUNCH_THROUGH &&
       pass_type != AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH)
      return NULL;

   uint32_t fragment_control = raster_state->fragment_control |
                               (stencil_test ? UINT32_C(0x000c0000) : 0);
   uint32_t secondary_control = UINT32_C(0x00000200);
   const struct agx_ppp_hal200_fragment_control_fields control_fields = {
      .linked_control = linked_fragment_control,
      .shader_tri_merge_disable =
         ctx->fs->b.info.needs_quad_derivatives ||
         (batch->tilebuffer_layout.nr_samples > 1 &&
          ctx->fs->epilog_key.write_z) ||
         /* The linker's own merge-disable bit. */
         (!agx_chip_tri_merge_across_side_effects(dev->chip) &&
          (linked_fragment_control & (UINT32_C(1) << 26))),
      .coverage_tri_merge_disable =
         pass_type == AGX_PASS_TYPE_PUNCH_THROUGH ||
         pass_type == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH,
      .primitive_tri_merge_disable =
         object_type != AGX_OBJECT_TYPE_TRIANGLE ||
         rast->polygon_mode != AGX_POLYGON_MODE_FILL,
   };
   if (!agx_ppp_hal200_fragment_controls_pack(
          dev->chip, &control_fields, fragment_control, secondary_control,
          &fragment_control, &secondary_control))
      return NULL;

   uint32_t occlusion_query;
   if (!agx_ppp_hal200_occlusion_pack(dev->chip, visibility, query_index,
                                      fragment_control, &fragment_control,
                                      &occlusion_query))
      return NULL;

   struct agx_fragment_face_packed front_face, back_face;
   agx_pack(&front_face, FRAGMENT_FACE, cfg) {
      cfg.stencil_reference = ctx->stencil_ref.ref_value[0];
      cfg.line_width = rast->line_width;
      cfg.polygon_mode = rast->polygon_mode;
   }
   agx_pack(&back_face, FRAGMENT_FACE, cfg) {
      cfg.stencil_reference =
         ctx->stencil_ref.ref_value[ctx->zs->base.stencil[1].enabled ? 1 : 0];
      cfg.line_width = rast->line_width;
      cfg.polygon_mode = rast->polygon_mode;
   }
   const bool depth_test =
      batch->key.zsbuf.texture && ctx->zs->base.depth_enabled;
   if (!agx_ppp_hal200_depth_face_pack(
          dev->chip, depth_test, ctx->zs->base.depth_writemask,
          ctx->zs->base.depth_func, front_face.opaque[0],
          &front_face.opaque[0]) ||
       !agx_ppp_hal200_depth_face_pack(
          dev->chip, depth_test, ctx->zs->base.depth_writemask,
          ctx->zs->base.depth_func, back_face.opaque[0],
          &back_face.opaque[0]))
      return NULL;

   struct agx_fragment_face_2_packed face_2;
   agx_pack_fragment_face_2(&face_2, object_type, &ctx->fs->b.info);
   /* G17P's native depth contract must preserve late shader side effects.
    * An unchanged depth output alone does not permit early rejection when
    * image or SSBO stores precede the fragment tests. Match Honeykrisp's
    * native face packing rather than inheriting the AGX2 depth-only word. */
   if (!agx_ppp_hal200_fragment_face_depth_pack(
          dev->chip, agx_translate_depth_layout(ctx->fs->b.info.depth_layout),
          ctx->fs->b.info.early_fragment_tests,
          ctx->fs->so && ctx->fs->so->info.writes_memory,
          face_2.opaque[0], &face_2.opaque[0]))
      return NULL;

   /* The output select carries the vertex program's clip-distance planes,
    * point size and viewport/layer selectors alongside the fragment bits;
    * Apple's clip3 pipeline keeps the first three plane bits in this word.
    * The G17P layout matches AGX2's OUTPUT_SELECT, so the compiler's packed
    * selects merge as they do on other chips.
    */
   uint32_t output_select;
   if (!agx_ppp_hal200_output_select_pack(
          dev->chip, vs->uvs.osel.opaque[0], ctx->linked.fs->cf.nr_bindings,
          ctx->linked.fs->cf.reads_z, &output_select))
      return NULL;

   const struct agx_ppp_hal200_graphics_fields output_fields = {
      .words = {output_select, 0, viewport_count_word, output_size,
                AGX_PPP_HAL200_PRIMARY_FACE_HEADER, fragment_control},
   };
   const struct agx_ppp_hal200_graphics_fields primary_fields = {
      .words = {fragment_control, front_face.opaque[0],
                agx_g17p_stencil_word(&ctx->zs->base.stencil[0]),
                back_face.opaque[0],
                agx_g17p_stencil_word(ctx->zs->base.stencil[1].enabled
                                        ? &ctx->zs->base.stencil[1]
                                        : &ctx->zs->base.stencil[0]),
                occlusion_query},
   };
   const struct agx_ppp_hal200_graphics_fields secondary_fields = {
      .words = {secondary_control, face_2.opaque[0], face_2.opaque[0],
                UINT32_C(0x0001ffff)},
   };

   uint32_t cull = UINT32_C(0x00000480);
   if (!agx_ppp_hal200_cull_mode_pack(dev->chip, rast->base.cull_face,
                                      agx_front_face_ccw(ctx), cull, &cull) ||
       !agx_ppp_hal200_cull_provoking_pack(
          dev->chip, rast->base.flatshade_first ? 0 : 2, cull, &cull) ||
       !agx_ppp_hal200_depth_clip_pack(dev->chip, rast->base.depth_clip_near,
                                       cull, &cull) ||
       !agx_ppp_hal200_rasterizer_discard_pack(
          dev->chip, rast->base.rasterizer_discard, secondary_control, cull,
          &secondary_control, &cull))
      return NULL;

   struct agx_ppp_hal200_graphics_core output_record, primary_record;
   struct agx_ppp_hal200_graphics_core secondary_record, cull_record;
   struct agx_ppp_hal200_graphics_fields final_secondary = secondary_fields;
   final_secondary.words[0] = secondary_control;
   const struct agx_ppp_hal200_graphics_fields cull_fields = {
      .words = {cull},
   };
   const struct agx_ppp_w_clamp_cull2_fields w_cull2_fields = {
      .w_clamp_raw = 0,
      .cull2_raw = UINT32_C(0x80) |
                   (batch->generate_primitive_id ? UINT32_C(0x1000) : 0),
   };
   struct agx_ppp_w_clamp_cull2_core w_cull2;
   if (!agx_ppp_hal200_graphics_core_pack(
          dev->chip, AGX_PPP_HAL200_GRAPHICS_OUTPUT, &output_fields,
          &output_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->chip, AGX_PPP_HAL200_GRAPHICS_PRIMARY_FACE, &primary_fields,
          &primary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->chip, AGX_PPP_HAL200_GRAPHICS_SECONDARY_FACE, &final_secondary,
          &secondary_record) ||
       !agx_ppp_hal200_graphics_core_pack(
          dev->chip, AGX_PPP_HAL200_GRAPHICS_CULL, &cull_fields,
          &cull_record) ||
       !agx_ppp_w_clamp_cull2_core_pack(dev->chip, &w_cull2_fields,
                                         &w_cull2))
      return NULL;

   uint8_t bind_group[AGX_G17P_BIND_GROUP_BYTES] = {0};
   memcpy(&bind_group[0x00], fs_record.bytes, fs_record.length);
   memcpy(&bind_group[0x1c], output_record.bytes, output_record.length);
   memcpy(&bind_group[0x30], primary_record.bytes, primary_record.length);
   memcpy(&bind_group[0x4c], secondary_record.bytes, secondary_record.length);
   memcpy(&bind_group[0x60], w_cull2.bytes, w_cull2.length);
   memcpy(&bind_group[0x6c], cull_record.bytes, cull_record.length);

   const size_t state_size = agx_g17p_bind_state_size(viewport.length);
   struct agx_ptr state =
      agx_pool_alloc_aligned(&batch->pool, state_size, 64);
   uint64_t compact_state;
   struct agx_g17p_bind_state_upload uploaded;
   if (!state_size || !state.cpu ||
       !agx_gallium_vdm_compact(batch, state.gpu, &compact_state) ||
       !agx_g17p_upload_bind_state(
          state.cpu, state_size, compact_state, bind_group,
          viewport.bytes, viewport.length,
          batch->linked_varyings.counts_32.opaque[0],
          batch->linked_varyings.counts_16.opaque[0], &uploaded))
      return NULL;

   struct agx_ptr indices = agx_pool_alloc_aligned(&batch->pool, 8, 4);
   uint64_t compact_indices;
   if (!indices.cpu ||
       !agx_gallium_vdm_compact(batch, indices.gpu, &compact_indices) ||
       !agx_g17p_upload_draw_indices(
          indices.cpu, 8, compact_indices, batch->scissor.size,
          batch->depth_bias.size, viewport_count, draw_state_offset))
      return NULL;

   memcpy(out, uploaded.tail.bytes, uploaded.tail.length);
   return out + uploaded.tail.length;
}

/* HAL300 uses full Ioto program pointers and separate viewport/base/graphics
 * PPP records. Use the same captured codecs as Honeykrisp; the legacy USC and
 * HAL200 bind-tail layouts are not compatible with this command processor.
 */
static bool
agx_g18_viewports_overlap(const struct pipe_viewport_state *viewports,
                          unsigned count)
{
   /* Native597: control bit 0 is set when any two viewport rectangles overlap.
    * Use double precision for their endpoints, which matches the native
    * results. Negative scales reverse the transform without changing its
    * bounds. */
   for (unsigned i = 1; i < count; ++i) {
      const double ix = viewports[i].translate[0];
      const double iy = viewports[i].translate[1];
      const double iw = fabs((double)viewports[i].scale[0]);
      const double ih = fabs((double)viewports[i].scale[1]);
      for (unsigned j = 0; j < i; ++j) {
         const double jx = viewports[j].translate[0];
         const double jy = viewports[j].translate[1];
         const double jw = fabs((double)viewports[j].scale[0]);
         const double jh = fabs((double)viewports[j].scale[1]);
         if (ix + iw > jx - jw && ix - iw < jx + jw &&
             iy + ih > jy - jh && iy - ih < jy + jh)
            return true;
      }
   }
   return false;
}

static bool
agx_g17g_viewport_scissor_records(struct agx_batch *batch,
                                 const struct agx_compiled_shader *vs,
                                 struct agx_vdm_ppp_state_core records[2])
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   const struct agx_rasterizer *rast = ctx->rast;
   /* Gallium keeps the API-visible array, without a highest-initialized
    * index. Match the ordinary AGX path: an indexed shader binds all 16
    * records, while shaders that only address viewport zero bind one. */
   const unsigned viewport_count =
      vs->b.info.nonzero_viewport ? AGX_MAX_VIEWPORTS : 1;
   static_assert(AGX_MAX_VIEWPORTS == AGX_PPP_VIEWPORT_PAYLOAD_MAX_VIEWPORTS);
   uint8_t scissors[AGX_MAX_VIEWPORTS * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH];
   struct agx_ppp_viewport_payload_fields viewport_fields = {
      .viewport_count = viewport_count,
   };
   for (unsigned i = 0; i < viewport_count; ++i) {
      const struct pipe_viewport_state *vp = &ctx->viewport[i];
      unsigned minx, miny, maxx, maxy;
      agx_get_scissor_extents(vp, rast->base.scissor ? &ctx->scissor[i] : NULL,
                             &batch->key, &minx, &miny, &maxx, &maxy);
      float minz, maxz;
      util_viewport_zmin_zmax(vp, rast->base.clip_halfz, &minz, &maxz);
      const struct agx_ppp_hal300_scissor_record_fields scissor_fields = {
         .min_x = minx, .max_x = maxx, .min_y = miny, .max_y = maxy,
         .min_z_raw = fui(minz), .max_z_raw = fui(maxz),
      };
      struct agx_ppp_hal300_scissor_record scissor;
      if (maxx > UINT16_MAX || maxy > UINT16_MAX ||
          !agx_ppp_hal300_scissor_record_pack(dev->chip, &scissor_fields, &scissor))
         return false;
      /* Native597 uploads consecutive 16-byte scissor records,
       * including each viewport's depth range. */
      memcpy(scissors + i * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH,
             scissor.bytes, AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH);
      uint8_t region_clip[AGX_REGION_CLIP_LENGTH];
      agx_pack(region_clip, REGION_CLIP, cfg) {
         cfg.enable = true;
         cfg.min_x = minx / 32; cfg.min_y = miny / 32;
         cfg.max_x = DIV_ROUND_UP(MAX2(maxx, 1), 32);
         cfg.max_y = DIV_ROUND_UP(MAX2(maxy, 1), 32);
      }
      memcpy(&viewport_fields.region_clip_raw[i], region_clip, sizeof(region_clip));
      viewport_fields.viewports[i] = (struct agx_ppp_viewport_transform_raw){
         .translate_x = fui(vp->translate[0]), .scale_x = fui(vp->scale[0]),
         .translate_y = fui(vp->translate[1]), .scale_y = fui(vp->scale[1]),
         .translate_z = fui(rast->base.clip_halfz ? vp->translate[2]
                                : vp->translate[2] - vp->scale[2]),
         .scale_z = fui(rast->base.clip_halfz ? vp->scale[2] : vp->scale[2] * 2.0f),
      };
   }
   viewport_fields.viewport_control_raw =
      dev->chip == AGX_CHIP_G18G &&
      agx_g18_viewports_overlap(ctx->viewport, viewport_count);
   /* HAL300 binds a complete bias record, not the legacy array index. Keep
    * it current on raster state changes even for a non-triangle draw: a
    * later triangle may reuse it, while its ISP enable is per primitive.
    */
   const bool bias_active = dev->chip == AGX_CHIP_G18G && rast->depth_bias &&
      (rast->base.offset_units != 0.0f || rast->base.offset_scale != 0.0f);
   float bias = bias_active ? rast->base.offset_units * 2.0f : 0.0f;
   if (bias_active && batch->key.zsbuf.texture &&
       agx_resource(batch->key.zsbuf.texture)->layout.format == PIPE_FORMAT_Z16_UNORM) {
      /* Native G18 depth bias pairs integer ISP bias with normalized Z16 depth
       * units. */
      const float resolution = 1.0f / 65535.0f;
      bias = fmaf(rast->base.offset_units, resolution,
                  copysignf(resolution, rast->base.offset_units));
   }
   const struct agx_ppp_hal300_depth_bias_record_fields bias_fields = {
      .depth_bias_raw = fui(bias),
      .slope_scale_raw = fui(bias_active ? rast->base.offset_scale : 0.0f),
      .clamp_raw = fui(bias_active ? rast->base.offset_clamp : 0.0f),
   };
   struct agx_ppp_hal300_depth_bias_record bias_record;
   if (!agx_ppp_hal300_depth_bias_record_pack(dev->chip, &bias_fields, &bias_record))
      return false;
   const struct agx_ppp_hal300_depth_bias_scissor_bases_fields base_fields = {
      .scissor_base = agx_pool_upload_aligned(
         &batch->pool, scissors,
         viewport_count * AGX_PPP_HAL300_SCISSOR_RECORD_LENGTH, 64),
      .depth_bias_base = agx_pool_upload_aligned(&batch->pool, bias_record.bytes,
                                                  bias_record.length, 64),
   };
   struct agx_ppp_hal300_depth_bias_scissor_bases bases;
   if (!base_fields.scissor_base || !base_fields.depth_bias_base ||
       !agx_ppp_hal300_depth_bias_scissor_bases_pack(dev->chip, &base_fields, &bases))
      return false;

   struct agx_ppp_viewport_payload_core viewport;
   if (!agx_ppp_viewport_payload_core_pack(dev->chip, &viewport_fields, &viewport))
      return false;

   const uint64_t vp_gpu = agx_pool_upload_aligned(&batch->pool, viewport.bytes,
                                                   viewport.length, 64);
   const uint64_t bases_gpu = agx_pool_upload_aligned(&batch->pool, bases.bytes,
                                                      bases.length, 64);
   return vp_gpu && bases_gpu &&
          agx_gallium_ppp_record(dev->chip, vp_gpu, viewport.length / 4, &records[0]) &&
          agx_gallium_ppp_record(dev->chip, bases_gpu, bases.length / 4, &records[1]);
}


/* Native1060 (probe725, captured by native batch1026 and verified by1036)
 * renders one single-sample four-byte colour array beside an uncompressed D16
 * array at 4x4x4, 8x4x4 and 8x8x48: whole level-0 arrays whose layers occupy
 * one 128-byte cacheline each, which agx_zls_layer_stride encodes as zero.
 * Every layer's captured backing matches the sampled readback. Page-strided
 * or compressed D16 arrays, depth-only and partial bindings have no native
 * witness and stay outside this profile.
 *
 * Layered1105: the remaining color_depth_attachments D16 cube-map arrays --
 * 64x64x18 (8-KiB layers, 64 cachelines) and 117x117x6, 256x256x6 and
 * 173x173x18 (2, 8 and 6 16-KiB pages per layer) -- are the uncompressed D16
 * shapes of native batch1094's layered group, beside RGBA8 and R32Sint colour
 * arrays. They join under the same envelope, and only at the layer stride
 * their M5 layout gives; agx_zls_layer_stride and the D16 clear encode them
 * as for every other ZLS array. */

static bool
agx_g18_layered_d16_native(const struct agx_device *dev,
                           const struct agx_batch *batch)
{
   static const struct {
      uint16_t width, height, layers;
      uint32_t layer_stride_B;
   } shapes[] = {
      {4, 4, 4, AIL_CACHELINE},
      {8, 4, 4, AIL_CACHELINE},
      {8, 8, 48, AIL_CACHELINE},
      {64, 64, 18, 0x2000},
      {117, 117, 6, 0x8000},
      {256, 256, 6, 0x20000},
      {173, 173, 18, 0x18000},
   };
   const struct pipe_framebuffer_state *key = &batch->key;
   const struct pipe_surface *zs = &key->zsbuf;
   const struct pipe_surface *cb = &key->cbufs[0];
   const struct agx_tilebuffer_layout *tib = &batch->tilebuffer_layout;

   if (dev->chip != AGX_CHIP_G18G || !zs->texture || key->nr_cbufs != 1 ||
       !cb->texture || tib->nr_samples != 1 || tib->spilled[0] ||
       util_format_get_blocksize(tib->logical_format[0]) != 4)
      return false;

   const struct agx_resource *z = agx_resource(zs->texture);
   const struct ail_layout *layout = &z->layout;
   const unsigned layers = zs->last_layer - zs->first_layer + 1;
   if (layout->format != PIPE_FORMAT_Z16_UNORM || z->separate_stencil ||
       layout->policy != AIL_LAYOUT_POLICY_M5 ||
       layout->tiling != AIL_TILING_GPU || layout->compressed ||
       layout->sample_count_sa != 1 || layout->levels != 1 || zs->level != 0 ||
       zs->first_layer != 0 || layers != layout->depth_px ||
       cb->level != 0 || cb->first_layer != 0 ||
       (unsigned)(cb->last_layer - cb->first_layer + 1) != layers ||
       util_framebuffer_get_num_layers(key) != layers ||
       (unsigned)key->layers != layers || key->width != layout->width_px ||
       key->height != layout->height_px)
      return false;

   for (unsigned i = 0; i < ARRAY_SIZE(shapes); ++i) {
      if (key->width == shapes[i].width && key->height == shapes[i].height &&
          layers == (unsigned)shapes[i].layers &&
          layout->layer_stride_B == shapes[i].layer_stride_B)
         return true;
   }

   return false;
}

/* Four-sample colour and D32/S8 attachments share the native tile geometry.
 * Depth-only sample shading is statically unrolled into SR39/DEPTHF writes.
 * Colour sample shading expands both shader parts over distinct exports;
 * the linked prolog applies the API mask before the masked colour stores. */
static bool
agx_g18_msaa_memory_shader(struct agx_context *ctx, struct agx_batch *batch)
{
   const struct agx_device *dev = agx_device(ctx->base.screen);
   const struct agx_compiled_shader *fs = ctx->fs;
   const unsigned samples = batch->tilebuffer_layout.nr_samples;

   /* The no-colour per-sample main is statically expanded before compilation.
    * Native564/567 place covered-sample memory writes before late depth, and
    * the epilog preserves each unwritten colour sample independently. Colour
    * exports, stencil and external sample loops are different contracts.
    * write_z with writes_sample_mask is the main's own DEPTHF feedback;
    * agx_update_fs clears that duplicate epilog export before linking. */
   const bool admitted = dev->chip == AGX_CHIP_G18G &&
          (samples == 2 || samples == 4) && fs && fs->so && fs->so->info.writes_memory &&
          fs->so->info.uses_sample_shading && !fs->so->info.uses_fbfetch &&
          !fs->b.info.early_fragment_tests &&
          !fs->epilog_key.sample_shading && !fs->epilog_key.loc_written &&
          (!fs->epilog_key.write_z || fs->b.info.writes_sample_mask) &&
          !fs->epilog_key.write_s;
   if (!admitted && fs && fs->so && fs->so->info.writes_memory &&
       (dev->debug & AGX_DBG_FAULTTRACE)) {
      fprintf(stderr,
              "G18_MSAA_MEMORY_PROFILE samples=%u enabled=%u sample=%u "
              "fbfetch=%u early=%u epilog_sample=%u colours=%#x "
              "write_z=%u write_s=%u write_mask=%u reads_tib=%u\n",
              samples, dev->chip == AGX_CHIP_G18G,
              fs->so->info.uses_sample_shading,
              fs->so->info.uses_fbfetch, fs->b.info.early_fragment_tests,
              fs->epilog_key.sample_shading, fs->epilog_key.loc_written,
              fs->epilog_key.write_z, fs->epilog_key.write_s,
              fs->b.info.writes_sample_mask, fs->b.info.reads_tib);
   }
   return admitted;
}

static bool
agx_g18_msaa4_draw_admitted(struct agx_context *ctx, struct agx_batch *batch,
                            unsigned fragment_pass)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   const struct agx_compiled_shader *fs = ctx->fs;
   const bool no_color = batch->key.nr_cbufs == 0;
   const bool unrolled_depth =
      no_color && fs && fs->so && fs->so->info.uses_sample_shading &&
      fs->epilog_key.write_z && !fs->epilog_key.sample_shading;
   const bool unrolled_colour =
      !no_color && fs && fs->so && fs->so->info.uses_sample_shading &&
      fs->epilog_key.sample_shading && !fs->epilog_key.write_z &&
      !fs->epilog_key.write_s && !fs->so->info.writes_memory;
   const bool unrolled_memory = agx_g18_msaa_memory_shader(ctx, batch);
   /* The compiler records every emitted mask form. Color unrolling may use
    * the same full-target late transaction as the per-pixel profile, with
    * optional kill-only updates, but never partial or early testing. */
   const bool unrolled_colour_mask =
      unrolled_colour && !fs->b.info.early_fragment_tests &&
      (fs->b.info.g18_sample_mask_forms & AGX_G18_SAMPLE_MASK_FULL) &&
      !(fs->b.info.g18_sample_mask_forms & AGX_G18_SAMPLE_MASK_OTHER);
   /* Native1121 psdiscard (Apple M6, verified capture 1121, run 1128): a
    * sample-shaded fragment discarding per sample at 2x/4x on one RGBA8
    * target without depth is a punch-through triangle draw (PPP 0x02000062
    * word 1 0x44000200) whose fragment holds class 1 from its first update
    * and kills/updates each sample with the staging-0/staging-2 movmsk words
    * (agx_compile.c, AGX_G18_SAMPLE_MASK_SAMPLE). Admit exactly that shape:
    * every mask update of the program is the per-sample form. */
   const bool unrolled_colour_sample_discard =
      unrolled_colour &&
      !fs->b.info.early_fragment_tests &&
      fs->b.info.g18_sample_mask_forms == AGX_G18_SAMPLE_MASK_SAMPLE &&
      !fs->so->info.uses_fbfetch && !fs->b.info.reads_tib &&
      fragment_pass == AGX_PASS_TYPE_PUNCH_THROUGH &&
      batch->reduced_prim == MESA_PRIM_TRIANGLES &&
      batch->key.nr_cbufs == 1 &&
      batch->key.cbufs[0].format == PIPE_FORMAT_R8G8B8A8_UNORM &&
      !batch->key.zsbuf.texture;
   /* The API mask prolog tests TARGET=0xff with LIVE=api_sample_mask.
    * Native666's constant masks at 2x/4x use this full-target coverage
    * transaction. Keep it separate from shader masks, explicit early tests
    * and Z/S exports, which have their own ordering requirements. */
   const bool per_pixel_api_mask =
      fs && fs->so && !fs->b.info.early_fragment_tests &&
      !fs->so->info.uses_sample_shading && !fs->so->info.writes_memory &&
      !fs->epilog_key.sample_shading && !fs->epilog_key.write_z &&
      !fs->epilog_key.write_s && !fs->b.info.writes_sample_mask &&
      !fs->b.info.reads_tib;
   /* Native G18 dynamic coverage uses the same late SR39/MOVMSK transaction
    * at 2x and 4x. The compiler proves full-testing versus non-testing words;
    * this per-pixel profile adds no sample loop, memory or Z/S export. */
   const bool per_pixel_mask =
      fs && fs->so && fs->b.info.writes_sample_mask &&
      !fs->b.info.early_fragment_tests && !fs->so->info.uses_sample_shading &&
      !fs->so->info.writes_memory && !fs->so->info.uses_fbfetch &&
      !fs->epilog_key.sample_shading && !fs->epilog_key.write_z &&
      !fs->epilog_key.write_s && !fs->b.info.reads_tib;
   /* API coverage runs in a kill-only prolog when the main shader writes
    * SampleMask: run_zs_tests is false in agx_update_fs, and the prolog's
    * LIVE operand is zero. The shader's full-target late transaction then
    * tests the survivors once. Coverage kills are cumulative, so the shader
    * cannot revive an API-disabled sample. Require the compiler's native
    * full-target certificate; other partial-target forms remain refused. */
   const bool per_pixel_api_shader_mask =
      per_pixel_mask &&
      (fs->b.info.g18_sample_mask_forms & AGX_G18_SAMPLE_MASK_FULL) &&
      !(fs->b.info.g18_sample_mask_forms & AGX_G18_SAMPLE_MASK_OTHER);
   /* Masked-off resident MRTs are preserved by the epilog's tile loads,
    * making the linked program translucent even when the API main reads no
    * TIB. Native454 grounds this per-sample read/modify/write transaction;
    * the compiler still proves complete sample/partition lifetimes. */
   const bool resident_mrt_preserve =
      fs &&
      util_bitcount(agx_g18g_ms_mrt_mask(&batch->tilebuffer_layout)) == 3 &&
      fragment_pass == AGX_PASS_TYPE_TRANSLUCENT;
   /* Native point captures at 1x/2x/4x reproduce the existing PPP
    * packer byte for byte. Keep this initial MSAA profile to the same
    * opaque per-pixel RGBA8 state; other point features need their own
    * output qualification. Point size and varying validation remain in
    * the ordinary draw-state builder below. */
   const bool per_pixel_points =
      batch->reduced_prim == MESA_PRIM_POINTS && per_pixel_api_mask &&
      !fs->so->info.uses_fbfetch && fragment_pass == AGX_PASS_TYPE_OPAQUE &&
      batch->key.nr_cbufs == 1 &&
      batch->key.cbufs[0].format == PIPE_FORMAT_R8G8B8A8_UNORM &&
      !batch->key.zsbuf.texture &&
      (ctx->sample_mask & BITFIELD_MASK(batch->tilebuffer_layout.nr_samples)) ==
         BITFIELD_MASK(batch->tilebuffer_layout.nr_samples);
   /* Native1094 a2c (Apple M6, verified): alpha-to-coverage at 2x and 4x on
    * one RGBA8 target without depth is a punch-through draw (PPP 0x02000062
    * word 1 0x44000200, set below) whose fragment computes its coverage, then
    * runs pixwait512 -> pixwait1 -> movmsk -> pixrel1 -> pixwait12 -> lstcol
    * -> pixrel524: the full-target late SR39/MOVMSK transaction of the
    * dynamic per-pixel mask profile above, byte for byte. Mesa computes the
    * coverage in the epilog (nir_lower_alpha_to_coverage), so the main writes
    * no mask; an API sample mask stays the kill-only prolog of
    * per_pixel_api_mask. The other words that move with A2C natively are
    * per-program: specTPR f22 (0x20) is the staging field Mesa already
    * derives from its own program (compiler_d), and R2 word 6 is Apple's
    * compiler metadata (METADATA root.f0.f32, read by
    * FragmentProgramVariantESLState), which Mesa leaves 0 for every program. */
   const bool per_pixel_a2c =
      ctx->blend->key.alpha_to_coverage &&
      per_pixel_api_mask && !fs->so->info.uses_fbfetch &&
      fragment_pass == AGX_PASS_TYPE_PUNCH_THROUGH &&
      batch->reduced_prim == MESA_PRIM_TRIANGLES &&
      batch->key.nr_cbufs == 1 &&
      batch->key.cbufs[0].format == PIPE_FORMAT_R8G8B8A8_UNORM &&
      !batch->key.zsbuf.texture;
   /* Native1094 fbfetch (Apple M6, verified): advanced-blend multiply at
    * 2x/4x and softlight at 4x on one RGBA8 target without depth are
    * translucent draws (PPP 0x02000062 word 1 0x20000200) whose fragment
    * holds one pixel transaction across every sample's tile read, blend and
    * write: pixwait512 -> pixwait8 -> lldcol ... pixwait12 -> lstcol ...
    * pixrel524. Mesa's sample-shaded colour main reads each sample and the
    * epilog writes each sample; the main takes the transaction entry before
    * its first read (agx3_pack_binary) and the linker drops the epilog's.
    * R2 word 6 (Apple's per-program compiler metadata) stays 0, as for every
    * Mesa program. */
   const bool g18_fbfetch_unrolled =
      fs && fs->so &&
      fs->so->info.uses_fbfetch && unrolled_colour &&
      fragment_pass == AGX_PASS_TYPE_TRANSLUCENT &&
      batch->reduced_prim == MESA_PRIM_TRIANGLES &&
      batch->key.nr_cbufs == 1 &&
      batch->key.cbufs[0].format == PIPE_FORMAT_R8G8B8A8_UNORM &&
      !batch->key.zsbuf.texture && !fs->b.info.early_fragment_tests &&
      !fs->b.info.writes_sample_mask;
   /* A translucent pixel-rate draw, blended or colour-masked: the epilog's
    * unrolled sample loop reads, merges and writes each covered sample whole
    * (build_g18_msaa4_fs_epilog; a partial mask is merged like a blend, and
    * F16 targets are read as F16, as the G18 sample transactions require).
    * Measured on the M6 against llvmpipe at 2x and 4x, with and without
    * depth/stencil, for one RGBA8, R11G11B10F or RGBA16F target: Google
    * Maps' area fills, which were dropped. A partial API sample mask makes
    * the pass translucent punch-through; the prolog kills the samples it
    * leaves out (per_pixel_api_mask), and the loop writes only covered ones. */
   const bool per_pixel_translucent =
      per_pixel_api_mask && !fs->so->info.uses_fbfetch &&
      (fragment_pass == AGX_PASS_TYPE_TRANSLUCENT ||
       fragment_pass == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH) &&
      batch->reduced_prim == MESA_PRIM_TRIANGLES &&
      batch->key.nr_cbufs == 1 &&
      (batch->key.cbufs[0].format == PIPE_FORMAT_R8G8B8A8_UNORM ||
       batch->key.cbufs[0].format == PIPE_FORMAT_R11G11B10_FLOAT ||
       batch->key.cbufs[0].format == PIPE_FORMAT_R16G16B16A16_FLOAT) &&
      ctx->blend->key.rt[0].colormask != 0;
   const bool admitted = agx_g18_msaa4_tile_layout(dev, &batch->tilebuffer_layout) &&
          (no_color || agx_g18_msaa_color_surfaces(batch)) &&
          agx_g18_msaa4_zs_surface(dev, &batch->key.zsbuf) &&
          (batch->reduced_prim == MESA_PRIM_TRIANGLES ||
           batch->reduced_prim == MESA_PRIM_LINES || per_pixel_points) &&
          (fragment_pass == AGX_PASS_TYPE_OPAQUE ||
           fragment_pass == AGX_PASS_TYPE_PUNCH_THROUGH ||
           (unrolled_memory &&
            (fragment_pass == AGX_PASS_TYPE_TRANSLUCENT ||
             fragment_pass == AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH)) ||
           resident_mrt_preserve || g18_fbfetch_unrolled ||
           per_pixel_translucent) &&
          ctx->rast->base.multisample &&
           (per_pixel_api_mask || per_pixel_api_shader_mask ||
            unrolled_colour || unrolled_memory ||
           (ctx->sample_mask & BITFIELD_MASK(batch->tilebuffer_layout.nr_samples)) ==
              BITFIELD_MASK(batch->tilebuffer_layout.nr_samples)) &&
          (!ctx->blend->key.alpha_to_coverage || per_pixel_a2c) &&
          fs && fs->so && (!fs->so->info.uses_fbfetch || g18_fbfetch_unrolled) &&
          (!fs->so->info.writes_memory || unrolled_memory) &&
          (!fs->so->info.uses_sample_shading || unrolled_depth ||
           unrolled_colour || unrolled_memory) &&
          (!fs->epilog_key.sample_shading || unrolled_colour) &&
          !fs->epilog_key.write_s &&
          (!fs->b.info.writes_sample_mask || per_pixel_mask ||
            unrolled_depth || unrolled_memory || unrolled_colour_mask ||
            unrolled_colour_sample_discard) &&
          (!fs->b.info.reads_tib || g18_fbfetch_unrolled);
   if (admitted && per_pixel_api_shader_mask &&
       (ctx->sample_mask & BITFIELD_MASK(batch->tilebuffer_layout.nr_samples)) !=
          BITFIELD_MASK(batch->tilebuffer_layout.nr_samples) &&
       (dev->debug & AGX_DBG_FAULTTRACE)) {
      fprintf(stderr,
              "G18_MSAA_API_SHADER_MASK api=%#x forms=%#x samples=%u "
              "kill_only_prolog=1 late_full_target=1\n",
              ctx->sample_mask, fs->b.info.g18_sample_mask_forms,
              batch->tilebuffer_layout.nr_samples);
   }
   if (!admitted && (dev->debug & AGX_DBG_FAULTTRACE)) {
      fprintf(stderr,
              "G18_MSAA_DRAW_REJECT layout=%u color=%u zs=%u prim=%u "
              "pass=%u opaque=%u punch=%u raster_ms=%u mask=%#x "
              "a2c=%u fs=%u so=%u fbfetch=%u memory=%u sample=%u "
              "epilog_sample=%u write_s=%u write_mask=%u reads_tib=%u "
              "unrolled_depth=%u unrolled_colour=%u unrolled_memory=%u\n",
              agx_g18_msaa4_tile_layout(dev, &batch->tilebuffer_layout),
              no_color ? batch->key.zsbuf.texture != NULL
                       : agx_g18_msaa_color_surfaces(batch),
              agx_g18_msaa4_zs_surface(dev, &batch->key.zsbuf),
              batch->reduced_prim, fragment_pass, AGX_PASS_TYPE_OPAQUE,
              AGX_PASS_TYPE_PUNCH_THROUGH, ctx->rast->base.multisample,
              ctx->sample_mask, ctx->blend->key.alpha_to_coverage,
              !!fs, fs && fs->so, fs && fs->so && fs->so->info.uses_fbfetch,
              fs && fs->so && fs->so->info.writes_memory,
              fs && fs->so && fs->so->info.uses_sample_shading,
              fs && fs->epilog_key.sample_shading,
              fs && fs->epilog_key.write_s,
              fs && fs->b.info.writes_sample_mask, fs && fs->b.info.reads_tib,
              unrolled_depth, unrolled_colour, unrolled_memory);
   }
   return admitted;
}

static uint8_t *
agx_encode_g17g_state(struct agx_batch *batch, uint8_t *out)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   /* A geometry-assisted draw -- transform feedback included -- rasterizes
    * through the GS copy program, which is then the hardware vertex shader,
    * as in agx_encode_state(). The profile below admits that only on G18G. */
   struct agx_compiled_shader *hw_vs = ctx->gs ? ctx->gs->gs_copy : ctx->vs;
   struct agx_linked_shader *hw_vs_linked =
      ctx->gs ? ctx->linked.gs_copy : ctx->linked.vs;
   const struct agx_compiled_shader *vs = hw_vs;
   struct agx_linked_shader *fs = ctx->linked.fs;
   const struct agx_rasterizer *rast = ctx->rast;
   struct agx_varyings_fs *cf = fs ? &fs->cf : NULL;
   const struct agx_cf_binding *color =
      cf && cf->nr_bindings >= 1 && cf->nr_bindings <= 3
         ? &cf->bindings[cf->nr_bindings - 1] : NULL;
   /* The UVS groups are laid out in enum order: position, varyings, point
    * size, layer/viewport, clip distance. The profile admits a point-size
    * group, a packed layer/viewport word on G18, and a clip-distance group.
    * Clip distances run from their group offset to the end. Native597
    * vertex references preserve the combined system-output word. */
   const bool psiz = vs && vs->uvs.group_offs[UVS_LAYER_VIEWPORT] ==
                              vs->uvs.group_offs[UVS_PSIZ] + 1;
   const bool layer_viewport =
      vs && vs->uvs.group_offs[UVS_CLIP_DIST] ==
               vs->uvs.group_offs[UVS_LAYER_VIEWPORT] + 1;
   const unsigned clip_dist_words =
      vs && vs->uvs.size > vs->uvs.group_offs[UVS_CLIP_DIST]
         ? vs->uvs.size - vs->uvs.group_offs[UVS_CLIP_DIST]
         : 0;
   const bool vertex_color_output =
      vs && vs->uvs.user_size >= 1 && vs->uvs.user_size <= 8 &&
      vs->uvs.size ==
         4 + vs->uvs.user_size + psiz + layer_viewport + clip_dist_words;
   bool color_varying = vertex_color_output && color;
   unsigned coefficient_count = 0;
   if (color_varying) {
      for (unsigned i = 0; i < cf->nr_bindings; ++i) {
         const struct agx_cf_binding *b = &cf->bindings[i];
         coefficient_count += b->count;
         if (b->slot == VARYING_SLOT_POS) {
            color_varying &= i == 0 && b->offset == 3 && b->count == 1 &&
                             b->smooth && !b->perspective;
         } else {
            /* Every binding must interpolate the same way as the colour
             * binding; smooth and perspective are independent, and a
             * smooth binding that is not perspective-correct is a
             * noperspective varying. */
            color_varying &= b->slot < ARRAY_SIZE(vs->uvs.components) &&
                             b->count > 0 &&
                             b->offset + b->count <= vs->uvs.components[b->slot] &&
                             b->smooth == color->smooth &&
                             b->perspective == color->perspective;
         }
      }
      /* Only a perspective-correct varying needs the position binding that
       * supplies its W; a flat or noperspective one does not, which is why
       * a noperspective blit arrives with a single binding. */
      color_varying &= coefficient_count == cf->nr_cf &&
         (color->perspective ? (cf->nr_bindings >= 2 &&
                               cf->bindings[0].slot == VARYING_SLOT_POS)
                             : cf->nr_bindings <= 2);
   }
   /* Shared utility vertex shaders can export a color that a constant-color
    * fragment shader ignores. Keep the vertex UVS layout intact while emitting
    * zero fragment interpolation bindings for that independently linked stage. */
   const bool no_varyings = vs && cf && !cf->nr_bindings && !cf->nr_cf &&
      ((vs->uvs.user_size == 0 &&
        vs->uvs.size == 4 + psiz + clip_dist_words) || vertex_color_output);
   /* G18G encodes any varying layout the linker produces: the VIT is the
    * AGX2 CF binding table and R1/R4 take the linker's VARYING_COUNTS and
    * OUTPUT_SELECT words, including an empty fragment
    * binding table. Vertex outputs retained for transform feedback still
    * contribute to the UVS counts, even when the fragment ignores them. */
   const bool general_varyings =
      dev->chip == AGX_CHIP_G18G && cf &&
      (cf->nr_bindings || !cf->nr_cf);
   const bool no_bias = !rast->depth_bias ||
      (rast->base.offset_units == 0 && rast->base.offset_scale == 0 &&
       rast->base.offset_clamp == 0);
   const unsigned fragment_pass = fs ? fs->fragment_control.opaque[0] >> 29 : 0;
   /*
    * A D32 depth attachment is admitted. Its per-draw state is the graphics
    * record's depth group, decoded from one controlled render per comparison
    * on 25G76 (gpu/m5-state-matrix-reference.m); the attachment itself, its
    * clear value and its load/store are the ZLS registers, which this
    * driver already fills generically and the kernel already forwards.
    * Depth testing against an attachment that was never bound is refused
    * rather than silently ignored. Stencil storage still needs its own
    * witnesses for the operation encodings and stays out.
    */
   /*
    * A render with no colour attachment at all is admitted, which is how
    * the depth-format conformance tests draw. The state that moves with it
    * is the graphics record's varying-count field and the ISP record's last
    * word, both taken from a controlled colour-free capture; everything
    * else, including the colour-free tile programs, is untouched because
    * there is no colour to load or store.
    */
   const bool color_attachment_ok =
      batch->key.nr_cbufs == 0 ||
      (batch->key.nr_cbufs == 1 && batch->key.cbufs[0].texture) ||
       (dev->chip == AGX_CHIP_G18G &&
        (agx_g18g_word_mrt_mask(&batch->tilebuffer_layout) ||
         agx_g18g_raw32_mrt_mask(&batch->tilebuffer_layout) ||
         agx_g18g_ms_mrt_mask(&batch->tilebuffer_layout) ||
         agx_g18g_native_mixed_mask(&batch->tilebuffer_layout)));
   const struct pipe_surface *zsbuf = &batch->key.zsbuf;
   /* With the comparison disabled the depth unit performs no test and, per
    * the API, no write either, and a disabled stencil unit does nothing at
    * all. The captured record for that state is the inactive depth group
    * regardless of what is attached, and the attachment itself is described
    * only by the ZLS registers, which Mesa packs from the layout for every
    * depth format and the kernel forwards unchanged. So an attachment this
    * render never reads or writes is admitted whatever its format -- which
    * is how a conformance surface with a depth-stencil buffer draws for
    * every case that does not ask for depth. Actually testing against an
    * attachment still needs the witnessed D32 depth group.
    */
   const bool zs_units_idle = !ctx->zs->base.depth_enabled &&
                              !ctx->zs->base.stencil[0].enabled &&
                              !ctx->zs->base.stencil[1].enabled;
   /* Which depth formats may be tested against. Two controlled renders on
    * 25G76 answered this (gpu/m5-state-matrix-reference.m modes depth16 and
    * depth32s8, captured 2026-09-14): a 16-bit depth plane moves exactly the
    * ZLS control's format field, the clear register from a float to a 0..
    * 65535 integer, and one ISP flag -- all three of which the driver and
    * the kernel already derive from the layout -- while no PPP record moves
    * at all. A combined depth-stencil plane used for depth alone is
    * byte-identical to a plain D32 one. G18 stencil state is encoded below
    * using the shared operation words and native per-face references.
    */
   /* The attachment's hardware format, from the driver's layout: the
    * pipe_resource keeps the API format, but u_transfer_helper allocates the
    * API's Z24X8 as a Z32F plane and Z24S8 as Z32F plus separate S8, so
    * glxgears' 24-bit depth visual renders into exactly the planes above. */
   const enum pipe_format zs_format =
      zsbuf->texture ? agx_resource(zsbuf->texture)->layout.format
                     : zsbuf->format;
   const bool depth_format_ok =
      zs_format == PIPE_FORMAT_Z32_FLOAT ||
      zs_format == PIPE_FORMAT_Z16_UNORM ||
      zs_format == PIPE_FORMAT_Z32_FLOAT_S8X24_UINT ||
      (dev->chip == AGX_CHIP_G18G && zs_format == PIPE_FORMAT_S8_UINT &&
       !ctx->zs->base.depth_enabled);
   /* An attachmentless framebuffer still rasterizes fragment image stores.
    * Reuse the colour-free tile programs and inactive ZLS state; neither
    * requires a backing attachment. Native1061's1x/2x/4x visibility renders
    * use the same empty colour programs and inactive ZLS state. The ordinary
    * multisample draw checks below still validate the actual shader/mask.
    */
   const bool attachmentless =
      dev->chip == AGX_CHIP_G18G && batch->key.nr_cbufs == 0 &&
      !zsbuf->texture && zs_units_idle &&
      (batch->tilebuffer_layout.nr_samples == 1 ||
       agx_g18_msaa4_tile_layout(dev, &batch->tilebuffer_layout));
   const bool depth_attachment_ok =
      attachmentless ||
      ((batch->key.nr_cbufs || zsbuf->texture) &&
       (zsbuf->texture ? (zs_units_idle || depth_format_ok)
                      : !ctx->zs->base.depth_enabled));
   /* A fragment that writes gl_FragDepth is carried by the record encoder's
    * fragment_writes_depth; one that writes stencil moves fields with no
    * witness, and encoding it as though it did not would be silently wrong
    * rendering rather than a refusal. */
   /* Twenty-odd independent conditions; a draw that misses the profile is
    * worth naming the one it missed, exactly as the program path does. */
   const char *outside_profile = NULL;
#define M5_DRAW_OUTSIDE(cond, reason)                                          \
   do {                                                                        \
      if (!outside_profile && (cond))                                          \
         outside_profile = (reason);                                           \
   } while (0)

   M5_DRAW_OUTSIDE(!vs, "no vertex shader");
   M5_DRAW_OUTSIDE(!fs, "no fragment shader");
   /* G18G rasterizes a geometry-assisted draw through the GS copy program,
    * whose UVS layout the terms below check like any vertex shader's. The
    * indexed shapes draw a restarted U32 list (dynamic topology) or a U16
    * list widened from the static topology, the direct indexed forms the
    * J873 captures show with restart (agx_vdm_index_list_core.h). */
   M5_DRAW_OUTSIDE(ctx->gs && dev->chip != AGX_CHIP_G18G, "geometry shader");
   if (vs && fs && cf) {
      M5_DRAW_OUTSIDE(!general_varyings && !no_varyings && !color_varying,
                      "varyings outside the colour shape");
      M5_DRAW_OUTSIDE(!general_varyings && cf->reads_z, "fragment reads Z");
      M5_DRAW_OUTSIDE(ctx->fs->epilog_key.write_s, "fragment writes stencil");
      M5_DRAW_OUTSIDE(vs->b.info.has_preamble || vs->b.info.nr_preamble_gprs ||
                         vs->b.info.preamble_scratch_size,
                      "vertex preamble");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_POSITION] != 0,
                      "UVS position group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_VARYINGS] != 4,
                      "UVS varying group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_PSIZ] != 4 + vs->uvs.user_size,
                      "UVS point-size group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_LAYER_VIEWPORT] !=
                         4 + vs->uvs.user_size + psiz,
                      "UVS layer/viewport group");
      M5_DRAW_OUTSIDE(vs->uvs.group_offs[UVS_CLIP_DIST] !=
                         vs->uvs.group_offs[UVS_LAYER_VIEWPORT] + layer_viewport ||
                         (layer_viewport && dev->chip != AGX_CHIP_G18G),
                      "UVS clip-distance group");
      /* The output selector's low sixteen bits are the sixteen named
       * clip-distance planes; bit 18 is point size. Each enabled plane
       * contributes one UVS word, which is the same count the graphics
       * record carries, so the two have to agree. Native597's UVS constructor
       * also emits bits 19/20 for the shared layer/viewport word. Gallium
       * packs both halves together, including a zero for an absent output.
       * The vertex varying bit remains clear; the fragment contributes it. */
      M5_DRAW_OUTSIDE((vs->uvs.osel.opaque[0] & ~UINT32_C(0xffff)) !=
                         ((psiz ? 0x40000 : 0) |
                          (layer_viewport ? 0x180000 : 0)),
                      "UVS output selector");
      M5_DRAW_OUTSIDE(util_bitcount(vs->uvs.osel.opaque[0] & 0xffff) !=
                         clip_dist_words,
                      "UVS clip-distance planes");
      M5_DRAW_OUTSIDE(vs->b.info.nonzero_viewport && !layer_viewport,
                      "viewport index without UVS system output");
      M5_DRAW_OUTSIDE(fragment_pass != AGX_PASS_TYPE_OPAQUE &&
                         fragment_pass != AGX_PASS_TYPE_TRANSLUCENT &&
                         fragment_pass != AGX_PASS_TYPE_PUNCH_THROUGH &&
                         !(dev->chip == AGX_CHIP_G18G &&
                           fragment_pass ==
                              AGX_PASS_TYPE_TRANSLUCENT_PUNCH_THROUGH),
                      "fragment pass type");
      const bool g18_late_memory =
         dev->chip == AGX_CHIP_G18G &&
         (batch->tilebuffer_layout.nr_samples == 1 ||
          agx_g18_msaa_memory_shader(ctx, batch)) &&
         !ctx->fs->b.info.early_fragment_tests &&
         !(ctx->zs->base.depth_enabled &&
           ctx->zs->base.depth_func == PIPE_FUNC_NEVER);
      const bool g18_early_memory =
         dev->chip == AGX_CHIP_G18G &&
         batch->tilebuffer_layout.nr_samples == 1 &&
         ctx->fs->b.info.early_fragment_tests &&
         ctx->fs->so &&
         !ctx->fs->so->info.writes_api_sample_mask;
      /* The main acquires the pixel transaction before helper-predicated
       * writes. One-sample explicit early tests use the native MOVMSK
       * transaction before those writes and suppress epilog retesting;
       * late shaders supply depth feedback afterward. Late depth-NEVER
       * emulation and explicit early multisample paths remain separate.
       * The statically expanded MS memory shader keeps the late ordering
       * in each unrolled body. */
      M5_DRAW_OUTSIDE(ctx->fs->so && ctx->fs->so->info.writes_memory &&
                         !g18_late_memory && !g18_early_memory,
                      "fragment writes memory");
   }
   /* Points join triangles: the object type is witnessed end to end (see
    * agx_ppp_hal300_graphics_fields::object_type) and Apple's own point
    * draw exports a point size, which the profile's UVS terms above already
    * require to agree with the output selector. */
   /* G18G draws lines too: the J873 "lines" render moves exactly the words
    * points do, with the early-test words' 0x10000000 and no point size
    * (agx_ppp_hal300_graphics_fields::object_type). Line width is carried
    * separately in the native-grounded front/back face records. */
   M5_DRAW_OUTSIDE(batch->reduced_prim != MESA_PRIM_TRIANGLES &&
                      batch->reduced_prim != MESA_PRIM_POINTS &&
                      !(dev->chip == AGX_CHIP_G18G &&
                        batch->reduced_prim == MESA_PRIM_LINES),
                   "primitive is neither triangles nor points");
   M5_DRAW_OUTSIDE(batch->reduced_prim == MESA_PRIM_POINTS && !psiz,
                   "point draw without a point size");
   M5_DRAW_OUTSIDE(!color_attachment_ok, "colour attachment shape");
   M5_DRAW_OUTSIDE(!depth_attachment_ok, "depth attachment shape");
   M5_DRAW_OUTSIDE(batch->key.layers > 1 &&
                        (dev->chip != AGX_CHIP_G18G ||
                         (batch->key.nr_cbufs != 1 && !attachmentless &&
                          !agx_g18g_word_mrt_mask(&batch->tilebuffer_layout)) ||
                        /* D16 joins Z32 only in native1060's array shapes. */
                        (batch->key.zsbuf.texture &&
                         (batch->tilebuffer_layout.nr_samples != 1 ||
                          (zs_format != PIPE_FORMAT_Z32_FLOAT &&
                           zs_format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT &&
                           !agx_g18_layered_d16_native(dev, batch)))) ||
                        (batch->tilebuffer_layout.nr_samples != 1 &&
                         !agx_g18_msaa4_tile_layout(dev, &batch->tilebuffer_layout))),
                   "layered render");
   M5_DRAW_OUTSIDE(batch->tilebuffer_layout.nr_samples != 1 &&
                      !agx_g18_msaa4_draw_admitted(ctx, batch, fragment_pass),
                   "multisampled");
   /* Grounded on G18G only: the VDM variant and the cull-2 word below. */
   M5_DRAW_OUTSIDE(rast->base.rasterizer_discard && dev->chip != AGX_CHIP_G18G,
                   "rasterizer discard");
   M5_DRAW_OUTSIDE(rast->base.fill_front != PIPE_POLYGON_MODE_FILL ||
                      rast->base.fill_back != PIPE_POLYGON_MODE_FILL,
                   "polygon fill mode");
   M5_DRAW_OUTSIDE(!no_bias && dev->chip != AGX_CHIP_G18G, "depth bias");
   M5_DRAW_OUTSIDE(rast->base.depth_clamp, "depth clamp");
   M5_DRAW_OUTSIDE(!rast->base.depth_clip_near || !rast->base.depth_clip_far,
                   "depth clip disabled");
   M5_DRAW_OUTSIDE(dev->chip != AGX_CHIP_G18G &&
                      (ctx->zs->base.stencil[0].enabled ||
                       ctx->zs->base.stencil[1].enabled),
                   "stencil test");
   M5_DRAW_OUTSIDE(dev->chip != AGX_CHIP_G18G &&
                      ctx->active_queries && ctx->occlusion_query,
                   "occlusion query");
#undef M5_DRAW_OUTSIDE

   if (outside_profile) {
      mesa_loge("G17G Gallium draw exceeds the native triangle profile: %s",
                outside_profile);
      if ((dev->debug & AGX_DBG_FAULTTRACE) && vs && fs) {
         fprintf(stderr, "M5_DRAW_REJECT varyings=%u/%u psiz=%u uvs=%u user=%u groups=%u,%u,%u,%u,%u osel=%#x cf=%u,%u reads_z=%u preamble=%u,%u,%u viewport=%u\n",
                 no_varyings, color_varying, psiz, vs->uvs.size, vs->uvs.user_size,
                 vs->uvs.group_offs[UVS_POSITION], vs->uvs.group_offs[UVS_VARYINGS],
                 vs->uvs.group_offs[UVS_PSIZ], vs->uvs.group_offs[UVS_LAYER_VIEWPORT],
                 vs->uvs.group_offs[UVS_CLIP_DIST], vs->uvs.osel.opaque[0],
                 cf->nr_bindings, cf->nr_cf, cf->reads_z, vs->b.info.has_preamble,
                 vs->b.info.nr_preamble_gprs, vs->b.info.preamble_scratch_size,
                 vs->b.info.nonzero_viewport);
         if (color)
            fprintf(stderr, "M5_DRAW_REJECT_CF offset=%u count=%u smooth=%u perspective=%u\n",
                    color->offset, color->count, color->smooth, color->perspective);
         fprintf(stderr, "M5_DRAW_REJECT_STATE prim=%u cbufs=%u zs=%s layers=%u samples=%u pass=%u discard=%u fill=%u,%u bias=%u clamp=%u clip=%u,%u depth=%u stencil=%u,%u query=%u\n",
                 batch->reduced_prim, batch->key.nr_cbufs,
                 batch->key.zsbuf.texture
                    ? util_format_name(batch->key.zsbuf.format)
                    : "none",
                 batch->key.layers, batch->tilebuffer_layout.nr_samples,
                 fs->fragment_control.opaque[0] >> 29, rast->base.rasterizer_discard,
                 rast->base.fill_front, rast->base.fill_back, !no_bias,
                 rast->base.depth_clamp, rast->base.depth_clip_near,
                 rast->base.depth_clip_far, ctx->zs->base.depth_enabled,
                 ctx->zs->base.stencil[0].enabled, ctx->zs->base.stencil[1].enabled,
                 !!(ctx->active_queries && ctx->occlusion_query));
      } else if (dev->debug & AGX_DBG_FAULTTRACE) {
         /* A draw without a fragment shader still has attachment state worth
          * seeing; the varying report above simply has nothing to say. */
         fprintf(stderr,
                 "M5_DRAW_REJECT_STATE prim=%u cbufs=%u zs=%s layers=%u "
                 "samples=%u vs=%u fs=%u\n",
                 batch->reduced_prim, batch->key.nr_cbufs,
                 util_format_name(batch->key.zsbuf.format), batch->key.layers,
                 batch->tilebuffer_layout.nr_samples, !!vs, !!fs);
      }
      return NULL;
   }

   batch->varyings = 0;
   batch->generate_primitive_id = false;
   /* The empty-table path must observe the same 32-bit-only restriction. */
   if (general_varyings && batch->linked_varyings.counts_16.opaque[0] != 0)
      return NULL;
   if (cf->nr_bindings) {
      const size_t size = AGX_CF_BINDING_HEADER_LENGTH +
                          cf->nr_bindings * AGX_CF_BINDING_LENGTH;
      struct agx_ptr t = agx_pool_alloc_aligned(&batch->pipeline_pool, size, 64);
      if (!t.cpu)
         return NULL;
      agx_link_varyings_vs_fs(t.cpu, &batch->linked_varyings, vs->uvs.user_size,
                              cf, rast->base.flatshade_first ? 0 : 2,
                              general_varyings &&
                                    batch->reduced_prim == MESA_PRIM_POINTS
                                 ? rast->base.sprite_coord_enable
                                 : 0,
                              &batch->generate_primitive_id);
      if (agx_chip_uses_shade_model_5(dev->chip))
         agx_cf_bindings_select_perspective(t.cpu, cf);
      if (!general_varyings) {
         /* Varying Counts packs smooth at bit 0, flat at bit 8 and linear
          * (noperspective) at bit 16; the record below uses the same
          * layout. */
         const unsigned interpolation_shift =
            color->perspective ? 0 : color->smooth ? 16 : 8;
         if (batch->generate_primitive_id ||
             batch->linked_varyings.counts_16.opaque[0] != 0 ||
             batch->linked_varyings.counts_32.opaque[0] !=
                (vs->uvs.user_size << interpolation_shift))
            return NULL;
      }
      batch->varyings = t.gpu;
   }

   uint64_t vertex_program = 0, fragment_program = 0;
   const char *vertex_why = NULL, *fragment_why = NULL;
   agx_build_pipeline(batch, hw_vs, hw_vs_linked,
                      MESA_SHADER_VERTEX, 0, 0, &vertex_program, &vertex_why);
   agx_build_pipeline(batch, agx_fs_launch_shader(ctx), ctx->linked.fs,
                      MESA_SHADER_FRAGMENT, 0, 0, &fragment_program, &fragment_why);
   const bool dropped_vertex = agx_drop_for_null_g17p_program(
      ctx, hw_vs, vertex_program, "vertex", vertex_why);
   const bool dropped_fragment = agx_drop_for_null_g17p_program(
      ctx, ctx->fs, fragment_program, "fragment", fragment_why);
   if (dropped_vertex || dropped_fragment)
      return NULL;

   /* A pipeline built with Metal's rasterizationEnabled = NO uses the
    * rasterization-disabled vertex program variant; native G18 does the
    * same. */
   const struct agx_vdm_state_update_draw_info draw_info = {
      .source_layout = AGX_VDM_STATE_UPDATE_SOURCE_LAYOUT_AGX3_VERTEX_PROGRAM_VARIANT,
      .draw_path = AGX_VDM_STATE_UPDATE_DRAW_DIRECT_NONINDEXED,
      .rasterization_enabled = !rast->base.rasterizer_discard,
      .raw_provoking_selector = rast->base.flatshade_first ? 0 : 2,
      .template_info = {
         .pipeline = vertex_program,
         .raw_vs_output_count_0 = vs->uvs.size,
         /* Generated primitive ID occupies an extra hardware output. Keep
          * allocation and generation coupled, as in the Vulkan draw path. */
         .raw_vs_output_count_1_extra = batch->generate_primitive_id,
         .raw_vs_output_flag_6 = batch->generate_primitive_id,
         .raw_active_context_selector = UINT32_MAX,
      },
   };
   struct agx_vdm_state_update vertex;
   if (!agx_vdm_state_update_draw_pack(dev->chip, &draw_info, &vertex))
      return NULL;

   enum agx_visibility_mode visibility = AGX_VISIBILITY_MODE_NONE;
   uint16_t query_index = 0;
   if (ctx->active_queries && ctx->occlusion_query) {
      visibility = ctx->occlusion_query->type == PIPE_QUERY_OCCLUSION_COUNTER
                      ? AGX_VISIBILITY_MODE_COUNTING
                      : AGX_VISIBILITY_MODE_BOOLEAN;
      /* This also retains the heap BO and records the batch as a writer,
       * so result reads and slot reuse wait for the query draws. */
      query_index = agx_get_oq_index(batch, ctx->occlusion_query);
   }

   const struct agx_ppp_hal300_graphics_fields graphics_fields = {
      .fragment_program = fragment_program,
      .occlusion_query_mode = visibility,
      .occlusion_query_index = query_index,
      .scissor_enable = dev->chip == AGX_CHIP_G18G,
      .depth_bias_enable = dev->chip == AGX_CHIP_G18G && rast->depth_bias &&
         batch->reduced_prim == MESA_PRIM_TRIANGLES &&
         (rast->base.offset_units != 0.0f || rast->base.offset_scale != 0.0f),
      .no_color_attachment = batch->key.nr_cbufs == 0,
      /* Run 1139: the mixed-format epilog neither loads nor stores a target
       * whose colour mask is empty, so with every mask empty the linked
       * program writes nothing and the linker marks it no_op. Apple draws
       * that with the colour-free ISP word (see the field). */
      .fragment_writes_nothing =
         dev->chip == AGX_CHIP_G18G && batch->key.nr_cbufs != 0 && fs->no_op,
      .fragment_pass = fragment_pass,
      /* Bit 26 of the shader's own fragment control; a discarding shader
       * carries it beside its punch-through pass type. Native1094's 2x/4x
       * alpha-to-coverage draws (Apple M6, verified) set it in PPP
       * 0x02000062 word 1 (0x44000200) for every draw; Mesa's coverage is
       * lowered into the epilog, so it is set here for the admitted A2C
       * profile (agx_g18_msaa4_draw_admitted). Native1121's 2x/4x
       * per-sample discard draws (run 1128) set the same word 1
       * (0x44000200); Mesa's unrolled program does not carry the bit, so it
       * is set for the admitted per-sample discard profile, whose only mask
       * form is AGX_G18_SAMPLE_MASK_SAMPLE. */
      .disable_tri_merging =
         (fs->fragment_control.opaque[0] & BITFIELD_BIT(26)) != 0 ||
         (dev->chip == AGX_CHIP_G18G &&
          batch->tilebuffer_layout.nr_samples > 1 &&
          ctx->blend->key.alpha_to_coverage) ||
         (dev->chip == AGX_CHIP_G18G &&
          batch->tilebuffer_layout.nr_samples > 1 &&
          ctx->fs->b.info.g18_sample_mask_forms ==
             AGX_G18_SAMPLE_MASK_SAMPLE),
      .fragment_writes_depth = ctx->fs->epilog_key.write_z,
      .varying_table = batch->varyings, .cf_binding_count = cf->nr_bindings,
      .cull_mode = rast->base.cull_face,
      .front_face_ccw = agx_front_face_ccw(ctx), .point_size_output = psiz,
      .clip_distance_words = clip_dist_words,
      .line_width_valid = dev->chip == AGX_CHIP_G18G &&
                          batch->reduced_prim == MESA_PRIM_LINES,
      .line_width = rast->line_width,
      .object_type = batch->reduced_prim == MESA_PRIM_POINTS
                        ? AGX_PPP_HAL300_POINTS
                        : batch->reduced_prim == MESA_PRIM_LINES
                        ? AGX_PPP_HAL300_LINES
                        : AGX_PPP_HAL300_TRIANGLES,
      .vertex_varying_components = vs->uvs.user_size,
      .fragment_varying_components = color_varying ? vs->uvs.user_size : 0,
      .interpolation = !color_varying || color->perspective
                          ? AGX_PPP_HAL300_SMOOTH
                          : color->smooth ? AGX_PPP_HAL300_LINEAR
                                          : AGX_PPP_HAL300_FLAT,
      /* PIPE_FUNC_* and the record share the never..always order. A depth
       * attachment with the test disabled encodes exactly like no
       * attachment at all, which the M5 capture matrix established. */
      .depth_active = ctx->zs->base.depth_enabled,
      .depth_compare =
         ctx->zs->base.depth_enabled ? ctx->zs->base.depth_func : 0,
      .depth_write =
         ctx->zs->base.depth_enabled && ctx->zs->base.depth_writemask,
      .stencil_active = ctx->zs->base.stencil[0].enabled,
      .front_stencil_reference = ctx->zs->base.stencil[0].enabled
                                   ? ctx->stencil_ref.ref_value[0] : 0,
      .back_stencil_reference = ctx->zs->base.stencil[0].enabled
         ? ctx->stencil_ref.ref_value[ctx->zs->base.stencil[1].enabled ? 1 : 0]
         : 0,
      .front_stencil = agx_g17p_stencil_word(&ctx->zs->base.stencil[0]),
      .back_stencil = agx_g17p_stencil_word(ctx->zs->base.stencil[1].enabled
                                             ? &ctx->zs->base.stencil[1]
                                             : &ctx->zs->base.stencil[0]),
      .rasterizer_discard = rast->base.rasterizer_discard,
      .general_varyings = general_varyings,
      .varying_counts_32 = batch->linked_varyings.counts_32.opaque[0],
      .varying_counts_16 = batch->linked_varyings.counts_16.opaque[0],
      .output_select = vs->uvs.osel.opaque[0] | fs->osel.opaque[0],
      .uvs_words = vs->uvs.size,
      .needs_primitive_id = general_varyings && batch->generate_primitive_id,
      /* GL's default is the last vertex; the J873 captures carry 0x580 for
       * it and 0x480 for first. The VIT copy above already follows the same
       * selector. */
      .flat_shading_vertex =
         dev->chip == AGX_CHIP_G18G && !rast->base.flatshade_first ? 2 : 0,
   };
   struct agx_ppp_hal300_graphics_core graphics;
   if (!agx_ppp_hal300_graphics_core_pack(dev->chip, &graphics_fields, &graphics))
      return NULL;

   struct agx_vdm_ppp_state_core records[2 + AGX_PPP_HAL300_GRAPHICS_RECORDS];
   unsigned record_count = 0;
   /* Shader rebinding can change the viewport count, including when a GS
    * copy program becomes the hardware vertex shader. As on AGX2, keep the
    * full VS dirty dependency as well as rasterizer/viewport/scissor state.
    * The pool and encoded state live for the batch; batch switches dirty all.
    */
   if (ctx->dirty & (AGX_DIRTY_VIEWPORT | AGX_DIRTY_SCISSOR_ZBIAS |
                     AGX_DIRTY_RS | AGX_DIRTY_VS)) {
      if (!agx_g17g_viewport_scissor_records(batch, vs, records))
         return NULL;
      record_count = 2;
   }
   const uint64_t graphics_gpu = agx_pool_upload_aligned(&batch->pool, graphics.bytes,
                                                         sizeof(graphics.bytes), 64);
   if (!graphics_gpu)
      return NULL;
   for (unsigned i = 0; i < AGX_PPP_HAL300_GRAPHICS_RECORDS; ++i) {
      if (!agx_gallium_ppp_record(dev->chip, graphics_gpu + graphics.offsets[i],
                                      graphics.lengths[i] / 4, &records[record_count]))
         return NULL;
      record_count++;
   }

   memcpy(out, vertex.bytes, vertex.length);
   out += vertex.length;
   for (unsigned i = 0; i < record_count; ++i) {
      memcpy(out, records[i].bytes, records[i].length);
      out += records[i].length;
   }
   return out;
}

static uint8_t *
agx_g17p_emit_vs_state_update(struct agx_batch *batch, uint8_t *out,
                              const struct agx_compiled_shader *vs,
                              uint32_t raw_word0, uint64_t pipeline,
                              uint32_t outputs_out[2])
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   const uint32_t provoking = ctx->rast->base.flatshade_first ? 0 : 2;
   const uint32_t outputs[2] = {
      vs->uvs.vdm.opaque[0] +
         (batch->generate_primitive_id ? UINT32_C(0x100) : 0),
      (ctx->rast->base.rasterizer_discard ? 0x30 + provoking : provoking) |
         (batch->generate_primitive_id ? UINT32_C(0x40) : 0),
   };
   const struct agx_vdm_state_update_info info = {
      .raw_vertex_shader_word0 = raw_word0,
      .pipeline = pipeline,
      .raw_pipeline_template = 0,
      .raw_vertex_outputs = {outputs[0], outputs[1]},
      .raw_vertex_unknown = 0,
   };
   struct agx_vdm_state_update encoded;
   const bool packed = agx_vdm_state_update_pack(dev->chip, &info, &encoded);
   assert(packed);
   if (!packed)
      return NULL;

   memcpy(out, encoded.bytes, encoded.length);
   if (outputs_out) {
      outputs_out[0] = outputs[0];
      outputs_out[1] = outputs[1];
   }
   return out + encoded.length;
}

static uint8_t *
agx_encode_state(struct agx_batch *batch, uint8_t *out,
                 uint32_t *draw_state_offset)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);

   /* If nothing is dirty, encode nothing */
   if (!ctx->dirty)
      return out;

   if (draw_state_offset)
      *draw_state_offset = 0;

   if (agx_chip_hal300_path(dev->chip)) {
      uint8_t *encoded = agx_chip_g17_m5(dev->chip)
                            ? agx_encode_native_g17g_state(batch, out)
                            : agx_encode_g17g_state(batch, out);
      if (encoded == NULL)
         batch->draws_refused++;
      return encoded;
   }

   struct agx_rasterizer *rast = ctx->rast;
   unsigned ppp_updates = 0;

   struct agx_compiled_shader *vs = ctx->vs;
   if (ctx->gs)
      vs = ctx->gs->gs_copy;

   bool varyings_dirty = false;

   if (IS_DIRTY(VS_PROG) || IS_DIRTY(FS_PROG) || IS_DIRTY(RS) ||
       IS_DIRTY(PRIM)) {

      unsigned bindings = ctx->linked.fs->cf.nr_bindings;
      if (bindings) {
         size_t linkage_size =
            AGX_CF_BINDING_HEADER_LENGTH + (bindings * AGX_CF_BINDING_LENGTH);

         struct agx_ptr t =
            agx_pool_alloc_aligned(&batch->pipeline_pool, linkage_size,
                                   agx_chip_native_agx3(dev->chip) ? 64 : 16);

         agx_link_varyings_vs_fs(t.cpu, &batch->linked_varyings,
                                 vs->uvs.user_size, &ctx->linked.fs->cf,
                                 ctx->rast->base.flatshade_first ? 0 : 2,
                                 (batch->reduced_prim == MESA_PRIM_POINTS)
                                    ? ctx->rast->base.sprite_coord_enable
                                    : 0,
                                 &batch->generate_primitive_id);

         batch->varyings =
            agx_chip_native_agx3(dev->chip) ? t.gpu : agx_usc_addr(dev, t.gpu);
      } else {
         batch->varyings = 0;
         batch->generate_primitive_id = false;
      }

      varyings_dirty = true;
      ppp_updates++;
   }

   bool vs_emitted = false;
   uint32_t vs_word0 = 0;
   uint32_t vs_outputs[2] = {0, 0};

   /* HAL200 retains the vertex state update independently of its bind tail
    * and per-draw selector, just as the older VDM retains VS state. The old
    * immutable uniform upload remains valid until vertex inputs change. */
   if (IS_DIRTY(VS) || varyings_dirty) {
      struct agx_vdm_state_vertex_shader_word_0_packed counts;
      agx_pack(&counts, VDM_STATE_VERTEX_SHADER_WORD_0, cfg) {
         cfg.uniform_register_count = vs->b.info.push_count;
         cfg.preshader_register_count = vs->b.info.nr_preamble_gprs;
         cfg.texture_state_register_count = agx_nr_tex_descriptors(batch, vs);
         cfg.sampler_state_register_count =
            translate_sampler_state_count(ctx, vs->stage);
      }

      uint64_t pipeline = 0;
      const uint32_t packed_pipeline =
         agx_build_pipeline(batch, vs,
                            ctx->gs ? ctx->linked.gs_copy : ctx->linked.vs,
                            MESA_SHADER_VERTEX, 0, 0, &pipeline, NULL);

      if (agx_chip_native_agx3(dev->chip)) {
         if (agx_drop_for_null_g17p_program(ctx, vs, pipeline, "vertex",
                                            NULL))
            return NULL;

         vs_word0 = counts.opaque[0];
         out = agx_g17p_emit_vs_state_update(batch, out, vs, vs_word0,
                                             pipeline, vs_outputs);
         if (!out)
            return NULL;
         vs_emitted = true;
      } else {
         agx_push(out, VDM_STATE, cfg) {
            cfg.vertex_shader_word_0_present = true;
            cfg.vertex_shader_word_1_present = true;
            cfg.vertex_outputs_present = true;
            cfg.vertex_unknown_present = true;
         }
         agx_push_packed(out, counts, VDM_STATE_VERTEX_SHADER_WORD_0);
         agx_push(out, VDM_STATE_VERTEX_SHADER_WORD_1, cfg) {
            cfg.pipeline = packed_pipeline;
         }
         agx_push_packed(out, vs->uvs.vdm, VDM_STATE_VERTEX_OUTPUTS);
         agx_push(out, VDM_STATE_VERTEX_UNKNOWN, cfg) {
            cfg.flat_shading_control = ctx->rast->base.flatshade_first
                                          ? AGX_VDM_VERTEX_0
                                          : AGX_VDM_VERTEX_2;
            cfg.unknown_4 = cfg.unknown_5 =
               ctx->rast->base.rasterizer_discard;
            cfg.generate_primitive_id = batch->generate_primitive_id;
         }

         /* Pad up to a multiple of 8 bytes */
         memset(out, 0, 4);
         out += 4;
      }

      if (unlikely(dev->debug & AGX_DBG_PERF))
         ctx->perf.vs_emits++;
   }

   if (agx_chip_native_agx3(dev->chip)) {
      /* Elide the HAL200 bind tail when only vertex uniforms or VBO bindings
       * changed and the current VDM chunk already latched the matching PPP
       * bind tail and vertex metadata. */
      const bool need_bind_tail =
         !batch->g17p_bind_tail_latched || varyings_dirty ||
         (ctx->dirty & ~(AGX_DIRTY_VS | AGX_DIRTY_VERTEX)) != 0 ||
         batch->g17p_latched_vs_shader != vs ||
         batch->g17p_fragment_shader != agx_fs_launch_shader(ctx) ||
         batch->g17p_fragment_linked != ctx->linked.fs ||
         (vs_emitted &&
          (!batch->g17p_have_latched_vs ||
           vs_word0 != batch->g17p_latched_vs_word0 ||
           vs_outputs[0] != batch->g17p_latched_vs_outputs[0] ||
           vs_outputs[1] != batch->g17p_latched_vs_outputs[1]));
      if (!need_bind_tail)
         return out;

      out = agx_g17p_emit_bind_state(batch, out, vs, draw_state_offset);
      if (!out)
         return NULL;

      batch->g17p_bind_tail_latched = true;
      batch->g17p_latched_vs_shader = vs;
      if (vs_emitted) {
         batch->g17p_have_latched_vs = true;
         batch->g17p_latched_vs_word0 = vs_word0;
         batch->g17p_latched_vs_outputs[0] = vs_outputs[0];
         batch->g17p_latched_vs_outputs[1] = vs_outputs[1];
      }
      return out;
   }

   struct agx_pool *pool = &batch->pool;

   if ((ctx->dirty & AGX_DIRTY_RS) && ctx->rast->depth_bias) {
      agx_upload_depth_bias(batch, &ctx->rast->base);
      ctx->dirty |= AGX_DIRTY_SCISSOR_ZBIAS;
   }

   if (ctx->dirty & (AGX_DIRTY_VIEWPORT | AGX_DIRTY_SCISSOR_ZBIAS |
                     AGX_DIRTY_RS | AGX_DIRTY_VS)) {

      agx_upload_viewport_scissor(pool, batch, &out, ctx->viewport,
                                  ctx->rast->base.scissor ? ctx->scissor : NULL,
                                  ctx->rast->base.clip_halfz,
                                  vs->b.info.nonzero_viewport);
   }

   bool is_points = batch->reduced_prim == MESA_PRIM_POINTS;
   bool is_lines = batch->reduced_prim == MESA_PRIM_LINES;

   bool object_type_dirty =
      IS_DIRTY(PRIM) || (is_points && IS_DIRTY(SPRITE_COORD_MODE));

   bool fragment_face_dirty =
      IS_DIRTY(ZS) || IS_DIRTY(STENCIL_REF) || IS_DIRTY(RS);

   enum agx_object_type object_type = is_points  ? agx_point_object_type(rast)
                                      : is_lines ? AGX_OBJECT_TYPE_LINE
                                                 : AGX_OBJECT_TYPE_TRIANGLE;

   struct AGX_PPP_HEADER dirty = {
      .fragment_control =
         IS_DIRTY(ZS) || IS_DIRTY(RS) || IS_DIRTY(PRIM) || IS_DIRTY(QUERY),
      .fragment_control_2 = IS_DIRTY(FS_PROG) || IS_DIRTY(RS),
      .fragment_front_face = fragment_face_dirty,
      .fragment_front_face_2 = object_type_dirty || IS_DIRTY(FS_PROG),
      .fragment_front_stencil = IS_DIRTY(ZS),
      .fragment_back_face = fragment_face_dirty,
      .fragment_back_face_2 = object_type_dirty || IS_DIRTY(FS_PROG),
      .fragment_back_stencil = IS_DIRTY(ZS),
      .output_select = varyings_dirty,
      .varying_counts_32 = varyings_dirty,
      .varying_counts_16 = varyings_dirty,
      /* Also dirty with tess but agx_draw_patches dirties RS for that */
      .cull = IS_DIRTY(RS),
      .cull_2 = varyings_dirty,
      .fragment_shader =
         (IS_DIRTY(FS) || varyings_dirty || IS_DIRTY(SAMPLE_MASK)) &&
         !ctx->linked.fs->no_op,
      .occlusion_query = IS_DIRTY(QUERY),
      .output_size = IS_DIRTY(VS_PROG),
      .viewport_count = 1, /* irrelevant */
   };

   size_t size = agx_ppp_update_size(&dirty);
   struct agx_ptr T =
      agx_pool_alloc_aligned(&batch->pool, size, AGX_PPP_HEADER_ALIGN);
   struct agx_ppp_update ppp = agx_new_ppp_update(T, size, &dirty);

   if (dirty.fragment_control) {
      agx_ppp_push(&ppp, FRAGMENT_CONTROL, cfg) {
         if (ctx->active_queries && ctx->occlusion_query) {
            if (ctx->occlusion_query->type == PIPE_QUERY_OCCLUSION_COUNTER)
               cfg.visibility_mode = AGX_VISIBILITY_MODE_COUNTING;
            else
               cfg.visibility_mode = AGX_VISIBILITY_MODE_BOOLEAN;
         }

         cfg.stencil_test_enable = ctx->zs->base.stencil[0].enabled;
         cfg.two_sided_stencil = ctx->zs->base.stencil[1].enabled;
         cfg.depth_bias_enable =
            rast->depth_bias && object_type == AGX_OBJECT_TYPE_TRIANGLE;

         /* Always enable scissoring so we may scissor to the viewport (TODO:
          * optimize this out if the viewport is the default and the app does
          * not use the scissor test)
          */
         cfg.scissor_enable = true;

         /* This avoids broken derivatives along primitive edges */
         cfg.disable_tri_merging = is_lines || is_points;
      }
   }

   if (dirty.fragment_control_2) {
      /* Annoying, rasterizer_discard seems to be ignored (sometimes?) in the
       * main fragment control word and has to be combined into the secondary
       * word for reliable behaviour.
       */
      agx_ppp_push_merged(&ppp, FRAGMENT_CONTROL, cfg,
                          ctx->linked.fs->fragment_control) {
         cfg.tag_write_disable = rast->base.rasterizer_discard;
      }
   }

   if (dirty.fragment_front_face) {
      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, ctx->zs->depth) {
         cfg.stencil_reference = ctx->stencil_ref.ref_value[0];
         cfg.line_width = rast->line_width;
         cfg.polygon_mode = rast->polygon_mode;
      }
   }

   if (dirty.fragment_front_face_2)
      agx_ppp_fragment_face_2(&ppp, object_type, &ctx->fs->b.info);

   if (dirty.fragment_front_stencil) {
      agx_ppp_push_packed(&ppp, ctx->zs->front_stencil.opaque,
                          FRAGMENT_STENCIL);
   }

   if (dirty.fragment_back_face) {
      agx_ppp_push_merged(&ppp, FRAGMENT_FACE, cfg, ctx->zs->depth) {
         bool twosided = ctx->zs->base.stencil[1].enabled;
         cfg.stencil_reference = ctx->stencil_ref.ref_value[twosided ? 1 : 0];
         cfg.line_width = rast->line_width;
         cfg.polygon_mode = rast->polygon_mode;
      }
   }

   if (dirty.fragment_back_face_2)
      agx_ppp_fragment_face_2(&ppp, object_type, &ctx->fs->b.info);

   if (dirty.fragment_back_stencil)
      agx_ppp_push_packed(&ppp, ctx->zs->back_stencil.opaque, FRAGMENT_STENCIL);

   assert(dirty.varying_counts_32 == dirty.varying_counts_16);
   assert(dirty.varying_counts_32 == dirty.output_select);

   if (dirty.output_select) {
      agx_ppp_push_merged_blobs(&ppp, AGX_OUTPUT_SELECT_LENGTH, &vs->uvs.osel,
                                &ctx->linked.fs->osel);

      agx_ppp_push_packed(&ppp, &batch->linked_varyings.counts_32,
                          VARYING_COUNTS);

      agx_ppp_push_packed(&ppp, &batch->linked_varyings.counts_16,
                          VARYING_COUNTS);
   }

   if (dirty.cull) {
      agx_ppp_push_merged(&ppp, CULL, cfg, ctx->rast->cull) {
         cfg.front_face_ccw = agx_front_face_ccw(ctx);
      }
   }

   if (dirty.cull_2) {
      agx_ppp_push(&ppp, CULL_2, cfg) {
         cfg.needs_primitive_id = batch->generate_primitive_id;
         cfg.clamp_w = true;
      }
   }

   if (dirty.fragment_shader) {
      unsigned frag_tex_count = ctx->stage[MESA_SHADER_FRAGMENT].texture_count;

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_0, cfg) {
         cfg.uniform_register_count = ctx->fs->b.info.push_count;
         cfg.preshader_register_count = ctx->fs->b.info.nr_preamble_gprs;
         cfg.texture_state_register_count =
            agx_nr_tex_descriptors(batch, ctx->fs);
         cfg.sampler_state_register_count =
            translate_sampler_state_count(ctx, MESA_SHADER_FRAGMENT);
         cfg.cf_binding_count = ctx->linked.fs->cf.nr_bindings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_1, cfg) {
         cfg.pipeline = agx_build_pipeline(batch, agx_fs_launch_shader(ctx),
                                           ctx->linked.fs,
                                           MESA_SHADER_FRAGMENT, 0, 0, NULL,
                                           NULL);
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_2, cfg) {
         cfg.cf_bindings = batch->varyings;
      }

      agx_ppp_push(&ppp, FRAGMENT_SHADER_WORD_3, cfg) {
         /* XXX: This is wrong */
         cfg.unknown = frag_tex_count >= 4;
      }
   }

   if (dirty.occlusion_query) {
      agx_ppp_push(&ppp, FRAGMENT_OCCLUSION_QUERY, cfg) {
         if (ctx->active_queries && ctx->occlusion_query) {
            cfg.index = agx_get_oq_index(batch, ctx->occlusion_query);
         }
      }
   }

   if (dirty.output_size) {
      agx_ppp_push(&ppp, OUTPUT_SIZE, cfg)
         cfg.count = vs->uvs.size;
   }

   bool emitted = agx_gallium_ppp_fini(batch, &out, &ppp);
   assert(emitted);
   (void)emitted;
   ppp_updates++;

   assert(ppp_updates <= MAX_PPP_UPDATES);
   return out;
}

static enum agx_primitive
agx_primitive_for_pipe(enum mesa_prim mode)
{
   switch (mode) {
   case MESA_PRIM_POINTS:
      return AGX_PRIMITIVE_POINTS;
   case MESA_PRIM_LINES:
      return AGX_PRIMITIVE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return AGX_PRIMITIVE_LINE_STRIP;
   case MESA_PRIM_LINE_LOOP:
      return AGX_PRIMITIVE_LINE_LOOP;
   case MESA_PRIM_TRIANGLES:
      return AGX_PRIMITIVE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return AGX_PRIMITIVE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return AGX_PRIMITIVE_TRIANGLE_FAN;
   case MESA_PRIM_QUADS:
      return AGX_PRIMITIVE_QUADS;
   case MESA_PRIM_QUAD_STRIP:
      return AGX_PRIMITIVE_QUAD_STRIP;
   default:
      UNREACHABLE("todo: other primitive types");
   }
}

static uint8_t *
agx_g17p_emit_draw_body(enum agx_chip chip, uint8_t *out,
                        const struct agx_draw *draw,
                        enum agx_primitive primitive, uint32_t restart_index,
                        uint32_t draw_state_offset)
{
   if (draw->indexed) {
      const struct agx_vdm_hal200_indexed_draw_fields fields = {
         .primitive = primitive,
         .index_size_B = agx_index_size_to_B(draw->index_size),
         .restart_enable = draw->restart,
         .restart_index = restart_index,
         .draw_state_context_offset = draw_state_offset,
         .index_buffer = draw->index_buffer,
         .index_buffer_range_B = draw->index_buffer_range_B,
         .first_index = draw->start,
         .index_count = draw->b.count[0],
         .instance_count = draw->b.count[1],
         .base_vertex = (int32_t)draw->index_bias,
      };
      if (draw_state_offset) {
         struct agx_vdm_hal200_indexed_draw packed;
         const bool valid = agx_vdm_hal200_indexed_draw_pack(
            chip, &fields, &packed);
         assert(valid);
         if (!valid)
            return out;
         memcpy(out, packed.bytes, packed.length);
         return out + packed.length;
      } else {
         struct agx_vdm_hal200_indexed_draw_body packed;
         const bool valid = agx_vdm_hal200_indexed_draw_body_pack(
            chip, &fields, &packed);
         assert(valid);
         if (!valid)
            return out;
         memcpy(out, packed.bytes, packed.length);
         return out + packed.length;
      }
   }

   const struct agx_vdm_hal200_direct_draw_fields fields = {
      .draw_state_context_offset = draw_state_offset,
      .primitive = primitive,
      .vertex_count = draw->b.count[0],
      .instance_count = draw->b.count[1],
      .vertex_start = draw->start,
   };
   if (draw_state_offset) {
      struct agx_vdm_hal200_direct_draw packed;
      const bool valid = agx_vdm_hal200_direct_draw_pack(
         chip, &fields, &packed);
      if (!valid) {
         mesa_loge("asahi: invalid G17P direct draw: primitive=%u vertices=%u "
                   "instances=%u start=%u state_offset=0x%x",
                   fields.primitive, fields.vertex_count, fields.instance_count,
                   fields.vertex_start, fields.draw_state_context_offset);
      }
      assert(valid);
      if (!valid)
         return out;
      memcpy(out, packed.bytes, packed.length);
      return out + packed.length;
   } else {
      struct agx_vdm_hal200_direct_draw_body packed;
      const bool valid = agx_vdm_hal200_direct_draw_body_pack(
         chip, &fields, &packed);
      assert(valid);
      if (!valid)
         return out;
      memcpy(out, packed.bytes, packed.length);
      return out + packed.length;
   }
}

static uint8_t *
agx_g17p_emit_indirect_draw(enum agx_chip chip, uint8_t *out,
                            const struct agx_draw *draw,
                            enum agx_primitive primitive,
                            uint32_t restart_index,
                            uint32_t draw_state_offset)
{
   const bool tail_emitted = draw_state_offset != 0;
   const struct agx_vdm_hal200_indirect_draw_fields fields = {
      /* Not emitted without a fresh tail, but packed regardless. */
      .draw_state_context_offset = tail_emitted ? draw_state_offset : 4,
      .primitive = primitive,
      .indexed = draw->indexed,
      .index_size_B = draw->indexed ? agx_index_size_to_B(draw->index_size) : 0,
      .restart_enable = draw->restart,
      .restart_index = restart_index,
      .index_buffer = draw->index_buffer,
      .index_buffer_range_B = draw->index_buffer_range_B,
      .indirect_buffer = draw->b.ptr,
   };
   struct agx_vdm_hal200_indirect_draw packed;
   size_t emitted = 0;
   const bool valid =
      agx_vdm_hal200_indirect_draw_pack(chip, &fields, &packed) &&
      agx_vdm_hal200_draw_emit(chip, packed.bytes, packed.length, tail_emitted,
                               out, packed.length, &emitted);
   assert(valid);
   if (!valid)
      return out;

   return out + emitted;
}

static uint64_t
agx_index_buffer_rsrc_ptr(struct agx_batch *batch,
                          const struct pipe_draw_info *info, size_t *extent)
{
   assert(!info->has_user_indices && "cannot use user pointers with indirect");

   struct agx_resource *rsrc = agx_resource(info->index.resource);
   agx_batch_reads(batch, rsrc);
   agx_batch_note_vtx_read(batch, rsrc);

   *extent =
      ALIGN_POT(rsrc->layout.size_B - rsrc->layout.level_offsets_B[0], 4);
   return agx_map_gpu(rsrc);
}

static uint64_t
agx_index_buffer_direct_ptr(struct agx_batch *batch,
                            const struct pipe_draw_start_count_bias *draw,
                            const struct pipe_draw_info *info, size_t *extent)
{
   off_t offset = draw->start * info->index_size;
   uint32_t max_extent = draw->count * info->index_size;

   if (!info->has_user_indices) {
      uint64_t base = agx_index_buffer_rsrc_ptr(batch, info, extent);

      /* G17P fetches indices from a dword-aligned address: the low two bits
       * of the index list pointer are dropped, so a u16 draw starting at an
       * odd index (or a u8 draw at index % 4 != 0) reads from the preceding
       * aligned address. Copy such ranges to an aligned pool upload.
       */
      if (agx_chip_native_agx3(agx_device(batch->ctx->base.screen)->chip) &&
          ((base + offset) & 3)) {
         struct agx_resource *rsrc = agx_resource(info->index.resource);
         const uint8_t *cpu = (const uint8_t *)agx_bo_map(rsrc->bo) +
                              rsrc->layout.level_offsets_B[0];

         *extent = ALIGN_POT(max_extent, 4);
         return agx_pool_upload_aligned(&batch->pool, cpu + offset,
                                        max_extent, 64);
      }

      *extent = ALIGN_POT(MIN2(*extent - offset, max_extent), 4);
      return base + offset;
   } else {
      *extent = ALIGN_POT(max_extent, 4);

      return agx_pool_upload_aligned(&batch->pool,
                                     ((uint8_t *)info->index.user) + offset,
                                     draw->count * info->index_size, 64);
   }
}

static uint64_t
agx_index_buffer_ptr(struct agx_batch *batch, const struct pipe_draw_info *info,
                     const struct pipe_draw_start_count_bias *draw,
                     size_t *extent)
{
   if (draw)
      return agx_index_buffer_direct_ptr(batch, draw, info, extent);
   else
      return agx_index_buffer_rsrc_ptr(batch, info, extent);
}

static void
agx_ensure_cmdbuf_has_space(struct agx_batch *batch, struct agx_encoder *enc,
                            size_t space)
{
   bool vdm = enc == &batch->vdm;
   assert(vdm || (enc == &batch->cdm));

   size_t link_length =
      vdm ? AGX_VDM_STREAM_LINK_LENGTH : AGX_CDM_STREAM_LINK_LENGTH;

   /* Assert that we have space for a link tag */
   assert((enc->current + link_length) <= enc->end && "Encoder overflowed");

   /* Always leave room for a link tag, in case we run out of space later,
    * plus padding because VDM apparently overreads?
    *
    * 0x200 is not enough. 0x400 seems to work. 0x800 for safety.
    */
   space += link_length + 0x800;

   /* If there is room in the command buffer, we're done */
   if (likely((enc->end - enc->current) >= space))
      return;

   /* Otherwise, we need to allocate a new command buffer. We use memory owned
    * by the batch to simplify lifetime management for the BO.
    */
   size_t size = 65536;
   struct agx_bo *target_bo = NULL;
   struct agx_ptr T = agx_pool_alloc_aligned_with_bo(
      &batch->pool, size, 256, &target_bo);

   /* Jump from the old command buffer to the new command buffer */
   struct agx_device *dev = agx_device(batch->ctx->base.screen);
   if (vdm && agx_chip_native_agx3(dev->chip)) {
      const uint64_t alias =
         agx_batch_get_g17p_vdm_alias(batch, target_bo, T.gpu);
      uint32_t compact = 0;
      struct agx_vdm_stream_link_core link;
      const bool packed = agx_g17p_vdm_alias_compact(alias, &compact) &&
                          agx_vdm_stream_link_core_pack(dev->chip, compact,
                                                       &link);
      assert(packed);
      if (!packed)
         return;

      memcpy(enc->current, link.bytes, link.length);
      batch->g17p_bind_tail_latched = false;
      batch->g17p_have_latched_vs = false;
   } else if (vdm && agx_chip_g17_m5(dev->chip)) {
      /* The M5 command processor adds the render-context base to a Stream
       * Link target, as it does to a PPP payload pointer: an absolute target
       * is read 0x1000000000 above the new chunk. Honeykrisp makes the same
       * translation in hk_cmd_buffer_emit_cs_jump(). */
      uint64_t compact = 0;
      struct agx_vdm_stream_link_core link;
      const bool packed = agx_m5_vdm_compact(T.gpu, &compact) &&
                          agx_vdm_stream_link_core_pack(dev->chip, compact,
                                                       &link);
      assert(packed);
      if (!packed)
         return;

      memcpy(enc->current, link.bytes, link.length);
   } else {
      agx_cs_jump((uint32_t *)enc->current, T.gpu, vdm);
   }

   /* Swap out the command buffer */
   enc->current = T.cpu;
   enc->end = enc->current + size;
}

static void
agx_ia_update(struct agx_batch *batch, const struct pipe_draw_info *info,
              uint64_t draw, uint64_t ib, uint64_t ib_range_el)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (!batch->cdm.bo) {
      batch->cdm = agx_encoder_allocate(batch, dev);
   }

   uint64_t ia_vertices = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_IA_VERTICES]);

   uint64_t ia_primitives = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_IA_PRIMITIVES]);

   uint64_t vs_invocations = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_VS_INVOCATIONS]);

   uint64_t c_prims = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_C_PRIMITIVES]);

   uint64_t c_invs = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_C_INVOCATIONS]);

   /* With a geometry/tessellation shader, clipper counters are written by the
    * pre-GS/tess prefix sum kernel since they depend on the output on the
    * geometry/tessellation shader. Without a geometry/tessellation shader,
    * they are written along with IA.
    */
   if (ctx->stage[MESA_SHADER_GEOMETRY].shader ||
       ctx->stage[MESA_SHADER_TESS_EVAL].shader) {

      c_prims = AGX_SCRATCH_PAGE_ADDRESS;
      c_invs = AGX_SCRATCH_PAGE_ADDRESS;
   }

   if (info->primitive_restart) {
      perf_debug(dev, "Input assembly counters with primitive restart");

      libagx_increment_ia_restart(batch, agx_1d(1024), AGX_BARRIER_ALL,
                                  ia_vertices, ia_primitives, vs_invocations,
                                  c_prims, c_invs, draw, ib, ib_range_el,
                                  info->restart_index, info->index_size,
                                  info->mode, ctx->patch_vertices);
   } else {
      perf_debug(dev, "Input assembly counters");

      libagx_increment_ia(batch, agx_1d(1), AGX_BARRIER_ALL, ia_vertices,
                          ia_primitives, vs_invocations, c_prims, c_invs, draw,
                          info->mode, ctx->patch_vertices);
   }
}

static uint64_t
agx_batch_heap(struct agx_batch *batch)
{
   struct agx_context *ctx = batch->ctx;

   if (!batch->heap) {
      uint32_t size = 128 * 1024 * 1024;

      if (!ctx->heap) {
         ctx->heap = pipe_buffer_create(ctx->base.screen, PIPE_BIND_GLOBAL,
                                        PIPE_USAGE_DEFAULT, size);
      }

      struct poly_heap heap = {
         .base = agx_resource(ctx->heap)->bo->va->addr,
         .size = size,
      };

      agx_batch_writes(batch, agx_resource(ctx->heap), 0);

      batch->heap =
         agx_pool_upload_aligned(&batch->pool, &heap, sizeof(heap), 8);
   }

   return batch->heap;
}

static uint64_t
agx_batch_geometry_params(struct agx_batch *batch, uint64_t input_index_buffer,
                          size_t index_buffer_size_B,
                          const struct pipe_draw_info *info,
                          const struct pipe_draw_start_count_bias *draw,
                          const struct pipe_draw_indirect_info *indirect)
{
   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params vp;
   poly_vertex_params_init(&vp, batch->ctx->vs->b.info.outputs, wg_size);

   if (info->index_size) {
      vp.index_size_B = info->index_size;
      vp.index_buffer = input_index_buffer;
      vp.index_buffer_range_el = index_buffer_size_B / info->index_size;
   }

   struct poly_geometry_params params;
   poly_geometry_params_init(&params, info->mode, wg_size);

   params.flat_outputs =
      batch->ctx->stage[MESA_SHADER_FRAGMENT].shader->info.inputs_flat_shaded;

   params.xfb_offs_ptrs[0] = AGX_ZERO_PAGE_ADDRESS;
   params.xfb_offs_ptrs[1] = AGX_ZERO_PAGE_ADDRESS;
   params.xfb_offs_ptrs[2] = AGX_ZERO_PAGE_ADDRESS;
   params.xfb_offs_ptrs[3] = AGX_ZERO_PAGE_ADDRESS;

   for (unsigned i = 0; i < ARRAY_SIZE(batch->ctx->streamout.targets); ++i) {
      struct agx_streamout_target *so =
         agx_so_target(batch->ctx->streamout.targets[i]);
      struct agx_resource *rsrc = so ? agx_resource(so->offset) : NULL;

      uint32_t size;
      params.xfb_base_original[i] = agx_batch_get_so_address(batch, i, &size);
      params.xfb_size[i] = size;

      if (rsrc) {
         params.xfb_offs_ptrs[i] = agx_map_gpu(rsrc);
         agx_batch_writes(batch, rsrc, 0);
         batch->incoherent_writes = true;
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(batch->ctx->prims_generated); ++i) {
      params.prims_generated_counter[i] =
         agx_get_query_address(batch, batch->ctx->prims_generated[i]);
   }

   /* The pre-GS kernel also adds to query_prims_generated_counter, which
    * backs Vulkan's VK_EXT_primitives_generated_query. GL's primitives
    * generated query already counts through prims_generated_counter, so
    * point this one at the scratch page rather than leaving it zero.
    */
   for (unsigned i = 0; i < ARRAY_SIZE(params.query_prims_generated_counter);
        ++i) {
      params.query_prims_generated_counter[i] = AGX_SCRATCH_PAGE_ADDRESS;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(batch->ctx->tf_prims_generated); ++i) {
      params.xfb_prims_generated_counter[i] =
         agx_get_query_address(batch, batch->ctx->tf_prims_generated[i]);
   }

   if (batch->ctx->active_queries && batch->ctx->streamout.num_targets > 0) {
      for (unsigned i = 0; i < ARRAY_SIZE(batch->ctx->tf_overflow); ++i) {
         params.xfb_overflow[i] =
            agx_get_query_address(batch, batch->ctx->tf_overflow[i]);
      }

      params.xfb_any_overflow =
         agx_get_query_address(batch, batch->ctx->tf_any_overflow);
   } else {
      for (unsigned i = 0; i < ARRAY_SIZE(batch->ctx->tf_overflow); ++i) {
         params.xfb_overflow[i] = AGX_SCRATCH_PAGE_ADDRESS;
      }

      params.xfb_any_overflow = AGX_SCRATCH_PAGE_ADDRESS;
   }

   /* Calculate input primitive count for direct draws, and allocate the vertex
    * & count buffers. GPU calculates and allocates for indirect draws.
    */
   batch->uniforms.vertex_outputs = vp.outputs;
   params.count_buffer_stride = batch->ctx->gs->gs.count_words * 4;

   bool prefix_sum = batch->ctx->gs->gs.prefix_sum;
   if (!prefix_sum && params.count_buffer_stride) {
      struct agx_ptr T = agx_pool_alloc_aligned(&batch->pool, 16, 4);
      memset(T.cpu, 0, 16);
      params.count_buffer = T.gpu;
   }

   if (!indirect) {
      poly_vertex_params_set_draw(&vp, draw->count, info->instance_count);

      struct poly_gs_info *gsi = &batch->ctx->gs->gs;
      poly_geometry_params_set_draw(&params, info->mode, gsi->shape,
                                    gsi->max_indices, draw->count,
                                    info->instance_count);

      unsigned vb_size = poly_tcs_in_size(draw->count * info->instance_count,
                                          batch->uniforms.vertex_outputs);
      unsigned size = params.input_primitives * params.count_buffer_stride;

      if (size && prefix_sum) {
         params.count_buffer =
            agx_pool_alloc_aligned(&batch->pool, size, 4).gpu;
      }

      /* Allocate if there are any outputs, or use the scratch page to trap
       * speculative reads (such as lowered gl_PrimitiveIDIn checking
       * vp.outputs) when the vertex shader writes no varyings. Matches
       * Borealis hk_upload_vertex_params().
       */
      vp.output_buffer =
         vb_size ? agx_pool_alloc_aligned(&batch->pool, vb_size, 4).gpu
                 : AGX_SCRATCH_PAGE_ADDRESS;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         unsigned idx_size = params.input_primitives * gsi->max_indices;

         params.output_index_buffer =
            agx_pool_alloc_aligned_with_bo(&batch->pool, idx_size * 4, 4,
                                           &batch->geom_index_bo)
               .gpu;
         batch->geom_index = params.output_index_buffer;
      }
   }

   batch->uniforms.vertex_params =
      agx_pool_upload_aligned(&batch->pool, &vp, sizeof(vp), 8);

   return agx_pool_upload_aligned_with_bo(&batch->pool, &params, sizeof(params),
                                          8, &batch->geom_params_bo);
}

static uint64_t
agx_indirect_buffer_ptr(struct agx_batch *batch,
                        const struct pipe_draw_indirect_info *indirect)
{
   assert(indirect->buffer && "drawauto already handled");

   struct agx_resource *rsrc = agx_resource(indirect->buffer);
   agx_batch_reads(batch, rsrc);
   agx_batch_note_vtx_read(batch, rsrc);
   return agx_map_gpu(rsrc) + indirect->offset;
}

static void
agx_launch_gs_prerast(struct agx_batch *batch,
                      const struct pipe_draw_info *info,
                      const struct pipe_draw_start_count_bias *draws,
                      const struct pipe_draw_indirect_info *indirect)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_compiled_shader *gs = ctx->gs;

   if (ctx->stage[MESA_SHADER_GEOMETRY].shader->is_xfb_passthrough)
      perf_debug(dev, "Transform feedbck");
   else
      perf_debug(dev, "Geometry shader");

   /* This is a graphics batch, so it may not have had a CDM encoder allocated
    * yet. Allocate that so we can start enqueueing compute work.
    */
   if (!batch->cdm.bo) {
      batch->cdm = agx_encoder_allocate(batch, dev);
   }

   agx_ensure_cmdbuf_has_space(
      batch, &batch->cdm,
      8 * (AGX_CDM_LAUNCH_WORD_0_LENGTH + AGX_CDM_LAUNCH_WORD_1_LENGTH +
           AGX_CDM_UNK_G14X_LENGTH + AGX_CDM_INDIRECT_LENGTH +
           AGX_CDM_GLOBAL_SIZE_LENGTH + AGX_CDM_LOCAL_SIZE_LENGTH +
           AGX_CDM_BARRIER_LENGTH));

   assert(!info->primitive_restart && "should have been lowered");

   uint64_t vp = batch->uniforms.vertex_params;
   uint64_t gp = batch->uniforms.geometry_params;
   struct agx_grid grid_vs, grid_gs;
   struct agx_workgroup wg = agx_workgroup(64, 1, 1);

   /* Setup grids */
   if (indirect) {
      uint64_t ib = 0;
      size_t ib_extent = 0;

      if (info->index_size) {
         ib = agx_index_buffer_ptr(batch, info, indirect ? NULL : draws,
                                   &ib_extent);
      }

      struct libagx_gs_setup_indirect_args gsi = {
         .index_buffer = ib,
         .index_buffer_range_el = ib_extent / info->index_size,
         .draw = agx_indirect_buffer_ptr(batch, indirect),
         .vp = batch->uniforms.vertex_params,
         .p = batch->uniforms.geometry_params,
         .heap = agx_batch_heap(batch),
         .vs_outputs = batch->uniforms.vertex_outputs,
         .index_size_B = info->index_size,
         .prim = info->mode,
         .is_prefix_summing = gs->gs.prefix_sum,
         .max_indices = gs->gs.max_indices,
         .shape = gs->gs.shape,
      };

      libagx_gs_setup_indirect_struct(batch, agx_1d(1), AGX_BARRIER_ALL, gsi);

      grid_vs = agx_grid_indirect_local(
         vp + offsetof(struct poly_vertex_params, grid));

      grid_gs = agx_grid_indirect_local(
         gp + offsetof(struct poly_geometry_params, grid));
   } else {
      grid_vs = agx_3d(draws->count, info->instance_count, 1);

      grid_gs =
         agx_3d(u_decomposed_prims_for_vertices(info->mode, draws->count),
                info->instance_count, 1);
   }

   /* Launch the vertex shader first */
   agx_launch(batch, grid_vs, wg, ctx->vs, ctx->linked.vs, ctx->vs->stage, 0, 0);

   /* Transform feedback and various queries require extra dispatching,
    * determine if we need that here.
    */
   enum pipe_statistics_query_index gs_queries[] = {
      PIPE_STAT_QUERY_GS_INVOCATIONS,
      PIPE_STAT_QUERY_GS_PRIMITIVES,
      PIPE_STAT_QUERY_C_PRIMITIVES,
      PIPE_STAT_QUERY_C_INVOCATIONS,
   };

   bool xfb_or_queries = ctx->stage[MESA_SHADER_GEOMETRY].shader->has_xfb_info;

   for (unsigned i = 0; i < ARRAY_SIZE(gs_queries); ++i) {
      xfb_or_queries |= (ctx->pipeline_statistics[gs_queries[i]] != NULL);
   }

   for (unsigned i = 0; i < ARRAY_SIZE(ctx->prims_generated); ++i) {
      xfb_or_queries |= (ctx->prims_generated[i] != NULL);
   }

   /* If there is a count shader, launch it and prefix sum the results. */
   if (gs->gs_count && xfb_or_queries) {
      perf_debug(dev, "Geometry shader count");
      agx_launch(batch, grid_gs, wg, gs->gs_count, NULL, MESA_SHADER_GEOMETRY,
                 0, 0);
   }

   if (gs->gs.prefix_sum) {
      libagx_prefix_sum_geom(batch, agx_1d(1024 * gs->gs.count_words),
                             AGX_BARRIER_ALL, gp);
   }

   /* Pre-GS shader */
   if (xfb_or_queries) {
      perf_debug(dev, "Geometry shader transform feedback / query program");
      agx_launch(batch, agx_1d(1), agx_workgroup(1, 1, 1), gs->pre_gs, NULL,
                 MESA_SHADER_COMPUTE, 0, 0);
   }

   /* Pre-rast geometry shader */
   agx_launch(batch, grid_gs, wg, gs, NULL, MESA_SHADER_GEOMETRY, 0, 0);
}

static void
agx_draw_without_restart(struct agx_batch *batch,
                         const struct pipe_draw_info *info,
                         unsigned drawid_offset,
                         const struct pipe_draw_indirect_info *indirect,
                         const struct pipe_draw_start_count_bias *draw)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);

   perf_debug(dev, "Unrolling primitive restart due to GS/XFB");

   agx_batch_init_state(batch);

   size_t ib_extent = 0;
   uint64_t ib;

   /* The rest of this function handles only the general case of indirect
    * multidraws, so synthesize an indexed indirect draw now if we need one for
    * a direct draw (necessarily only one). This unifies the code paths.
    */
   struct pipe_draw_indirect_info indirect_synthesized = {.draw_count = 1};

   if (!indirect) {
      /* Adds in the offset so set to 0 in the desc */
      ib = agx_index_buffer_direct_ptr(batch, draw, info, &ib_extent);

      uint32_t desc[5] = {draw->count, info->instance_count, 0,
                          draw->index_bias, info->start_instance};

      u_upload_data_ref(ctx->base.const_uploader, 0, sizeof(desc), 4, &desc,
                        &indirect_synthesized.offset,
                        &indirect_synthesized.buffer);

      indirect = &indirect_synthesized;
   } else {
      /* Does not add in offset, the unroll kernel uses the desc's offset */
      ib = agx_index_buffer_rsrc_ptr(batch, info, &ib_extent);
   }

   /* Next, we unroll the index buffer used by the indirect draw */
   if (!batch->cdm.bo)
      batch->cdm = agx_encoder_allocate(batch, dev);

   /* Allocate output indirect draw descriptors. This is exact. */
   struct agx_resource out_draws_rsrc = {0};
   struct agx_ptr out_draws = agx_pool_alloc_aligned_with_bo(
      &batch->pool, 5 * sizeof(uint32_t) * indirect->draw_count, 4,
      &out_draws_rsrc.bo);

   struct libagx_unroll_restart_args unroll = {
      .heap = agx_batch_heap(batch),
      .index_buffer = ib,
      .out_draw = out_draws.gpu,
      .restart_index = info->restart_index,
      .index_buffer_size_el = ib_extent / info->index_size,
      .index_size_log2 = util_logbase2(info->index_size),
      .flatshade_first = batch->ctx->rast->base.flatshade_first,
      .in_draw = agx_indirect_buffer_ptr(batch, indirect),
   };

   /* Unroll the index buffer for each draw */
   libagx_unroll_restart_struct(batch, agx_1d(1024 * indirect->draw_count),
                                AGX_BARRIER_ALL, unroll,
                                poly_compact_prim(info->mode));

   /* Now draw the results without restart */
   struct pipe_draw_info new_info = {
      .mode = u_decomposed_prim(info->mode),
      .index_size = info->index_size,
      .index.resource = ctx->heap,
      .increment_draw_id = info->increment_draw_id,
      .index_bias_varies = info->index_bias_varies,
   };

   struct pipe_draw_indirect_info new_indirect = *indirect;
   new_indirect.buffer = &out_draws_rsrc.base.b;
   new_indirect.offset = out_draws.gpu - out_draws_rsrc.bo->va->addr;
   new_indirect.stride = 5 * sizeof(uint32_t);

   ctx->active_draw_without_restart = true;
   ctx->base.draw_vbo(&ctx->base, &new_info, drawid_offset, &new_indirect, NULL,
                      1);
   ctx->active_draw_without_restart = false;
}

static bool
agx_needs_passthrough_gs(struct agx_context *ctx,
                         const struct pipe_draw_info *info,
                         const struct pipe_draw_indirect_info *indirect,
                         bool *xfb_only, bool xfb_on_cpu)
{
   /* If there is already a geometry shader in the pipeline, we do not need to
    * apply a passthrough GS of our own.
    */
   if (ctx->stage[MESA_SHADER_GEOMETRY].shader)
      return false;

   /* Rendering adjacency requires a GS, add a passthrough since we don't have
    * one.
    */
   if (mesa_prim_has_adjacency(info->mode)) {
      perf_debug_ctx(ctx, "Using passthrough GS due to adjacency primitives");
      return true;
   }

   /* TODO: Handle fans properly, we need to plumb a sysval. */
   if (info->mode == MESA_PRIM_TRIANGLE_FAN &&
       ctx->rast->base.flatshade_first &&
       ctx->stage[MESA_SHADER_FRAGMENT].shader->info.inputs_flat_shaded) {

      perf_debug_ctx(ctx, "Using passthrough GS due to first tri fans");
      return true;
   }

   /* TODO: this is really sloppy, we should add a VDM kernel for this.
    *
    * On M5, the tessellation prepass counts primitives directly
    * (agx_draw_patches), as Borealis does, so patch draws do not need a
    * passthrough GS for indirect primitive queries.
    */
   if ((indirect || info->mode == MESA_PRIM_PATCHES) && ctx->active_queries &&
       ctx->prims_generated[0] &&
       !(agx_m5_gpu_tess(ctx) &&
         (info->mode == MESA_PRIM_PATCHES || ctx->in_tess))) {
      perf_debug_ctx(ctx, "Using passthrough GS due to indirect prim query");
      return true;
   }

   /* Edge flags are emulated with a geometry shader */
   if (has_edgeflags(ctx, info->mode)) {
      perf_debug_ctx(ctx, "Using passthrough GS due to edge flags");
      return true;
   }

   /* Transform feedback is layered on geometry shaders, so if transform
    * feedback is used, we need a GS.
    */
   struct agx_uncompiled_shader *last_vtx =
      ctx->stage[MESA_SHADER_TESS_EVAL].shader
         ?: ctx->stage[MESA_SHADER_VERTEX].shader;

   if (last_vtx->has_xfb_info && ctx->streamout.num_targets && !xfb_on_cpu) {
      *xfb_only = true;
      return true;
   }

   /* Otherwise, we don't need one */
   return false;
}

static enum mesa_prim
agx_tess_output_prim(struct agx_uncompiled_shader *tcs,
                     struct agx_uncompiled_shader *tes)
{
   if ((tcs && tcs->tess.point_mode) || tes->tess.point_mode) {
      return MESA_PRIM_POINTS;
   } else if (TESS_PRIMITIVE_ISOLINES ==
              MAX2(tcs ? tcs->tess.primitive : 0, tes->tess.primitive)) {
      return MESA_PRIM_LINES;
   } else {
      return MESA_PRIM_TRIANGLES;
   }
}

static struct agx_uncompiled_shader *
agx_get_passthrough_gs(struct agx_context *ctx,
                       struct agx_uncompiled_shader *prev_cso,
                       enum mesa_prim mode, bool xfb_passthrough)
{
   bool edgeflags = has_edgeflags(ctx, mode);

   if (mode == MESA_PRIM_PATCHES) {
      mode = agx_tess_output_prim(ctx->stage[MESA_SHADER_TESS_CTRL].shader,
                                  ctx->stage[MESA_SHADER_TESS_EVAL].shader);
   }

   /* Only handle the polygon mode when edge flags are in use, because
    * nir_passthrough_gs doesn't handle transform feedback + polygon mode
    * properly. Technically this can break edge flags + transform feedback
    * but that's firmly in "doctor, it hurts when I do this" territory, and
    * I'm not sure that's even possible to hit. TODO: Reevaluate.
    */
   unsigned poly_mode =
      edgeflags ? ctx->rast->base.fill_front : PIPE_POLYGON_MODE_FILL;

   if (prev_cso->passthrough_progs[mode][poly_mode][edgeflags])
      return prev_cso->passthrough_progs[mode][poly_mode][edgeflags];

   struct blob_reader reader;
   blob_reader_init(&reader, prev_cso->early_serialized_nir.data,
                    prev_cso->early_serialized_nir.size);
   nir_shader *prev = nir_deserialize(NULL, &agx_nir_options, &reader);

   nir_shader *gs = nir_create_passthrough_gs(
      &agx_nir_options, prev, mode, rast_prim(mode, poly_mode), edgeflags,
      false /* force line strip out */, false);

   /* The passthrough program takes the vertex shader's place in front of the
    * fragment shader, so it inherits what linking told the vertex shader
    * about the fragment interpolation qualifiers. The copy program derived
    * from it can then inline its UVS indices like the vertex shader does,
    * which the AGX3 vertex writeout requires. Other chips keep the generic
    * passthrough program: there, inheriting the qualifiers breaks flat
    * varyings drawn through transform feedback. G17G needs them like the
    * AGX3 chips.
    */
   if (agx_chip_native_or_m5(agx_device(ctx->base.screen)->chip)) {
      gs->info.known_interpolation_qualifiers =
         prev->info.known_interpolation_qualifiers;
      gs->info.linear_varyings = prev->info.linear_varyings;
      gs->info.perspective_varyings = prev->info.perspective_varyings;
   }
   ralloc_free(prev);

   struct agx_uncompiled_shader *cso = pipe_shader_from_nir(&ctx->base, gs);
   cso->is_xfb_passthrough = xfb_passthrough;
   prev_cso->passthrough_progs[mode][poly_mode][edgeflags] = cso;
   return cso;
}

static void
agx_apply_passthrough_gs(struct agx_context *ctx,
                         const struct pipe_draw_info *info,
                         unsigned drawid_offset,
                         const struct pipe_draw_indirect_info *indirect,
                         const struct pipe_draw_start_count_bias *draws,
                         unsigned num_draws, bool xfb_passthrough)
{
   mesa_shader_stage prev_stage = ctx->stage[MESA_SHADER_TESS_EVAL].shader
                                     ? MESA_SHADER_TESS_EVAL
                                     : MESA_SHADER_VERTEX;
   struct agx_uncompiled_shader *prev_cso = ctx->stage[prev_stage].shader;

   assert(ctx->stage[MESA_SHADER_GEOMETRY].shader == NULL);

   /* Draw with passthrough */
   ctx->base.bind_gs_state(
      &ctx->base,
      agx_get_passthrough_gs(ctx, prev_cso, info->mode, xfb_passthrough));
   ctx->base.draw_vbo(&ctx->base, info, drawid_offset, indirect, draws,
                      num_draws);
   ctx->base.bind_gs_state(&ctx->base, NULL);
}

static void
util_draw_multi_unroll_indirect(struct pipe_context *pctx,
                                const struct pipe_draw_info *info,
                                const struct pipe_draw_indirect_info *indirect,
                                const struct pipe_draw_start_count_bias *draws)
{
   for (unsigned i = 0; i < indirect->draw_count; ++i) {
      const struct pipe_draw_indirect_info subindirect = {
         .buffer = indirect->buffer,
         .count_from_stream_output = indirect->count_from_stream_output,
         .offset = indirect->offset + (i * indirect->stride),
         .draw_count = 1,
      };

      pctx->draw_vbo(pctx, info, i, &subindirect, draws, 1);
   }
}

static void
util_draw_multi_upload_indirect(struct pipe_context *pctx,
                                const struct pipe_draw_info *info,
                                const struct pipe_draw_indirect_info *indirect,
                                const struct pipe_draw_start_count_bias *draws)
{
   struct pipe_draw_indirect_info indirect_ = *indirect;
   u_upload_data_ref(pctx->const_uploader, 0, 4, 4, &indirect->draw_count,
                     &indirect_.indirect_draw_count_offset,
                     &indirect_.indirect_draw_count);

   pctx->draw_vbo(pctx, info, 0, &indirect_, draws, 1);
}

static void
agx_upload_draw_params(struct agx_batch *batch,
                       const struct pipe_draw_indirect_info *indirect,
                       const struct pipe_draw_start_count_bias *draws,
                       const struct pipe_draw_info *info)
{
   if (indirect) {
      uint64_t address = agx_indirect_buffer_ptr(batch, indirect);

      /* To implement draw parameters, we use the last 2 words of the
       * indirect draw descriptor. Offset by 3 words for indexed draw (5
       * total) and 2 words for non-indexed (4 total).  See the layouts of
       * indexed vs non-indexed draw descriptors.
       *
       * This gives us a consistent layout
       *
       *    uint32_t first_vertex;
       *    uint32_t base_instance;
       *
       * and we can implement load_first_vertex & load_base_instance without
       * checking for indexing.
       */
      uint32_t offset = info->index_size ? 3 : 2;
      batch->uniforms.tables[AGX_SYSVAL_TABLE_PARAMS] = address + offset * 4;
   } else {
      /* Upload just those two words. */
      uint32_t params[2] = {
         info->index_size ? draws->index_bias : draws->start,
         info->start_instance,
      };

      batch->uniforms.tables[AGX_SYSVAL_TABLE_PARAMS] =
         agx_pool_upload_aligned(&batch->pool, params, sizeof(params), 4);
   }
}

static void
agx_draw_patches(struct agx_context *ctx, const struct pipe_draw_info *info,
                 unsigned drawid_offset,
                 const struct pipe_draw_indirect_info *indirect,
                 const struct pipe_draw_start_count_bias *draws,
                 unsigned num_draws)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   perf_debug(dev, "Tessellation");

   struct agx_uncompiled_shader *tcs = ctx->stage[MESA_SHADER_TESS_CTRL].shader;
   struct agx_uncompiled_shader *tes = ctx->stage[MESA_SHADER_TESS_EVAL].shader;

   assert(tes != NULL && "required with patches");

   unsigned patch_vertices = ctx->patch_vertices;

   /* OpenGL allows omitting the tcs, fill in a passthrough program if needed.
    * In principle, we could optimize this case, but I don't think it matters.
    */
   bool unbind_tcs_when_done = false;
   if (!tcs) {
      struct agx_uncompiled_shader *vs = ctx->stage[MESA_SHADER_VERTEX].shader;

      assert(patch_vertices >= 1 &&
             patch_vertices <= ARRAY_SIZE(vs->passthrough_tcs));

      if (!vs->passthrough_tcs[patch_vertices - 1]) {
         struct blob_reader reader;
         blob_reader_init(&reader, vs->early_serialized_nir.data,
                          vs->early_serialized_nir.size);
         nir_shader *vs_nir = nir_deserialize(NULL, &agx_nir_options, &reader);
         nir_shader *nir = nir_create_passthrough_tcs(&agx_nir_options, vs_nir,
                                                      patch_vertices);
         ralloc_free(vs_nir);

         /* Lower the tess level sysvals and gather info, since mesa/st won't do
          * either for us.
          */
         NIR_PASS(_, nir, nir_lower_system_values);

         nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

         vs->passthrough_tcs[patch_vertices - 1] =
            pipe_shader_from_nir(&ctx->base, nir);
      }

      tcs = vs->passthrough_tcs[patch_vertices - 1];
      ctx->base.bind_tcs_state(&ctx->base, tcs);
      unbind_tcs_when_done = true;
   }

   enum tess_primitive_mode mode =
      MAX2(tcs->tess.primitive, tes->tess.primitive);
   enum gl_tess_spacing spacing = MAX2(tcs->tess.spacing, tes->tess.spacing);

   enum pipe_tess_spacing pspacing = spacing == TESS_SPACING_EQUAL
                                        ? PIPE_TESS_SPACING_EQUAL
                                     : spacing == TESS_SPACING_FRACTIONAL_ODD
                                        ? PIPE_TESS_SPACING_FRACTIONAL_ODD
                                        : PIPE_TESS_SPACING_FRACTIONAL_EVEN;

   bool point_mode = MAX2(tcs->tess.point_mode, tes->tess.point_mode);
   enum mesa_prim out_prim = agx_tess_output_prim(tcs, tes);

   enum poly_tess_partitioning partitioning =
      (enum poly_tess_partitioning)pspacing;

   struct agx_bo *draw_bo = NULL;
   size_t draw_stride = 5 * sizeof(uint32_t);

   struct agx_batch *batch = agx_get_batch(ctx);

   /* M5 submits each dispatch from where it was recorded, so its control
    * stream cannot be continued in another buffer: start a new batch first. */
   const bool m5 = agx_chip_g17_m5(dev->chip);
   if (m5 && batch->cdm.bo && batch->cdm.end - batch->cdm.current < 0x10000) {
      agx_flush_batch_for_reason(ctx, batch, "M5 compute stream full");
      batch = agx_get_batch(ctx);
      /* Still a render of its own (agx_draw_vbo()). */
      batch->g17g_prims = AGX_G17G_RENDER_PRIMS;
      batch->g17g_small_budget = true;
   }

   agx_batch_init_state(batch);

   if (!batch->cdm.bo) {
      batch->cdm = agx_encoder_allocate(batch, dev);
   }

   agx_upload_draw_params(batch, indirect, draws, info);
   agx_upload_vbos(batch);
   agx_update_vs(batch, info->index_size);
   agx_update_tcs(ctx, info);

   /* Bail before building any tessellation state on a program that cannot
    * run, undoing the passthrough TCS this function may have bound. The
    * vertex CSO and ctx->in_tess are not touched until later, so there is
    * nothing else to restore here.
    */
   if (agx_drop_for_unusable_program(ctx, ctx->vs, ctx->linked.vs != NULL,
                                     "vertex") ||
       agx_drop_for_unusable_program(ctx, ctx->tcs, true,
                                     "tessellation control")) {
      if (unbind_tcs_when_done)
         ctx->base.bind_tcs_state(&ctx->base, NULL);
      return;
   }

   /* XXX */
   ctx->stage[MESA_SHADER_TESS_CTRL].dirty = ~0;
   ctx->stage[MESA_SHADER_TESS_EVAL].dirty = ~0;
   agx_update_descriptors(batch, ctx->vs);
   agx_update_descriptors(batch, ctx->tcs);

   batch->uniforms.vertex_outputs = ctx->vs->b.info.outputs;

   const uint32_t wg_size[3] = {64, 1, 1};

   struct poly_vertex_params vp;
   poly_vertex_params_init(&vp, batch->ctx->vs->b.info.outputs, wg_size);
   vp.instance_offset = ctx->tess_split.first_instance;

   if (info->index_size) {
      size_t ib_extent = 0;
      vp.index_size_B = info->index_size;
      vp.index_buffer = agx_index_buffer_ptr(batch, info, draws, &ib_extent);
      vp.index_buffer_range_el = ib_extent / info->index_size;
   }

   /* Setup parameters */
   uint64_t heap = agx_batch_heap(batch);
   assert((tcs->tess.output_stride & 3) == 0 && "must be aligned");

   struct poly_tess_params params = {
      .heap = heap,
      .tcs_stride_el = tcs->tess.output_stride / 4,
      .statistic = agx_get_query_address(
         batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_DS_INVOCATIONS]),
      .input_patch_size = patch_vertices,
      .output_patch_size = tcs->tess.output_patch_size,
      .tcs_patch_constants = tcs->tess.nr_patch_outputs,
      .tcs_per_vertex_outputs = tcs->tess.per_vertex_outputs,
      .patch_coord_buffer = agx_resource(ctx->heap)->bo->va->addr,
      .partitioning = partitioning,
      .points_mode = point_mode,
      .isolines = mode == TESS_PRIMITIVE_ISOLINES,
      .first_patch = ctx->tess_split.first_patch,
   };

   if (!point_mode && tes->tess.primitive != TESS_PRIMITIVE_ISOLINES) {
      params.ccw = !tes->tess.ccw;
   }

   memcpy(&params.tess_level_outer_default, ctx->default_outer_level,
          sizeof(ctx->default_outer_level));
   memcpy(&params.tess_level_inner_default, ctx->default_inner_level,
          sizeof(ctx->default_inner_level));

   struct agx_grid vs_grid, tcs_grid, tess_grid;

   if (indirect == NULL) {
      poly_vertex_params_set_draw(&vp, draws->count, info->instance_count);

      unsigned in_patches = draws->count / patch_vertices;
      if (in_patches == 0)
         return;

      /* TCS invocation counter increments once per-patch. A draw drawn in
       * parts counts its first instance's parts, as it would whole. */
      if (!ctx->tess_split.first_instance) {
         agx_query_increment_cpu(
            ctx, ctx->pipeline_statistics[PIPE_STAT_QUERY_HS_INVOCATIONS],
            in_patches);
      }

      unsigned unrolled_patches = in_patches * info->instance_count;

      uint32_t alloc = 0;
      uint32_t tcs_out_offs = alloc;
      alloc += unrolled_patches * tcs->tess.output_stride;

      uint32_t patch_coord_offs = alloc;
      alloc += unrolled_patches * 4;

      uint32_t count_offs = alloc;
      alloc += unrolled_patches * sizeof(uint32_t);

      uint32_t draw_offs = alloc;
      alloc += draw_stride;

      struct agx_ptr blob =
         agx_pool_alloc_aligned_with_bo(&batch->pool, alloc, 4, &draw_bo);

      params.tcs_buffer = blob.gpu + tcs_out_offs;
      params.patches_per_instance = in_patches;
      params.coord_allocs = blob.gpu + patch_coord_offs;
      params.nr_patches = unrolled_patches;
      params.out_draws = blob.gpu + draw_offs;
      params.counts = blob.gpu + count_offs;

      unsigned vb_size = poly_tcs_in_size(draws->count * info->instance_count,
                                          batch->uniforms.vertex_outputs);
      vp.output_buffer =
         vb_size ? agx_pool_alloc_aligned(&batch->pool, vb_size, 4).gpu
                 : AGX_SCRATCH_PAGE_ADDRESS;

      vs_grid = agx_3d(draws->count, info->instance_count, 1);
      tcs_grid = agx_3d(in_patches * tcs->tess.output_patch_size,
                        info->instance_count, 1);

      tess_grid = agx_1d(unrolled_patches);
   } else if (indirect) {
      params.out_draws =
         agx_pool_alloc_aligned_with_bo(&batch->pool, draw_stride, 4, &draw_bo)
            .gpu;
   }

   uint64_t vertex_state =
      agx_pool_upload_aligned(&batch->pool, &vp, sizeof(vp), 8);
   batch->uniforms.vertex_params = vertex_state;

   uint64_t state =
      agx_pool_upload_aligned(&batch->pool, &params, sizeof(params), 4);
   batch->uniforms.tess_params = state;

   if (indirect) {
      perf_debug(dev, "Indirect tessellation");

      uint32_t grid_stride = sizeof(uint32_t) * 6;

      uint64_t indirect_ptr = agx_indirect_buffer_ptr(batch, indirect);

      uint64_t tcs_statistic = agx_get_query_address(
         batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_HS_INVOCATIONS]);

      /* Allocate 3x indirect global+local grids for VS/TCS/tess */
      uint64_t grids =
         agx_pool_alloc_aligned(&batch->pool, grid_stride * 3, 4).gpu;

      libagx_tess_setup_indirect(batch, agx_1d(1), AGX_BARRIER_ALL, state,
                                 grids, vertex_state, indirect_ptr, 0, 0,
                                 0 /* XXX: Index buffer */,
                                 ctx->vs->b.info.outputs, tcs_statistic);

      vs_grid = agx_grid_indirect_local(grids + 0 * grid_stride);
      tcs_grid = agx_grid_indirect_local(grids + 1 * grid_stride);
      tess_grid = agx_grid_indirect_local(grids + 2 * grid_stride);
   }

   agx_launch(batch, vs_grid, agx_workgroup(64, 1, 1), ctx->vs, ctx->linked.vs,
              MESA_SHADER_VERTEX, 0, 0);

   agx_launch(batch, tcs_grid, agx_workgroup(tcs->tess.output_patch_size, 1, 1),
              ctx->tcs, NULL, MESA_SHADER_TESS_CTRL, 0, 0);

   uint64_t c_prims = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_C_PRIMITIVES]);
   uint64_t c_invs = agx_get_query_address(
      batch, ctx->pipeline_statistics[PIPE_STAT_QUERY_C_INVOCATIONS]);

   /* Primitives generated: elsewhere a passthrough geometry shader counts
    * them (agx_needs_passthrough_gs), on M5 the prefix sum does, as in hk.
    * poly_increment_counters() skips a null pointer. */
   uint64_t prims_generated =
      m5 && ctx->active_queries && ctx->prims_generated[0]
         ? agx_get_query_address(batch, ctx->prims_generated[0])
         : 0;

   /* If there's a geometry shader, it will increment the clipper stats.
    * Otherwise, we do when tessellating.
    */
   if (ctx->stage[MESA_SHADER_GEOMETRY].shader) {
      c_prims = AGX_SCRATCH_PAGE_ADDRESS;
      c_invs = AGX_SCRATCH_PAGE_ADDRESS;
      prims_generated = 0;
   }

   /* Generate counts, then prefix sum them, then finally tessellate. */
   libagx_tessellate(batch, tess_grid, AGX_BARRIER_ALL, mode,
                     POLY_TESS_MODE_COUNT, state);
   libagx_prefix_sum_tess(batch, agx_1d(1024), AGX_BARRIER_ALL, state, c_prims,
                          c_invs, prims_generated,
                          c_prims || c_invs || prims_generated);
   libagx_tessellate(batch, tess_grid, AGX_BARRIER_ALL, mode,
                     POLY_TESS_MODE_WITH_COUNTS, state);

   /* Face culling state needs to be specialized for tess */
   ctx->dirty |= AGX_DIRTY_RS;

   /* M5 refuses a draw with rasterization discarded (agx_draw_vbo()): the
    * prepass above has already counted, and with no geometry shader there
    * is nothing else to do. */
   if (m5 && ctx->rast->base.rasterizer_discard &&
       !ctx->stage[MESA_SHADER_GEOMETRY].shader) {
      if (unbind_tcs_when_done)
         ctx->base.bind_tcs_state(&ctx->base, NULL);
      return;
   }

   /* Run TES as VS */
   void *vs_cso = ctx->stage[MESA_SHADER_VERTEX].shader;
   void *tes_cso = ctx->stage[MESA_SHADER_TESS_EVAL].shader;
   ctx->base.bind_vs_state(&ctx->base, tes_cso);
   ctx->in_tess = true;

   struct pipe_draw_info draw_info = {
      .mode = out_prim,
      .index_size = 4,
      .index.resource = ctx->heap,
      .instance_count = 1,
   };

   /* Wrap the pool allocation in a fake resource for meta-Gallium use */
   struct agx_resource indirect_rsrc = {.bo = draw_bo};

   struct pipe_draw_indirect_info copy_indirect = {
      .buffer = &indirect_rsrc.base.b,
      .offset = params.out_draws - draw_bo->va->addr,
      .stride = draw_stride,
      .draw_count = 1,
   };

   ctx->base.draw_vbo(&ctx->base, &draw_info, 0, &copy_indirect, NULL, 1);

   /* Restore vertex state */
   ctx->base.bind_vs_state(&ctx->base, vs_cso);
   ctx->in_tess = false;

   if (unbind_tcs_when_done) {
      ctx->base.bind_tcs_state(&ctx->base, NULL);
   }
}

/*
 * From the ARB_texture_barrier spec:
 *
 *  Specifically, the values of rendered fragments are undefined if any
 *  shader stage fetches texels and the same texels are written via fragment
 *  shader outputs, even if the reads and writes are not in the same Draw
 *  call, unless any of the following exceptions apply:
 *
 *  - The reads and writes are from/to disjoint sets of texels (after
 *    accounting for texture filtering rules).
 *
 *  - There is only a single read and write of each texel, and the read is in
 *    the fragment shader invocation that writes the same texel (e.g. using
 *    "texelFetch2D(sampler, ivec2(gl_FragCoord.xy), 0);").
 *
 *  - If a texel has been written, then in order to safely read the result
 *    a texel fetch must be in a subsequent Draw separated by the command
 *
 *      void TextureBarrier(void);
 *
 *    TextureBarrier() will guarantee that writes have completed and caches
 *    have been invalidated before subsequent Draws are executed."
 *
 * The wording is subtle, but we are not required to flush implicitly for
 * feedback loops, even though we're a tiler. What we are required to do is
 * decompress framebuffers involved in feedback loops, because otherwise
 * the hardware will race itself with exception #1, where we have a disjoint
 * group texels that intersects a compressed tile being written out.
 */
static void
agx_legalize_feedback_loop_surf(struct agx_context *ctx,
                                struct agx_resource *rsrc,
                                struct pipe_surface *surf, unsigned bit)
{
   if (agx_resource(surf->texture) != rsrc || !rsrc->layout.compressed)
      return;

   /* Decompress if we can and shadow if we can't. */
   if (agx_resource_is_shared(rsrc)) {
      struct agx_batch *batch = agx_get_batch(ctx);

      /* If we already did in-place decompression for this one */
      if (batch->feedback & bit)
         return;

      /* Use our current context batch. If it already touched
       * this buffer, that will have been flushed above.
       */
      agx_decompress_inplace(batch, surf, "Texture feedback loop");

      /* Mark it as a feedback cbuf, so it will be written to
       * uncompressed despite having a compressed layout.
       */
      batch->feedback |= bit;
   } else {
      agx_decompress(ctx, rsrc, "Texture feedback loop");
   }

   /* Not required by the spec, just for debug */
   if (agx_device(ctx->base.screen)->debug & AGX_DBG_FEEDBACK)
      agx_flush_writer(ctx, rsrc, "Feedback loop");
}

/*
 * The sampler views a stage's shader can read: the bindful textures, the only
 * views agx_upload_textures() gives the hardware. A stage without a shader
 * reads none.
 */
static unsigned
agx_nr_textures_read(const struct agx_context *ctx, mesa_shader_stage stage)
{
   const struct agx_stage *s = &ctx->stage[stage];

   if (!s->shader)
      return 0;

   return MIN2(s->texture_count, s->shader->info.nr_bindful_textures);
}

/*
 * Only views the draw's shaders read can form a loop. Gallium keeps a view
 * bound until it is replaced, and st/mesa leaves a stage's views alone while
 * the bound program samples nothing there, so a view of an attachment is
 * often still bound when nothing reads it: a shadow map sampled by the main
 * pass, then rendered again by a depth-only program without samplers.
 * Counting such a view would decompress the attachment for good. So skip the
 * compute stage, whose dispatches never run inside a draw, and the slots a
 * stage's shader does not read. Which slots those are depends on the shader,
 * so check a stage again when its shader changes as well as its views; a new
 * framebuffer or batch dirties every stage.
 */
static void
agx_legalize_feedback_loops(struct agx_context *ctx)
{
   const unsigned mask =
      ctx->stages_with_textures & ~BITFIELD_BIT(MESA_SHADER_COMPUTE);
   /* Trust that u_blitter knows what it's doing */
   if (!mask || ctx->blitter->running)
      return;

   u_foreach_bit(stage, mask) {
      if (!ctx->stage[stage].shader)
         continue;

      bool new_shader =
         (stage == MESA_SHADER_VERTEX && (ctx->dirty & AGX_DIRTY_VS_PROG)) ||
         (stage == MESA_SHADER_FRAGMENT && (ctx->dirty & AGX_DIRTY_FS_PROG));

      if (!(ctx->stage[stage].dirty & AGX_STAGE_DIRTY_IMAGE) && !new_shader)
         continue;

      unsigned nr_read = agx_nr_textures_read(ctx, stage);

      for (unsigned i = 0; i < nr_read; ++i) {
         if (!ctx->stage[stage].textures[i])
            continue;

         struct agx_resource *rsrc = ctx->stage[stage].textures[i]->rsrc;

         /* The bit names the colour buffer, as every reader of
          * batch->feedback indexes it by render target. */
         for (unsigned cb = 0; cb < ctx->framebuffer.nr_cbufs; ++cb) {
            agx_legalize_feedback_loop_surf(
               ctx, rsrc, &ctx->framebuffer.cbufs[cb], PIPE_CLEAR_COLOR0 << cb);
         }

         /* TODO: Separate stencil? */
         agx_legalize_feedback_loop_surf(ctx, rsrc, &ctx->framebuffer.zsbuf,
                                         PIPE_CLEAR_DEPTH);
      }
   }
}

/*
 * Usually, non-attachment stores must be barriered explicitly by the app using
 * glMemoryBarrier. Transform feedback buffers are annoyingly excluded from this
 * requirement, so we need to handle those hazards ourselves. Transform feedback
 * will barrier with itself so we only need to consider write-after-read
 * hazards.
 *
 * The general case does require a flush. Consider binding a buffer as a UBO,
 * drawing with a fragment shader reading that UBO, then rebinding as XFB, and
 * drawing with XFB stores. We have to split the batch in the middle.
 * gles-3.0-transform-feedback-uniform-buffer-object does this.
 */
static void
agx_legalize_xfb(struct agx_context *ctx)
{
   /* If this draw isn't writing transform feedback, there's nothing to worry
    * about.
    */
   if (!ctx->streamout.num_targets ||
       !agx_last_uncompiled_vgt(ctx)->has_xfb_info)
      return;

   /* Otherwise, flush the readers of anything written by transform feedback. */
   for (unsigned i = 0; i < ctx->streamout.num_targets; ++i) {
      struct agx_streamout_target *tgt =
         agx_so_target(ctx->streamout.targets[i]);

      if (tgt != NULL) {
         agx_flush_readers(ctx, agx_resource(tgt->base.buffer),
                           "Transform feedback buffer");
      }
   }
}

static bool
agx_keep_one_fs_output(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);
   if (sem.location < FRAG_RESULT_DATA0 || !nir_src_is_const(*offset))
      return false;

   if (sem.location - FRAG_RESULT_DATA0 + nir_src_as_uint(*offset) ==
       *(unsigned *)data) {
      sem.location = FRAG_RESULT_DATA0;
      sem.num_slots = 1;
      nir_intrinsic_set_io_semantics(intr, sem);
      b->cursor = nir_before_instr(&intr->instr);
      nir_src_rewrite(offset, nir_imm_int(b, 0));
   } else {
      nir_instr_remove(&intr->instr);
   }
   return true;
}

static bool
agx_drop_unread_varying(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_store_output)
      return false;

   nir_io_semantics sem = nir_intrinsic_io_semantics(intr);
   nir_src *offset = nir_get_io_offset_src(intr);
   if (sem.location < VARYING_SLOT_VAR0 || !nir_src_is_const(*offset))
      return false;

   const unsigned slot = sem.location + nir_src_as_uint(*offset);
   if (*(uint64_t *)data & BITFIELD64_BIT(slot))
      return false;

   nir_instr_remove(&intr->instr);
   return true;
}

/* The varyings the split MRT pass for attachment rt reads: the fragment
 * program with only that colour output, as agx_compile_variant builds it,
 * once dead code is gone. */
static uint64_t
agx_g17g_mrt_inputs(struct agx_uncompiled_shader *so, unsigned rt)
{
   if (so->g17g_mrt_inputs_valid & BITFIELD_BIT(rt))
      return so->g17g_mrt_inputs[rt];

   struct blob_reader reader;
   blob_reader_init(&reader, so->serialized_nir.data, so->serialized_nir.size);
   nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, agx_keep_one_fs_output,
            nir_metadata_control_flow, &rt);
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));

   so->g17g_mrt_inputs[rt] = nir->info.inputs_read;
   so->g17g_mrt_inputs_valid |= BITFIELD_BIT(rt);
   ralloc_free(nir);
   return so->g17g_mrt_inputs[rt];
}

/*
 * A G17G render keeps several colour attachments in the tilebuffer only in
 * the layouts its multi-target tile programs move (agx_g17g_mrt_resident),
 * and a render of more than one attachment in any other is dropped. Draw
 * such a framebuffer one attachment at a time: bind each attachment alone at
 * slot 0 with the depth/stencil attachment, and let the epilog store that
 * attachment's output location with its blend state (agx_update_fs).
 * Blending, colour masks and outputs are per attachment, so the colour
 * results are exact. Depth and stencil are updated by the first pass only;
 * later passes test depth for equality against what it wrote, with all depth
 * and stencil writes off.
 */
static void
agx_g17g_split_mrt(struct agx_context *ctx, bool app_draw,
                   void (*pass)(struct agx_context *ctx, unsigned rt,
                                bool first, void *data),
                   void *data)
{
   struct pipe_context *pctx = &ctx->base;
   struct pipe_framebuffer_state saved = {0};
   util_copy_framebuffer_state(&saved, &ctx->framebuffer);

   bool first = true;
   for (unsigned rt = 0; rt < saved.nr_cbufs; ++rt) {
      if (!saved.cbufs[rt].texture)
         continue;

      /* set_framebuffer_state takes its own references */
      struct pipe_framebuffer_state fb = saved;
      fb.nr_cbufs = 1;
      fb.cbufs[0] = saved.cbufs[rt];
      memset(&fb.cbufs[1], 0, sizeof(fb.cbufs) - sizeof(fb.cbufs[0]));
      pctx->set_framebuffer_state(pctx, &fb);

      ctx->g17g_mrt_split = app_draw;
      ctx->g17g_mrt_rt = rt;
      pass(ctx, rt, first, data);
      ctx->g17g_mrt_split = false;
      ctx->g17g_mrt_pass++;
      first = false;
   }

   ctx->g17g_mrt_pass = 0;
   pctx->set_framebuffer_state(pctx, &saved);
   util_unreference_framebuffer_state(&saved);
}

struct agx_g17g_mrt_draw {
   const struct pipe_draw_info *info;
   unsigned drawid_offset;
   const struct pipe_draw_indirect_info *indirect;
   const struct pipe_draw_start_count_bias *draws;
};

static void
agx_g17g_mrt_draw_pass(struct agx_context *ctx, unsigned rt, bool first,
                       void *data)
{
   struct pipe_context *pctx = &ctx->base;
   const struct agx_g17g_mrt_draw *d = data;
   struct agx_zsa *zs = ctx->zs;
   void *later = NULL;

   if (!first && zs &&
       ((zs->base.depth_enabled && zs->base.depth_writemask) ||
        (zs->base.stencil[0].enabled && zs->base.stencil[0].writemask) ||
        (zs->base.stencil[1].enabled && zs->base.stencil[1].writemask))) {
      struct pipe_depth_stencil_alpha_state state = zs->base;
      if (state.depth_enabled && state.depth_writemask) {
         state.depth_func = PIPE_FUNC_EQUAL;
         state.depth_writemask = false;
      }
      state.stencil[0].writemask = 0;
      state.stencil[1].writemask = 0;
      later = pctx->create_depth_stencil_alpha_state(pctx, &state);
      pctx->bind_depth_stencil_alpha_state(pctx, later);
   }

   pctx->draw_vbo(pctx, d->info, d->drawid_offset, d->indirect, d->draws, 1);

   if (later) {
      pctx->bind_depth_stencil_alpha_state(pctx, zs);
      pctx->delete_depth_stencil_alpha_state(pctx, later);
   }
}

static void
agx_g17g_split_mrt_draw(struct agx_context *ctx,
                        const struct pipe_draw_info *info,
                        unsigned drawid_offset,
                        const struct pipe_draw_indirect_info *indirect,
                        const struct pipe_draw_start_count_bias *draws)
{
   struct agx_g17g_mrt_draw d = {info, drawid_offset, indirect, draws};
   agx_g17g_split_mrt(ctx, true, agx_g17g_mrt_draw_pass, &d);
}

/*
 * The split repeats a draw once per attachment, and with it every side
 * effect of its shaders: an atomic counter counts each invocation once per
 * attachment (dEQP-GLES31 opaque_type_indexing.atomic_counter). A
 * framebuffer fetch reads attachments the split does not bind
 * (framebuffer_fetch.basic.multiple_render_targets). Such a draw runs once
 * over the whole framebuffer instead: attachment 0 stays in the tilebuffer
 * and the others are spilled, loaded and stored by the fragments as images
 * (agx_build_tilebuffer_layout_rt0_resident). Single-sample only, and
 * attachment 0 must exist to be the resident one.
 *
 * Every other draw runs in one pass too when its spilled attachments can be
 * described (agx_g17g_mrt_spills_encodable). The split costs far more than
 * the repeated draw: its passes share the depth attachment, the first
 * writing it and the others testing against it, so each pass flushes the
 * batch of the one before and every application draw becomes a render per
 * attachment.
 */

/* The fragment shader addresses at most four attachments as images: with
 * more, attachments 4 and up keep their old contents (piglit
 * bound-resource-limits, arb_texture_barrier-blending-in-shader with eight
 * textures). */
#define AGX_G17G_ONE_PASS_MAX_CBUFS 4

/*
 * Whether each attachment has the texture and PBE descriptors the fragments
 * load and store the spilled ones through
 * (agx_upload_spilled_rt_descriptors). A
 * batch with one that has none is dropped whole, so such a framebuffer
 * keeps the split.
 */
static bool
agx_g17g_mrt_spills_encodable(struct agx_context *ctx)
{
   struct agx_device *dev = agx_device(ctx->base.screen);
   const enum agx_descriptor_generation generation =
      agx_chip_descriptor_generation(dev->chip);
   /* The packers only read the device through the batch. */
   struct agx_batch scratch = {.ctx = ctx};

   if (ctx->framebuffer.nr_cbufs > AGX_G17G_ONE_PASS_MAX_CBUFS)
      return false;

   /* Attachment 0 too: the table holds an entry for every attachment. */
   for (unsigned rt = 0; rt < ctx->framebuffer.nr_cbufs; ++rt) {
      const struct pipe_surface *surf = &ctx->framebuffer.cbufs[rt];
      if (!surf->texture)
         continue;

      struct agx_gallium_texture_descriptor texture;
      struct agx_gallium_pbe_descriptor pbe;
      struct pipe_image_view view = image_view_for_surface(surf);
      struct pipe_sampler_view sampler_view = sampler_view_for_surface(surf);
      sampler_view.target = PIPE_TEXTURE_2D_ARRAY;

      if (!agx_pack_texture(generation, &texture, agx_resource(surf->texture),
                            surf->format, &sampler_view) ||
          !agx_batch_upload_pbe_internal(&scratch, &pbe, &view, false, false,
                                         true, true, false, NULL))
         return false;
   }

   return true;
}

bool
agx_g17g_mrt_resident(struct agx_context *ctx)
{
   if (!ctx->g17g_mrt_resident_known) {
      ctx->g17g_mrt_resident_ok = agx_g17g_fb_mrt_resident(
         agx_device(ctx->base.screen), &ctx->framebuffer);
      ctx->g17g_mrt_resident_known = true;
   }
   return ctx->g17g_mrt_resident_ok;
}

static bool
agx_g17g_mrt_one_pass(struct agx_context *ctx)
{
   struct agx_uncompiled_shader *vs = ctx->stage[MESA_SHADER_VERTEX].shader;
   struct agx_uncompiled_shader *fs = ctx->stage[MESA_SHADER_FRAGMENT].shader;

   if (!ctx->framebuffer.cbufs[0].texture ||
       util_framebuffer_get_num_samples(&ctx->framebuffer) > 1 ||
       util_framebuffer_get_num_layers(&ctx->framebuffer) > 1)
      return false;

   if ((vs && vs->info.writes_memory) ||
       (fs && (fs->info.writes_memory || fs->info.uses_fbfetch)))
      return true;

   if (!ctx->g17g_mrt_spills_known) {
      ctx->g17g_mrt_spills_ok = agx_g17g_mrt_spills_encodable(ctx);
      ctx->g17g_mrt_spills_known = true;
   }
   return ctx->g17g_mrt_spills_ok;
}

static void
agx_g17g_one_pass_mrt_draw(struct agx_context *ctx,
                           const struct pipe_draw_info *info,
                           unsigned drawid_offset,
                           const struct pipe_draw_indirect_info *indirect,
                           const struct pipe_draw_start_count_bias *draws)
{
   ctx->g17g_mrt_spill = true;
   ctx->base.draw_vbo(&ctx->base, info, drawid_offset, indirect, draws, 1);
   ctx->g17g_mrt_spill = false;
}

struct agx_g17g_mrt_clear {
   unsigned buffers;
   uint32_t color_clear_mask;
   uint8_t stencil_clear_mask;
   const union pipe_color_union *color;
   double depth;
   unsigned stencil;
};

static void
agx_g17g_mrt_clear_pass(struct agx_context *ctx, unsigned rt, bool first,
                        void *data)
{
   const struct agx_g17g_mrt_clear *c = data;
   unsigned buffers = (c->buffers & (PIPE_CLEAR_COLOR0 << rt)) ? PIPE_CLEAR_COLOR0 : 0;
   if (first)
      buffers |= c->buffers & PIPE_CLEAR_DEPTHSTENCIL;

   if (buffers) {
      ctx->base.clear(&ctx->base, buffers, (c->color_clear_mask >> (4 * rt)) & 0xf,
                      c->stencil_clear_mask, NULL, c->color, c->depth,
                      c->stencil);
   }
}

void
agx_g17g_split_mrt_clear(struct agx_context *ctx, unsigned buffers,
                         uint32_t color_clear_mask, uint8_t stencil_clear_mask,
                         const union pipe_color_union *color, double depth,
                         unsigned stencil)
{
   struct agx_g17g_mrt_clear c = {
      buffers, color_clear_mask, stencil_clear_mask, color, depth, stencil,
   };
   agx_g17g_split_mrt(ctx, false, agx_g17g_mrt_clear_pass, &c);
}

/*
 * G17G renders one layer: the kernel and the tile programs refuse a layered
 * render. A layered framebuffer is drawn a layer at a time through each
 * attachment's single-layer view. Primitives without a layer output go to
 * layer 0, as in a layered render; agx_swtnl.c draws the primitives of a CPU
 * geometry stage, or of a vertex shader that writes gl_Layer, into the layer
 * each selects.
 */
void
agx_g17g_bind_layer(struct agx_context *ctx,
                    const struct pipe_framebuffer_state *layered, unsigned layer)
{
   struct pipe_framebuffer_state fb = *layered;
   struct pipe_surface *surfaces[PIPE_MAX_COLOR_BUFS + 1];
   unsigned nr = 0;
   for (unsigned i = 0; i < fb.nr_cbufs; ++i)
      surfaces[nr++] = &fb.cbufs[i];
   surfaces[nr++] = &fb.zsbuf;

   for (unsigned i = 0; i < nr; ++i) {
      struct pipe_surface *surf = surfaces[i];
      if (!surf->texture)
         continue;
      /* An attachment with fewer layers has none to draw into here. */
      if (surf->first_layer + layer > surf->last_layer) {
         memset(surf, 0, sizeof(*surf));
         continue;
      }
      surf->first_layer += layer;
      surf->last_layer = surf->first_layer;
   }
   fb.layers = 1;
   ctx->base.set_framebuffer_state(&ctx->base, &fb);
   ctx->g17g_layer = layer;
}

static void
agx_g17g_layered_draw(struct agx_context *ctx, const struct pipe_draw_info *info,
                      unsigned drawid_offset,
                      const struct pipe_draw_indirect_info *indirect,
                      const struct pipe_draw_start_count_bias *draws,
                      unsigned num_draws)
{
   struct pipe_context *pctx = &ctx->base;
   struct pipe_framebuffer_state saved = {0};
   util_copy_framebuffer_state(&saved, &ctx->framebuffer);

   ctx->g17g_layer_split = true;
   if (!indirect && agx_swtnl_applies(ctx)) {
      ctx->g17g_layered_fb = &saved;
      agx_swtnl_draw(ctx, info, draws);
      ctx->g17g_layered_fb = NULL;
   } else {
      agx_g17g_bind_layer(ctx, &saved, 0);
      pctx->draw_vbo(pctx, info, drawid_offset, indirect, draws, num_draws);
   }
   ctx->g17g_layer_split = false;
   ctx->g17g_layer = 0;

   pctx->set_framebuffer_state(pctx, &saved);
   util_unreference_framebuffer_state(&saved);
}

void
agx_g17g_layered_clear(struct agx_context *ctx, unsigned buffers,
                       uint32_t color_clear_mask, uint8_t stencil_clear_mask,
                       const union pipe_color_union *color, double depth,
                       unsigned stencil)
{
   struct pipe_context *pctx = &ctx->base;
   struct pipe_framebuffer_state saved = {0};
   util_copy_framebuffer_state(&saved, &ctx->framebuffer);

   ctx->g17g_layer_split = true;
   const unsigned layers = util_framebuffer_get_num_layers(&saved);
   for (unsigned layer = 0; layer < layers; ++layer) {
      agx_g17g_bind_layer(ctx, &saved, layer);
      pctx->clear(pctx, buffers, color_clear_mask, stencil_clear_mask, NULL,
                  color, depth, stencil);
   }
   ctx->g17g_layer_split = false;
   ctx->g17g_layer = 0;

   pctx->set_framebuffer_state(pctx, &saved);
   util_unreference_framebuffer_state(&saved);
}

/* Most primitives one patch can tessellate into. A patch cut into at most n
 * segments per edge yields at most 2 (n + 1)^2 primitives; n is 64 when a
 * control shader writes the levels, else the default levels bound it.
 */
static unsigned
agx_tess_max_prims_per_patch(const struct agx_context *ctx)
{
   unsigned segments = 64;

   if (!ctx->stage[MESA_SHADER_TESS_CTRL].shader) {
      const float levels[6] = {
         ctx->default_outer_level[0], ctx->default_outer_level[1],
         ctx->default_outer_level[2], ctx->default_outer_level[3],
         ctx->default_inner_level[0], ctx->default_inner_level[1],
      };
      float level = 1.0f;
      bool bounded = true;

      /* A NaN or larger level leaves the bound at 64. */
      for (unsigned i = 0; i < ARRAY_SIZE(levels); ++i) {
         bounded &= levels[i] <= 63.0f;
         level = bounded ? MAX2(level, levels[i]) : level;
      }

      /* Fractional partitioning rounds up to the next odd or even count. */
      if (bounded)
         segments = MIN2((unsigned)level + 2, 64);
   }

   return 2 * (segments + 1) * (segments + 1);
}

static_assert(2 * 65 * 65 <= AGX_G17G_RENDER_PRIMS,
              "one patch fits an M5 render");

/* Draw a direct patch draw in consecutive parts whose worst case each fits
 * one M5 render (agx_draw_vbo() gives each its own): whole instances when
 * one instance fits, else patch ranges of one instance. The parts run in API
 * order and keep the draw's IDs: gl_PrimitiveID counts from the part's first
 * patch and gl_InstanceID from its first instance (agx_draw_patches), while
 * start_instance, and with it gl_BaseInstance and instanced fetches, stays.
 */
static void
agx_draw_patches_split(struct agx_context *ctx,
                       const struct pipe_draw_info *info, unsigned drawid_offset,
                       const struct pipe_draw_start_count_bias *draw,
                       unsigned patches, unsigned max_patches)
{
   const unsigned patch_vertices = ctx->patch_vertices;
   const unsigned patch_step = MIN2(patches, max_patches);
   const unsigned instance_step = max_patches / patch_step;

   struct pipe_draw_info part = *info;
   struct pipe_draw_start_count_bias part_draw = *draw;

   /* The render condition holds for the whole draw, checked once. */
   struct pipe_query *cond_query = ctx->cond_query;
   ctx->cond_query = NULL;

   for (unsigned i = 0; i < info->instance_count;) {
      part.instance_count = MIN2(instance_step, info->instance_count - i);

      for (unsigned p = 0; p < patches;) {
         const unsigned n = MIN2(patch_step, patches - p);

         part_draw.start = draw->start + p * patch_vertices;
         part_draw.count = n * patch_vertices;

         ctx->tess_split.first_patch = p;
         ctx->tess_split.first_instance = i;
         ctx->base.draw_vbo(&ctx->base, &part, drawid_offset, NULL,
                            &part_draw, 1);
         p += n;
      }

      i += part.instance_count;
   }

   ctx->tess_split.first_patch = 0;
   ctx->tess_split.first_instance = 0;
   ctx->cond_query = cond_query;
}

/* Multi-draws (glMultiDrawElements*, Sodium's terrain) arrive as one
 * pipe_draw_vbo with hundreds of sub-draws that share every bound state.
 * util_draw_multi() sends each through agx_draw_vbo(), which re-validates
 * shaders, descriptors, feedback loops and queries per sub-draw: ~0.1 us of
 * CPU each, 1.5 ms per frame at Minecraft's 15k draws. Once a full draw has
 * left the context clean, a following sub-draw with the same state only needs
 * its index list record: agx_encode_state() would emit nothing for it anyway.
 * Anything that varies per sub-draw (draw ID or base parameters in uniforms,
 * statistics, geometry or tessellation stages, transform feedback, misaligned
 * indices) keeps the full path. The record written is the HAL200 draw body
 * of agx_g17p_emit_draw_body(), which is what every agx_chip_native_agx3()
 * chip encodes; other chips keep util_draw_multi().
 */

static void agx_draw_vbo(struct pipe_context *pctx,
                         const struct pipe_draw_info *info,
                         unsigned drawid_offset,
                         const struct pipe_draw_indirect_info *indirect,
                         const struct pipe_draw_start_count_bias *draws,
                         unsigned num_draws);

static bool
agx_multidraw_state_clean(const struct agx_context *ctx)
{
   if (ctx->dirty)
      return false;

   for (unsigned i = 0; i < ARRAY_SIZE(ctx->stage); ++i) {
      if (i != MESA_SHADER_COMPUTE && ctx->stage[i].dirty)
         return false;
   }

   return true;
}

/* Is everything agx_draw_vbo() redoes per sub-draw a function of the bound
 * state alone? Checked once per multi-draw, after its first full draw. */
static bool
agx_multidraw_can_skip_common(struct agx_context *ctx,
                              const struct pipe_draw_info *info,
                              bool allow_draw_params)
{
   if (!ctx->vs || !ctx->linked.vs || !ctx->fs || !ctx->linked.fs ||
       ctx->gs || ctx->stage[MESA_SHADER_GEOMETRY].shader ||
       ctx->stage[MESA_SHADER_TESS_EVAL].shader || ctx->in_tess ||
       ctx->swtnl_inner || ctx->active_draw_without_restart ||
       ctx->streamout.num_targets || info->mode == MESA_PRIM_PATCHES)
      return false;

   if (!allow_draw_params &&
       (ctx->vs->b.info.uses_draw_id || ctx->linked.vs->uses_base_param))
      return false;

   if (ctx->active_queries) {
      if (ctx->prims_generated[0])
         return false;

      for (unsigned i = 0; i < ARRAY_SIZE(ctx->pipeline_statistics); ++i) {
         if (ctx->pipeline_statistics[i])
            return false;
      }
   }

   bool xfb_passthrough = false;
   return !agx_needs_passthrough_gs(ctx, info, NULL, &xfb_passthrough, false);
}

static bool
agx_multidraw_can_skip(struct agx_context *ctx,
                       const struct pipe_draw_info *info)
{
   return agx_multidraw_can_skip_common(ctx, info, false);
}

static void
agx_multidraw_emit_body(struct agx_context *ctx, struct agx_batch *batch,
                        const struct pipe_draw_info *info,
                        const struct pipe_draw_start_count_bias *d,
                        uint64_t ib, size_t ib_extent)
{
   struct agx_device *dev = agx_device(ctx->base.screen);

   if (ctx->time_elapsed)
      agx_batch_add_timestamp_query(batch, ctx->time_elapsed);

   agx_ensure_cmdbuf_has_space(
      batch, &batch->vdm,
      AGX_VDM_STATE_UPDATE_LENGTH + AGX_VDM_HAL200_BIND_TAIL_LENGTH +
         AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH +
         AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH + AGX_VDM_BARRIER_LENGTH);

   struct agx_draw draw = {0};
   if (info->index_size) {
      draw.index_size = agx_translate_index_size(info->index_size);
      draw.index_buffer = ib;
      draw.index_buffer_range_B = ib_extent;
      draw.restart = info->primitive_restart;
      draw.indexed = true;
      draw.index_bias = d->index_bias;
   } else {
      draw.start = d->start;
   }
   draw.b = agx_3d(d->count, info->instance_count, 1);

   batch->vdm.current = agx_g17p_emit_draw_body(
      dev->chip, batch->vdm.current, &draw, agx_primitive_for_pipe(info->mode),
      info->restart_index, 0);
   assert((batch->vdm.current + AGX_VDM_STREAM_LINK_LENGTH) <= batch->vdm.end &&
          "Failed to reserve sufficient space in encoder");

   batch->draws++;

   if (unlikely(dev->debug & AGX_DBG_PERF))
      ctx->perf.draws++;

   /* Same limits as the end of agx_draw_vbo() */
   if (unlikely(
          (((batch->scissor.size / AGX_SCISSOR_LENGTH) + AGX_MAX_VIEWPORTS) >
           UINT16_MAX) ||
          (batch->depth_bias.size / AGX_DEPTH_BIAS_LENGTH) >= UINT16_MAX)) {
      agx_flush_batch_for_reason(ctx, batch, "Scissor/depth bias overflow");
   } else if (unlikely(batch->draws > 100000)) {
      agx_flush_batch_for_reason(ctx, batch, "Absurd number of draws");
   }
}

static void
agx_draw_multi(struct pipe_context *pctx, const struct pipe_draw_info *info,
               unsigned drawid_offset,
               const struct pipe_draw_indirect_info *indirect,
               const struct pipe_draw_start_count_bias *draws,
               unsigned num_draws)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);

   if (indirect || !agx_chip_native_agx3(dev->chip)) {
      util_draw_multi(pctx, info, drawid_offset, indirect, draws, num_draws);
      return;
   }

   unsigned drawid = drawid_offset;
   struct agx_batch *clean = NULL;
   int can_skip = -1;
   /* Index buffer of the clean batch: its read is already recorded */
   uint64_t ib_base = 0;
   size_t ib_size = 0;

   for (unsigned i = 0; i < num_draws; i++) {
      const struct pipe_draw_start_count_bias *d = &draws[i];

      if (d->count && info->instance_count) {
         const uint64_t offset = (uint64_t)d->start * info->index_size;
         uint64_t ib = 0;
         size_t ib_extent = 0;
         bool fast = clean != NULL;

         if (fast && info->index_size) {
            if (info->has_user_indices) {
               ib = agx_index_buffer_ptr(clean, info, d, &ib_extent);
            } else if (((ib_base + offset) & 3) || offset >= ib_size) {
               /* agx_draw_vbo() realigns these through a synchronizing map */
               fast = false;
            } else {
               ib = ib_base + offset;
               ib_extent =
                  ALIGN_POT(MIN2(ib_size - offset,
                                 (uint64_t)d->count * info->index_size), 4);
            }
         }

         if (fast) {
            agx_multidraw_emit_body(ctx, clean, info, d, ib, ib_extent);
         } else {
            struct agx_batch *before = ctx->batch;
            const unsigned before_draws = before ? before->draws : 0;

            agx_draw_vbo(pctx, info, drawid, NULL, d, 1);

            clean = NULL;
            if (ctx->batch && !agx_batch_is_compute(ctx->batch) &&
                (ctx->batch != before || ctx->batch->draws != before_draws) &&
                agx_multidraw_state_clean(ctx)) {
               if (can_skip < 0)
                  can_skip = agx_multidraw_can_skip(ctx, info);
               if (can_skip)
                  clean = ctx->batch;
            }

            /* Bindings do not change within the multi-draw, so the checks
             * agx_draw_vbo() makes before acquiring its batch give the same
             * answer for every later sub-draw: make them once, now that the
             * batch exists. A feedback loop flushes or dirties here and keeps
             * the full path. */
            if (clean) {
               agx_legalize_feedback_loops(ctx);
               if (ctx->batch != clean || !agx_multidraw_state_clean(ctx))
                  clean = NULL;
            }

            if (clean && info->index_size && !info->has_user_indices)
               ib_base = agx_index_buffer_rsrc_ptr(clean, info, &ib_size);
         }

         if (clean && ctx->batch != clean)
            clean = NULL;
      }

      if (info->increment_draw_id)
         drawid++;
   }
}

/* Return the smallest 4-byte-aligned address >= start such that every draw_id
 * slice within [addr, addr + span] lies more than INT16_MAX bytes away from
 * every other uniform range. Keeping the per-draw draw_id range in its own
 * Ioto address-cache component avoids merging it with static pool tables
 * (whose changing pairwise offset would otherwise miss the template cache on
 * every draw). */
static uint64_t
agx_mdi_isolated_va(uint64_t start, uint64_t span,
                    const struct agx3_ioto_uniform_range *ranges,
                    size_t range_count, int draw_id_range, int params_range,
                    uint64_t params_start, uint64_t params_end)
{
   uint64_t cand = start;
   for (size_t iter = 0; iter <= range_count; ++iter) {
      bool moved = false;
      for (size_t r = 0; r < range_count; ++r) {
         if ((int)r == draw_id_range)
            continue;
         uint64_t r_start, r_end;
         const uint64_t r_bytes = (uint64_t)ranges[r].size_halfs * 2;
         if ((int)r == params_range && params_start <= params_end) {
            r_start = params_start;
            r_end = params_end;
         } else {
            r_start = ranges[r].address;
            r_end = r_bytes >= 4 ? r_start + r_bytes - 4 : r_start;
         }
         if (r_end > UINT64_MAX - INT16_MAX - 4)
            continue;
         const uint64_t cand_end = cand + span;
         if (cand_end + INT16_MAX >= r_start && cand <= r_end + INT16_MAX) {
            cand = ALIGN_POT(r_end + INT16_MAX + 1, 4);
            moved = true;
         }
      }
      if (!moved)
         break;
   }
   return cand;
}

static struct agx_ptr
agx_mdi_alloc_draw_id_chunk(struct agx_pool *pool, size_t total_bytes,
                            const struct agx3_ioto_uniform_range *ranges,
                            size_t range_count, int draw_id_range,
                            int params_range, uint64_t params_start,
                            uint64_t params_end)
{
   const uint64_t span = total_bytes >= 4 ? total_bytes - 4 : 0;
   struct agx_ptr fallback = {0};

   for (unsigned attempt = 0; attempt < 2; ++attempt) {
      size_t pad = 0;
      if (pool->transient_bo && pool->transient_bo->va) {
         const uint64_t cur_va =
            pool->transient_bo->va->addr + ALIGN_POT(pool->transient_offset, 4);
         const uint64_t safe_va = agx_mdi_isolated_va(
            cur_va, span, ranges, range_count, draw_id_range, params_range,
            params_start, params_end);
         if (safe_va > cur_va && safe_va - cur_va <= 128 * 1024)
            pad = (size_t)(safe_va - cur_va);
      }

      struct agx_ptr ptr =
         agx_pool_alloc_aligned(pool, pad + total_bytes, 4);
      if (!ptr.cpu)
         return ptr;

      const uint64_t safe_va = agx_mdi_isolated_va(
         ptr.gpu, span, ranges, range_count, draw_id_range, params_range,
         params_start, params_end);
      if (safe_va >= ptr.gpu && safe_va - ptr.gpu <= pad) {
         const size_t delta = (size_t)(safe_va - ptr.gpu);
         return (struct agx_ptr){
            .cpu = (uint8_t *)ptr.cpu + delta,
            .gpu = ptr.gpu + delta,
         };
      }
      fallback = ptr;
   }
   return fallback;
}

struct agx_mdi_dyn_reloc {
   uint64_t dest[2];
   uint32_t delta;
   uint16_t offset;
   uint8_t short_mask;
   bool is_draw_id;
};

static bool
agx_mdi_prepare_batch_ioto(
   struct agx_device *dev, struct agx_linked_shader *linked,
   const struct agx3_g17p_graphics_program_desc *vs_desc,
   const struct agx3_ioto_uniform_range *base_ranges, size_t range_count,
   const struct agx_g17p_uniform_range_locs *locs, bool uses_draw_id,
   bool uses_base_param, uint64_t first_draw_id_va, uint64_t last_draw_id_va,
   uint64_t first_params_va, uint64_t last_params_va, uint8_t tpl_bytes[512],
   uint16_t *tpl_len_out,
   struct agx_mdi_dyn_reloc dyn_relocs[HK_G17P_PATCH_MAX_RELOCS],
   unsigned *dyn_reloc_count_out)
{
   struct agx3_ioto_uniform_range first_ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   struct agx3_ioto_uniform_range last_ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   memcpy(first_ranges, base_ranges, range_count * sizeof(base_ranges[0]));
   memcpy(last_ranges, base_ranges, range_count * sizeof(base_ranges[0]));

   uint32_t draw_id_mask = 0;
   uint32_t params_mask = 0;
   if (uses_draw_id && locs->draw_id_range >= 0) {
      first_ranges[locs->draw_id_range].address = first_draw_id_va;
      last_ranges[locs->draw_id_range].address = last_draw_id_va;
      draw_id_mask |= BITFIELD_BIT(locs->draw_id_range);
   }
   if (uses_base_param && locs->params_range >= 0) {
      first_ranges[locs->params_range].address = first_params_va;
      last_ranges[locs->params_range].address = last_params_va;
      params_mask |= BITFIELD_BIT(locs->params_range);
   }
   const uint32_t dyn_mask = draw_id_mask | params_mask;
   if (!dyn_mask ||
       !agx_g17p_graphics_template_ranges_valid(first_ranges, range_count) ||
       !agx_g17p_graphics_template_ranges_valid(last_ranges, range_count))
      return false;

   struct agx3_g17p_graphics_program_desc find_desc = *vs_desc;
   find_desc.uniform_ranges = first_ranges;
   find_desc.program_va = 0;
   const struct agx_g17p_graphics_template *plan =
      agx_g17p_graphics_program_find_plan(linked, &find_desc);
   if (!plan) {
      const uint64_t dummy_va = dev->shader_base + 0x10000;
      if (!agx_va_in_usc_window(dev, dummy_va) ||
          (dummy_va & (AGX_IOTO_PROGRAM_ALIGNMENT - 1)))
         return false;
      find_desc.program_va = dummy_va;
      uint8_t scratch[512];
      size_t scratch_written = 0;
      if (!agx_g17p_graphics_program_generate_cached(
             dev->chip, linked, &find_desc, scratch, sizeof(scratch),
             &scratch_written))
         return false;
      find_desc.program_va = 0;
      plan = agx_g17p_graphics_program_find_plan(linked, &find_desc);
   }
   if (!plan || !plan->has_plan)
      return false;

   const uint16_t len = plan->program.program.length;
   if (!len || len > 512)
      return false;

   if (!hk_g17p_patch_plan_fits(&plan->plan, first_ranges) ||
       !hk_g17p_patch_plan_fits(&plan->plan, last_ranges))
      return false;

   /* Require that no dynamic range shares an address-cache cluster with any
    * other range so per-draw patching cannot invalidate cluster offsets. */
   if (plan->plan.shared) {
      for (unsigned r = 0; r < plan->plan.range_count; ++r) {
         const unsigned leader = plan->plan.leader[r];
         if (dyn_mask & BITFIELD_BIT(r)) {
            if (leader != r)
               return false;
         } else if (dyn_mask & BITFIELD_BIT(leader)) {
            return false;
         }
      }
   }

   unsigned dyn_count = 0;
   for (unsigned r = 0; r < plan->plan.reloc_count; ++r) {
      const struct hk_g17p_patch_reloc *rel = &plan->plan.relocs[r];
      if (!(dyn_mask & BITFIELD_BIT(rel->range)))
         continue;
      const uint64_t base_first = first_ranges[rel->range].address + rel->delta;
      const uint64_t base_last = last_ranges[rel->range].address + rel->delta;
      if (base_last < base_first)
         return false;
      if ((rel->short_mask & 1u) && (base_first >> 32) != (base_last >> 32))
         return false;
      if (dyn_count >= HK_G17P_PATCH_MAX_RELOCS)
         return false;
      dyn_relocs[dyn_count++] = (struct agx_mdi_dyn_reloc){
         .dest = {rel->dest[0], rel->dest[1]},
         .delta = rel->delta,
         .offset = rel->offset,
         .short_mask = rel->short_mask,
         .is_draw_id = (draw_id_mask & BITFIELD_BIT(rel->range)) != 0,
      };
   }
   if (!dyn_count)
      return false;

   /* Bake all static relocations into the template once. */
   memcpy(tpl_bytes, plan->program.program.bytes, len);
   hk_g17p_patch_plan_apply(&plan->plan, first_ranges, tpl_bytes);
   *tpl_len_out = len;
   *dyn_reloc_count_out = dyn_count;
   return true;
}

/* Indirect multi-draws (glMultiDraw*Indirect) share every bound state across
 * sub-draws, varying only the indirect argument address and, when consumed,
 * gl_DrawID and base vertex/instance parameters. After the first full draw
 * validates state and latches the HAL200 bind tail, emit subsequent sub-draws
 * directly: body-only when vertex parameters do not vary, or a batched VS Ioto
 * template patch + pre-packed 28-byte VDM_STATE_UPDATE without a bind tail
 * (re-latching only after a 64 KiB VDM STREAM_LINK). */
static void
agx_draw_multi_indirect(struct pipe_context *pctx,
                        const struct pipe_draw_info *info,
                        const struct pipe_draw_indirect_info *indirect,
                        const struct pipe_draw_start_count_bias *draws)
{
   struct agx_context *ctx = agx_context(pctx);
   struct agx_device *dev = agx_device(pctx->screen);
   const enum agx_primitive primitive = agx_primitive_for_pipe(info->mode);
   const uint32_t arg_size_B = info->index_size ? 20u : 16u;
   const uint32_t params_offset_B = info->index_size ? 12u : 8u;
   const size_t command_space =
      AGX_VDM_STATE_UPDATE_LENGTH + AGX_VDM_HAL200_BIND_TAIL_LENGTH +
      AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH + AGX_VDM_BARRIER_LENGTH;

   /* Compute the exact prefix of sub-draws whose 64-bit byte offset fits both
    * pipe_draw_indirect_info::offset (uint32_t) and indirect->buffer->width0,
    * keeping the fast path and per-draw fallback on identical bounds. */
   const uint64_t buf_size = indirect->buffer->width0;
   const uint64_t base_offset = indirect->offset;
   const uint64_t indirect_stride = indirect->stride;
   if ((base_offset & 3u) || (indirect_stride & 3u) ||
       base_offset > buf_size || arg_size_B > buf_size - base_offset)
      return;

   const uint64_t max_delta =
      MIN2((uint64_t)UINT32_MAX - base_offset,
           buf_size - arg_size_B - base_offset);
   const unsigned valid_draw_count =
      indirect_stride
         ? (unsigned)MIN2((uint64_t)indirect->draw_count,
                          (max_delta / indirect_stride) + 1)
         : indirect->draw_count;
   uint32_t last_sub_offset = 0;
   if (!valid_draw_count ||
       !agx_mdi_subdraw_offset(indirect, valid_draw_count - 1, arg_size_B,
                               &last_sub_offset))
      return;

   struct agx_batch *clean = NULL;
   int can_skip = -1;
   uint64_t indirect_rsrc_base = 0;
   uint64_t ib_base = 0;
   size_t ib_size = 0;

   bool uses_draw_id = false;
   bool uses_base_param = false;
   bool needs_vs_update = false;
   struct agx3_ioto_uniform_range ranges[AGX3_G17P_GRAPHICS_MAX_RANGES];
   size_t range_count = 0;
   struct agx_g17p_uniform_range_locs locs = {
      .draw_id_range = -1,
      .params_range = -1,
   };
   size_t vs_capacity = 0;
   struct agx3_g17p_graphics_program_desc vs_desc = {0};
   uint32_t vs_word0 = 0;
   uint64_t draw_id_gpu = 0;
   unsigned draw_id_start = 0;
   unsigned draw_id_end = 0;
   size_t draw_id_stride = 0;

   /* Pre-packed VDM templates and batched Ioto program state for the active
    * clean batch / draw_id chunk. */
   struct agx_vdm_state_update vs_state_tpl = {0};
   uint32_t vs_outputs_tpl[2] = {0, 0};
   bool vs_state_tpl_valid = false;
   uint8_t draw_body_tpl[AGX_VDM_HAL200_INDIRECT_INDEXED_DRAW_LENGTH];
   size_t draw_body_len = 0;
   const size_t draw_body_addr_off = info->index_size ? 16u : 4u;
   bool draw_body_tpl_valid = false;
   uint8_t ioto_tpl[512];
   uint16_t ioto_tpl_len = 0;
   struct agx_mdi_dyn_reloc dyn_relocs[HK_G17P_PATCH_MAX_RELOCS];
   unsigned dyn_reloc_count = 0;
   bool ioto_tpl_valid = false;
   bool batch_ioto_failed = false;
   uint8_t *prog_batch_cpu = NULL;
   uint64_t prog_batch_gpu = 0;
   size_t prog_batch_stride = 0;
   unsigned prog_batch_start = 0;
   unsigned prog_batch_end = 0;

   for (unsigned i = 0; i < valid_draw_count; ++i) {
      uint32_t sub_offset = 0;
      if (!agx_mdi_subdraw_offset(indirect, i, arg_size_B, &sub_offset))
         break;

      bool fast = clean != NULL;

      if (fast) {
         const uint64_t indirect_ptr = indirect_rsrc_base + sub_offset;
         agx_ensure_cmdbuf_has_space(clean, &clean->vdm, command_space);
         uint8_t *out = clean->vdm.current;
         uint32_t draw_state_offset = 0;

         if (needs_vs_update) {
            uint64_t draw_id_va = 0;
            uint64_t params_va = 0;
            if (uses_draw_id && locs.draw_id_range >= 0) {
               if (unlikely(i >= draw_id_end)) {
                  const size_t stride = ALIGN_POT(locs.draw_id_size, 4);
                  const unsigned max_chunk = MAX2(1u, (unsigned)(65536u / stride));
                  const unsigned chunk =
                     MIN2(valid_draw_count - i, max_chunk);
                  uint64_t params_start = 1, params_end = 0;
                  if (uses_base_param && locs.params_range >= 0) {
                     const uint64_t p_bytes =
                        (uint64_t)ranges[locs.params_range].size_halfs * 2;
                     params_start =
                        indirect_rsrc_base + base_offset + params_offset_B;
                     params_end =
                        indirect_rsrc_base + last_sub_offset + params_offset_B +
                        (p_bytes >= 4 ? p_bytes - 4 : 0);
                  }
                  struct agx_ptr ids = agx_mdi_alloc_draw_id_chunk(
                     &clean->pool, (size_t)chunk * stride, ranges, range_count,
                     locs.draw_id_range,
                     uses_base_param ? locs.params_range : -1, params_start,
                     params_end);
                  if (!ids.cpu) {
                     fast = false;
                  } else {
                     const uint32_t draw_id_rel =
                        offsetof(struct agx_draw_uniforms, draw_id) -
                        locs.draw_id_offset;
                     if (stride == sizeof(uint32_t) && draw_id_rel == 0) {
                        uint32_t *dst = ids.cpu;
                        for (unsigned k = 0; k < chunk; ++k)
                           dst[k] = i + k;
                     } else {
                        const uint8_t *src =
                           (const uint8_t *)&clean->uniforms +
                           locs.draw_id_offset;
                        uint8_t *dst = ids.cpu;
                        for (unsigned k = 0; k < chunk; ++k, dst += stride) {
                           memcpy(dst, src, locs.draw_id_size);
                           uint32_t val = i + k;
                           memcpy(dst + draw_id_rel, &val, sizeof(val));
                        }
                     }
                     draw_id_gpu = ids.gpu;
                     draw_id_start = i;
                     draw_id_end = i + chunk;
                     draw_id_stride = stride;
                     ioto_tpl_valid = false;
                     batch_ioto_failed = false;
                     prog_batch_end = 0;
                  }
               }
               if (fast) {
                  draw_id_va =
                     draw_id_gpu + (uint64_t)(i - draw_id_start) * draw_id_stride;
                  clean->uniforms.draw_id = i;
                  ranges[locs.draw_id_range].address = draw_id_va;
               }
            }
            if (fast && uses_base_param && locs.params_range >= 0) {
               params_va = indirect_ptr + params_offset_B;
               clean->uniforms.tables[AGX_SYSVAL_TABLE_PARAMS] = params_va;
               ranges[locs.params_range].address = params_va;
            }
            if (fast && vs_state_tpl_valid && !batch_ioto_failed &&
                unlikely(i >= prog_batch_end)) {
               const unsigned sub_limit =
                  (uses_draw_id && locs.draw_id_range >= 0) ? draw_id_end
                                                            : valid_draw_count;
               uint32_t chunk_last_off = 0;
               if (!ioto_tpl_valid) {
                  if (sub_limit > i &&
                      agx_mdi_subdraw_offset(indirect, sub_limit - 1,
                                             arg_size_B, &chunk_last_off)) {
                     const uint64_t last_draw_id_va =
                        (uses_draw_id && locs.draw_id_range >= 0)
                           ? draw_id_gpu +
                                (uint64_t)(sub_limit - 1 - draw_id_start) *
                                   draw_id_stride
                           : 0;
                     const uint64_t last_params_va =
                        (uses_base_param && locs.params_range >= 0)
                           ? indirect_rsrc_base + chunk_last_off +
                                params_offset_B
                           : 0;
                     ioto_tpl_valid = agx_mdi_prepare_batch_ioto(
                        dev, ctx->linked.vs, &vs_desc, ranges, range_count,
                        &locs, uses_draw_id, uses_base_param, draw_id_va,
                        last_draw_id_va, params_va, last_params_va, ioto_tpl,
                        &ioto_tpl_len, dyn_relocs, &dyn_reloc_count);
                  }
                  if (!ioto_tpl_valid)
                     batch_ioto_failed = true;
               }
               if (ioto_tpl_valid && sub_limit > i) {
                  const size_t stride =
                     ALIGN_POT(ioto_tpl_len, AGX_IOTO_PROGRAM_ALIGNMENT);
                  const unsigned max_sub =
                     MAX2(1u, (unsigned)(65536u / stride));
                  const unsigned sub_chunk = MIN2(sub_limit - i, max_sub);
                  const size_t total_bytes = (size_t)sub_chunk * stride;
                  struct agx_bo *block_bo = NULL;
                  struct agx_ptr block = agx_pool_alloc_aligned_with_bo(
                     &clean->pipeline_pool, total_bytes,
                     AGX_IOTO_PROGRAM_ALIGNMENT, &block_bo);
                  if (block.cpu && block_bo &&
                      agx_va_in_usc_window(dev, block.gpu) &&
                      agx_va_in_usc_window(dev, block.gpu + total_bytes - 1)) {
                     agx_bo_assert_range(
                        block_bo, block.gpu - block_bo->va->addr, total_bytes);
                     prog_batch_cpu = block.cpu;
                     prog_batch_gpu = block.gpu;
                     prog_batch_stride = stride;
                     prog_batch_start = i;
                     prog_batch_end = i + sub_chunk;
                  } else {
                     batch_ioto_failed = true;
                  }
               }
            }
            if (fast && i < prog_batch_end && vs_state_tpl_valid) {
               for (unsigned r = 0; r < dyn_reloc_count; ++r) {
                  const struct agx_mdi_dyn_reloc *rel = &dyn_relocs[r];
                  const uint64_t base =
                     (rel->is_draw_id ? draw_id_va : params_va) + rel->delta;
                  uint8_t *patch = ioto_tpl + rel->offset;
                  const bool low_short = rel->short_mask & 1;
                  hk_g17p_patch_ldimm(patch, low_short, rel->dest[0],
                                      (uint32_t)base);
                  hk_g17p_patch_ldimm(patch + (low_short ? 2 : 8),
                                      rel->short_mask & 2, rel->dest[1],
                                      (uint32_t)(base >> 32));
               }
               const size_t slot_off =
                  (size_t)(i - prog_batch_start) * prog_batch_stride;
               memcpy(prog_batch_cpu + slot_off, ioto_tpl, ioto_tpl_len);
               const uint64_t prog_gpu = prog_batch_gpu + slot_off;

               struct agx_vdm_state_vertex_pipeline_pair pair;
               if (!agx_vdm_state_vertex_pipeline_pair_pack(
                      dev->chip, prog_gpu, 0, &pair)) {
                  fast = false;
               } else {
                  memcpy(&vs_state_tpl.bytes[8], pair.bytes,
                         sizeof(pair.bytes));
                  memcpy(out, vs_state_tpl.bytes, vs_state_tpl.length);
                  out += vs_state_tpl.length;

                  if (unlikely(!clean->g17p_bind_tail_latched ||
                               !clean->g17p_have_latched_vs)) {
                     out = agx_g17p_emit_bind_state(
                        clean, out, ctx->vs, &draw_state_offset);
                     if (!out) {
                        fast = false;
                     } else {
                        clean->g17p_bind_tail_latched = true;
                        clean->g17p_have_latched_vs = true;
                        clean->g17p_latched_vs_shader = ctx->vs;
                        clean->g17p_latched_vs_word0 = vs_word0;
                        clean->g17p_latched_vs_outputs[0] = vs_outputs_tpl[0];
                        clean->g17p_latched_vs_outputs[1] = vs_outputs_tpl[1];
                     }
                  }
                  if (fast) {
                     if (unlikely(dev->debug & AGX_DBG_PERF))
                        ctx->perf.vs_emits++;
                     clean->vs_root_valid = false;
                  }
               }
            } else if (fast) {
               struct agx_bo *program_bo = NULL;
               struct agx_ptr t = agx_pool_alloc_aligned_with_bo(
                  &clean->pipeline_pool, vs_capacity,
                  AGX_IOTO_PROGRAM_ALIGNMENT, &program_bo);
               size_t program_length = 0;
               if (!t.cpu || !program_bo || !agx_va_in_usc_window(dev, t.gpu)) {
                  fast = false;
               } else {
                  vs_desc.program_va = t.gpu;
                  if (!agx_g17p_graphics_program_cached(
                         dev->chip, ctx->linked.vs, &vs_desc, t.cpu,
                         vs_capacity, &program_length)) {
                     fast = false;
                  } else {
                     agx_bo_assert_range(program_bo,
                                         t.gpu - program_bo->va->addr,
                                         program_length);
                     uint32_t vs_outputs[2] = {0, 0};
                     out = agx_g17p_emit_vs_state_update(
                        clean, out, ctx->vs, vs_word0, t.gpu, vs_outputs);
                     if (!out) {
                        fast = false;
                     } else {
                        if (unlikely(!clean->g17p_bind_tail_latched ||
                                     !clean->g17p_have_latched_vs)) {
                           out = agx_g17p_emit_bind_state(
                              clean, out, ctx->vs, &draw_state_offset);
                           if (!out) {
                              fast = false;
                           } else {
                              clean->g17p_bind_tail_latched = true;
                              clean->g17p_have_latched_vs = true;
                              clean->g17p_latched_vs_shader = ctx->vs;
                              clean->g17p_latched_vs_word0 = vs_word0;
                              clean->g17p_latched_vs_outputs[0] = vs_outputs[0];
                              clean->g17p_latched_vs_outputs[1] = vs_outputs[1];
                           }
                        }
                        if (fast) {
                           if (unlikely(dev->debug & AGX_DBG_PERF))
                              ctx->perf.vs_emits++;
                           clean->vs_root_valid = false;
                        }
                     }
                  }
               }
            }
         }

         if (fast) {
            if (likely(draw_state_offset == 0 && draw_body_tpl_valid &&
                       indirect_ptr < (UINT64_C(1) << 48) &&
                       (indirect_ptr & 3u) == 0)) {
               const uint32_t addr_hi =
                  util_cpu_to_le32((uint32_t)((indirect_ptr >> 32) & 0xffffu));
               const uint32_t addr_lo =
                  util_cpu_to_le32((uint32_t)indirect_ptr);
               memcpy(draw_body_tpl + draw_body_addr_off, &addr_hi,
                      sizeof(addr_hi));
               memcpy(draw_body_tpl + draw_body_addr_off + 4, &addr_lo,
                      sizeof(addr_lo));
               memcpy(out, draw_body_tpl, draw_body_len);
               out += draw_body_len;
            } else {
               struct agx_draw draw = {0};
               if (info->index_size) {
                  draw.index_size = agx_translate_index_size(info->index_size);
                  draw.index_buffer = ib_base;
                  draw.index_buffer_range_B = ib_size;
                  draw.restart = info->primitive_restart;
                  draw.indexed = true;
               }
               draw.b = agx_grid_indirect(indirect_ptr);

               out = agx_g17p_emit_indirect_draw(
                  dev->chip, out, &draw, primitive, info->restart_index,
                  draw_state_offset);
            }
            clean->vdm.current = out;
            assert((clean->vdm.current + AGX_VDM_STREAM_LINK_LENGTH) <=
                      clean->vdm.end &&
                   "Failed to reserve sufficient space in encoder");

            clean->draws++;
            if (unlikely(dev->debug & AGX_DBG_PERF))
               ctx->perf.draws++;

            if (unlikely(
                   (((clean->scissor.size / AGX_SCISSOR_LENGTH) +
                     AGX_MAX_VIEWPORTS) > UINT16_MAX) ||
                   (clean->depth_bias.size / AGX_DEPTH_BIAS_LENGTH) >=
                      UINT16_MAX)) {
               agx_flush_batch_for_reason(ctx, clean,
                                          "Scissor/depth bias overflow");
            } else if (unlikely(clean->draws > 100000)) {
               agx_flush_batch_for_reason(ctx, clean,
                                          "Absurd number of draws");
            }
            if (ctx->batch != clean) {
               clean = NULL;
               draw_id_end = 0;
               prog_batch_end = 0;
               ioto_tpl_valid = false;
               batch_ioto_failed = false;
               vs_state_tpl_valid = false;
               draw_body_tpl_valid = false;
            }
            continue;
         }
         clean = NULL;
         draw_id_end = 0;
         prog_batch_end = 0;
         ioto_tpl_valid = false;
         batch_ioto_failed = false;
         vs_state_tpl_valid = false;
         draw_body_tpl_valid = false;
      }

      const struct pipe_draw_indirect_info subindirect = {
         .buffer = indirect->buffer,
         .count_from_stream_output = indirect->count_from_stream_output,
         .offset = sub_offset,
         .draw_count = 1,
      };
      struct agx_batch *before = ctx->batch;
      const unsigned before_draws = before ? before->draws : 0;

      agx_draw_vbo(pctx, info, i, &subindirect, draws, 1);

      clean = NULL;
      draw_id_end = 0;
      prog_batch_end = 0;
      ioto_tpl_valid = false;
      batch_ioto_failed = false;
      vs_state_tpl_valid = false;
      draw_body_tpl_valid = false;
      if (ctx->batch && !agx_batch_is_compute(ctx->batch) &&
          (ctx->batch != before || ctx->batch->draws != before_draws) &&
          agx_multidraw_state_clean(ctx) &&
          !(dev->debug & AGX_DBG_DIRTY) &&
          (!info->index_size ||
           (!info->has_user_indices && info->index.resource))) {
         if (can_skip < 0)
            can_skip = agx_multidraw_can_skip_common(ctx, info, true);
         if (can_skip)
            clean = ctx->batch;
      }

      if (clean) {
         agx_legalize_feedback_loops(ctx);
         if (ctx->batch != clean || !agx_multidraw_state_clean(ctx))
            clean = NULL;
      }

      if (clean) {
         struct agx_resource *indirect_rsrc = agx_resource(indirect->buffer);
         agx_batch_reads(clean, indirect_rsrc);
         agx_batch_note_vtx_read(clean, indirect_rsrc);
         indirect_rsrc_base = agx_map_gpu(indirect_rsrc);
         if (info->index_size)
            ib_base = agx_index_buffer_rsrc_ptr(clean, info, &ib_size);
         if (ctx->time_elapsed)
            agx_batch_add_timestamp_query(clean, ctx->time_elapsed);

         const struct agx_vdm_hal200_indirect_draw_fields body_fields = {
            .draw_state_context_offset = 4,
            .primitive = primitive,
            .indexed = info->index_size != 0,
            .index_size_B = info->index_size
                               ? agx_index_size_to_B(
                                    agx_translate_index_size(info->index_size))
                               : 0,
            .restart_enable = info->primitive_restart,
            .restart_index = info->restart_index,
            .index_buffer = ib_base,
            .index_buffer_range_B = ib_size,
            .indirect_buffer = indirect_rsrc_base + base_offset,
         };
         struct agx_vdm_hal200_indirect_draw packed_body;
         if (agx_vdm_hal200_indirect_draw_pack(dev->chip, &body_fields,
                                               &packed_body) &&
             packed_body.length > AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH &&
             agx_vdm_hal200_draw_emit(dev->chip, packed_body.bytes,
                                      packed_body.length, false, draw_body_tpl,
                                      sizeof(draw_body_tpl), &draw_body_len) &&
             draw_body_addr_off + 8 <= draw_body_len) {
            draw_body_tpl_valid = true;
         }

         uses_draw_id = ctx->vs->b.info.uses_draw_id;
         uses_base_param = ctx->linked.vs->uses_base_param;
         needs_vs_update = uses_draw_id || uses_base_param;
         if (needs_vs_update) {
            struct agx_compiled_shader *vs = ctx->vs;
            struct agx_linked_shader *linked = ctx->linked.vs;
            if (!linked || !linked->bo || !linked->bo->va || !linked->size ||
                vs->b.info.has_preamble || vs->b.info.preamble_scratch_size ||
                !agx_va_in_usc_window(dev,
                                      linked->bo->va->addr +
                                         linked->entry_offset) ||
                !agx_g17p_collect_graphics_uniform_ranges(
                   clean, vs, linked, ranges, &range_count, &locs) ||
                (uses_draw_id && locs.uses_indirect_root) ||
                !agx3_g17p_graphics_program_bound(dev->chip, ranges,
                                                  range_count, &vs_capacity)) {
               clean = NULL;
               can_skip = 0;
            } else if ((uses_draw_id && locs.draw_id_range < 0) &&
                       (!uses_base_param || locs.params_range < 0)) {
               needs_vs_update = false;
            } else {
               struct agx_vdm_state_vertex_shader_word_0_packed counts;
               agx_pack(&counts, VDM_STATE_VERTEX_SHADER_WORD_0, cfg) {
                  cfg.uniform_register_count = vs->b.info.push_count;
                  cfg.preshader_register_count = vs->b.info.nr_preamble_gprs;
                  cfg.texture_state_register_count =
                     agx_nr_tex_descriptors(clean, vs);
                  cfg.sampler_state_register_count =
                     translate_sampler_state_count(ctx, vs->stage);
               }
               vs_word0 = counts.opaque[0];
               const uint32_t provoking =
                  ctx->rast->base.flatshade_first ? 0 : 2;
               vs_outputs_tpl[0] =
                  vs->uvs.vdm.opaque[0] +
                  (clean->generate_primitive_id ? UINT32_C(0x100) : 0);
               vs_outputs_tpl[1] =
                  (ctx->rast->base.rasterizer_discard ? 0x30 + provoking
                                                      : provoking) |
                  (clean->generate_primitive_id ? UINT32_C(0x40) : 0);
               const struct agx_vdm_state_update_info vs_info = {
                  .raw_vertex_shader_word0 = vs_word0,
                  .pipeline = AGX_IOTO_PROGRAM_ALIGNMENT,
                  .raw_pipeline_template = 0,
                  .raw_vertex_outputs = {vs_outputs_tpl[0], vs_outputs_tpl[1]},
                  .raw_vertex_unknown = 0,
               };
               vs_state_tpl_valid =
                  agx_vdm_state_update_pack(dev->chip, &vs_info, &vs_state_tpl);
               vs_desc = (struct agx3_g17p_graphics_program_desc){
                  .shader_va = linked->bo->va->addr + linked->entry_offset,
                  .uniform_ranges = ranges,
                  .uniform_range_count = range_count,
                  .shader_binary_size = linked->size,
                  .scratch_size = MAX2(linked->scratch_size,
                                       linked->agx3_spectpr_scratch_size),
                  .nr_gprs = linked->nr_gprs,
                  .compiler_d_valid = linked->g17p_compiler_d_valid,
                  .compiler_d = linked->g17p_compiler_d,
                  .fragment = false,
                  .writes_sample_mask = false,
                  .colour_classes = linked->g17p_colour_classes,
                  .imageblock = {
                     .sample_size_B = clean->tilebuffer_layout.sample_size_B,
                     .nr_samples = clean->tilebuffer_layout.nr_samples,
                     .tile_size_px = clean->tilebuffer_layout.tile_size,
                  },
               };
            }
         }
      }
   }
}

static void
agx_try_elide_color_reload(struct agx_context *ctx, struct agx_batch *batch,
                           const struct pipe_draw_info *info,
                           const struct pipe_draw_indirect_info *indirect,
                           const struct pipe_draw_start_count_bias *draws)
{
   if ((batch->load & ~batch->clear & PIPE_CLEAR_COLOR) == 0)
      return;

   if (indirect || info->index_size || info->mode != MESA_PRIM_TRIANGLES ||
       draws->start != 0 || draws->count != 3 || info->instance_count < 1)
      return;

   if (ctx->gs || ctx->in_tess || ctx->cond_query || ctx->zs_merge ||
       ctx->g17g_mrt_split || !ctx->rast || !ctx->blend || !ctx->zs)
      return;

   if (util_framebuffer_get_num_layers(&batch->key) != 1 ||
       batch->key.width == 0 || batch->key.height == 0)
      return;

   const struct pipe_rasterizer_state *rs = &ctx->rast->base;
   if (rs->rasterizer_discard || rs->poly_stipple_enable ||
       rs->fill_front != PIPE_POLYGON_MODE_FILL ||
       rs->fill_back != PIPE_POLYGON_MODE_FILL ||
       rs->clip_plane_enable != 0 ||
       rs->cull_face == PIPE_FACE_FRONT_AND_BACK)
      return;

   const struct agx_uncompiled_shader *vs =
      ctx->stage[MESA_SHADER_VERTEX].shader;
   const struct agx_uncompiled_shader *fs =
      ctx->stage[MESA_SHADER_FRAGMENT].shader;
   if (!vs || !fs || vs->info.fullscreen_tri_winding == 0 ||
       vs->info.writes_clip_distance || vs->info.cull_distance_size != 0 ||
       vs->info.writes_layer || vs->info.writes_viewport)
      return;

   if (rs->depth_clip_near || rs->depth_clip_far) {
      const bool z_ok = rs->clip_halfz ? vs->info.fullscreen_tri_z_in_01
                                       : vs->info.fullscreen_tri_z_in_m11;
      if (!z_ok)
         return;
   }

   const struct pipe_viewport_state *vp = &ctx->viewport[0];
   if (!isfinite(vp->scale[0]) || !isfinite(vp->scale[1]) ||
       !isfinite(vp->translate[0]) || !isfinite(vp->translate[1]) ||
       vp->scale[0] == 0.0f || vp->scale[1] == 0.0f)
      return;

   const float abs_sx = fabsf(vp->scale[0]);
   const float abs_sy = fabsf(vp->scale[1]);
   if (vp->translate[0] - abs_sx > 0.0f ||
       vp->translate[0] + abs_sx < (float)batch->key.width ||
       vp->translate[1] - abs_sy > 0.0f ||
       vp->translate[1] + abs_sy < (float)batch->key.height)
      return;

   if (rs->scissor) {
      const struct pipe_scissor_state *ss = &ctx->scissor[0];
      if (ss->minx > 0 || ss->miny > 0 || ss->maxx < batch->key.width ||
          ss->maxy < batch->key.height)
         return;
   }

   if (rs->cull_face != PIPE_FACE_NONE) {
      const bool ndc_ccw = (vs->info.fullscreen_tri_winding == 1);
      const bool flip_xy = (vp->scale[0] * vp->scale[1]) < 0.0f;
      /* Window coordinates are Y-down, so cross_win < 0 is CCW. */
      const bool tri_ccw = (ndc_ccw == flip_xy);
      const bool is_front = (tri_ccw == agx_front_face_ccw(ctx));
      if ((is_front && (rs->cull_face & PIPE_FACE_FRONT)) ||
          (!is_front && (rs->cull_face & PIPE_FACE_BACK)))
         return;
   }

   const unsigned nr_samples = util_framebuffer_get_num_samples(&batch->key);
   if ((ctx->sample_mask & BITFIELD_MASK(nr_samples)) !=
       BITFIELD_MASK(nr_samples))
      return;

   if ((ctx->zs->base.depth_enabled &&
        ctx->zs->base.depth_func != PIPE_FUNC_ALWAYS) ||
       ctx->zs->base.stencil[0].enabled || ctx->zs->base.alpha_enabled)
      return;

   if (fs->info.fs_uses_discard || fs->info.uses_fbfetch ||
       fs->info.writes_api_sample_mask)
      return;

   uint32_t elide = 0;
   for (unsigned rt = 0; rt < batch->key.nr_cbufs; ++rt) {
      const struct pipe_surface *surf = &batch->key.cbufs[rt];
      if (!surf->texture)
         continue;
      const uint8_t fmt_mask =
         util_format_colormask(util_format_description(surf->format)) &
         PIPE_MASK_RGBA;
      if (!fmt_mask)
         continue;
      if ((fs->info.fs_unconditional_color_masks[rt] & fmt_mask) == fmt_mask &&
          (ctx->blend->opaque_overwrite_mask[rt] & fmt_mask) == fmt_mask)
         elide |= (PIPE_CLEAR_COLOR0 << rt);
   }

   batch->load &= ~elide;
   if (elide & PIPE_CLEAR_COLOR0)
      batch->g17p_resolve_reload = false;
}

static void
agx_draw_vbo(struct pipe_context *pctx, const struct pipe_draw_info *info,
             unsigned drawid_offset,
             const struct pipe_draw_indirect_info *indirect,
             const struct pipe_draw_start_count_bias *draws, unsigned num_draws)
{
   struct agx_context *ctx = agx_context(pctx);

   if (unlikely(!agx_render_condition_check(ctx)))
      return;

   /* Finish compiling any graphics shader that was bound before its
    * asynchronous compile completed (agx_bind_shader_state()). Every other
    * bound shader is already ready, so most draws skip this.
    */
   const uint32_t compiling = ctx->stages_compiling;
   if (unlikely(compiling)) {
      u_foreach_bit(s, compiling)
         agx_shader_wait_ready(ctx->stage[s].shader);
      ctx->stages_compiling = 0;
   }

   if (num_draws > 1) {
      agx_draw_multi(pctx, info, drawid_offset, indirect, draws, num_draws);
      return;
   }

   if (indirect && indirect->draw_count > 1 && !indirect->indirect_draw_count) {
      assert(drawid_offset == 0);
      assert(num_draws == 1);

      if (agx_chip_native_agx3(agx_device(pctx->screen)->chip) &&
          !indirect->count_from_stream_output && indirect->buffer) {
         agx_draw_multi_indirect(pctx, info, indirect, draws);
         return;
      }

      util_draw_multi_unroll_indirect(pctx, info, indirect, draws);
      return;
   }

   if (indirect && indirect->count_from_stream_output) {
      agx_draw_vbo_from_xfb(pctx, info, drawid_offset, indirect);
      return;
   }

   /* DrawAuto can resolve to an empty direct draw after reading its stream
    * output count. Empty draws have no shader or query side effects, and the
    * G17P command encoder requires nonzero vertex and instance counts. Keep
    * indirect draws intact: their counts are supplied by the GPU buffer.
    */
   if (!indirect && (!draws->count || !info->instance_count))
      return;

   /* G17P's index fetch starts at a dword boundary even for U8/U16
    * indices. Qt's image strips commonly start at the second U16 in an
    * element buffer, which otherwise fetches the preceding restart index
    * and loses one triangle. Realign before acquiring a batch: mapping with
    * PIPE_MAP_READ must synchronize any GPU writer of the original indices.
    * Aligned draws, indirect draws, and uploaded user indices keep their
    * existing paths.
    */
   const uint64_t index_offset = indirect ? 0 :
      (uint64_t)draws->start * info->index_size;
   if (!indirect && info->index_size && !info->has_user_indices &&
       agx_chip_native_agx3(agx_device(pctx->screen)->chip) &&
       (index_offset & 3)) {
      const uint64_t bytes = (uint64_t)draws->count * info->index_size;
      struct pipe_resource *source = info->index.resource;
      if (index_offset >= source->width0 ||
          bytes > source->width0 - index_offset)
         return;

      struct pipe_transfer *transfer = NULL;
      const void *indices = pipe_buffer_map_range(
         pctx, source, index_offset, bytes, PIPE_MAP_READ, &transfer);
      if (!indices)
         return;

      struct pipe_draw_info aligned = *info;
      aligned.index.resource = NULL;
      unsigned offset = 0;
      u_upload_data_ref(pctx->stream_uploader, 0, bytes, 64, indices,
                        &offset, &aligned.index.resource);
      pipe_buffer_unmap(pctx, transfer);
      if (aligned.index.resource) {
         struct pipe_draw_start_count_bias draw = *draws;
         draw.start = offset / info->index_size;
         agx_draw_vbo(pctx, &aligned, drawid_offset, NULL, &draw, 1);
         pipe_resource_reference(&aligned.index.resource, NULL);
      }
      return;
   }

   /* TODO: stop cheating */
   if (indirect && indirect->indirect_draw_count) {
      perf_debug_ctx(ctx, "multi-draw indirect");
      util_draw_indirect(pctx, info, drawid_offset, indirect);
      return;
   }

   /* No J873 capture holds an indirect Index List (test-ppp-hal300-g18g.c),
    * so agx_vdm_draw() has no G18G encoding for one. Read the arguments of
    * an API draw on the CPU, as the paths above and draw-auto do, until one
    * is witnessed. The tessellation draw's arguments live in a batch-private
    * buffer; it is refused below instead. */
   if (indirect && !ctx->in_tess &&
       agx_device(pctx->screen)->chip == AGX_CHIP_G18G &&
       !agx_vdm_index_list_indexed_or_indirect_supported_inline(
          agx_device(pctx->screen)->chip)) {
      perf_debug_ctx(ctx, "indirect draw without an indirect Index List");
      util_draw_indirect(pctx, info, drawid_offset, indirect);
      return;
   }

   /* HAL300's grounded index-list encodings consume U16 or U32 indices.
    * Preserve U8 API draws by widening through Gallium's ordinary index
    * utility before command construction. Vertex values, index bias and
    * the restart index remain unchanged; only the storage width changes. */
   if (info->index_size == 1 &&
       agx_chip_hal300_path(agx_device(pctx->screen)->chip) &&
       agx_vdm_index_list_direct_is_hal300_inline(agx_device(pctx->screen)->chip)) {
      if (indirect) {
         util_draw_indirect(pctx, info, drawid_offset, indirect);
         return;
      }

      if (!draws->count || !info->instance_count)
         return;
      if (draws->count > UINT32_MAX / sizeof(uint16_t)) {
         mesa_loge("asahi: U8 index expansion exceeds buffer size limit");
         return;
      }

      struct pipe_resource *indices = NULL;
      unsigned offset = 0;
      void *map = NULL;
      u_upload_alloc_ref(pctx->stream_uploader, 0,
                         draws->count * sizeof(uint16_t), 64,
                         &offset, &indices, &map);
      if (!indices || !map) {
         pipe_resource_reference(&indices, NULL);
         return;
      }
      util_shorten_ubyte_elts_to_userptr(pctx, info, 0, 0,
                                        draws->start, draws->count, map);
      u_upload_unmap(pctx->stream_uploader);

      struct pipe_draw_info widened = *info;
      struct pipe_draw_start_count_bias draw = *draws;
      widened.index_size = sizeof(uint16_t);
      widened.has_user_indices = false;
      widened.index.resource = indices;
      draw.start = offset / sizeof(uint16_t);
      agx_draw_vbo(pctx, &widened, drawid_offset, NULL, &draw, 1);
      pipe_resource_reference(&indices, NULL);
      return;
   }

   /* TODO: stop cheating.
    *
    * libagx supports this, just needs test coverage and gallium side wiring.
    * M5 reads every indirect patch draw back: its tessellation prepass
    * launches direct grids only.
    */
   if (indirect && info->mode == MESA_PRIM_PATCHES &&
       (info->index_size || agx_m5_gpu_tess(ctx))) {
      perf_debug_ctx(ctx, "indexed indirect with tess");
      util_draw_indirect(pctx, info, drawid_offset, indirect);
      return;
   }

   /* The GPU writes a tessellated draw's primitive count, and one M5 render
    * holds AGX_G17G_RENDER_PRIMS: past that the submission times out. Draw
    * a patch draw that could exceed it in parts that cannot. Indirect patch
    * draws were read back above; drop anything that still cannot be split.
    */
   if (info->mode == MESA_PRIM_PATCHES && agx_m5_gpu_tess(ctx)) {
      const unsigned max_patches =
         AGX_G17G_RENDER_PRIMS / agx_tess_max_prims_per_patch(ctx);

      if (indirect || !max_patches) {
         static bool reported;
         if (!reported) {
            reported = true;
            mesa_loge("asahi: dropping a tessellated draw that could exceed "
                      "the M5 primitive budget of one render");
         }
         return;
      }

      const unsigned patches = draws->count / MAX2(ctx->patch_vertices, 1);
      if (patches > max_patches ||
          (patches && info->instance_count > max_patches / patches)) {
         agx_draw_patches_split(ctx, info, drawid_offset, draws, patches,
                                max_patches);
         return;
      }
   }

   if (agx_chip_g17_m5(agx_device(pctx->screen)->chip) &&
       !ctx->g17g_layer_split &&
       util_framebuffer_get_num_layers(&ctx->framebuffer) > 1) {
      agx_g17g_layered_draw(ctx, info, drawid_offset, indirect, draws,
                            num_draws);
      return;
   }

   /* G17G runs geometry and tessellation stages on the CPU. */
   if (!indirect && agx_swtnl_applies(ctx)) {
      agx_swtnl_draw(ctx, info, draws);
      return;
   }

   /* G17G captures transform feedback on the CPU and then draws without the
    * passthrough geometry stage, whose rasterization it cannot encode. */
   bool xfb_on_cpu = ctx->g17g_mrt_split && ctx->g17g_mrt_pass;
   if (ctx->streamout.num_targets &&
       agx_last_uncompiled_vgt(ctx)->has_xfb_info &&
       agx_xfb_cpu_applies(ctx, indirect) && !xfb_on_cpu) {
      agx_xfb_cpu_draw(ctx, info, draws);
      xfb_on_cpu = true;
   }

   /* With rasterization discarded a G17G draw has nothing left for the GPU
    * (the draw profile refuses it, and a render of refused draws is dropped
    * with its clears). Only the primitives-generated count remains. */
   if (agx_chip_g17_m5(agx_device(pctx->screen)->chip) &&
       ctx->rast->base.rasterizer_discard && !indirect &&
       !ctx->stage[MESA_SHADER_GEOMETRY].shader &&
       !ctx->stage[MESA_SHADER_TESS_EVAL].shader &&
       (xfb_on_cpu || !ctx->streamout.num_targets ||
        !agx_last_uncompiled_vgt(ctx)->has_xfb_info)) {
      if (ctx->active_queries && ctx->prims_generated[0])
         agx_primitives_update_direct(ctx, info, draws);
      return;
   }

   if (agx_chip_g17_m5(agx_device(pctx->screen)->chip) &&
       !ctx->g17g_mrt_split && !ctx->g17g_mrt_spill &&
       ctx->framebuffer.nr_cbufs > 1 && !agx_g17g_mrt_resident(ctx)) {
      if (agx_g17g_mrt_one_pass(ctx))
         agx_g17g_one_pass_mrt_draw(ctx, info, drawid_offset, indirect, draws);
      else
         agx_g17g_split_mrt_draw(ctx, info, drawid_offset, indirect, draws);
      return;
   }

   bool xfb_passthrough = false;
   if (agx_needs_passthrough_gs(ctx, info, indirect, &xfb_passthrough,
                                xfb_on_cpu)) {
      agx_apply_passthrough_gs(ctx, info, drawid_offset, indirect, draws,
                               num_draws, xfb_passthrough);
      return;
   }

   /* We must legalize feedback loops and transform feedback writes before
    * getting the batch, since once we have the batch we're not allowed to flush
    * the bound render targets.
    */
   agx_check_zs_merge(ctx);
   agx_legalize_feedback_loops(ctx);
   agx_legalize_xfb(ctx);

   /* A G17G render holds a bounded number of primitives: two million
    * tessellated triangles in one render timed out and left the GPU dead
    * (tessellation.invariance). Start a new render past the budget.
    *
    * The small budget is for the output of a tessellation stage, where the
    * timeouts were seen, on the GPU or on the CPU (agx_swtnl.c). A render of
    * plain draws takes AGX_G17G_RENDER_PRIMS_PLAIN: every render started for
    * the budget loads and stores the whole framebuffer again. */
   const bool g17g_budget =
      agx_chip_g17_m5(agx_device(pctx->screen)->chip) && !indirect;
   const bool g17g_small = info->mode == MESA_PRIM_PATCHES || ctx->swtnl_inner;
   /* A tessellated draw's output count is unknown here: give it a render of
    * its own. */
   const uint64_t prims =
      !g17g_budget ? 0
      : info->mode == MESA_PRIM_PATCHES
         ? AGX_G17G_RENDER_PRIMS
         : (uint64_t)u_decomposed_prims_for_vertices(info->mode,
                                                     draws->count) *
              info->instance_count;
   if (g17g_budget && ctx->batch && ctx->batch->g17g_prims &&
       ctx->batch->g17g_prims + prims >
          (g17g_small || ctx->batch->g17g_small_budget
              ? AGX_G17G_RENDER_PRIMS
              : AGX_G17G_RENDER_PRIMS_PLAIN))
      agx_flush_batch_for_reason(ctx, ctx->batch, "G17G primitive budget");

   struct agx_batch *batch = agx_get_batch(ctx);
   batch->g17g_prims += prims;
   batch->g17g_small_budget |= g17g_budget && g17g_small;
   uint64_t ib = 0;
   size_t ib_extent = 0;

   if (info->index_size) {
      ib =
         agx_index_buffer_ptr(batch, info, indirect ? NULL : draws, &ib_extent);
   }

   /* Increment IA statistics before lowering tessellation. This ensures we
    * count the patches instead of counting the tessellated outputs.
    */
   if (ctx->active_queries && !ctx->in_tess &&
       !ctx->active_draw_without_restart &&
       (ctx->pipeline_statistics[PIPE_STAT_QUERY_IA_VERTICES] ||
        ctx->pipeline_statistics[PIPE_STAT_QUERY_IA_PRIMITIVES] ||
        ctx->pipeline_statistics[PIPE_STAT_QUERY_VS_INVOCATIONS] ||
        ((ctx->pipeline_statistics[PIPE_STAT_QUERY_C_PRIMITIVES] ||
          ctx->pipeline_statistics[PIPE_STAT_QUERY_C_INVOCATIONS]) &&
         !ctx->stage[MESA_SHADER_TESS_EVAL].shader &&
         !ctx->stage[MESA_SHADER_GEOMETRY].shader))) {

      uint64_t ptr;
      if (indirect) {
         ptr = agx_indirect_buffer_ptr(batch, indirect);
      } else {
         uint32_t desc[] = {draws->count, info->instance_count, 0};
         ptr = agx_pool_upload(&batch->pool, &desc, sizeof(desc));
      }

      agx_ia_update(batch, info, ptr, ib,
                    info->index_size ? ib_extent / info->index_size : 1);
   }

   if (info->mode == MESA_PRIM_PATCHES) {
      /* Where the CPU path can tessellate, the M5 GPUs tessellate only
       * there (agx_m5_gpu_tess(), agx_swtnl_applies()): a patch draw that
       * path cannot take (indirect, or images before rasterization) has no
       * encoding here. */
      if (agx_chip_g17_m5(agx_device(ctx->base.screen)->chip) &&
          !agx_m5_gpu_tess(ctx)) {
         agx_native_descriptor_reject_batch(batch, "tessellation outside the CPU path");
         return;
      }
      agx_draw_patches(ctx, info, drawid_offset, indirect, draws, num_draws);
      return;
   }

   /* Only the rasterization stream counts. The CPU stages' own GPU draw of
    * their output was counted by agx_swtnl_draw already (its geometry
    * shader is unbound by then, so it would count twice). */
   if (ctx->active_queries && ctx->prims_generated[0] &&
       !ctx->stage[MESA_SHADER_GEOMETRY].shader && !ctx->swtnl_inner &&
       !ctx->in_tess && !(ctx->g17g_mrt_split && ctx->g17g_mrt_pass)) {

      assert(!indirect && "we force a passthrough GS for this");
      agx_primitives_update_direct(ctx, info, draws);
   }

   if (ctx->stage[MESA_SHADER_GEOMETRY].shader && info->primitive_restart &&
       info->index_size) {

      agx_draw_without_restart(batch, info, drawid_offset, indirect, draws);
      return;
   }

   agx_batch_add_timestamp_query(batch, ctx->time_elapsed);

#ifndef NDEBUG
   if (unlikely(agx_device(pctx->screen)->debug & AGX_DBG_DIRTY))
      agx_dirty_all(ctx);
#endif

   agx_batch_init_state(batch);

   /* Dirty track the reduced prim: lines vs points vs triangles. Happens before
    * agx_update_vs/agx_update_fs, which specialize based on primitive.
    */
   enum mesa_prim reduced_prim = u_reduced_prim(info->mode);
   if (reduced_prim != batch->reduced_prim)
      ctx->dirty |= AGX_DIRTY_PRIM;
   batch->reduced_prim = reduced_prim;

   /* Update shaders first so we can use them after */
   if (agx_update_vs(batch, info->index_size)) {
      ctx->dirty |= AGX_DIRTY_VS | AGX_DIRTY_VS_PROG;
      ctx->stage[MESA_SHADER_VERTEX].dirty = ~0;
   } else if (ctx->stage[MESA_SHADER_VERTEX].dirty ||
              (ctx->dirty & AGX_DIRTY_VERTEX))
      ctx->dirty |= AGX_DIRTY_VS;

   if (agx_drop_for_unusable_program(ctx, ctx->vs, ctx->linked.vs != NULL,
                                     "vertex"))
      return;

   /* This is subtle. But agx_update_vs will be true at least once per batch. */
   assert(agx_batch_uses_bo(batch, ctx->vs->bo));
   assert(!ctx->linked.vs || agx_batch_uses_bo(batch, ctx->linked.vs->bo));

   agx_update_gs(ctx, info, indirect);

   if (ctx->gs &&
       agx_drop_for_unusable_program(ctx, ctx->gs, true, "geometry"))
      return;

   if (ctx->gs) {
      batch->uniforms.geometry_params =
         agx_batch_geometry_params(batch, ib, ib_extent, info, draws, indirect);

      agx_batch_add_bo(batch, ctx->gs->bo);
      agx_batch_add_bo(batch, ctx->gs->gs_copy->bo);
   }

   agx_update_gs_copy_link(ctx, batch);

   if (ctx->dirty & (AGX_DIRTY_VS_PROG | AGX_DIRTY_FS_PROG)) {
      struct agx_compiled_shader *vs = ctx->vs;
      if (ctx->gs)
         vs = ctx->gs->gs_copy;

      agx_assign_uvs(
         &batch->linked_varyings, &vs->uvs,
         ctx->stage[MESA_SHADER_FRAGMENT].shader->info.inputs_flat_shaded,
         ctx->stage[MESA_SHADER_FRAGMENT].shader->info.inputs_linear_shaded);

      for (unsigned i = 0; i < VARYING_SLOT_MAX; ++i) {
         batch->uniforms.uvs_index[i] = batch->linked_varyings.slots[i];
      }
   }

   /* Set draw ID */
   if (ctx->vs->b.info.uses_draw_id) {
      batch->uniforms.draw_id = drawid_offset;

      ctx->dirty |= AGX_DIRTY_VS;
   }

   if (agx_update_fs(batch)) {
      ctx->dirty |= AGX_DIRTY_FS | AGX_DIRTY_FS_PROG;
      ctx->stage[MESA_SHADER_FRAGMENT].dirty = ~0;
   } else if ((ctx->stage[MESA_SHADER_FRAGMENT].dirty) ||
              (ctx->dirty & (AGX_DIRTY_BLEND_COLOR | AGX_DIRTY_SAMPLE_MASK))) {
      ctx->dirty |= AGX_DIRTY_FS;
   }

   if (agx_drop_for_unusable_program(ctx, ctx->fs, ctx->linked.fs != NULL,
                                     "fragment"))
      return;

   /* This is subtle. But agx_update_fs will be true at least once per batch. */
   assert(!ctx->fs->bo || agx_batch_uses_bo(batch, ctx->fs->bo));
   assert(agx_batch_uses_bo(batch, ctx->linked.fs->bo));

   if (ctx->linked.vs->uses_base_param || ctx->gs) {
      agx_upload_draw_params(batch, indirect, draws, info);

      batch->uniforms.is_indexed_draw = (info->index_size > 0);
      ctx->dirty |= AGX_DIRTY_VS;
   }

   agx_update_descriptors(batch, ctx->vs);
   agx_update_descriptors(batch, ctx->gs);
   agx_update_descriptors(batch, ctx->fs);
   if (batch->native_rejected)
      return;

   if (ctx->gs || ctx->in_tess || ctx->streamout.num_targets ||
       ctx->cond_query ||
       (ctx->vs && ctx->vs->so && ctx->vs->so->info.writes_memory))
      batch->vtx_needs_render_barrier = true;

   if (IS_DIRTY(VS) || IS_DIRTY(FS) || ctx->gs || IS_DIRTY(VERTEX) ||
       IS_DIRTY(BLEND_COLOR) || IS_DIRTY(QUERY) || IS_DIRTY(POLY_STIPPLE) ||
       IS_DIRTY(RS) || IS_DIRTY(PRIM) || ctx->in_tess) {

      if (IS_DIRTY(VERTEX)) {
         agx_upload_vbos(batch);
      }

      if (IS_DIRTY(BLEND_COLOR)) {
         memcpy(batch->uniforms.blend_constant, &ctx->blend_color,
                sizeof(ctx->blend_color));
      }

      if (IS_DIRTY(RS)) {
         struct pipe_rasterizer_state *rs = &ctx->rast->base;

         batch->uniforms.fixed_point_size =
            rs->point_size_per_vertex ? 0.0 : rs->point_size;

         /* TODO: tri fans */
         batch->uniforms.provoking_vertex = !rs->flatshade_first ? 2 : 0;
      }

      batch->uniforms.layer_id = ctx->g17g_layer;

      if (IS_DIRTY(QUERY)) {
         for (unsigned i = 0; i < ARRAY_SIZE(ctx->pipeline_statistics); ++i) {
            struct agx_query *query = ctx->pipeline_statistics[i];
            batch->uniforms.pipeline_statistics[i] =
               agx_get_query_address(batch, query);
         }
      }

      if (IS_DIRTY(POLY_STIPPLE)) {
         STATIC_ASSERT(sizeof(ctx->poly_stipple) == 32 * 4);

         batch->uniforms.polygon_stipple = agx_pool_upload_aligned(
            &batch->pool, ctx->poly_stipple, sizeof(ctx->poly_stipple), 4);
      }

      agx_upload_uniforms(batch, ctx->gs ? ctx->gs->gs_copy : ctx->vs);
   }

   struct pipe_draw_info info_gs;
   struct pipe_draw_indirect_info indirect_gs;
   struct pipe_draw_start_count_bias draw_gs;

   /* Wrap the pool allocation in a fake resource for meta-Gallium use.
    * Defer initialization until ctx->gs is active to avoid ~2 KiB of stack
    * zeroing on every non-GS draw. */
   struct agx_resource indirect_rsrc;
   struct agx_resource index_rsrc;

   if (unlikely(ctx->gs)) {
      indirect_rsrc = (struct agx_resource){.bo = batch->geom_params_bo};
      index_rsrc = (struct agx_resource){0};
      /* Launch the pre-rasterization parts of the geometry shader */
      agx_launch_gs_prerast(batch, info, draws, indirect);

      /* Setup to rasterize the GS results */
      struct poly_gs_info *gsi = &ctx->gs->gs;
      info_gs = (struct pipe_draw_info){
         .mode = gsi->mode,
         .index_size = poly_gs_index_size(gsi->shape),
         .primitive_restart = poly_gs_indexed(gsi->shape),
         .restart_index = poly_gs_index_size(gsi->shape) == 1 ? 0xFF : ~0,
         .index.resource = &index_rsrc.base.b,
         .instance_count = 1,
      };

      if (indirect) {
         indirect_gs = (struct pipe_draw_indirect_info){
            .draw_count = 1,
            .buffer = &indirect_rsrc.base.b,
            .offset = batch->uniforms.geometry_params -
                      indirect_rsrc.bo->va->addr +
                      offsetof(struct poly_geometry_params, draw),
         };

         indirect = &indirect_gs;

         batch->geom_index_bo = agx_resource(batch->ctx->heap)->bo;
         batch->geom_index = batch->geom_index_bo->va->addr;
      } else {
         unsigned prims =
            u_decomposed_prims_for_vertices(info->mode, draws->count);

         draw_gs = (struct pipe_draw_start_count_bias){
            .count = poly_gs_rast_vertices(gsi->shape, gsi->max_indices, prims,
                                           info->instance_count),
         };

         info_gs.instance_count = poly_gs_rast_instances(
            gsi->shape, gsi->max_indices, prims, info->instance_count);

         draws = &draw_gs;
      }

      info = &info_gs;
      index_rsrc.bo = batch->geom_index_bo;

      /* TODO: Deduplicate? */
      batch->reduced_prim = u_reduced_prim(info->mode);
      ctx->dirty |= AGX_DIRTY_PRIM;

      if (gsi->shape == POLY_GS_SHAPE_DYNAMIC_INDEXED) {
         ib = batch->geom_index;
         ib_extent = index_rsrc.bo->size - (batch->geom_index - ib);
      } else if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED &&
                 agx_chip_hal300_path(agx_device(ctx->base.screen)->chip) &&
                 agx_vdm_index_list_direct_is_hal300_inline(
                    agx_device(ctx->base.screen)->chip)) {
         /* HAL300 index lists take U16 or U32 indices only, as for the U8
          * API draws widened above; widen the static topology the same way,
          * its 0xff restarts included. */
         uint16_t wide[ARRAY_SIZE(gsi->topology)];
         for (unsigned i = 0; i < gsi->max_indices; ++i)
            wide[i] = gsi->topology[i] == 0xff ? 0xffff : gsi->topology[i];
         ib = agx_pool_upload(&batch->pool, wide, gsi->max_indices * 2);
         ib_extent = gsi->max_indices * 2;
         info_gs.index_size = 2;
         info_gs.restart_index = 0xffff;
      } else if (gsi->shape == POLY_GS_SHAPE_STATIC_INDEXED) {
         ib = agx_pool_upload(&batch->pool, gsi->topology, gsi->max_indices);
         ib_extent = gsi->max_indices;
      }

      /* We need to reemit geometry descriptors since the txf sampler may change
       * between the GS prepass and the GS rast program.
       */
      agx_update_descriptors(batch, ctx->gs->gs_copy);
   }

   /* The geometry-shader and tessellation rasterization draws are indirect
    * by construction and cannot take the CPU path above. Without an indirect
    * Index List fail them closed; the prepass work stays in the batch. */
   if (indirect && agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G &&
       !agx_vdm_index_list_indexed_or_indirect_supported_inline(
          agx_device(ctx->base.screen)->chip)) {
      mesa_loge("asahi: dropping an internal indirect draw: %s has no "
                "witnessed indirect Index List",
                agx_chip_name(agx_device(ctx->base.screen)->chip));
      return;
   }

   /* Nothing left to rasterize -- a geometry or transform feedback pass with
    * an empty rasterized output, or too few vertices for one primitive. The
    * pre-rasterization work is already launched above; a native HAL200
    * direct draw cannot encode a zero count, and would draw nothing. */
   if (agx_chip_native_agx3(agx_device(ctx->base.screen)->chip) && !indirect &&
       (!draws->count || !info->instance_count))
      return;

   assert((!indirect || !indirect->indirect_draw_count) && "multidraw handled");

   /* Update batch masks based on current state */
   const bool first_draw = (batch->draws == 0 && batch->draw == 0);
   if (ctx->dirty & AGX_DIRTY_BLEND) {
      uint32_t store = ctx->blend->store;
      if (ctx->g17g_mrt_split) {
         store = (store & (PIPE_CLEAR_COLOR0 << ctx->g17g_mrt_rt))
                    ? PIPE_CLEAR_COLOR0
                    : 0;
      }

      /* TODO: Any point to tracking load? */
      if (!ctx->zs_merge) {
         batch->draw |= store;
         batch->resolve |= store;
      }
   }

   if (ctx->dirty & AGX_DIRTY_ZS) {
      batch->load |= ctx->zs->load;
      batch->draw |= ctx->zs->store;
      batch->resolve |= ctx->zs->store;
   }

   /* When we approach the end of a command buffer, cycle it out for a new one.
    * We only need to do this once per draw as long as we conservatively
    * estimate the maximum bytes of VDM commands that this draw will emit.
    */
   struct agx_device *dev = agx_device(ctx->base.screen);
   /* HAL300 emits a vertex state update and the viewport, bases and graphics
    * PPP records before its Index List. The AGX2 estimate is smaller than
    * that state block alone; overread padding is not command storage.
    */
   const size_t command_space =
      agx_chip_hal300_path(dev->chip)
         ? AGX_VDM_STATE_UPDATE_LENGTH +
              (2 + AGX_PPP_HAL300_GRAPHICS_RECORDS) * AGX_VDM_PPP_STATE_CORE_LENGTH +
              AGX_VDM_STATE_LENGTH + AGX_VDM_STATE_RESTART_INDEX_LENGTH +
              MAX2(AGX_VDM_INDEX_LIST_DIRECT_INDEXED_MAX_LENGTH,
                   AGX_VDM_INDEX_LIST_INDIRECT_INDEXED_MAX_LENGTH) +
              AGX_VDM_BARRIER_CORE_LENGTH
      : agx_chip_native_agx3(dev->chip)
         ? AGX_VDM_STATE_UPDATE_LENGTH + AGX_VDM_HAL200_BIND_TAIL_LENGTH +
              AGX_VDM_HAL200_DRAW_STATE_POINTER_LENGTH +
              AGX_VDM_HAL200_INDEXED_DRAW_BODY_LENGTH + AGX_VDM_BARRIER_LENGTH
         : (AGX_VDM_STATE_LENGTH * 2) +
              (AGX_PPP_STATE_LENGTH * MAX_PPP_UPDATES) +
              AGX_VDM_STATE_RESTART_INDEX_LENGTH +
              AGX_VDM_STATE_VERTEX_SHADER_WORD_0_LENGTH +
              AGX_VDM_STATE_VERTEX_SHADER_WORD_1_LENGTH +
              AGX_VDM_STATE_VERTEX_OUTPUTS_LENGTH +
              AGX_VDM_STATE_VERTEX_UNKNOWN_LENGTH + 4 /* padding */ +
              AGX_INDEX_LIST_LENGTH + AGX_INDEX_LIST_BUFFER_LO_LENGTH +
              AGX_INDEX_LIST_COUNT_LENGTH + AGX_INDEX_LIST_INSTANCES_LENGTH +
              AGX_INDEX_LIST_START_LENGTH + AGX_INDEX_LIST_BUFFER_SIZE_LENGTH;
   agx_ensure_cmdbuf_has_space(batch, &batch->vdm, command_space);

   uint32_t draw_state_offset = 0;
   const uint64_t skipped_before = ctx->skipped_no_program;
   uint8_t *out = agx_encode_state(batch, batch->vdm.current,
                                   &draw_state_offset);
   if (!out) {
      if (dev->chip == AGX_CHIP_G18G) {
         if (ctx->skipped_no_program == skipped_before)
            ctx->skipped_no_program++;
         mesa_loge("asahi: dropping work: G18 draw state encoding refused");
      }
      return;
   }

   if (!agx_chip_native_agx3(dev->chip) && info->index_size &&
       info->primitive_restart) {
      agx_push(out, VDM_STATE, cfg)
         cfg.restart_index_present = true;

      agx_push(out, VDM_STATE_RESTART_INDEX, cfg)
         cfg.value = info->restart_index;
   }

   struct agx_draw draw = {0};
   if (info->index_size) {
      draw.index_size = agx_translate_index_size(info->index_size);
      draw.index_buffer = ib;
      draw.index_buffer_range_B = ib_extent;
      draw.restart = info->primitive_restart;
      draw.indexed = true;
   } else {
      draw.start = indirect ? 0 : draws->start;
   }

   if (indirect) {
      draw.b = agx_grid_indirect(agx_indirect_buffer_ptr(batch, indirect));
   } else {
      draw.b = agx_3d(draws->count, info->instance_count, 1);
      if (info->index_size)
         draw.index_bias = draws->index_bias;
   }

   const enum agx_primitive primitive = agx_primitive_for_pipe(info->mode);
   if (agx_chip_native_agx3(dev->chip)) {
      out = indirect ? agx_g17p_emit_indirect_draw(
                          dev->chip, out, &draw, primitive,
                          info->restart_index, draw_state_offset)
                     : agx_g17p_emit_draw_body(
                          dev->chip, out, &draw, primitive,
                          info->restart_index, draw_state_offset);
   } else {
      if (dev->chip == AGX_CHIP_G18G) {
         /* The HAL300 Index List word follows the bound hardware vertex
          * program: its UVS record word 4 is the output count, 4 + point
          * size + clip words + user varyings, which is uvs.size for every
          * layout agx_encode_g17g_state() admits. */
         const struct agx_compiled_shader *hw_vs =
            ctx->gs ? ctx->gs->gs_copy : ctx->vs;
         const uint32_t unk2 = agx_vdm_index_list_hal300_unk2(
            dev->chip, hw_vs ? hw_vs->uvs.size : 0);
         out = (void *)agx_vdm_draw_unk2(
            (uint32_t *)out, dev->chip, draw, primitive,
            draw.restart ? info->restart_index
                         : agx_primitive_restart_index(draw.index_size),
            unk2);
      } else {
         out = (void *)agx_vdm_draw(
            (uint32_t *)out, dev->chip, draw, primitive,
            draw.restart ? info->restart_index
                         : agx_primitive_restart_index(draw.index_size));
      }
   }

   /* Barrier transform feedback writes on themselves for consistency.
    * This is the other half of agx_legalize_xfb.
    */
   if (ctx->gs && ctx->streamout.num_targets > 0) {
      struct agx_device *dev = agx_device(ctx->base.screen);
      out = (void *)agx_vdm_barrier(
         (uint32_t *)out, dev->chip,
         dev->params.features & DRM_ASAHI_FEATURE_VDM_USC_INVALIDATE);
   }

   assert(!agx_chip_hal300_path(dev->chip) ||
          (size_t)(out - batch->vdm.current) <= command_space);
   batch->vdm.current = out;
   assert((batch->vdm.current + AGX_VDM_STREAM_LINK_LENGTH) <= batch->vdm.end &&
          "Failed to reserve sufficient space in encoder");
   agx_dirty_reset_graphics(ctx);

   assert(batch == agx_get_batch(ctx) && "batch should not change under us");

   if (first_draw)
      agx_try_elide_color_reload(ctx, batch, info, indirect, draws);

   batch->draws++;
   if (ctx->stage[MESA_SHADER_FRAGMENT].shader)
      batch->last_fs_source = ctx->stage[MESA_SHADER_FRAGMENT].shader->source_hash;

   if (unlikely(dev->debug & AGX_DBG_PERF))
      ctx->perf.draws++;

   /* The scissor/zbias arrays are indexed with 16-bit integers, imposigin a
    * maximum of UINT16_MAX descriptors. Flush if the next draw would overflow
    */
   if (unlikely(
          (((batch->scissor.size / AGX_SCISSOR_LENGTH) + AGX_MAX_VIEWPORTS) >
           UINT16_MAX) ||
          (batch->depth_bias.size / AGX_DEPTH_BIAS_LENGTH) >= UINT16_MAX)) {
      agx_flush_batch_for_reason(ctx, batch, "Scissor/depth bias overflow");
   } else if (unlikely(batch->draws > 100000)) {
      /* Mostly so drawoverhead doesn't OOM */
      agx_flush_batch_for_reason(ctx, batch, "Absurd number of draws");
   } else if (unlikely(batch->sampler_heap.count >
                       (AGX_SAMPLER_HEAP_SIZE - (PIPE_MAX_SAMPLERS * 6)))) {
      agx_flush_batch_for_reason(ctx, batch, "Sampler heap overflow");
   }
}

static void
agx_texture_barrier(struct pipe_context *pipe, unsigned flags)
{
   struct agx_context *ctx = agx_context(pipe);

   /* Framebuffer fetch is coherent, so barriers are a no-op. */
   if (flags == PIPE_TEXTURE_BARRIER_FRAMEBUFFER)
      return;

   agx_flush_all(ctx, "Texture barrier");
}

void
agx_launch(struct agx_batch *batch, struct agx_grid grid,
           struct agx_workgroup wg, struct agx_compiled_shader *cs,
           struct agx_linked_shader *linked, mesa_shader_stage stage,
           unsigned variable_shared_mem, uint64_t indirect_num_workgroups)
{
   struct agx_context *ctx = batch->ctx;

   if (agx_drop_for_unusable_program(ctx, cs, true,
                                     _mesa_shader_stage_to_abbrev(stage)))
      return;

   /* A G18 stop-only compute shader still has a valid native program.
    * Skipping it after batch setup leaves a command stream with no recorded
    * dispatch, which the submission guard must reject. Keep query handling
    * and the ordinary metadata/launch validation by taking the normal path. */
   const bool native_g18_compute =
      stage == MESA_SHADER_COMPUTE &&
      agx_device(ctx->base.screen)->chip == AGX_CHIP_G18G;
   if (!linked && agx_is_shader_empty(&cs->b) && !native_g18_compute)
      return;

   /* To implement load_num_workgroups, the number of workgroups needs to be
    * available in GPU memory. This is either the indirect buffer, or just a
    * buffer we upload ourselves if not indirect.
    */
   if (indirect_num_workgroups) {
      /* G17P expands indirect workgroup counts into a separate hardware
       * record of global thread counts. Keep the API counts for the shader.
       */
      batch->uniforms.tables[AGX_SYSVAL_TABLE_GRID] = indirect_num_workgroups;
   } else if (grid.mode == AGX_CDM_MODE_DIRECT) {
      uint32_t groups[3] = {
         grid.count[0] / wg.x,
         grid.count[1] / wg.y,
         grid.count[2] / wg.z,
      };

      batch->uniforms.tables[AGX_SYSVAL_TABLE_GRID] =
         agx_pool_upload_aligned(&batch->pool, groups, sizeof(groups), 4);
   } else {
      batch->uniforms.tables[AGX_SYSVAL_TABLE_GRID] = grid.ptr;
   }

   util_dynarray_foreach(&ctx->global_buffers, struct pipe_resource *, res) {
      if (!*res)
         continue;

      struct agx_resource *buffer = agx_resource(*res);
      agx_batch_writes_storage(batch, buffer, 0);
      batch->incoherent_writes = true;
   }

   agx_update_descriptors(batch, cs);
   if (batch->native_rejected)
      return;
   agx_upload_uniforms(batch, NULL);

   struct agx_device *dev = agx_device(ctx->base.screen);
   if (dev->chip == AGX_CHIP_G18G) {
      struct agx3_ioto_uniform_range ranges[AGX_MAX_PUSH_RANGES + 2];
      size_t range_count = 0;
      const bool resource_pair = batch->texture_count[stage] != 0;

      if (resource_pair) {
         if (!cs->g17p_texture_resources ||
             !batch->stage_uniforms[stage].texture_base ||
             (!cs->g17p_storage_write &&
               (!batch->samplers[stage] ||
                (batch->ctx->stage[stage].custom_borders &&
                 !cs->g17p_custom_border_emulation))))
            {
               agx_native_descriptor_reject_batch(batch, "compute texture resources");
               return;
            }

         uint64_t pointers[4] = {0};
         unsigned pointer_halfs = AGX_ABI_G17P_TEXTURE_POINTER_HALFS;
         if (cs->g17p_storage_write) {
            const bool g18g = dev->chip == AGX_CHIP_G18G;
            const unsigned descriptor_size =
               agx_gallium_texture_descriptor_size(
                  agx_chip_descriptor_generation(dev->chip));
            const uint64_t write_control = agx_pool_upload_aligned(
               &batch->pool,
               g18g ? agx_g18g_native_write_state1
                    : agx_g17p_native_write_control,
               g18g ? sizeof(agx_g18g_native_write_state1)
                    : sizeof(agx_g17p_native_write_control),
               32);
            if (!descriptor_size || !write_control) {
               agx_native_descriptor_reject_batch(batch,
                                              "compute storage resources");
               return;
            }

            /* This lowering accepts exactly image unit 0 with no sampled
             * textures, whose paired PBE is descriptor 1. State 0 therefore
             * receives that resolved descriptor and the native write control
             * directly, matching Honeykrisp's write-only ABI. G18G's root is
             * the PBE and its sampler-shaped state record only, eight halves,
             * as native run112 kick 19 launch 1 loads it. */
            pointers[0] = batch->stage_uniforms[stage].texture_base +
                          descriptor_size;
            pointers[1] = write_control;
            pointer_halfs = g18g ? AGX_G18G_STORAGE_ROOT_HALFS
                                 : AGX_G17P_STORAGE_ROOT_HALFS;
         } else {
            pointers[0] = batch->stage_uniforms[stage].texture_base;
            pointers[1] = batch->samplers[stage];
         }
         const uint64_t pointer_va = agx_pool_upload_aligned(
            &batch->pool, pointers, pointer_halfs * 2, 8);
         if (!pointer_va)
            {
               agx_native_descriptor_reject_batch(batch, "compute texture pointers");
               return;
            }

         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = 0,
            .size_halfs = pointer_halfs,
            .address = pointer_va,
         };
      } else if (cs->g17p_texture_resources) {
         {
            agx_native_descriptor_reject_batch(batch, "compute texture table");
            return;
         }
      }

      for (unsigned i = 0; i < cs->push_range_count; ++i) {
         const struct agx_push_range *push = &cs->push[i];
         const uint64_t table = batch->uniforms.tables[push->table];
         if (push->table == AGX_SYSVAL_TABLE_PARAMS && !table)
            continue;
         if (resource_pair && push->uniform <
                                AGX_ABI_G17P_TEXTURE_POINTER_HALFS)
            {
               agx_native_descriptor_reject_batch(batch, "compute texture ABI overlap");
               return;
            }
         if (!table || (push->uniform & 1) || (push->length & 1) ||
             !push->length || range_count == ARRAY_SIZE(ranges))
            {
               mesa_loge("asahi: compute uniform range fields: index=%u "
                         "table=%u address=0x%llx uniform=%u length=%u "
                         "offset=%u ranges=%zu capacity=%zu",
                         i, push->table, (unsigned long long)table,
                         push->uniform, push->length, push->offset,
                         range_count, ARRAY_SIZE(ranges));
               agx_native_descriptor_reject_batch(batch, "compute uniform range");
               return;
            }
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = push->uniform,
            .size_halfs = push->length,
            .address = table + push->offset,
         };
      }

      if (cs->b.info.rodata.size_16) {
         const struct agx_rodata *rodata = &cs->b.info.rodata;
         const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
         if (!cs->bo || !cs->bo->va || (rodata->base_uniform & 1) ||
             (rodata->size_16 & 1) || !rodata->size_16 ||
             range_count == ARRAY_SIZE(ranges) ||
             rodata->offset > cs->bo->size ||
             size_B > cs->bo->size - rodata->offset)
            {
               agx_native_descriptor_reject_batch(batch, "compute constant range");
               return;
            }
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = rodata->base_uniform,
            .size_halfs = rodata->size_16,
            .address = cs->bo->va->addr + rodata->offset,
         };
      }

      if (dev->chip == AGX_CHIP_G18G && cs->b.info.local_size) {
         for (unsigned i = 0; i < range_count; ++i) {
            if (ranges[i].start_halfs <= AGX_ABI_G18_SHARED_BASE_HALF &&
                ranges[i].start_halfs + ranges[i].size_halfs >
                   AGX_ABI_G18_SHARED_BASE_HALF) {
               agx_native_descriptor_reject_batch(batch,
                                              "compute shared-base overlap");
               return;
            }
         }
      }

      if (!agx_g18_append_compute(batch, &cs->b.info, cs->b.binary, cs->bo,
                                  linked, grid, wg, ranges, range_count,
                                  variable_shared_mem)) {
         /* G18G has no fallback: agx_cdm_launch() writes the AGX2 compute
          * packet, and G18G's records are HAL300's. */
         agx_native_descriptor_reject_batch(batch, "compute Ioto dispatch");
         return;
      } else {
         /* The dispatch just recorded. Native run112 selects compute mode
          * 0x14c08a60 (desc+0x62a) for every command that uses texture or
          * image instructions and 0x10c08860 for every buffer-only one;
          * resource_pair is exactly "this dispatch bound a texture table". */
         if (dev->chip == AGX_CHIP_G18G && batch->cdm_dispatch.size)
            util_dynarray_top_ptr(&batch->cdm_dispatch, struct agx_cdm_dispatch)
               ->texture_state = resource_pair;
         return;
      }
   }

   if (agx_chip_native_agx3(dev->chip)) {
      struct agx3_ioto_uniform_range ranges[AGX_MAX_PUSH_RANGES + 2];
      size_t range_count = 0;
      const bool resource_pair = batch->texture_count[stage] != 0;

      if (resource_pair) {
         if (!cs->g17p_texture_resources ||
             !batch->stage_uniforms[stage].texture_base ||
             !batch->samplers[stage] ||
             (batch->ctx->stage[stage].custom_borders &&
              !cs->g17p_custom_border_emulation))
            agx_native_descriptor_rejected(dev,
                                           "compute texture resources");

         const uint64_t pointers[2] = {
            batch->stage_uniforms[stage].texture_base,
            batch->samplers[stage],
         };
         const uint64_t pointer_va = agx_pool_upload_aligned(
            &batch->pool, pointers, sizeof(pointers), 8);
         if (!pointer_va)
            agx_native_descriptor_rejected(dev,
                                           "compute texture pointers");

         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = 0,
            .size_halfs = AGX_ABI_G17P_TEXTURE_POINTER_HALFS,
            .address = pointer_va,
         };
      } else if (cs->g17p_texture_resources) {
         agx_native_descriptor_rejected(dev, "compute texture table");
      }

      for (unsigned i = 0; i < cs->push_range_count; ++i) {
         const struct agx_push_range *push = &cs->push[i];
         const uint64_t table = batch->uniforms.tables[push->table];
         if (push->table == AGX_SYSVAL_TABLE_PARAMS && !table)
            continue;
         if (resource_pair && push->uniform <
                                AGX_ABI_G17P_TEXTURE_POINTER_HALFS)
            agx_native_descriptor_rejected(dev,
                                           "compute texture ABI overlap");
         if (!table || (push->uniform & 1) || (push->length & 1) ||
             !push->length || range_count == ARRAY_SIZE(ranges))
            agx_native_descriptor_rejected(dev, "compute uniform range");
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = push->uniform,
            .size_halfs = push->length,
            .address = table + push->offset,
         };
      }

      if (cs->b.info.rodata.size_16) {
         const struct agx_rodata *rodata = &cs->b.info.rodata;
         const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
         if (!cs->bo || !cs->bo->va || (rodata->base_uniform & 1) ||
             (rodata->size_16 & 1) || !rodata->size_16 ||
             range_count == ARRAY_SIZE(ranges) ||
             rodata->offset > cs->bo->size ||
             size_B > cs->bo->size - rodata->offset)
            agx_native_descriptor_rejected(dev, "compute constant range");
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = rodata->base_uniform,
            .size_halfs = rodata->size_16,
            .address = cs->bo->va->addr + rodata->offset,
         };
      }

      if (!agx_g17p_append_compute(batch, cs, &cs->b.info, cs->b.binary, cs->bo,
                                   linked, grid, wg, ranges, range_count,
                                   variable_shared_mem))
         agx_native_descriptor_rejected(dev, "compute Ioto dispatch");
      return;
   }

   /* Only G17G reaches this path: every native AGX3 part, G17P included,
    * took the block above. dev is the declaration that block uses. */
   if (agx_chip_g17p_or_m5(dev->chip)) {
      struct agx3_ioto_uniform_range ranges[AGX_MAX_PUSH_RANGES + 2];
      size_t range_count = 0;
      const bool resource_pair = batch->texture_count[stage] != 0;

      if (resource_pair) {
         if (!cs->g17p_texture_resources ||
             !batch->stage_uniforms[stage].texture_base ||
             !batch->samplers[stage] ||
             batch->ctx->stage[stage].custom_borders)
            {
               agx_native_descriptor_reject_batch(batch, "compute texture resources");
               return;
            }

         const uint64_t pointers[2] = {
            batch->stage_uniforms[stage].texture_base,
            batch->samplers[stage],
         };
         const uint64_t pointer_va = agx_pool_upload_aligned(
            &batch->pool, pointers, sizeof(pointers), 8);
         if (!pointer_va)
            {
               agx_native_descriptor_reject_batch(batch, "compute texture pointers");
               return;
            }

         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = 0,
            .size_halfs = AGX_ABI_G17P_TEXTURE_POINTER_HALFS,
            .address = pointer_va,
         };
      } else if (cs->g17p_texture_resources) {
         {
            agx_native_descriptor_reject_batch(batch, "compute texture table");
            return;
         }
      }

      for (unsigned i = 0; i < cs->push_range_count; ++i) {
         const struct agx_push_range *push = &cs->push[i];
         const uint64_t table = batch->uniforms.tables[push->table];
         if (push->table == AGX_SYSVAL_TABLE_PARAMS && !table)
            continue;
         if (resource_pair && push->uniform <
                                AGX_ABI_G17P_TEXTURE_POINTER_HALFS)
            {
               agx_native_descriptor_reject_batch(batch, "compute texture ABI overlap");
               return;
            }
         if (!table || (push->uniform & 1) || (push->length & 1) ||
             !push->length || range_count == ARRAY_SIZE(ranges))
            {
               agx_native_descriptor_reject_batch(batch, "compute uniform range");
               return;
            }
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = push->uniform,
            .size_halfs = push->length,
            .address = table + push->offset,
         };
      }

      if (cs->b.info.rodata.size_16) {
         const struct agx_rodata *rodata = &cs->b.info.rodata;
         const uint64_t size_B = (uint64_t)rodata->size_16 * 2;
         if (!cs->bo || !cs->bo->va || (rodata->base_uniform & 1) ||
             (rodata->size_16 & 1) || !rodata->size_16 ||
             range_count == ARRAY_SIZE(ranges) ||
             rodata->offset > cs->bo->size ||
             size_B > cs->bo->size - rodata->offset)
            {
               agx_native_descriptor_reject_batch(batch, "compute constant range");
               return;
            }
         ranges[range_count++] = (struct agx3_ioto_uniform_range){
            .start_halfs = rodata->base_uniform,
            .size_halfs = rodata->size_16,
            .address = cs->bo->va->addr + rodata->offset,
         };
      }

      if (!agx_g17p_append_compute(batch, cs, &cs->b.info, cs->b.binary, cs->bo,
                                   linked, grid, wg, ranges, range_count,
                                   variable_shared_mem)) {
         /* No fallback on either AGX3 part: the AGX2 encoder below writes an
          * AGX2 dispatch into the AGX3 control stream, which the G17G
          * firmware runs and never retires (indirect dispatches, shaders
          * with scratch), leaving the GPU dead until reboot. */
         agx_native_descriptor_reject_batch(batch, "compute Ioto dispatch");
         return;
      } else {
         return;
      }
   }

   // TODO: This is broken.
   size_t subgroups_per_core = 0;
#if 0
   if (!info->indirect) {
      size_t subgroups_per_workgroup =
         DIV_ROUND_UP(info->block[0] * info->block[1] * info->block[2], 32);
      subgroups_per_core =
         local_workgroups *
         DIV_ROUND_UP(info->grid[0] * info->grid[1] * info->grid[2],
                     ctx->scratch_cs.num_cores);
   }
#endif

   uint32_t usc = agx_build_pipeline(batch, cs, linked, MESA_SHADER_COMPUTE,
                                     variable_shared_mem, subgroups_per_core,
                                     NULL, NULL);

   if (cs)
      agx_batch_add_bo(batch, cs->bo);

   struct agx_cdm_launch_word_0_packed launch;
   agx_pack(&launch, CDM_LAUNCH_WORD_0, cfg) {
      cfg.uniform_register_count = cs->b.info.push_count;
      cfg.preshader_register_count = cs->b.info.nr_preamble_gprs;
      cfg.texture_state_register_count =
         cs ? agx_nr_tex_descriptors(batch, cs) : 0;
      cfg.sampler_state_register_count =
         translate_sampler_state_count(ctx, stage);
   }

   agx_launch_internal(batch, grid, wg, launch, stage, usc);
}

static void
agx_launch_grid(struct pipe_context *pipe, const struct pipe_grid_info *info)
{
   struct agx_context *ctx = agx_context(pipe);
   if (unlikely(!ctx->compute_blitter.active &&
                !agx_render_condition_check(ctx)))
      return;

   /* G17G has no proven indirect-grid compute dispatch (the compute Ioto
    * program refuses grid=1). Read the group counts back -- waiting for
    * whatever wrote them -- and dispatch directly. */
   if (info->indirect &&
       agx_chip_g17_m5(agx_device(pipe->screen)->chip)) {
      uint32_t groups[3] = {0};
      pipe_buffer_read(pipe, info->indirect, info->indirect_offset,
                       sizeof(groups), groups);
      if (!groups[0] || !groups[1] || !groups[2])
         return;
      struct pipe_grid_info direct = *info;
      direct.indirect = NULL;
      direct.indirect_offset = 0;
      for (unsigned i = 0; i < 3; ++i) {
         direct.grid[i] = groups[i];
         direct.last_block[i] = 0;
      }
      agx_launch_grid(pipe, &direct);
      return;
   }

   struct agx_batch *batch = agx_get_compute_batch(ctx);

   uint64_t indirect = 0;
   if (info->indirect) {
      struct agx_resource *rsrc = agx_resource(info->indirect);
      agx_batch_reads(batch, rsrc);
      indirect = agx_map_gpu(rsrc) + info->indirect_offset;
   }

   /* Increment the pipeline stats query.
    *
    * TODO: Can we use the hardware counter for this?
    */
   struct agx_query *statistic =
      ctx->pipeline_statistics[PIPE_STAT_QUERY_CS_INVOCATIONS];

   struct agx_workgroup wg =
      agx_workgroup(info->block[0], info->block[1], info->block[2]);

   if (statistic) {
      if (indirect) {
         uint64_t addr = agx_get_query_address(batch, statistic);

         libagx_increment_cs_invocations(batch, agx_1d(1), AGX_BARRIER_ALL,
                                         indirect, addr,
                                         agx_workgroup_threads(wg));
      } else {
         agx_query_increment_cpu(ctx, statistic,
                                 agx_workgroup_threads(wg) * info->grid[0] *
                                    info->grid[1] * info->grid[2]);
      }
   }

   agx_batch_add_timestamp_query(batch, ctx->time_elapsed);

   agx_batch_init_state(batch);

   struct agx_uncompiled_shader *uncompiled =
      ctx->stage[MESA_SHADER_COMPUTE].shader;
   agx_shader_wait_ready(uncompiled);

   union asahi_shader_key key = {
      .cs.custom_borders =
         agx_chip_native_agx3(agx_device(pipe->screen)->chip) &&
         ctx->stage[MESA_SHADER_COMPUTE].custom_borders,
   };
   const unsigned cs_idx = key.cs.custom_borders ? 1u : 0u;
   struct agx_compiled_shader *cs = uncompiled->cached_cs[cs_idx];
   if (unlikely(!cs)) {
      /* The variant compiled at link time, or by an earlier dispatch, is in
       * the shader's table; agx_get_shader_variant() would compile (or fetch
       * from the disk cache) and insert it again on every dispatch. */
      struct hash_entry *variant =
         _mesa_hash_table_search(uncompiled->variants, &key);
      cs = variant ? variant->data
                   : agx_get_shader_variant(agx_screen(pipe->screen),
                                            &pipe->debug, uncompiled, &key);
      uncompiled->cached_cs[cs_idx] = cs;
   }

   struct agx_grid grid;
   if (indirect && agx_chip_native_agx3(agx_device(pipe->screen)->chip)) {
      /* G17P launches an indirect dispatch from a six-word grid -- three
       * global thread counts then the three local sizes -- so the API's three
       * workgroup counts are not the record the hardware reads. Expand them
       * with the same one-invocation setup kernel the tessellator already
       * uses for its indirect launches, and dispatch through the grounded
       * indirect-local record rather than grounding a second launch form. */
      uint64_t expanded =
         agx_pool_alloc_aligned(&batch->pool, 6 * sizeof(uint32_t), 4).gpu;

      libagx_setup_indirect_dispatch(batch, agx_1d(1), AGX_BARRIER_ALL,
                                     indirect, expanded, wg.x, wg.y, wg.z);

      grid = agx_grid_indirect_local(expanded);
   } else if (indirect) {
      grid = agx_grid_indirect(indirect);
   } else {
      grid = agx_3d(0, 0, 0);

      for (unsigned d = 0; d < 3; ++d) {
         grid.count[d] = ((info->grid[d] - 1) * info->block[d]) +
                         (info->last_block[d] ?: info->block[d]);
      }
   }

   agx_launch(batch, grid, wg, cs, NULL, MESA_SHADER_COMPUTE,
              info->variable_shared_mem, indirect);

   /* Count a plain application dispatch: a direct grid with no query. The
    * compute blitter's launches are driver work. See
    * agx_batch_sibling_queue(). */
   if (ctx->compute_blitter.active || indirect || statistic ||
       ctx->time_elapsed) {
      batch->compute_internal = true;
   } else {
      const uint64_t threads = (uint64_t)grid.count[0] * grid.count[1] *
                               grid.count[2];
      batch->storage_dispatches++;
      batch->storage_min_threads = MIN2(batch->storage_min_threads, threads);
   }

   /* TODO: Dirty tracking? */
   agx_dirty_all(ctx);

   batch->uniforms.tables[AGX_SYSVAL_TABLE_GRID] = 0;

   /* If the next dispatch might overflow, flush now. TODO: If this is ever hit
    * in practice, we can use CDM stream links.
    */
   size_t dispatch_upper_bound =
      AGX_CDM_LAUNCH_WORD_0_LENGTH + AGX_CDM_LAUNCH_WORD_1_LENGTH +
      AGX_CDM_UNK_G14X_LENGTH + AGX_CDM_INDIRECT_LENGTH +
      AGX_CDM_GLOBAL_SIZE_LENGTH + AGX_CDM_LOCAL_SIZE_LENGTH +
      AGX_CDM_BARRIER_LENGTH;

   if (batch->cdm.current + dispatch_upper_bound >= batch->cdm.end)
      agx_flush_batch_for_reason(ctx, batch, "CDM overfull");
}

static void
agx_set_global_binding(struct pipe_context *pipe, unsigned first,
                       unsigned count, struct pipe_resource **resources,
                       uint32_t **handles)
{
   struct agx_context *ctx = agx_context(pipe);
   unsigned old_size =
      util_dynarray_num_elements(&ctx->global_buffers, *resources);

   if (old_size < first + count) {
      /* we are screwed no matter what */
      if (!util_dynarray_grow(&ctx->global_buffers, *resources,
                              (first + count) - old_size))
         UNREACHABLE("out of memory");

      for (unsigned i = old_size; i < first + count; i++)
         *util_dynarray_element(&ctx->global_buffers, struct pipe_resource *,
                                i) = NULL;
   }

   for (unsigned i = 0; i < count; ++i) {
      struct pipe_resource **res = util_dynarray_element(
         &ctx->global_buffers, struct pipe_resource *, first + i);
      if (resources && resources[i]) {
         pipe_resource_reference(res, resources[i]);

         /* The handle points to uint32_t, but space is allocated for 64
          * bits. We need to respect the offset passed in. This interface
          * is so bad.
          */
         uint64_t addr = 0;
         struct agx_resource *rsrc = agx_resource(resources[i]);

         memcpy(&addr, handles[i], sizeof(addr));
         addr += agx_map_gpu(rsrc);
         memcpy(handles[i], &addr, sizeof(addr));
      } else {
         pipe_resource_reference(res, NULL);
      }
   }
}

void agx_init_state_functions(struct pipe_context *ctx);

void
agx_decompress_inplace(struct agx_batch *batch, struct pipe_surface *surf,
                       const char *reason)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   struct agx_resource *rsrc = agx_resource(surf->texture);
   struct ail_layout *layout = &rsrc->layout;
   unsigned level = surf->level;
   const enum agx_descriptor_generation descriptor_generation =
      agx_chip_descriptor_generation(dev->chip);

   perf_debug(dev, "Decompressing in-place due to: %s", reason);

   /* libagx_decompress_images is a byte-exact AGX2 24+24 ABI.  A native
    * AGX3 version needs a 32+32 argument block and matching precompiled shader;
    * do not dispatch the legacy kernel against a wider descriptor generation.
    */
   if (!agx_gallium_descriptor_generation_is_agx2(descriptor_generation)) {
      {
         agx_native_descriptor_reject_batch(batch, "decompression descriptor block");
         return;
      }
   }
   if (!batch->cdm.bo)
      batch->cdm = agx_encoder_allocate(batch, dev);

   struct agx_ptr images = agx_pool_alloc_aligned(
      &batch->pool, sizeof(struct libagx_decompress_images), 64);
   struct libagx_decompress_images *img = images.cpu;

   struct pipe_sampler_view sampler_view = sampler_view_for_surface(surf);
   sampler_view.target = PIPE_TEXTURE_2D_ARRAY;
   struct pipe_image_view view = image_view_for_surface(surf);
   struct agx_gallium_texture_descriptor native_texture;
   struct agx_gallium_pbe_descriptor native_pbe;
   if (!agx_pack_texture(descriptor_generation, &native_texture, rsrc,
                         surf->format, &sampler_view))
      {
         agx_native_descriptor_reject_batch(batch, "decompression texture");
         return;
      }
   if (!agx_batch_upload_pbe(batch, &native_pbe, &view, false, true, true,
                             true))
      {
         agx_native_descriptor_reject_batch(batch, "decompression PBE");
         return;
      }
   memcpy(&img->compressed, &native_texture, AGX_TEXTURE_LENGTH);
   memcpy(&img->uncompressed, &native_pbe, AGX_PBE_LENGTH);

   struct agx_grid grid = agx_3d(ail_metadata_width_tl(layout, level) * 32,
                                 ail_metadata_height_tl(layout, level),
                                 surf->last_layer - surf->first_layer + 1);

   libagx_decompress(batch, grid, AGX_BARRIER_ALL, layout, surf->first_layer,
                     level, agx_map_gpu(rsrc), images.gpu);
}

void
agx_init_state_functions(struct pipe_context *ctx)
{
   ctx->create_blend_state = agx_create_blend_state;
   ctx->create_depth_stencil_alpha_state = agx_create_zsa_state;
   ctx->create_fs_state = agx_create_shader_state;
   ctx->create_rasterizer_state = agx_create_rs_state;
   ctx->create_sampler_state = agx_create_sampler_state;
   ctx->create_sampler_view = agx_create_sampler_view;
   ctx->create_vertex_elements_state = agx_create_vertex_elements;
   ctx->create_vs_state = agx_create_shader_state;
   ctx->create_gs_state = agx_create_shader_state;
   ctx->create_tcs_state = agx_create_shader_state;
   ctx->create_tes_state = agx_create_shader_state;
   ctx->create_compute_state = agx_create_compute_state;
   ctx->bind_blend_state = agx_bind_blend_state;
   ctx->bind_depth_stencil_alpha_state = agx_bind_zsa_state;
   ctx->bind_sampler_states = agx_bind_sampler_states;
   ctx->bind_fs_state = agx_bind_fs_state;
   ctx->bind_rasterizer_state = agx_bind_rasterizer_state;
   ctx->bind_vertex_elements_state = agx_bind_vertex_elements_state;
   ctx->bind_vs_state = agx_bind_vs_state;
   ctx->bind_gs_state = agx_bind_gs_state;
   ctx->bind_tcs_state = agx_bind_tcs_state;
   ctx->bind_tes_state = agx_bind_tes_state;
   ctx->bind_compute_state = agx_bind_cs_state;
   ctx->delete_blend_state = agx_delete_state;
   ctx->delete_depth_stencil_alpha_state = agx_delete_state;
   ctx->delete_fs_state = agx_delete_shader_state;
   ctx->delete_compute_state = agx_delete_shader_state;
   ctx->delete_rasterizer_state = agx_delete_state;
   ctx->delete_sampler_state = agx_delete_sampler_state;
   ctx->delete_vertex_elements_state = agx_delete_state;
   ctx->delete_vs_state = agx_delete_shader_state;
   ctx->delete_gs_state = agx_delete_shader_state;
   ctx->delete_tcs_state = agx_delete_shader_state;
   ctx->delete_tes_state = agx_delete_shader_state;
   ctx->set_blend_color = agx_set_blend_color;
   ctx->set_constant_buffer = agx_set_constant_buffer;
   ctx->set_shader_buffers = agx_set_shader_buffers;
   ctx->set_shader_images = agx_set_shader_images;
   ctx->set_sampler_views = agx_set_sampler_views;
   ctx->set_framebuffer_state = agx_set_framebuffer_state;
   ctx->set_polygon_stipple = agx_set_polygon_stipple;
   ctx->set_patch_vertices = agx_set_patch_vertices;
   ctx->set_sample_mask = agx_set_sample_mask;
   ctx->set_scissor_states = agx_set_scissor_states;
   ctx->set_stencil_ref = agx_set_stencil_ref;
   ctx->set_vertex_buffers = agx_set_vertex_buffers;
   ctx->set_viewport_states = agx_set_viewport_states;
   ctx->sampler_view_destroy = agx_sampler_view_destroy;
   ctx->sampler_view_release = u_default_sampler_view_release;
   ctx->resource_release = u_default_resource_release;
   ctx->draw_vbo = agx_draw_vbo;
   ctx->launch_grid = agx_launch_grid;
   ctx->set_global_binding = agx_set_global_binding;
   ctx->texture_barrier = agx_texture_barrier;
   ctx->get_compute_state_info = agx_get_compute_state_info;
   ctx->set_tess_state = agx_set_tess_state;
}
