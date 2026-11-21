/*
 * Copyright 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */
#include <stdio.h>
#include <stdlib.h>
#include "pipe/p_state.h"
#include "util/format/u_format.h"
#include "util/half_float.h"
#include "util/macros.h"
#include "agx_device.h"
#include "agx_state.h"
#include "pool.h"

static uint64_t
agx_const_buffer_ptr(struct agx_batch *batch, struct pipe_constant_buffer *cb)
{
   if (cb->buffer) {
      struct agx_resource *rsrc = agx_resource(cb->buffer);
      agx_batch_reads_range(batch, rsrc, cb->buffer_offset, cb->buffer_size);
      return agx_map_gpu(rsrc) + cb->buffer_offset;
   } else {
      return 0;
   }
}

void
agx_upload_vbos(struct agx_batch *batch)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_vertex_elements *attribs = ctx->attributes;
   uint64_t buffers[PIPE_MAX_ATTRIBS] = {0};
   size_t buf_sizes[PIPE_MAX_ATTRIBS] = {0};

   u_foreach_bit(vbo, ctx->vb_mask) {
      struct pipe_vertex_buffer vb = ctx->vertex_buffers[vbo];
      assert(!vb.is_user_buffer);

      if (vb.buffer.resource) {
         struct agx_resource *rsrc = agx_resource(vb.buffer.resource);
         agx_batch_reads(batch, rsrc);

         buffers[vbo] = agx_map_gpu(rsrc) + vb.buffer_offset;
         buf_sizes[vbo] = rsrc->layout.size_B - vb.buffer_offset -
                          rsrc->layout.level_offsets_B[0];
      }
   }

   for (unsigned i = 0; i < PIPE_MAX_ATTRIBS; ++i) {
      unsigned buf = attribs->buffers[i];
      uint64_t addr;

      batch->uniforms.attrib_clamp[i] = agx_calculate_vbo_clamp(
         buffers[buf], attribs->key[i].format, buf_sizes[buf],
         attribs->key[i].stride, attribs->src_offsets[i], &addr);

      batch->uniforms.attrib_base[i] = addr;
   }
}

void
agx_upload_uniforms(struct agx_batch *batch,
                    struct agx_compiled_shader *hw_vs)
{
   struct agx_context *ctx = batch->ctx;

   struct agx_ptr root_ptr = agx_pool_alloc_aligned(
      &batch->pool, sizeof(struct agx_draw_uniforms), 16);

   batch->uniforms.tables[AGX_SYSVAL_TABLE_ROOT] = root_ptr.gpu;
   batch->uniforms.sample_mask = ctx->sample_mask;

   assert(_mesa_float_to_half(0.5) == 0x3800);
   batch->uniforms.clip_z_coeff =
      (ctx->rast && !ctx->rast->base.clip_halfz) ? 0x3800 : 0x0;

   batch->uniforms.sprite_mask =
      (batch->reduced_prim == MESA_PRIM_POINTS && ctx->rast)
         ? ctx->rast->base.sprite_coord_enable
         : 0;

   if (hw_vs && agx_chip_native_agx3(agx_device(ctx->base.screen)->chip)) {
      /* Keep the old immutable upload for unchanged vertex inputs, including
       * fragment-only root updates. The unconditional native VS update used
       * to cover root inputs not represented by VS dirty bits. Compare the
       * ranges actually consumed by the hardware VS so query/statistic and
       * indirect table pointer changes cannot leave it reading stale data.
       * Use cached CPU memory: GPU pool mappings are write-combined. */
      if (!(ctx->dirty & AGX_DIRTY_VS) && batch->vs_root_valid) {
         for (unsigned i = 0; i < hw_vs->push_range_count; ++i) {
            const struct agx_push_range *push = &hw_vs->push[i];
            if (push->table != AGX_SYSVAL_TABLE_ROOT)
               continue;

            assert((size_t)push->offset + push->length * 2 <=
                   sizeof(batch->uniforms));
            if (memcmp((const uint8_t *)&batch->vs_root + push->offset,
                       (const uint8_t *)&batch->uniforms + push->offset,
                       push->length * 2) != 0) {
               ctx->dirty |= AGX_DIRTY_VS;
               break;
            }
         }
      }
      batch->vs_root = batch->uniforms;
      batch->vs_root_valid = true;
   }

   memcpy(root_ptr.cpu, &batch->uniforms, sizeof(batch->uniforms));
}

void
agx_set_sampler_uniforms(struct agx_batch *batch, mesa_shader_stage stage)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_stage *st = &ctx->stage[stage];
   struct agx_stage_uniforms *unif = &batch->stage_uniforms[stage];
   struct agx_device *dev = agx_device(ctx->base.screen);

   const bool custom_border = agx_chip_agx3_path(dev->chip);
   if (custom_border) {
      for (unsigned s = 0; s < PIPE_MAX_SAMPLERS; ++s) {
         unif->custom_border_sampler_handle[s] = -1;
         memset(unif->custom_border[s], 0, sizeof(unif->custom_border[s]));
         if (dev->chip == AGX_CHIP_G18G)
            unif->custom_border_compare[s] = 0;
      }
   }

   u_foreach_bit(s, st->valid_samplers) {
      unif->lod_bias[s] = st->samplers[s]->lod_bias_as_fp16;

      if (custom_border && st->samplers[s]->uses_custom_border) {
         uint16_t handle;
         if (!agx_gallium_g17p_custom_border_handle(dev->chip, s, &handle))
            UNREACHABLE("G17P custom-border sampler index outside table");

         unif->custom_border_sampler_handle[s] = handle;
         memcpy(unif->custom_border[s], st->samplers[s]->custom_border,
                sizeof(unif->custom_border[s]));

         if (agx_shadow_gather_custom_border(dev->chip)) {
            const struct pipe_sampler_state *sampler = &st->samplers[s]->base;
            const struct util_format_description *desc =
               util_format_description(sampler->border_color_format);
            unif->custom_border_compare_func[s] = sampler->compare_func;
            if (util_format_has_depth(desc) &&
                desc->channel[0].type != UTIL_FORMAT_TYPE_FLOAT)
               unif->custom_border_compare_func[s] |= AGX_BORDER_COMPARE_CLAMP_REF;
         }

         if (dev->chip == AGX_CHIP_G18G) {
            const struct pipe_sampler_state *state = &st->samplers[s]->base;
            const bool fixed_depth =
               util_format_is_depth_or_stencil(state->border_color_format) &&
               state->border_color_format != PIPE_FORMAT_Z32_FLOAT &&
               state->border_color_format != PIPE_FORMAT_Z32_FLOAT_S8X24_UINT;
            unif->custom_border_compare[s] = state->compare_func |
                                            (fixed_depth ? 8 : 0);
         }
      }
   }

   /* Materialize indirect sampler handles. AGX2 uses its global heap; G17P
    * and G17G index the native per-stage table uploaded with the graphics
    * program (G17G's sixteenth API sampler, handle 16, is past the
    * immediate field and read the heap handle's empty entry instead:
    * GLES3 texture.units.all_units). */
   if (st->shader && st->shader->uses_bindless_samplers) {
      u_foreach_bit(s, st->valid_samplers) {
         if (agx_chip_agx3_path(dev->chip)) {
            /* The G17P graphics Ioto program publishes the per-stage HAL200
             * table directly. Indirect operands therefore name the same
             * i+1 entries as immediate operands; the old 28+index handle
             * belongs to AGX2's separate 8-byte global heap. */
            uint16_t native_handle;
            if (!agx_gallium_g17p_sampler_handle(dev->chip, s,
                                                  &native_handle))
               UNREACHABLE("G17P sampler index outside the native table");
            unif->sampler_handle[s] = native_handle;
         } else {
            unif->sampler_handle[s] =
               28 +
               agx_sampler_heap_add(
                  dev, &batch->sampler_heap,
                  &st->samplers[s]->desc_without_custom_border);
         }
      }
   }
}

void
agx_set_cbuf_uniforms(struct agx_batch *batch, mesa_shader_stage stage)
{
   struct agx_stage *st = &batch->ctx->stage[stage];
   struct agx_stage_uniforms *unif = &batch->stage_uniforms[stage];

   u_foreach_bit(cb, st->cb_mask) {
      unif->ubo_base[cb] = agx_const_buffer_ptr(batch, &st->cb[cb]);
      unif->ubo_size[cb] = st->cb[cb].buffer_size;
   }
}

void
agx_set_ssbo_uniforms(struct agx_batch *batch, mesa_shader_stage stage)
{
   struct agx_stage *st = &batch->ctx->stage[stage];
   struct agx_stage_uniforms *unif = &batch->stage_uniforms[stage];

   /* Single element sink. TODO: Optimize with soft fault. */
   uint32_t zeroes[4] = {0};
   uint64_t sink = agx_pool_upload_aligned(&batch->pool, &zeroes, 16, 16);

   /* Consider all shader buffers, needed to avoid faults with
    * e.g. arb_shader_storage_buffer_object-array-ssbo-binding.
    */
   for (unsigned cb = 0; cb < PIPE_MAX_SHADER_BUFFERS; ++cb) {
      struct pipe_shader_buffer *sb = &st->ssbo[cb];

      if (sb->buffer && st->ssbo[cb].buffer_size) {
         struct agx_resource *rsrc = agx_resource(sb->buffer);

         if (st->ssbo_writable_mask & BITFIELD_BIT(cb)) {
            agx_batch_writes_range(batch, rsrc, sb->buffer_offset,
                                   sb->buffer_size);
            batch->incoherent_writes = true;
         } else {
            agx_batch_reads_range(batch, rsrc, sb->buffer_offset,
                                  sb->buffer_size);
         }

         unif->ssbo_base[cb] = agx_map_gpu(rsrc) + sb->buffer_offset;
         unif->ssbo_size[cb] = st->ssbo[cb].buffer_size;
      } else {
         /* Invalid, so use the sink */
         unif->ssbo_base[cb] = sink;
         unif->ssbo_size[cb] = 0;
      }
   }
}
