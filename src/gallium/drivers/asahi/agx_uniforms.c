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

/* 32 zero bytes, uploaded once per batch, for a constant or storage buffer
 * binding with nothing in range. Robust access redirects an out-of-bounds
 * load to offset 0 instead of dropping it, so the sink must hold the widest
 * load, a dvec4.
 */
static uint64_t
agx_batch_zero_sink(struct agx_batch *batch)
{
   if (!batch->zero_sink) {
      static const uint32_t zeroes[8] = {0};
      batch->zero_sink =
         agx_pool_upload_aligned(&batch->pool, zeroes, sizeof(zeroes), 16);
   }

   return batch->zero_sink;
}

static uint64_t
agx_const_buffer_ptr(struct agx_batch *batch, mesa_shader_stage stage,
                     struct pipe_constant_buffer *cb)
{
   if (cb->buffer) {
      /* Nothing of the range is inside the buffer (agx_set_constant_buffer()
       * clamps it), so every load is out of bounds: read the sink, not past
       * the end of the BO.
       */
      if (cb->buffer_size == 0)
         return agx_batch_zero_sink(batch);

      struct agx_resource *rsrc = agx_resource(cb->buffer);
      agx_batch_reads_range(batch, rsrc, cb->buffer_offset, cb->buffer_size);
      if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_COMPUTE)
         agx_batch_note_vtx_read(batch, rsrc);
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
         agx_batch_note_vtx_read(batch, rsrc);

         buffers[vbo] = agx_map_gpu(rsrc) + vb.buffer_offset;
         buf_sizes[vbo] = rsrc->layout.size_B - vb.buffer_offset -
                          rsrc->layout.level_offsets_B[0];
      }
   }

   const unsigned count = attribs->num_attribs;
   for (unsigned i = 0; i < count; ++i) {
      unsigned buf = attribs->buffers[i];
      uint64_t addr;

      batch->uniforms.attrib_clamp[i] = agx_calculate_vbo_clamp(
         buffers[buf], attribs->key[i].format, buf_sizes[buf],
         attribs->key[i].stride, attribs->src_offsets[i], &addr);

      batch->uniforms.attrib_base[i] = addr;
   }

   for (unsigned i = count; i < batch->uploaded_attribs; ++i) {
      batch->uniforms.attrib_base[i] = 0;
      batch->uniforms.attrib_clamp[i] = 0;
   }
   batch->uploaded_attribs = count;
}

void
agx_upload_uniforms(struct agx_batch *batch,
                    struct agx_compiled_shader *hw_vs)
{
   struct agx_context *ctx = batch->ctx;
   struct agx_device *dev = agx_device(ctx->base.screen);
   size_t root_size = sizeof(struct agx_draw_uniforms);

   batch->uniforms.sample_mask = ctx->sample_mask;

   assert(_mesa_float_to_half(0.5) == 0x3800);
   batch->uniforms.clip_z_coeff =
      (ctx->rast && !ctx->rast->base.clip_halfz) ? 0x3800 : 0x0;

   batch->uniforms.sprite_mask =
      (batch->reduced_prim == MESA_PRIM_POINTS && ctx->rast)
         ? ctx->rast->base.sprite_coord_enable
         : 0;

   /* Graphics draws without geometry/tessellation read only the pushed prefix
    * of agx_draw_uniforms unless a shader loads the root table address itself
    * (root_uniforms_upload_size == UINT16_MAX). On incremental encoders, a
    * vertex-only update after the first draw reuses the latched fragment
    * program, so only hw_vs consumes the root upload. When only VS stage
    * descriptors changed (e.g. a dynamic UBO binding) and none of hw_vs's
    * pushed root ranges changed since batch->vs_root, reuse the existing root
    * upload directly. */
   if (hw_vs && !ctx->gs && !ctx->in_tess && ctx->fs) {
      const bool vs_only_update =
         batch->draws > 0 && !agx_chip_hal300_path(dev->chip) &&
         !(ctx->dirty & ~(AGX_DIRTY_VS | AGX_DIRTY_VERTEX));
      if (vs_only_update && !(ctx->dirty & AGX_DIRTY_VERTEX) &&
          batch->vs_root_valid &&
          batch->uniforms.tables[AGX_SYSVAL_TABLE_ROOT] &&
          hw_vs->root_uniforms_upload_size != UINT16_MAX) {
         bool root_changed = false;
         for (unsigned i = 0; i < hw_vs->push_range_count; ++i) {
            const struct agx_push_range *push = &hw_vs->push[i];
            if (push->table != AGX_SYSVAL_TABLE_ROOT)
               continue;

            assert((size_t)push->offset + push->length * 2 <=
                   sizeof(batch->uniforms));
            if (memcmp((const uint8_t *)&batch->vs_root + push->offset,
                       (const uint8_t *)&batch->uniforms + push->offset,
                       push->length * 2) != 0) {
               root_changed = true;
               break;
            }
         }
         if (!root_changed)
            return;
      }

      uint16_t max_upload = hw_vs->root_uniforms_upload_size;
      if (!vs_only_update) {
         const struct agx_compiled_shader *fs =
            (ctx->linked.fs && ctx->linked.fs->inline_blend_fs)
               ? ctx->linked.fs->inline_blend_fs
               : ctx->fs;
         max_upload = MAX2(max_upload, fs->root_uniforms_upload_size);
      }
      if (max_upload != UINT16_MAX)
         root_size = MAX2((size_t)max_upload, 16u);
   }

   struct agx_ptr root_ptr = agx_pool_alloc_aligned(&batch->pool, root_size, 16);

   batch->uniforms.tables[AGX_SYSVAL_TABLE_ROOT] = root_ptr.gpu;

   if (hw_vs && agx_chip_native_agx3(dev->chip)) {
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
      memcpy(&batch->vs_root, &batch->uniforms, root_size);
      batch->vs_root_valid = true;
   }

   memcpy(root_ptr.cpu, &batch->uniforms, root_size);
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
      unif->ubo_base[cb] = agx_const_buffer_ptr(batch, stage, &st->cb[cb]);
      unif->ubo_size[cb] = st->cb[cb].buffer_size;
   }
}

void
agx_set_ssbo_uniforms(struct agx_batch *batch, mesa_shader_stage stage)
{
   struct agx_stage *st = &batch->ctx->stage[stage];
   struct agx_stage_uniforms *unif = &batch->stage_uniforms[stage];

   /* TODO: Optimize the sink with soft fault. */
   uint64_t sink = agx_batch_zero_sink(batch);

   /* Consider all shader buffers, needed to avoid faults with
    * e.g. arb_shader_storage_buffer_object-array-ssbo-binding.
    */
   for (unsigned cb = 0; cb < PIPE_MAX_SHADER_BUFFERS; ++cb) {
      struct pipe_shader_buffer *sb = &st->ssbo[cb];

      if (sb->buffer && st->ssbo[cb].buffer_size) {
         struct agx_resource *rsrc = agx_resource(sb->buffer);

         if (st->ssbo_writable_mask & BITFIELD_BIT(cb)) {
            if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_COMPUTE)
               batch->vtx_needs_render_barrier = true;
            agx_batch_writes_storage_range(batch, rsrc, sb->buffer_offset,
                                           sb->buffer_size);
            batch->incoherent_writes = true;
         } else {
            agx_batch_reads_range(batch, rsrc, sb->buffer_offset,
                                  sb->buffer_size);
            if (stage != MESA_SHADER_FRAGMENT && stage != MESA_SHADER_COMPUTE)
               agx_batch_note_vtx_read(batch, rsrc);
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
