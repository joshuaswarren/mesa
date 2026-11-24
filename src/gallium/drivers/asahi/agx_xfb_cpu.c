/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Transform feedback on the CPU, for G17G.
 *
 * Elsewhere transform feedback is written by the geometry-shader
 * rasterization program (poly), whose draw G17G cannot encode, so every
 * capture came back empty. Here the vertex shader runs a second time on the
 * CPU through Gallium's draw module, whose stream-output stage writes the
 * captured outputs, and the GPU draw proceeds without a geometry stage.
 * Only plain vertex pipelines take this path: no geometry or tessellation
 * stage, no vertex-stage textures, images or storage buffers, and direct
 * draws.
 */

#include "draw/draw_context.h"
#include "draw/draw_nir.h"
#include "draw/draw_private.h"
#include "draw/draw_vbuf.h"
#include "draw/draw_vertex.h"
#include "compiler/nir/nir_xfb_info.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "agx_state.h"

struct agx_xfb_render {
   struct vbuf_render base;
   struct vertex_info vinfo;
   void *vertices;
   size_t size;
   uint64_t written[PIPE_MAX_VERTEX_STREAMS];
   uint64_t generated[PIPE_MAX_VERTEX_STREAMS];
};

static struct agx_xfb_render *
agx_xfb_render(struct vbuf_render *render)
{
   return (struct agx_xfb_render *)render;
}

static const struct vertex_info *
xfb_get_vertex_info(struct vbuf_render *render)
{
   return &agx_xfb_render(render)->vinfo;
}

static bool
xfb_allocate_vertices(struct vbuf_render *render, uint16_t vertex_size,
                      uint16_t nr_vertices)
{
   struct agx_xfb_render *r = agx_xfb_render(render);
   size_t size = (size_t)vertex_size * nr_vertices;

   if (size > r->size) {
      free(r->vertices);
      r->vertices = malloc(size);
      r->size = r->vertices ? size : 0;
   }

   return r->vertices != NULL;
}

static void *
xfb_map_vertices(struct vbuf_render *render)
{
   return agx_xfb_render(render)->vertices;
}

static void
xfb_unmap_vertices(struct vbuf_render *render, uint16_t min_index,
                   uint16_t max_index)
{
}

static void
xfb_set_primitive(struct vbuf_render *render, enum mesa_prim prim)
{
}

static void
xfb_draw_elements(struct vbuf_render *render, const uint16_t *indices,
                  unsigned count)
{
}

static void
xfb_draw_arrays(struct vbuf_render *render, unsigned start, unsigned nr)
{
}

static void
xfb_release_vertices(struct vbuf_render *render)
{
}

static void
xfb_set_stream_output_info(struct vbuf_render *render, unsigned stream,
                           unsigned primitive_count,
                           unsigned primitive_generated)
{
   struct agx_xfb_render *r = agx_xfb_render(render);

   r->written[stream] += primitive_count;
   r->generated[stream] += primitive_generated;
}

static void
xfb_destroy(struct vbuf_render *render)
{
   struct agx_xfb_render *r = agx_xfb_render(render);

   free(r->vertices);
   FREE(r);
}

static bool
agx_xfb_cpu_init(struct agx_context *ctx)
{
   if (ctx->xfb_draw)
      return true;

   /* The interpreter: the LLVM middle end crashed in its generated code on
    * the first capture, and speed does not matter here. */
   struct draw_context *draw = draw_create_no_llvm(&ctx->base);
   struct agx_xfb_render *r = CALLOC_STRUCT(agx_xfb_render);
   if (!draw || !r) {
      if (draw)
         draw_destroy(draw);
      FREE(r);
      return false;
   }

   r->base.max_indices = 16 * 1024;
   r->base.max_vertex_buffer_bytes = 1024 * 1024;
   r->base.get_vertex_info = xfb_get_vertex_info;
   r->base.allocate_vertices = xfb_allocate_vertices;
   r->base.map_vertices = xfb_map_vertices;
   r->base.unmap_vertices = xfb_unmap_vertices;
   r->base.set_primitive = xfb_set_primitive;
   r->base.draw_elements = xfb_draw_elements;
   r->base.draw_arrays = xfb_draw_arrays;
   r->base.release_vertices = xfb_release_vertices;
   r->base.set_stream_output_info = xfb_set_stream_output_info;
   r->base.destroy = xfb_destroy;

   /* Nothing past stream output is used: emit the position only. */
   r->vinfo.num_attribs = 1;
   r->vinfo.attrib[0].emit = EMIT_4F;
   r->vinfo.attrib[0].src_index = 0;
   draw_compute_vertex_size(&r->vinfo);

   struct draw_stage *stage = draw_vbuf_stage(draw, &r->base);
   if (!stage) {
      draw_destroy(draw);
      xfb_destroy(&r->base);
      return false;
   }

   draw_set_rasterize_stage(draw, stage);
   draw_set_render(draw, &r->base);
   draw_set_driver_clipping(draw, true, true, true, true);
   /* The fetch-shade-emit middle end, which draw picks when no pipeline
    * stage is needed, has no stream-output stage. */
   draw->pt.no_fse = true;

   ctx->xfb_draw = draw;
   ctx->xfb_render = &r->base;
   return true;
}

void
agx_xfb_cpu_destroy(struct agx_context *ctx)
{
   /* The vbuf stage owns the render. */
   if (ctx->xfb_draw)
      draw_destroy(ctx->xfb_draw);

   ctx->xfb_draw = NULL;
   ctx->xfb_render = NULL;
}

void
agx_xfb_cpu_create_vs(struct agx_context *ctx, struct agx_uncompiled_shader *so,
                      const struct pipe_shader_state *cso, const nir_shader *nir)
{
   /* The state tracker hands this driver I/O variables, and with them the
    * stream-output info the draw module consumes. */
   if (nir->info.stage != MESA_SHADER_VERTEX || nir->info.io_lowered ||
       !cso->stream_output.num_outputs)
      return;

   /* The CPU copy reads only vertex attributes and constants. */
   if (nir->info.num_textures || nir->info.num_images ||
       nir->info.num_ssbos || BITSET_COUNT(nir->info.textures_used))
      return;

   simple_mtx_lock(&ctx->xfb_cpu_lock);
   if (!agx_xfb_cpu_init(ctx)) {
      simple_mtx_unlock(&ctx->xfb_cpu_lock);
      return;
   }

   nir_shader *clone = nir_shader_clone(NULL, nir);
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .stream_output = cso->stream_output,
   };

   agx_swtnl_lower_for_draw(clone);

   state.ir.nir = clone;
   so->xfb_cpu_vs = draw_create_vertex_shader(ctx->xfb_draw, &state);
   if (!so->xfb_cpu_vs)
      ralloc_free(clone);
   simple_mtx_unlock(&ctx->xfb_cpu_lock);
}

void
agx_xfb_cpu_delete_vs(struct agx_context *ctx, struct agx_uncompiled_shader *so)
{
   simple_mtx_lock(&ctx->xfb_cpu_lock);
   if (so->xfb_cpu_vs && ctx->xfb_draw)
      draw_delete_vertex_shader(ctx->xfb_draw, so->xfb_cpu_vs);

   so->xfb_cpu_vs = NULL;
   simple_mtx_unlock(&ctx->xfb_cpu_lock);
}

/*
 * Can agx_xfb_cpu_draw() capture this draw? Called only when transform
 * feedback is bound and the last vertex stage has outputs to capture.
 */
bool
agx_xfb_cpu_applies(struct agx_context *ctx,
                    const struct pipe_draw_indirect_info *indirect)
{
   struct agx_uncompiled_shader *vs = ctx->stage[MESA_SHADER_VERTEX].shader;

   return agx_chip_g17_m5(agx_device(ctx->base.screen)->chip) && !indirect &&
          vs && vs->xfb_cpu_vs && ctx->attributes &&
          !ctx->stage[MESA_SHADER_GEOMETRY].shader &&
          !ctx->stage[MESA_SHADER_TESS_EVAL].shader;
}

static const void *
map_buffer(struct agx_context *ctx, struct pipe_resource *prsc, unsigned usage,
           struct pipe_transfer **transfer)
{
   return pipe_buffer_map_range(&ctx->base, prsc, 0, prsc->width0, usage,
                                transfer);
}

void
agx_xfb_cpu_draw(struct agx_context *ctx, const struct pipe_draw_info *info,
                 const struct pipe_draw_start_count_bias *draw_info)
{
   simple_mtx_lock(&ctx->xfb_cpu_lock);
   struct draw_context *draw = ctx->xfb_draw;
   struct agx_xfb_render *r = agx_xfb_render(ctx->xfb_render);
   struct agx_uncompiled_shader *vs = ctx->stage[MESA_SHADER_VERTEX].shader;
   struct pipe_transfer *vb_transfers[PIPE_MAX_ATTRIBS] = {0};
   struct pipe_transfer *cb_transfers[PIPE_MAX_CONSTANT_BUFFERS] = {0};
   struct pipe_transfer *so_transfers[PIPE_MAX_SO_BUFFERS] = {0};
   struct pipe_transfer *ib_transfer = NULL;
   struct draw_so_target targets[PIPE_MAX_SO_BUFFERS];
   struct draw_so_target *target_ptrs[PIPE_MAX_SO_BUFFERS] = {0};
   unsigned num_targets = ctx->streamout.num_targets;

   const struct pipe_rasterizer_state rast = {
      .flatshade_first = ctx->rast->base.flatshade_first,
      .half_pixel_center = true,
      .bottom_edge_rule = true,
      .depth_clip_near = true,
      .depth_clip_far = true,
      .rasterizer_discard = true,
      .cull_face = PIPE_FACE_FRONT_AND_BACK,
      .point_size = 1.0,
      .line_width = 1.0,
   };

   draw_set_rasterizer_state(draw, &rast, NULL);
   draw_bind_vertex_shader(draw, vs->xfb_cpu_vs);

   draw_set_vertex_elements(draw, ctx->attributes->num_attribs,
                            ctx->attributes->pipe);

   unsigned nr_vbs = util_last_bit(ctx->vb_mask);
   draw_set_vertex_buffers(draw, nr_vbs, ctx->vertex_buffers);
   for (unsigned i = 0; i < nr_vbs; ++i) {
      const struct pipe_vertex_buffer *vb = &ctx->vertex_buffers[i];
      const void *ptr = NULL;
      size_t size = 0;

      if (vb->is_user_buffer) {
         ptr = vb->buffer.user;
         size = ~0;
      } else if (vb->buffer.resource) {
         ptr = map_buffer(ctx, vb->buffer.resource, PIPE_MAP_READ,
                          &vb_transfers[i]);
         size = vb->buffer.resource->width0;
      }

      draw_set_mapped_vertex_buffer(draw, i, ptr, size);
   }

   const void *indices = NULL;
   if (info->index_size) {
      unsigned available = ~0;
      if (info->has_user_indices) {
         indices = info->index.user;
      } else {
         indices = map_buffer(ctx, info->index.resource, PIPE_MAP_READ,
                              &ib_transfer);
         available = info->index.resource->width0;
      }
      draw_set_indexes(draw, indices, info->index_size, available);
   }

   struct agx_stage *stage = &ctx->stage[MESA_SHADER_VERTEX];
   u_foreach_bit(i, stage->cb_mask) {
      const struct pipe_constant_buffer *cb = &stage->cb[i];
      const uint8_t *ptr = cb->user_buffer;

      if (!ptr && cb->buffer) {
         ptr = map_buffer(ctx, cb->buffer, PIPE_MAP_READ, &cb_transfers[i]);
         if (ptr)
            ptr += cb->buffer_offset;
      }

      draw_set_mapped_constant_buffer(draw, MESA_SHADER_VERTEX, i, ptr,
                                      cb->buffer_size);
   }

   for (unsigned i = 0; i < num_targets; ++i) {
      struct pipe_stream_output_target *tgt = ctx->streamout.targets[i];
      if (!tgt)
         continue;

      struct agx_streamout_target *atgt = agx_so_target(tgt);
      uint32_t offset_B = 0;
      pipe_buffer_read(&ctx->base, atgt->offset, 0, 4, &offset_B);

      targets[i] = (struct draw_so_target){
         .target = *tgt,
         .mapping = (void *)map_buffer(ctx, tgt->buffer,
                                       PIPE_MAP_READ | PIPE_MAP_WRITE,
                                       &so_transfers[i]),
         .internal_offset = offset_B,
      };
      atgt->stride = vs->xfb_strides[i] * 4;
      target_ptrs[i] = &targets[i];
   }

   draw_set_mapped_so_targets(draw, num_targets, target_ptrs);
   draw_collect_primitives_generated(draw, true);

   memset(r->written, 0, sizeof(r->written));
   memset(r->generated, 0, sizeof(r->generated));

   draw_vbo(draw, info, 0, NULL, draw_info, 1, 0);
   draw_flush(draw);

   for (unsigned i = 0; i < num_targets; ++i) {
      if (!target_ptrs[i])
         continue;

      uint32_t offset_B = targets[i].internal_offset;
      tc_buffer_write(&ctx->base, agx_so_target(ctx->streamout.targets[i])->offset,
                      0, 4, &offset_B);
   }

   /* Unbind before unmapping. */
   draw_set_mapped_so_targets(draw, 0, NULL);
   for (unsigned i = 0; i < nr_vbs; ++i)
      draw_set_mapped_vertex_buffer(draw, i, NULL, 0);
   if (indices)
      draw_set_indexes(draw, NULL, 0, 0);

   for (unsigned i = 0; i < ARRAY_SIZE(vb_transfers); ++i) {
      if (vb_transfers[i])
         pipe_buffer_unmap(&ctx->base, vb_transfers[i]);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(cb_transfers); ++i) {
      if (cb_transfers[i])
         pipe_buffer_unmap(&ctx->base, cb_transfers[i]);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(so_transfers); ++i) {
      if (so_transfers[i])
         pipe_buffer_unmap(&ctx->base, so_transfers[i]);
   }
   if (ib_transfer)
      pipe_buffer_unmap(&ctx->base, ib_transfer);

   /* The capture's queries: primitives written per stream, and overflow when
    * fewer were written than generated. */
   if (ctx->active_queries) {
      for (unsigned s = 0; s < ARRAY_SIZE(ctx->tf_prims_generated); ++s) {
         agx_query_increment_cpu(ctx, ctx->tf_prims_generated[s], r->written[s]);
         if (r->written[s] < r->generated[s]) {
            agx_query_increment_cpu(ctx, ctx->tf_overflow[s], 1);
            agx_query_increment_cpu(ctx, ctx->tf_any_overflow, 1);
         }
      }
   }
   simple_mtx_unlock(&ctx->xfb_cpu_lock);
}
