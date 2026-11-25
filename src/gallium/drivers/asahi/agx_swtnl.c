/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Geometry and tessellation stages on the CPU, for G17G.
 *
 * G17G cannot encode the geometry-assisted draw that poly builds (the
 * rasterization program and its draw), and it has no precompiled
 * tessellation kernels, so draws with either stage were dropped. Run every
 * stage before rasterization on the CPU with Gallium's draw module, and draw
 * its output vertices on the GPU through a vertex shader that passes them on
 * to the fragment shader. Transform feedback from the last stage is written
 * by the draw module's stream-output stage on the way.
 *
 * Tessellation runs here when the draw module can tessellate (an LLVM
 * build), otherwise on the GPU (agx_draw_patches, agx_m5_gpu_tess()).
 */

#include <math.h>
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "gallium/auxiliary/nir/pipe_nir.h"
#include "draw/draw_context.h"
#include "draw/draw_nir.h"
#include "draw/draw_private.h"
#include "draw/draw_vbuf.h"
#include "draw/draw_vertex.h"
#include "nir/nir_to_tgsi.h"
#include "tgsi/tgsi_exec.h"
#include "tgsi/tgsi_from_mesa.h"
#include "util/format/u_format.h"
#include "util/half_float.h"
#include "util/hash_table.h"
#include "util/u_inlines.h"
#include "util/u_math.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "util/u_dynarray.h"
#include "util/u_framebuffer.h"
#include "util/u_upload_mgr.h"
#include "agx_state.h"

/* Textures of the CPU stages: a linear copy of every level and layer the
 * view covers, read through the transfer path (which detiles), in the
 * layout the draw module samples from. */
struct swtnl_texture {
   uint8_t *data;
   enum pipe_texture_target target;
   enum pipe_format format;
   unsigned width0, height0, depth0;
   unsigned first_level, last_level;
   unsigned layers;
   unsigned buf_elements;
   uint32_t row_stride[PIPE_MAX_TEXTURE_LEVELS];
   uint32_t img_stride[PIPE_MAX_TEXTURE_LEVELS];
   uint32_t mip_offsets[PIPE_MAX_TEXTURE_LEVELS];
   uint8_t swizzle[4];
   bool bound;
};

struct swtnl_tgsi_buffer {
   struct tgsi_buffer base;
   void *ptrs[PIPE_MAX_SHADER_BUFFERS];
   uint32_t sizes[PIPE_MAX_SHADER_BUFFERS];
};

struct swtnl_tgsi_sampler {
   struct tgsi_sampler base;
   const struct swtnl_texture *textures;
   unsigned nr_textures;
   const struct pipe_sampler_state *samplers[PIPE_MAX_SAMPLERS];
   unsigned nr_samplers;
};

struct agx_swtnl_render {
   struct vbuf_render base;
   struct agx_context *ctx;
   struct vertex_info vinfo;

   /* TGSI interpreter hooks used when the draw module runs without LLVM.
    * Index 0 is the vertex stage, index 1 is the geometry stage. */
   struct swtnl_tgsi_buffer tgsi_buffers[2];
   struct swtnl_tgsi_sampler tgsi_samplers[2];
   /* Per-stage detiled texture descriptors kept off the draw stack. */
   struct swtnl_texture textures[MESA_SHADER_FRAGMENT][PIPE_MAX_SHADER_SAMPLER_VIEWS];

   /* What the GPU's vertex shader receives after the position */
   uint64_t slots;
   bool psiz;

   /* With a layered framebuffer: the attribute after all of those holding
    * the last stage's gl_Layer, and its byte offset in the vertex; -1 when
    * the stage writes none (every primitive goes to layer 0). */
   int layer_attrib;
   unsigned layer_offset_B;

   void *vertices;
   size_t size;
   uint16_t vertex_size, nr_vertices;
   enum mesa_prim prim;

   /* What the draw produced, drawn on the GPU in one go at its end: the
    * draw module hands its output over a primitive or a patch at a time. */
   struct util_dynarray acc_vertices, acc_indices;
   unsigned acc_count, cur_base;
   enum mesa_prim acc_prim;

   uint64_t written[PIPE_MAX_VERTEX_STREAMS];
   uint64_t generated[PIPE_MAX_VERTEX_STREAMS];
};

static struct agx_swtnl_render *
agx_swtnl_render(struct vbuf_render *render)
{
   return (struct agx_swtnl_render *)render;
}

static const struct vertex_info *
swtnl_get_vertex_info(struct vbuf_render *render)
{
   return &agx_swtnl_render(render)->vinfo;
}

static bool
swtnl_allocate_vertices(struct vbuf_render *render, uint16_t vertex_size,
                        uint16_t nr_vertices)
{
   struct agx_swtnl_render *r = agx_swtnl_render(render);
   size_t size = (size_t)vertex_size * nr_vertices;

   if (size > r->size) {
      free(r->vertices);
      r->vertices = malloc(size);
      r->size = r->vertices ? size : 0;
   }

   r->vertex_size = vertex_size;
   r->nr_vertices = nr_vertices;
   return r->vertices != NULL;
}

static void *
swtnl_map_vertices(struct vbuf_render *render)
{
   return agx_swtnl_render(render)->vertices;
}

static void
swtnl_unmap_vertices(struct vbuf_render *render, uint16_t min_index,
                     uint16_t max_index)
{
   struct agx_swtnl_render *r = agx_swtnl_render(render);

   r->cur_base = r->acc_count;
   util_dynarray_append_array(&r->acc_vertices, uint8_t, r->vertices,
                              (size_t)r->vertex_size * r->nr_vertices);
   r->acc_count += r->nr_vertices;
}

static void
swtnl_set_primitive(struct vbuf_render *render, enum mesa_prim prim)
{
   agx_swtnl_render(render)->prim = prim;
}

static void
swtnl_release_vertices(struct vbuf_render *render)
{
}

static void
swtnl_set_stream_output_info(struct vbuf_render *render, unsigned stream,
                             unsigned primitive_count,
                             unsigned primitive_generated)
{
   struct agx_swtnl_render *r = agx_swtnl_render(render);

   r->written[stream] += primitive_count;
   r->generated[stream] += primitive_generated;
}

static void
swtnl_destroy(struct vbuf_render *render)
{
   struct agx_swtnl_render *r = agx_swtnl_render(render);

   util_dynarray_fini(&r->acc_vertices);
   util_dynarray_fini(&r->acc_indices);
   free(r->vertices);
   FREE(r);
}

/* The vertex shader that hands the CPU's output vertices to the fragment
 * shader: attribute 0 is the position, then one vec4 per varying slot, then
 * the point size. */
static void *
agx_swtnl_passthrough_vs(struct agx_context *ctx, uint64_t slots, bool psiz)
{
   struct agx_uncompiled_shader *fs = ctx->stage[MESA_SHADER_FRAGMENT].shader;
   const uint64_t flat = fs ? fs->info.inputs_flat_shaded : 0;
   const uint64_t linear = fs ? fs->info.inputs_linear_shaded : 0;

   /* slots and psiz identify the layout; flat and linear the linkage */
   const uint64_t key = _mesa_hash_data(&(uint64_t[]){slots, flat, linear, psiz},
                                        4 * sizeof(uint64_t));
   if (!ctx->swtnl_vs)
      ctx->swtnl_vs = _mesa_hash_table_u64_create(NULL);

   void *cso = _mesa_hash_table_u64_search(ctx->swtnl_vs, key);
   if (cso)
      return cso;

   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                  &agx_nir_options,
                                                  "G17G CPU vertex passthrough");
   /* The order of the emitted vertex: position, the varyings, point size. */
   gl_varying_slot order[2 + 64];
   unsigned nr = 0;
   order[nr++] = VARYING_SLOT_POS;
   u_foreach_bit64(slot, slots)
      order[nr++] = slot;
   if (psiz)
      order[nr++] = VARYING_SLOT_PSIZ;

   for (unsigned attrib = 0; attrib < nr; ++attrib) {
      const gl_varying_slot slot = order[attrib];
      const bool scalar = slot == VARYING_SLOT_PSIZ;
      const struct glsl_type *type =
         scalar ? glsl_float_type() : glsl_vec4_type();

      nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, type,
                                             "in");
      in->data.location = VERT_ATTRIB_GENERIC0 + attrib;
      in->data.driver_location = attrib;

      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out,
                                              type, "out");
      out->data.location = slot;
      out->data.driver_location = attrib;

      nir_store_var(&b, out, nir_load_var(&b, in), scalar ? 0x1 : 0xf);
   }

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   b.shader->info.next_stage = MESA_SHADER_FRAGMENT;
   b.shader->info.known_interpolation_qualifiers = true;
   b.shader->info.linear_varyings = linear & slots;
   b.shader->info.perspective_varyings = slots & ~flat & ~linear;

   cso = pipe_shader_from_nir(&ctx->base, b.shader);
   if (cso)
      _mesa_hash_table_u64_insert(ctx->swtnl_vs, key, cso);
   return cso;
}

/* Draw an index list over the vertices bound by agx_swtnl_flush(). */
static void
swtnl_draw_list(struct agx_swtnl_render *r, const uint32_t *indices,
                unsigned count, unsigned nr_vertices)
{
   struct pipe_context *pctx = &r->ctx->base;
   struct pipe_resource *ib_rsrc = NULL;
   unsigned ib_offset = 0;
   u_upload_data_ref(pctx->stream_uploader, 0, count * sizeof(uint32_t), 4,
                     indices, &ib_offset, &ib_rsrc);
   u_upload_unmap(pctx->stream_uploader);
   if (!ib_rsrc)
      return;

   const struct pipe_draw_info info = {
      .mode = r->acc_prim,
      .index_size = 4,
      .instance_count = 1,
      .index.resource = ib_rsrc,
      .max_index = nr_vertices - 1,
   };
   const struct pipe_draw_start_count_bias draw = {
      .start = ib_offset / sizeof(uint32_t),
      .count = count,
   };

   /* At most AGX_G17G_RENDER_PRIMS primitives per draw, so the per-render
    * budget in agx_draw_vbo() can split a large output across renders. */
   const unsigned per_prim = mesa_vertices_per_prim(r->acc_prim);
   const unsigned chunk = AGX_G17G_RENDER_PRIMS * per_prim;
   for (unsigned first = 0; first < count; first += chunk) {
      struct pipe_draw_start_count_bias part = draw;
      part.start += first;
      part.count = MIN2(chunk, count - first);
      pctx->draw_vbo(pctx, &info, 0, NULL, &part, 1);
   }
   pipe_resource_reference(&ib_rsrc, NULL);
}

static void
agx_swtnl_flush(struct agx_swtnl_render *r)
{
   const uint32_t *indices = r->acc_indices.data;
   const unsigned count = r->acc_indices.size / sizeof(uint32_t);
   const unsigned nr_vertices = r->acc_count;

   struct agx_context *ctx = r->ctx;
   struct pipe_context *pctx = &ctx->base;

   if (!count || ctx->rast->base.rasterizer_discard)
      goto reset;

   void *vs = agx_swtnl_passthrough_vs(ctx, r->slots, r->psiz);
   if (!vs)
      goto reset;

   struct pipe_resource *vb_rsrc = NULL;
   unsigned vb_offset = 0;
   u_upload_data_ref(pctx->stream_uploader, 0, r->acc_vertices.size, 16,
                     r->acc_vertices.data, &vb_offset, &vb_rsrc);
   u_upload_unmap(pctx->stream_uploader);
   if (!vb_rsrc)
      goto out;

   struct pipe_vertex_element elements[PIPE_MAX_ATTRIBS];
   /* The layer attribute is the CPU's alone: the passthrough never reads it. */
   const unsigned nr_attribs =
      r->vinfo.num_attribs - (r->layer_attrib >= 0 ? 1 : 0);
   for (unsigned i = 0; i < nr_attribs; ++i) {
      elements[i] = (struct pipe_vertex_element){
         .src_offset = i * 16,
         .src_format = r->psiz && i == nr_attribs - 1
                          ? PIPE_FORMAT_R32_FLOAT
                          : PIPE_FORMAT_R32G32B32A32_FLOAT,
         .src_stride = r->vertex_size,
      };
   }
   void *velems = pctx->create_vertex_elements_state(pctx, nr_attribs, elements);

   /* Swap the application's vertex pipeline for the passthrough. */
   struct agx_uncompiled_shader *saved[MESA_SHADER_FRAGMENT];
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s)
      saved[s] = ctx->stage[s].shader;
   void *saved_velems = ctx->attributes;
   struct pipe_vertex_buffer saved_vbs[PIPE_MAX_ATTRIBS] = {0};
   const unsigned nr_vbs = util_last_bit(ctx->vb_mask);
   for (unsigned i = 0; i < nr_vbs; ++i)
      pipe_vertex_buffer_reference(&saved_vbs[i], &ctx->vertex_buffers[i]);
   const unsigned saved_targets = ctx->streamout.num_targets;

   pctx->bind_tcs_state(pctx, NULL);
   pctx->bind_tes_state(pctx, NULL);
   pctx->bind_gs_state(pctx, NULL);
   pctx->bind_vs_state(pctx, vs);
   pctx->bind_vertex_elements_state(pctx, velems);
   const struct pipe_vertex_buffer vb = {
      .buffer.resource = vb_rsrc,
      .buffer_offset = vb_offset,
   };
   pctx->set_vertex_buffers(pctx, 1, &vb);
   ctx->streamout.num_targets = 0;

   ctx->swtnl_inner = true;
   const struct pipe_framebuffer_state *layered = ctx->g17g_layered_fb;
   if (!layered) {
      swtnl_draw_list(r, indices, count, nr_vertices);
   } else if (r->layer_attrib < 0) {
      agx_g17g_bind_layer(ctx, layered, 0);
      swtnl_draw_list(r, indices, count, nr_vertices);
   } else {
      /* Each primitive goes to the layer its provoking vertex names. */
      const unsigned per_prim = mesa_vertices_per_prim(r->acc_prim);
      const unsigned provoking =
         ctx->rast && ctx->rast->base.flatshade_first ? 0 : per_prim - 1;
      const unsigned layers = util_framebuffer_get_num_layers(layered);
      const uint8_t *vertices = r->acc_vertices.data;
      uint32_t *subset = malloc(count * sizeof(uint32_t));
      for (unsigned layer = 0; subset && layer < layers; ++layer) {
         unsigned n = 0;
         for (unsigned p = 0; p + per_prim <= count; p += per_prim) {
            uint32_t value;
            memcpy(&value,
                   vertices + (size_t)indices[p + provoking] * r->vertex_size +
                      r->layer_offset_B,
                   sizeof(value));
            if (value != layer)
               continue;
            memcpy(subset + n, indices + p, per_prim * sizeof(uint32_t));
            n += per_prim;
         }
         if (!n)
            continue;
         agx_g17g_bind_layer(ctx, layered, layer);
         swtnl_draw_list(r, subset, n, nr_vertices);
      }
      free(subset);
   }
   ctx->swtnl_inner = false;

   ctx->streamout.num_targets = saved_targets;
   pctx->set_vertex_buffers(pctx, nr_vbs, saved_vbs);
   for (unsigned i = 0; i < nr_vbs; ++i)
      pipe_vertex_buffer_unreference(&saved_vbs[i]);
   pctx->bind_vertex_elements_state(pctx, saved_velems);
   pctx->delete_vertex_elements_state(pctx, velems);
   pctx->bind_vs_state(pctx, saved[MESA_SHADER_VERTEX]);
   pctx->bind_tcs_state(pctx, saved[MESA_SHADER_TESS_CTRL]);
   pctx->bind_tes_state(pctx, saved[MESA_SHADER_TESS_EVAL]);
   pctx->bind_gs_state(pctx, saved[MESA_SHADER_GEOMETRY]);

out:
   pipe_resource_reference(&vb_rsrc, NULL);
reset:
   util_dynarray_clear(&r->acc_vertices);
   util_dynarray_clear(&r->acc_indices);
   r->acc_count = r->cur_base = 0;
}


/* The list type a run of this primitive type is collected as: runs are
 * concatenated into one indexed draw, and concatenated strips, fans and
 * loops would join (a geometry shader's strips, emitted one call each,
 * drew the triangles between them). */
static enum mesa_prim
swtnl_list_prim(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_LINE_STRIP:
   case MESA_PRIM_LINE_LOOP:
      return MESA_PRIM_LINES;
   case MESA_PRIM_TRIANGLE_STRIP:
   case MESA_PRIM_TRIANGLE_FAN:
      return MESA_PRIM_TRIANGLES;
   default:
      return prim;
   }
}

static void
swtnl_push(struct agx_swtnl_render *r, uint32_t index)
{
   uint32_t *out = util_dynarray_grow(&r->acc_indices, uint32_t, 1);
   if (out)
      *out = index;
}

/* Queue a run of the current vertex buffer's vertices for the GPU. */
static void
swtnl_collect(struct agx_swtnl_render *r, const uint16_t *indices,
              unsigned start, unsigned count)
{
   if (!count)
      return;

   const enum mesa_prim list = swtnl_list_prim(r->prim);

   /* One GPU draw per primitive type: start over with this buffer. */
   if (r->acc_indices.size && r->acc_prim != list) {
      agx_swtnl_flush(r);
      util_dynarray_append_array(&r->acc_vertices, uint8_t, r->vertices,
                                 (size_t)r->vertex_size * r->nr_vertices);
      r->acc_count = r->nr_vertices;
   }
   r->acc_prim = list;

#define V(i) (r->cur_base + (indices ? indices[(i)] : start + (i)))
   if (list == r->prim) {
      uint32_t *out = util_dynarray_grow(&r->acc_indices, uint32_t, count);
      if (!out)
         return;
      for (unsigned i = 0; i < count; ++i)
         out[i] = V(i);
      return;
   }

   /* Decompose, keeping the provoking vertex of each primitive. */
   const bool first = r->ctx->rast && r->ctx->rast->base.flatshade_first;
   switch (r->prim) {
   case MESA_PRIM_LINE_STRIP:
   case MESA_PRIM_LINE_LOOP:
      for (unsigned i = 0; i + 1 < count; ++i) {
         swtnl_push(r, V(i));
         swtnl_push(r, V(i + 1));
      }
      if (r->prim == MESA_PRIM_LINE_LOOP && count > 1) {
         swtnl_push(r, V(count - 1));
         swtnl_push(r, V(0));
      }
      break;
   case MESA_PRIM_TRIANGLE_STRIP:
      for (unsigned i = 0; i + 2 < count; ++i) {
         const unsigned a = i, b = i + 1, c = i + 2;
         const bool odd = i & 1;
         const unsigned tri[3] = {
            odd ? (first ? a : b) : a,
            odd ? (first ? c : a) : b,
            odd ? (first ? b : c) : c,
         };
         for (unsigned k = 0; k < 3; ++k)
            swtnl_push(r, V(tri[k]));
      }
      break;
   case MESA_PRIM_TRIANGLE_FAN:
      for (unsigned i = 0; i + 2 < count; ++i) {
         const unsigned tri[3] = {
            first ? i + 1 : 0,
            first ? i + 2 : i + 1,
            first ? 0 : i + 2,
         };
         for (unsigned k = 0; k < 3; ++k)
            swtnl_push(r, V(tri[k]));
      }
      break;
   default:
      break;
   }
#undef V
}

static void
swtnl_draw_elements(struct vbuf_render *render, const uint16_t *indices,
                    unsigned count)
{
   swtnl_collect(agx_swtnl_render(render), indices, 0, count);
}

static void
swtnl_draw_arrays(struct vbuf_render *render, unsigned start, unsigned nr)
{
   swtnl_collect(agx_swtnl_render(render), NULL, start, nr);
}

static void *
swtnl_tgsi_buffer_lookup(const struct tgsi_buffer *buffer,
                         uint32_t unit, uint32_t *size)
{
   const struct swtnl_tgsi_buffer *buf =
      (const struct swtnl_tgsi_buffer *)buffer;

   if (unit >= PIPE_MAX_SHADER_BUFFERS || !buf->ptrs[unit]) {
      *size = 0;
      return NULL;
   }

   *size = buf->sizes[unit];
   return buf->ptrs[unit];
}

static void
swtnl_apply_swizzle(const uint8_t swz[4], bool is_int,
                    const uint32_t in[4], uint32_t out[4])
{
   for (unsigned c = 0; c < 4; ++c) {
      switch (swz[c]) {
      case PIPE_SWIZZLE_X: out[c] = in[0]; break;
      case PIPE_SWIZZLE_Y: out[c] = in[1]; break;
      case PIPE_SWIZZLE_Z: out[c] = in[2]; break;
      case PIPE_SWIZZLE_W: out[c] = in[3]; break;
      case PIPE_SWIZZLE_1: out[c] = is_int ? 1u : fui(1.0f); break;
      default:             out[c] = 0u; break;
      }
   }
}

static void
swtnl_tgsi_get_dims(struct tgsi_sampler *sampler,
                    const unsigned sview_index, int level, int dims[4])
{
   const struct swtnl_tgsi_sampler *ts =
      (const struct swtnl_tgsi_sampler *)sampler;

   dims[0] = dims[1] = dims[2] = dims[3] = 0;
   if (!ts->textures || sview_index >= ts->nr_textures)
      return;

   const struct swtnl_texture *t = &ts->textures[sview_index];
   if (!t->data)
      return;

   if (t->target == PIPE_BUFFER) {
      dims[0] = t->buf_elements;
      return;
   }

   dims[3] = t->last_level - t->first_level + 1;
   if (level < 0)
      return;

   unsigned lvl = t->first_level + (unsigned)level;
   if (lvl > t->last_level)
      return;

   dims[0] = u_minify(t->width0, lvl);
   switch (t->target) {
   case PIPE_TEXTURE_1D:
      break;
   case PIPE_TEXTURE_1D_ARRAY:
      dims[1] = t->layers;
      break;
   case PIPE_TEXTURE_2D:
   case PIPE_TEXTURE_RECT:
   case PIPE_TEXTURE_CUBE:
      dims[1] = u_minify(t->height0, lvl);
      break;
   case PIPE_TEXTURE_2D_ARRAY:
      dims[1] = u_minify(t->height0, lvl);
      dims[2] = t->layers;
      break;
   case PIPE_TEXTURE_CUBE_ARRAY:
      dims[1] = u_minify(t->height0, lvl);
      dims[2] = t->layers / 6;
      break;
   case PIPE_TEXTURE_3D:
      dims[1] = u_minify(t->height0, lvl);
      dims[2] = u_minify(t->depth0, lvl);
      break;
   default:
      break;
   }
}

static bool
swtnl_fetch_raw_texel(const struct swtnl_texture *t, unsigned lvl,
                      int x, int y, int z, uint32_t raw[4])
{
   if (lvl < t->first_level || lvl > t->last_level)
      return false;

   const int w = u_minify(t->width0, lvl);
   const int h = u_minify(t->height0, lvl);
   const int d = t->target == PIPE_TEXTURE_3D ? (int)u_minify(t->depth0, lvl)
                                              : (int)t->layers;
   if (x < 0 || x >= w || y < 0 || y >= h || z < 0 || z >= d)
      return false;

   const uint8_t *slice =
      t->data + t->mip_offsets[lvl] + (size_t)t->img_stride[lvl] * z;
   if (util_format_is_depth_or_stencil(t->format)) {
      const struct util_format_description *desc =
         util_format_description(t->format);
      const unsigned bs = util_format_get_blocksize(t->format);
      const uint8_t *texel =
         slice + (size_t)t->row_stride[lvl] * y + (size_t)x * bs;
      if (util_format_has_depth(desc)) {
         float zval = 0.0f;
         util_format_unpack_z_float(t->format, &zval, texel, 1);
         raw[0] = raw[1] = raw[2] = fui(zval);
         raw[3] = fui(1.0f);
      } else {
         uint8_t sval = 0;
         util_format_unpack_s_8uint(t->format, &sval, texel, 1);
         raw[0] = sval;
         raw[1] = raw[2] = 0;
         raw[3] = 1;
      }
      return true;
   }

   util_format_read_4(t->format, raw, 16, slice, t->row_stride[lvl],
                      x, y, 1, 1);
   return true;
}

static void
swtnl_tgsi_get_texel(struct tgsi_sampler *sampler,
                     const unsigned sview_index,
                     const int ci[TGSI_QUAD_SIZE],
                     const int cj[TGSI_QUAD_SIZE],
                     const int ck[TGSI_QUAD_SIZE],
                     const int lod[TGSI_QUAD_SIZE],
                     const int8_t offset[3],
                     float rgba[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE])
{
   const struct swtnl_tgsi_sampler *ts =
      (const struct swtnl_tgsi_sampler *)sampler;

   memset(rgba, 0, sizeof(float) * TGSI_NUM_CHANNELS * TGSI_QUAD_SIZE);
   if (!ts->textures || sview_index >= ts->nr_textures)
      return;

   const struct swtnl_texture *t = &ts->textures[sview_index];
   if (!t->data)
      return;

   const bool is_int = util_format_is_pure_integer(t->format);
   const unsigned bs = util_format_get_blocksize(t->format);

   for (unsigned q = 0; q < TGSI_QUAD_SIZE; ++q) {
      uint32_t raw[4] = {0}, swz[4] = {0};
      bool ok = false;

      if (t->target == PIPE_BUFFER) {
         int idx = ci[q];
         if (idx >= 0 && (unsigned)idx < t->buf_elements) {
            util_format_read_4(t->format, raw, 16, t->data + (size_t)idx * bs,
                               bs, 0, 0, 1, 1);
            ok = true;
         }
      } else {
         int rel_lod = t->target == PIPE_TEXTURE_RECT ? 0 : lod[q];
         if (rel_lod >= 0) {
            unsigned lvl = t->first_level + (unsigned)rel_lod;
            int x = ci[q] + offset[0];
            int y = 0, z = 0;
            if (t->target == PIPE_TEXTURE_1D_ARRAY) {
               z = cj[q];
            } else if (t->target == PIPE_TEXTURE_2D ||
                       t->target == PIPE_TEXTURE_RECT) {
               y = cj[q] + offset[1];
            } else if (t->target == PIPE_TEXTURE_2D_ARRAY) {
               y = cj[q] + offset[1];
               z = ck[q];
            } else if (t->target == PIPE_TEXTURE_3D) {
               y = cj[q] + offset[1];
               z = ck[q] + offset[2];
            }
            ok = swtnl_fetch_raw_texel(t, lvl, x, y, z, raw);
         }
      }

      if (ok) {
         swtnl_apply_swizzle(t->swizzle, is_int, raw, swz);
         for (unsigned c = 0; c < 4; ++c)
            memcpy(&rgba[c][q], &swz[c], sizeof(uint32_t));
      }
   }
}

static int
swtnl_wrap_int(unsigned wrap, int idx, int size, bool *border)
{
   *border = false;
   if (size <= 0) {
      *border = true;
      return 0;
   }

   switch (wrap) {
   case PIPE_TEX_WRAP_REPEAT: {
      int r = idx % size;
      return r < 0 ? r + size : r;
   }
   case PIPE_TEX_WRAP_MIRROR_REPEAT: {
      int p = 2 * size;
      int r = idx % p;
      if (r < 0)
         r += p;
      return r >= size ? p - 1 - r : r;
   }
   case PIPE_TEX_WRAP_CLAMP:
   case PIPE_TEX_WRAP_CLAMP_TO_EDGE:
      return CLAMP(idx, 0, size - 1);
   case PIPE_TEX_WRAP_CLAMP_TO_BORDER:
      if (idx < 0 || idx >= size) {
         *border = true;
         return 0;
      }
      return idx;
   case PIPE_TEX_WRAP_MIRROR_CLAMP:
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_EDGE: {
      int a = idx < 0 ? -idx - 1 : idx;
      return CLAMP(a, 0, size - 1);
   }
   case PIPE_TEX_WRAP_MIRROR_CLAMP_TO_BORDER: {
      int a = idx < 0 ? -idx - 1 : idx;
      if (a >= size) {
         *border = true;
         return 0;
      }
      return a;
   }
   default:
      return CLAMP(idx, 0, size - 1);
   }
}

static float
swtnl_shadow_compare(unsigned func, float ref, float tex)
{
   switch (func) {
   case PIPE_FUNC_NEVER:    return 0.0f;
   case PIPE_FUNC_LESS:     return ref < tex ? 1.0f : 0.0f;
   case PIPE_FUNC_EQUAL:    return ref == tex ? 1.0f : 0.0f;
   case PIPE_FUNC_LEQUAL:   return ref <= tex ? 1.0f : 0.0f;
   case PIPE_FUNC_GREATER:  return ref > tex ? 1.0f : 0.0f;
   case PIPE_FUNC_NOTEQUAL: return ref != tex ? 1.0f : 0.0f;
   case PIPE_FUNC_GEQUAL:   return ref >= tex ? 1.0f : 0.0f;
   case PIPE_FUNC_ALWAYS:   return 1.0f;
   default:                 return 0.0f;
   }
}

static void
swtnl_cube_face(float rx, float ry, float rz, float *sc, float *tc, int *face)
{
   const float ax = fabsf(rx), ay = fabsf(ry), az = fabsf(rz);
   float uc, vc, ma;

   if (ax >= ay && ax >= az) {
      ma = ax;
      if (rx >= 0.0f) {
         *face = 0; uc = -rz; vc = -ry;
      } else {
         *face = 1; uc = rz; vc = -ry;
      }
   } else if (ay >= ax && ay >= az) {
      ma = ay;
      if (ry >= 0.0f) {
         *face = 2; uc = rx; vc = rz;
      } else {
         *face = 3; uc = rx; vc = -rz;
      }
   } else {
      ma = az;
      if (rz >= 0.0f) {
         *face = 4; uc = rx; vc = -ry;
      } else {
         *face = 5; uc = -rx; vc = -ry;
      }
   }

   if (ma > 0.0f) {
      *sc = 0.5f * (uc / ma + 1.0f);
      *tc = 0.5f * (vc / ma + 1.0f);
   } else {
      *sc = *tc = 0.5f;
   }
}

static void
swtnl_sample_texel(const struct swtnl_texture *t,
                   const struct pipe_sampler_state *samp,
                   unsigned lvl, int x, int y, int z, bool border,
                   uint32_t out[4])
{
   const bool is_int = util_format_is_pure_integer(t->format);
   uint32_t raw[4] = {0};

   if (border || !swtnl_fetch_raw_texel(t, lvl, x, y, z, raw)) {
      for (unsigned c = 0; c < 4; ++c) {
         if (is_int) {
            out[c] = samp->border_color.ui[c];
         } else {
            float v = samp->border_color.f[c];
            if (util_format_is_unorm(t->format))
               v = CLAMP(v, 0.0f, 1.0f);
            else if (util_format_is_snorm(t->format))
               v = CLAMP(v, -1.0f, 1.0f);
            out[c] = fui(v);
         }
      }
      return;
   }

   swtnl_apply_swizzle(t->swizzle, is_int, raw, out);
}

static void
swtnl_sample_level(const struct swtnl_texture *tex,
                   const struct pipe_sampler_state *samp,
                   unsigned lvl, float sc, float tc, float rc, int layer,
                   const int8_t offset[3], bool unnorm, bool shadow,
                   bool linear, float ref, unsigned wrap_s, unsigned wrap_t,
                   unsigned wrap_r, uint32_t out[4])
{
   const int w = u_minify(tex->width0, lvl);
   const int h = u_minify(tex->height0, lvl);
   const int d = tex->target == PIPE_TEXTURE_3D ? (int)u_minify(tex->depth0, lvl)
                                                : (int)tex->layers;

   if (!linear) {
      int ix = (int)floorf(unnorm ? sc : sc * w) + offset[0];
      int iy = (tex->target == PIPE_TEXTURE_1D ||
                tex->target == PIPE_TEXTURE_1D_ARRAY)
                  ? 0
                  : (int)floorf(unnorm ? tc : tc * h) + offset[1];
      int iz = tex->target == PIPE_TEXTURE_3D
                  ? (int)floorf(unnorm ? rc : rc * d) + offset[2]
                  : layer;
      bool bx = false, by = false, bz = false;
      int wx = swtnl_wrap_int(wrap_s, ix, w, &bx);
      int wy = (tex->target == PIPE_TEXTURE_1D ||
                tex->target == PIPE_TEXTURE_1D_ARRAY)
                  ? 0
                  : swtnl_wrap_int(wrap_t, iy, h, &by);
      int wz = tex->target == PIPE_TEXTURE_3D
                  ? swtnl_wrap_int(wrap_r, iz, d, &bz)
                  : iz;
      uint32_t tx[4];
      swtnl_sample_texel(tex, samp, lvl, wx, wy, wz, bx || by || bz, tx);
      if (shadow) {
         float v = swtnl_shadow_compare(samp->compare_func, ref, uif(tx[0]));
         out[0] = out[1] = out[2] = fui(v);
         out[3] = fui(1.0f);
      } else {
         memcpy(out, tx, sizeof(tx));
      }
      return;
   }

   float u = (unnorm ? sc : sc * w) - 0.5f;
   int ix0 = (int)floorf(u);
   float xw = u - (float)ix0;
   bool bx0, bx1;
   int wx0 = swtnl_wrap_int(wrap_s, ix0 + offset[0], w, &bx0);
   int wx1 = swtnl_wrap_int(wrap_s, ix0 + 1 + offset[0], w, &bx1);

   if (tex->target == PIPE_TEXTURE_1D ||
       tex->target == PIPE_TEXTURE_1D_ARRAY) {
      uint32_t t0[4], t1[4];
      swtnl_sample_texel(tex, samp, lvl, wx0, 0, layer, bx0, t0);
      swtnl_sample_texel(tex, samp, lvl, wx1, 0, layer, bx1, t1);
      if (shadow) {
         float s0 = swtnl_shadow_compare(samp->compare_func, ref, uif(t0[0]));
         float s1 = swtnl_shadow_compare(samp->compare_func, ref, uif(t1[0]));
         float v = s0 * (1.0f - xw) + s1 * xw;
         out[0] = out[1] = out[2] = fui(v);
         out[3] = fui(1.0f);
      } else {
         for (unsigned c = 0; c < 4; ++c)
            out[c] = fui(uif(t0[c]) * (1.0f - xw) + uif(t1[c]) * xw);
      }
      return;
   }

   float v = (unnorm ? tc : tc * h) - 0.5f;
   int iy0 = (int)floorf(v);
   float yw = v - (float)iy0;
   bool by0, by1;
   int wy0 = swtnl_wrap_int(wrap_t, iy0 + offset[1], h, &by0);
   int wy1 = swtnl_wrap_int(wrap_t, iy0 + 1 + offset[1], h, &by1);

   int wz0 = layer, wz1 = layer;
   float zw = 0.0f;
   bool bz0 = false, bz1 = false;
   const bool is_3d = tex->target == PIPE_TEXTURE_3D;
   if (is_3d) {
      float p = (unnorm ? rc : rc * d) - 0.5f;
      int iz0 = (int)floorf(p);
      zw = p - (float)iz0;
      wz0 = swtnl_wrap_int(wrap_r, iz0 + offset[2], d, &bz0);
      wz1 = swtnl_wrap_int(wrap_r, iz0 + 1 + offset[2], d, &bz1);
   }

   float acc[4] = {0};
   for (unsigned slice = 0; slice < (is_3d ? 2u : 1u); ++slice) {
      int wz = slice ? wz1 : wz0;
      bool bz = slice ? bz1 : bz0;
      float sw = is_3d ? (slice ? zw : 1.0f - zw) : 1.0f;
      uint32_t t00[4], t10[4], t01[4], t11[4];
      swtnl_sample_texel(tex, samp, lvl, wx0, wy0, wz, bx0 || by0 || bz, t00);
      swtnl_sample_texel(tex, samp, lvl, wx1, wy0, wz, bx1 || by0 || bz, t10);
      swtnl_sample_texel(tex, samp, lvl, wx0, wy1, wz, bx0 || by1 || bz, t01);
      swtnl_sample_texel(tex, samp, lvl, wx1, wy1, wz, bx1 || by1 || bz, t11);
      if (shadow) {
         float s00 = swtnl_shadow_compare(samp->compare_func, ref, uif(t00[0]));
         float s10 = swtnl_shadow_compare(samp->compare_func, ref, uif(t10[0]));
         float s01 = swtnl_shadow_compare(samp->compare_func, ref, uif(t01[0]));
         float s11 = swtnl_shadow_compare(samp->compare_func, ref, uif(t11[0]));
         float top = s00 * (1.0f - xw) + s10 * xw;
         float bot = s01 * (1.0f - xw) + s11 * xw;
         acc[0] += sw * (top * (1.0f - yw) + bot * yw);
      } else {
         for (unsigned c = 0; c < 4; ++c) {
            float top = uif(t00[c]) * (1.0f - xw) + uif(t10[c]) * xw;
            float bot = uif(t01[c]) * (1.0f - xw) + uif(t11[c]) * xw;
            acc[c] += sw * (top * (1.0f - yw) + bot * yw);
         }
      }
   }
   if (shadow) {
      out[0] = out[1] = out[2] = fui(acc[0]);
      out[3] = fui(1.0f);
   } else {
      for (unsigned c = 0; c < 4; ++c)
         out[c] = fui(acc[c]);
   }
}

static void
swtnl_tgsi_get_samples(struct tgsi_sampler *sampler,
                       const unsigned sview_index,
                       const unsigned sampler_index,
                       const float s[TGSI_QUAD_SIZE],
                       const float t_coord[TGSI_QUAD_SIZE],
                       const float r[TGSI_QUAD_SIZE],
                       const float c0[TGSI_QUAD_SIZE],
                       const float c1[TGSI_QUAD_SIZE],
                       UNUSED float derivs[3][2][TGSI_QUAD_SIZE],
                       const int8_t offset[3],
                       enum tgsi_sampler_control control,
                       float rgba[TGSI_NUM_CHANNELS][TGSI_QUAD_SIZE])
{
   const struct swtnl_tgsi_sampler *ts =
      (const struct swtnl_tgsi_sampler *)sampler;

   memset(rgba, 0, sizeof(float) * TGSI_NUM_CHANNELS * TGSI_QUAD_SIZE);
   if (!ts->textures || sview_index >= ts->nr_textures ||
       sampler_index >= ts->nr_samplers || !ts->samplers[sampler_index])
      return;

   const struct swtnl_texture *tex = &ts->textures[sview_index];
   const struct pipe_sampler_state *samp = ts->samplers[sampler_index];
   if (!tex->data)
      return;

   const struct util_format_description *fdesc =
      util_format_description(tex->format);
   const bool is_int = util_format_is_pure_integer(tex->format) ||
                       (util_format_is_depth_or_stencil(tex->format) &&
                        !util_format_has_depth(fdesc));
   const bool unnorm =
      samp->unnormalized_coords || tex->target == PIPE_TEXTURE_RECT;
   const bool is_cube = tex->target == PIPE_TEXTURE_CUBE ||
                        tex->target == PIPE_TEXTURE_CUBE_ARRAY;
   const bool shadow = samp->compare_mode == PIPE_TEX_COMPARE_R_TO_TEXTURE;
   const bool gather = control == TGSI_SAMPLER_GATHER;
   const unsigned gather_comp = (*(const uint32_t *)c1) & 3u;
   const bool clamp_ref = util_format_is_unorm(tex->format) ||
                          (util_format_has_depth(fdesc) &&
                           !util_format_is_float(tex->format));
   /* Cube sampling clamps to edge per face; seamless cross-face filtering is
    * a known limitation of this non-LLVM TGSI fallback path. */
   const unsigned wrap_s = is_cube ? PIPE_TEX_WRAP_CLAMP_TO_EDGE : samp->wrap_s;
   const unsigned wrap_t = is_cube ? PIPE_TEX_WRAP_CLAMP_TO_EDGE : samp->wrap_t;
   const unsigned wrap_r = samp->wrap_r;

   for (unsigned q = 0; q < TGSI_QUAD_SIZE; ++q) {
      /* tgsi_exec currently passes derivs = NULL for explicit-derivative
       * fetches in vertex/geometry stages, so only explicit LOD/bias and
       * sampler lod_bias contribute here. */
      float lod = samp->lod_bias;
      if (control == TGSI_SAMPLER_LOD_EXPLICIT)
         lod = c1[q] + samp->lod_bias;
      else if (control == TGSI_SAMPLER_LOD_BIAS)
         lod = c1[q] + samp->lod_bias;
      else if (control == TGSI_SAMPLER_LOD_ZERO || gather)
         lod = 0.0f;

      if (samp->min_mip_filter == PIPE_TEX_MIPFILTER_NONE || gather || unnorm) {
         lod = 0.0f;
      } else {
         lod = CLAMP(lod, samp->min_lod, samp->max_lod);
      }

      const int max_rel = (int)(tex->last_level - tex->first_level);

      float sc = s[q], tc = t_coord[q], rc = r[q];
      int layer = 0;
      if (is_cube) {
         int face = 0;
         swtnl_cube_face(s[q], t_coord[q], r[q], &sc, &tc, &face);
         if (tex->target == PIPE_TEXTURE_CUBE_ARRAY) {
            int cube_idx = CLAMP((int)floorf(c0[q] + 0.5f), 0,
                                 MAX2((int)(tex->layers / 6) - 1, 0));
            layer = cube_idx * 6 + face;
         } else {
            layer = face;
         }
      } else if (tex->target == PIPE_TEXTURE_1D_ARRAY) {
         layer = CLAMP((int)floorf(tc + 0.5f), 0, MAX2((int)tex->layers - 1, 0));
         tc = 0.0f;
      } else if (tex->target == PIPE_TEXTURE_2D_ARRAY) {
         layer = CLAMP((int)floorf(rc + 0.5f), 0, MAX2((int)tex->layers - 1, 0));
      }

      float ref = 0.0f;
      if (shadow) {
         if (tex->target == PIPE_TEXTURE_2D_ARRAY ||
             tex->target == PIPE_TEXTURE_CUBE)
            ref = c0[q];
         else if (tex->target == PIPE_TEXTURE_CUBE_ARRAY)
            ref = c1[q];
         else
            ref = r[q];
         if (clamp_ref)
            ref = CLAMP(ref, 0.0f, 1.0f);
      }

      const unsigned img_filter =
         lod <= 0.0f ? samp->mag_img_filter : samp->min_img_filter;
      const bool linear = !is_int && img_filter == PIPE_TEX_FILTER_LINEAR;

      if (gather) {
         const unsigned lvl = tex->first_level;
         const int w = u_minify(tex->width0, lvl);
         const int h = u_minify(tex->height0, lvl);
         float u = (unnorm ? sc : sc * w) - 0.5f;
         float v = (unnorm ? tc : tc * h) - 0.5f;
         int ix0 = (int)floorf(u) + offset[0];
         int iy0 = (int)floorf(v) + offset[1];
         bool bx0, bx1, by0, by1;
         int wx0 = swtnl_wrap_int(wrap_s, ix0, w, &bx0);
         int wx1 = swtnl_wrap_int(wrap_s, ix0 + 1, w, &bx1);
         int wy0 = swtnl_wrap_int(wrap_t, iy0, h, &by0);
         int wy1 = swtnl_wrap_int(wrap_t, iy0 + 1, h, &by1);

         /* Order matching TGSI/OpenGL textureGather: (0,1), (1,1), (1,0), (0,0). */
         const int xs[4] = { wx0, wx1, wx1, wx0 };
         const int ys[4] = { wy1, wy1, wy0, wy0 };
         const bool bs[4] = { bx0 || by1, bx1 || by1, bx1 || by0, bx0 || by0 };
         for (unsigned k = 0; k < 4; ++k) {
            uint32_t tx[4];
            swtnl_sample_texel(tex, samp, lvl, xs[k], ys[k], layer, bs[k], tx);
            if (shadow) {
               rgba[k][q] = swtnl_shadow_compare(samp->compare_func, ref, uif(tx[0]));
            } else {
               memcpy(&rgba[k][q], &tx[gather_comp], sizeof(uint32_t));
            }
         }
         continue;
      }

      if (samp->min_mip_filter == PIPE_TEX_MIPFILTER_LINEAR && !is_int &&
          !unnorm) {
         const float clamped_lod = CLAMP(lod, 0.0f, (float)max_rel);
         const int rel0 = CLAMP((int)floorf(clamped_lod), 0, max_rel);
         const int rel1 = MIN2(rel0 + 1, max_rel);
         const float frac = clamped_lod - (float)rel0;
         uint32_t out0[4];
         swtnl_sample_level(tex, samp, tex->first_level + (unsigned)rel0,
                            sc, tc, rc, layer, offset, unnorm, shadow, linear,
                            ref, wrap_s, wrap_t, wrap_r, out0);
         if (rel1 != rel0 && frac > 0.0f) {
            uint32_t out1[4];
            swtnl_sample_level(tex, samp, tex->first_level + (unsigned)rel1,
                               sc, tc, rc, layer, offset, unnorm, shadow,
                               linear, ref, wrap_s, wrap_t, wrap_r, out1);
            for (unsigned c = 0; c < 4; ++c)
               rgba[c][q] = uif(out0[c]) * (1.0f - frac) + uif(out1[c]) * frac;
         } else {
            for (unsigned c = 0; c < 4; ++c)
               memcpy(&rgba[c][q], &out0[c], sizeof(uint32_t));
         }
      } else {
         const int rel0 = CLAMP((int)floorf(lod + 0.5f), 0, max_rel);
         uint32_t out0[4];
         swtnl_sample_level(tex, samp, tex->first_level + (unsigned)rel0,
                            sc, tc, rc, layer, offset, unnorm, shadow, linear,
                            ref, wrap_s, wrap_t, wrap_r, out0);
         for (unsigned c = 0; c < 4; ++c)
            memcpy(&rgba[c][q], &out0[c], sizeof(uint32_t));
      }
   }
}

static void
swtnl_tgsi_query_lod(const struct tgsi_sampler *sampler,
                     const unsigned sview_index,
                     const unsigned sampler_index,
                     const float s[TGSI_QUAD_SIZE],
                     const float t[TGSI_QUAD_SIZE],
                     const float p[TGSI_QUAD_SIZE],
                     const float c0[TGSI_QUAD_SIZE],
                     const enum tgsi_sampler_control control,
                     float mipmap[TGSI_QUAD_SIZE],
                     float lod[TGSI_QUAD_SIZE])
{
   for (unsigned q = 0; q < TGSI_QUAD_SIZE; ++q) {
      mipmap[q] = 0.0f;
      lod[q] = 0.0f;
   }
}

static bool
agx_swtnl_init(struct agx_context *ctx)
{
   if (ctx->swtnl_draw)
      return true;

   /* LLVM: the interpreter has no tessellation and no 16-bit types. */
   struct draw_context *draw = draw_create(&ctx->base);
   struct agx_swtnl_render *r = CALLOC_STRUCT(agx_swtnl_render);
   if (!draw || !r) {
      if (draw)
         draw_destroy(draw);
      FREE(r);
      return false;
   }

   /* gallivm bounds NIR constant loads in dwords, as llvmpipe sets up; the
    * default vec4 stride left only the first dword of a 16-byte buffer. */
   draw_set_constant_buffer_stride(draw, sizeof(float));

   /* The GPU rasterizes wide lines itself (GLES3 clipping.line.wide_*);
    * the draw module's wide-line stage would turn them into quads with
    * other coverage (primitive_bounding_box.wide_lines). */
   draw_wide_line_threshold(draw, 1000000.0f);

   r->ctx = ctx;
   util_dynarray_init(&r->acc_vertices, NULL);
   util_dynarray_init(&r->acc_indices, NULL);
   r->base.max_indices = 16 * 1024;
   r->base.max_vertex_buffer_bytes = 4 * 1024 * 1024;
   r->base.get_vertex_info = swtnl_get_vertex_info;
   r->base.allocate_vertices = swtnl_allocate_vertices;
   r->base.map_vertices = swtnl_map_vertices;
   r->base.unmap_vertices = swtnl_unmap_vertices;
   r->base.set_primitive = swtnl_set_primitive;
   r->base.draw_elements = swtnl_draw_elements;
   r->base.draw_arrays = swtnl_draw_arrays;
   r->base.release_vertices = swtnl_release_vertices;
   r->base.set_stream_output_info = swtnl_set_stream_output_info;
   r->base.destroy = swtnl_destroy;

   struct draw_stage *stage = draw_vbuf_stage(draw, &r->base);
   if (!stage) {
      draw_destroy(draw);
      swtnl_destroy(&r->base);
      return false;
   }

   draw_set_rasterize_stage(draw, stage);
   draw_set_render(draw, &r->base);
   /* The GPU clips, and maps the clip-space positions it receives. */
   draw_set_driver_clipping(draw, true, true, true, true);
   const struct pipe_viewport_state identity = {
      .scale = {1, 1, 1},
      .swizzle_x = PIPE_VIEWPORT_SWIZZLE_POSITIVE_X,
      .swizzle_y = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Y,
      .swizzle_z = PIPE_VIEWPORT_SWIZZLE_POSITIVE_Z,
      .swizzle_w = PIPE_VIEWPORT_SWIZZLE_POSITIVE_W,
   };
   draw_set_viewport_states(draw, 0, 1, &identity);
   draw->pt.no_fse = true;

   if (!draw->llvm) {
      const mesa_shader_stage stages[2] = {
         MESA_SHADER_VERTEX,
         MESA_SHADER_GEOMETRY,
      };
      for (unsigned i = 0; i < 2; ++i) {
         r->tgsi_buffers[i].base.lookup = swtnl_tgsi_buffer_lookup;
         draw_buffer(draw, stages[i], &r->tgsi_buffers[i].base);

         r->tgsi_samplers[i].base.get_dims = swtnl_tgsi_get_dims;
         r->tgsi_samplers[i].base.get_samples = swtnl_tgsi_get_samples;
         r->tgsi_samplers[i].base.get_texel = swtnl_tgsi_get_texel;
         r->tgsi_samplers[i].base.query_lod = swtnl_tgsi_query_lod;
         draw_texture_sampler(draw, stages[i], &r->tgsi_samplers[i].base);
      }
   }

   ctx->swtnl_draw = draw;
   ctx->swtnl_render = &r->base;
   return true;
}

void
agx_swtnl_destroy(struct agx_context *ctx)
{
   if (ctx->swtnl_vs) {
      hash_table_u64_foreach(ctx->swtnl_vs, entry)
         ctx->base.delete_vs_state(&ctx->base, entry.data);
      _mesa_hash_table_u64_destroy(ctx->swtnl_vs);
      ctx->swtnl_vs = NULL;
   }

   /* The vbuf stage owns the render. */
   if (ctx->swtnl_draw)
      draw_destroy(ctx->swtnl_draw);

   ctx->swtnl_draw = NULL;
   ctx->swtnl_render = NULL;
}

/* Arithmetic below 32 bits runs at 32 (LLVM's NIR translation lacks some
 * 16-bit forms, e.g. mediump bitfieldExtract). */
static unsigned
swtnl_widen_bit_size(const nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_alu)
      return 0;

   const nir_alu_instr *alu = nir_instr_as_alu(instr);
   /* Conversions and the type-agnostic moves are carried by
    * agx_swtnl_strip_16bit: widening a move zero-extends a half float. */
   if (nir_op_infos[alu->op].is_conversion || nir_op_is_vec(alu->op) ||
       alu->op == nir_op_mov || alu->op == nir_op_bcsel)
      return 0;

   bool narrow = alu->def.bit_size < 32 && alu->def.bit_size > 1;
   for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
      const unsigned size = nir_src_bit_size(alu->src[i].src);
      narrow |= size < 32 && size > 1;
   }
   return narrow ? 32 : 0;
}


/* After swtnl_widen_bit_size only conversions touch 16-bit values. gallivm
 * mistranslates those in vertex-pipeline stages (the mediump executors of
 * the geometry and tessellation builtins crashed in cast_type), so carry
 * every 16-bit value in 32 bits: floats rounded to half precision with
 * fquantize2f16, integers with their low 16 bits significant and extended
 * where they are converted back. */

/* fquantize2f16 in 32-bit arithmetic (gallivm has no fquantize2f16): round
 * the mantissa to ten bits, flush half denormals to signed zero and
 * overflow to signed infinity, pass NaN through. */
static nir_def *
strip16_quantize(nir_builder *b, nir_def *f)
{
   nir_def *u = f;
   nir_def *bias = nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, u, 13), 1), 0xfff);
   nir_def *r = nir_iand_imm(b, nir_iadd(b, u, bias), 0xffffe000);
   nir_def *sign = nir_iand_imm(b, u, 0x80000000);
   nir_def *mag = nir_fabs(b, f);
   r = nir_bcsel(b, nir_flt_imm(b, mag, 6.103515625e-05f), sign, r);
   r = nir_bcsel(b, nir_fge_imm(b, mag, 65520.0f),
                 nir_ior_imm(b, sign, 0x7f800000), r);
   return nir_bcsel(b, nir_fneu(b, f, f), f, r);
}

static nir_def *
strip16_extend(nir_builder *b, nir_def *raw, bool sign)
{
   return sign ? nir_ishr_imm(b, nir_ishl_imm(b, raw, 16), 16)
               : nir_iand_imm(b, raw, 0xffff);
}

static nir_def *
strip16_src(nir_builder *b, nir_alu_instr *alu, unsigned i, nir_def **map)
{
   nir_def *src = nir_ssa_for_alu_src(b, alu, i);
   if (src->bit_size != 16)
      return src;

   nir_def *def = alu->src[i].src.ssa;
   if (nir_def_instr(def)->type == nir_instr_type_load_const) {
      nir_load_const_instr *lc = nir_instr_as_load_const(nir_def_instr(def));
      /* only scalar 16-bit constants reach here in practice */
      if (def->num_components != 1)
         return NULL;
      const uint16_t v = lc->value[alu->src[i].swizzle[0]].u16;
      const nir_alu_type t = nir_alu_type_get_base_type(
         nir_op_infos[alu->op].input_types[i]);
      if (t == nir_type_float)
         return nir_imm_float(b, _mesa_half_to_float(v));
      return nir_imm_int(b, t == nir_type_int ? (int16_t)v : v);
   }

   nir_def *wide = map[def->index];
   if (!wide)
      return NULL;
   unsigned swizzle[NIR_MAX_VEC_COMPONENTS];
   for (unsigned c = 0; c < alu->def.num_components; ++c)
      swizzle[c] = alu->src[i].swizzle[c];
   return nir_swizzle(b, wide, swizzle, alu->def.num_components);
}

static bool
agx_swtnl_strip_16bit(nir_shader *nir)
{
   bool progress = false;

   nir_foreach_function_impl(impl, nir) {
      nir_index_ssa_defs(impl);
      nir_def **map = calloc(impl->ssa_alloc, sizeof(*map));
      nir_builder b = nir_builder_create(impl);

      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            const nir_op_info *info = &nir_op_infos[alu->op];

            /* Moves of 16-bit values: the same move of their 32-bit forms */
            if ((nir_op_is_vec(alu->op) || alu->op == nir_op_mov ||
                 alu->op == nir_op_bcsel) && alu->def.bit_size == 16) {
               b.cursor = nir_before_instr(instr);
               nir_def *srcs[NIR_MAX_VEC_COMPONENTS] = {0};
               bool ok = true;
               for (unsigned i = 0; i < info->num_inputs && ok; ++i) {
                  if (alu->op == nir_op_bcsel && i == 0) {
                     srcs[0] = nir_ssa_for_alu_src(&b, alu, 0);
                     continue;
                  }
                  srcs[i] = strip16_src(&b, alu, i, map);
                  ok = srcs[i] != NULL;
               }
               if (!ok)
                  continue;
               map[alu->def.index] =
                  alu->op == nir_op_bcsel ? nir_bcsel(&b, srcs[0], srcs[1], srcs[2])
                  : alu->op == nir_op_mov ? srcs[0]
                                          : nir_vec(&b, srcs, info->num_inputs);
               progress = true;
               continue;
            }

            if (!info->is_conversion || info->num_inputs != 1)
               continue;

            const bool dst16 = alu->def.bit_size == 16;
            const bool src16 = nir_src_bit_size(alu->src[0].src) == 16;
            if (!dst16 && !src16)
               continue;

            b.cursor = nir_before_instr(instr);
            nir_def *src = strip16_src(&b, alu, 0, map);
            if (!src)
               continue;

            const nir_alu_type in_t = nir_alu_type_get_base_type(info->input_types[0]);
            const nir_alu_type out_t = nir_alu_type_get_base_type(info->output_type);
            /* A 16-bit integer source carries its low half; extend it. */
            if (src16 && in_t != nir_type_float)
               src = strip16_extend(&b, src, in_t == nir_type_int);

            nir_def *wide;
            if (out_t == nir_type_float) {
               nir_def *f = in_t == nir_type_float ? src
                            : in_t == nir_type_int  ? nir_i2f32(&b, src)
                            : in_t == nir_type_bool ? nir_b2f32(&b, src)
                                                    : nir_u2f32(&b, src);
               if (f->bit_size == 64)
                  f = nir_f2f32(&b, f);
               wide = dst16 ? strip16_quantize(&b, f)
                            : alu->def.bit_size == 64 ? nir_f2f64(&b, f) : f;
            } else if (out_t == nir_type_bool) {
               continue;
            } else {
               const bool sign = out_t == nir_type_int;
               nir_def *v = in_t == nir_type_float
                               ? (sign ? nir_f2i32(&b, src) : nir_f2u32(&b, src))
                            : in_t == nir_type_bool ? nir_b2i32(&b, src)
                                                    : src;
               if (v->bit_size == 64)
                  v = nir_u2u32(&b, v);
               wide = alu->def.bit_size == 64
                         ? (sign ? nir_i2i64(&b, v) : nir_u2u64(&b, v))
                         : v;
            }

            if (dst16) {
               map[alu->def.index] = wide;
            } else {
               nir_def_rewrite_uses(&alu->def, wide);
               nir_instr_remove(instr);
            }
            progress = true;
         }
      }

      free(map);
      nir_progress(progress, impl, nir_metadata_control_flow);
   }

   if (progress)
      NIR_PASS(_, nir, nir_opt_dce);
   return progress;
}


/* The lowerings llvmpipe asks NIR for (lp_screen.c): gallivm translates
 * none of these ops. */
static const nir_shader_compiler_options *
swtnl_draw_options(void)
{
   static nir_shader_compiler_options options;
   static bool ready;
   if (!ready) {
      options = agx_nir_options;
      options.lower_scmp = true;
      options.lower_fsat = true;
      options.lower_fdph = true;
      options.float_mul_add16 = options.float_mul_add32 =
         options.float_mul_add64 = nir_float_muladd_support_keep_weak_ffma;
      options.lower_fmod = true;
      options.lower_hadd = true;
      options.lower_uadd_sat = options.lower_usub_sat = true;
      options.lower_iadd_sat = true;
      options.lower_bitfield_insert = true;
      options.lower_bitfield_extract = true;
      options.lower_bitfield_extract8 = options.lower_bitfield_extract16 = true;
      options.lower_pack_snorm_2x16 = options.lower_pack_snorm_4x8 = true;
      options.lower_pack_unorm_2x16 = options.lower_pack_unorm_4x8 = true;
      options.lower_pack_half_2x16 = true;
      options.lower_pack_64_4x16 = true;
      options.lower_pack_split = true;
      options.lower_unpack_snorm_2x16 = options.lower_unpack_snorm_4x8 = true;
      options.lower_unpack_unorm_2x16 = options.lower_unpack_unorm_4x8 = true;
      options.lower_unpack_half_2x16 = true;
      options.lower_extract_byte = options.lower_extract_word = true;
      options.lower_insert_byte = options.lower_insert_word = true;
      options.lower_uadd_carry = options.lower_usub_borrow = true;
      options.lower_mul_2x32_64 = true;
      options.lower_ifind_msb = true;
      options.lower_fquantize2f16 = true;
      options.lower_fminmax_signed_zero = true;
      options.lower_int64_options |=
         nir_lower_imul_2x32_64 | nir_lower_bitfield_extract64;
      ready = true;
   }
   return &options;
}

static bool
swtnl_algebraic(nir_shader *nir)
{
   bool progress = false, more;
   do {
      more = false;
      NIR_PASS(more, nir, nir_opt_algebraic);
      NIR_PASS(more, nir, nir_opt_constant_folding);
      NIR_PASS(more, nir, nir_opt_dce);
      progress |= more;
   } while (more);
   return progress;
}

/* The state tracker finalized the shader for this driver, which can leave
 * amul in it; the draw module knows only imul. */
/* The half-float bits of a 32-bit float in 32-bit arithmetic: the 16-bit
 * widening below carries f2f16 results as 32-bit floats, so the lowered
 * packHalf2x16 (a 16-bit pack of two f2f16) would pack float32 bits. Round
 * to nearest even, overflow to infinity, keep NaN; a half denormal is
 * round(|f| * 2^24). */
static nir_def *
swtnl_half_bits(nir_builder *b, nir_def *f)
{
   nir_def *sign = nir_iand_imm(b, nir_ushr_imm(b, f, 16), 0x8000);
   nir_def *abs = nir_iand_imm(b, f, 0x7fffffff);
   nir_def *r = nir_iadd_imm(b, abs, -0x38000000);
   r = nir_iadd(b, r,
                nir_iadd_imm(b, nir_iand_imm(b, nir_ushr_imm(b, r, 13), 1), 0xfff));
   r = nir_umin(b, nir_ushr_imm(b, r, 13), nir_imm_int(b, 0x7c00));
   nir_def *denormal = nir_f2u32(
      b, nir_fround_even(b, nir_fmul_imm(b, nir_fabs(b, f), 16777216.0)));
   r = nir_bcsel(b, nir_ult(b, abs, nir_imm_int(b, 0x38800000)), denormal, r);
   r = nir_bcsel(b, nir_ult(b, nir_imm_int(b, 0x7f800000), abs),
                 nir_imm_int(b, 0x7e00), r);
   return nir_ior(b, sign, r);
}

static bool
swtnl_lower_pack_half(nir_builder *b, nir_alu_instr *alu, UNUSED void *data)
{
   if ((alu->op != nir_op_pack_half_2x16 &&
        alu->op != nir_op_pack_half_2x16_split) ||
       alu->def.num_components != 1)
      return false;

   b->cursor = nir_before_instr(&alu->instr);
   nir_def *x, *y;
   if (alu->op == nir_op_pack_half_2x16) {
      nir_def *v = nir_ssa_for_alu_src(b, alu, 0);
      x = nir_channel(b, v, 0);
      y = nir_channel(b, v, 1);
   } else {
      x = nir_ssa_for_alu_src(b, alu, 0);
      y = nir_ssa_for_alu_src(b, alu, 1);
   }
   nir_def_replace(&alu->def,
                   nir_ior(b, swtnl_half_bits(b, x),
                           nir_ishl_imm(b, swtnl_half_bits(b, y), 16)));
   return true;
}

/* The draw module runs the vertex, evaluation and geometry stages one SIMD
 * vector at a time on one thread, so a barrier there has no invocations to
 * wait for, and gallivm can only suspend a coroutine, which only the
 * tessellation control stage is. Keep the memory semantics: gallivm emits
 * them as a fence. */
static bool
swtnl_drop_execution_barrier(nir_builder *b, nir_intrinsic_instr *intr,
                             void *data)
{
   if (intr->intrinsic != nir_intrinsic_barrier ||
       nir_intrinsic_execution_scope(intr) == SCOPE_NONE)
      return false;
   if (nir_intrinsic_memory_semantics(intr))
      nir_intrinsic_set_execution_scope(intr, SCOPE_NONE);
   else
      nir_instr_remove(&intr->instr);
   return true;
}

bool
agx_swtnl_lower_for_draw(nir_shader *nir)
{
   bool progress = false;
   NIR_PASS(progress, nir, draw_nir_lower_opcodes);

   if (nir->info.stage != MESA_SHADER_TESS_CTRL)
      NIR_PASS(progress, nir, nir_shader_intrinsics_pass,
               swtnl_drop_execution_barrier, nir_metadata_control_flow, NULL);

   /* gallivm has no quad operations, and its ballots are one 32-bit word:
    * lower the subgroup operations as lavapipe does. */
   const nir_lower_subgroups_options subgroups = {
      .ballot_bit_size = 32,
      .ballot_components = 1,
      .lower_quad = true,
      .lower_quad_vote = true,
      .lower_inverse_ballot = true,
      .lower_rotate_to_shuffle = true,
   };
   NIR_PASS(progress, nir, nir_lower_subgroups, &subgroups);

   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            /* amul, and the mediump conversions the driver would fold */
            nir_op op = alu->op;
            switch (alu->op) {
            case nir_op_amul: op = nir_op_imul; break;
            case nir_op_f2fmp: op = nir_op_f2f16; break;
            case nir_op_i2imp: op = nir_op_i2i16; break;
            case nir_op_f2imp: op = nir_op_f2i16; break;
            case nir_op_f2ump: op = nir_op_f2u16; break;
            case nir_op_i2fmp: op = nir_op_i2f16; break;
            case nir_op_u2fmp: op = nir_op_u2f16; break;
            default: break;
            }
            if (op != alu->op) {
               alu->op = op;
               progress = true;
            }
         }
      }
   }

   NIR_PASS(progress, nir, nir_shader_alu_pass, swtnl_lower_pack_half,
            nir_metadata_control_flow, NULL);

   /* The lowerings first: unpack_half_2x16 and friends become 16-bit
    * conversions, which the widening below then carries in 32 bits. */
   nir->options = swtnl_draw_options();
   progress |= swtnl_algebraic(nir);
   /* The pack/unpack_half lowerings live in the scalarizer. */
   NIR_PASS(progress, nir, nir_lower_alu_to_scalar, NULL, NULL);
   progress |= swtnl_algebraic(nir);
   NIR_PASS(progress, nir, nir_lower_bit_size, swtnl_widen_bit_size, NULL);
   NIR_PASS(progress, nir, nir_opt_dce);
   NIR_PASS(progress, nir, agx_swtnl_strip_16bit);
   progress |= swtnl_algebraic(nir);

   NIR_PASS(progress, nir, nir_lower_int64);

   /* gallivm sets up its 16- and 64-bit types from the gathered bit sizes */
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
   return progress;
}

/* The draw module's copy of a shader, created on first use. */
static void *
agx_swtnl_shader(struct agx_context *ctx, struct agx_uncompiled_shader *so)
{
   if (so->swtnl_shader || so->swtnl_failed)
      return so->swtnl_shader;

   struct blob_reader reader;
   blob_reader_init(&reader, so->early_serialized_nir.data,
                    so->early_serialized_nir.size);
   nir_shader *nir = nir_deserialize(NULL, &agx_nir_options, &reader);
   agx_swtnl_lower_for_draw(nir);

   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = nir,
      .stream_output = so->base.stream_output,
   };

   switch (so->type) {
   case MESA_SHADER_VERTEX:
      so->swtnl_shader = draw_create_vertex_shader(ctx->swtnl_draw, &state);
      break;
   case MESA_SHADER_TESS_CTRL:
      so->swtnl_shader = draw_create_tess_ctrl_shader(ctx->swtnl_draw, &state);
      break;
   case MESA_SHADER_TESS_EVAL:
      so->swtnl_shader = draw_create_tess_eval_shader(ctx->swtnl_draw, &state);
      break;
   case MESA_SHADER_GEOMETRY:
      /* The interpreter runs geometry shaders from TGSI only. nir_to_tgsi
       * takes the NIR; the draw module copies the tokens. */
      if (!ctx->swtnl_draw->llvm) {
         state.type = PIPE_SHADER_IR_TGSI;
         state.tokens = nir_to_tgsi(nir, ctx->base.screen);
         nir = NULL;
      }
      so->swtnl_shader = draw_create_geometry_shader(ctx->swtnl_draw, &state);
      if (state.type == PIPE_SHADER_IR_TGSI)
         FREE((void *)state.tokens);
      break;
   default:
      break;
   }

   /* On success the draw module owns the NIR. */
   if (!so->swtnl_shader) {
      ralloc_free(nir);
      so->swtnl_failed = true;
   }
   return so->swtnl_shader;
}

void
agx_swtnl_delete_shader(struct agx_context *ctx, struct agx_uncompiled_shader *so)
{
   if (!so->swtnl_shader || !ctx->swtnl_draw)
      return;

   switch (so->type) {
   case MESA_SHADER_VERTEX:
      draw_delete_vertex_shader(ctx->swtnl_draw, so->swtnl_shader);
      break;
   case MESA_SHADER_TESS_CTRL:
      draw_delete_tess_ctrl_shader(ctx->swtnl_draw, so->swtnl_shader);
      break;
   case MESA_SHADER_TESS_EVAL:
      draw_delete_tess_eval_shader(ctx->swtnl_draw, so->swtnl_shader);
      break;
   case MESA_SHADER_GEOMETRY:
      draw_delete_geometry_shader(ctx->swtnl_draw, so->swtnl_shader);
      break;
   default:
      break;
   }
   so->swtnl_shader = NULL;
}

/*
 * Does this draw take the CPU path? G17G, a geometry stage (or tessellation
 * the draw module can do, see agx_m5_gpu_tess()), and nothing the CPU copy
 * does not see: no textures, images or storage buffers before rasterization.
 *
 * A layered framebuffer is drawn a layer at a time (agx_g17g_layered_draw),
 * so the layer a vertex shader writes reaches the GPU only as the per-layer
 * subsets drawn here. Without this, every primitive of such a draw (layered
 * clears and blits through u_blitter, layered PBO transfers in st/mesa,
 * gl_Layer from a vertex shader) lands in layer 0.
 */
bool
agx_swtnl_applies(struct agx_context *ctx)
{
   if (!agx_chip_g17_m5(agx_device(ctx->base.screen)->chip) ||
       ctx->swtnl_inner)
      return false;

   const struct agx_uncompiled_shader *vs =
      ctx->stage[MESA_SHADER_VERTEX].shader;
   const bool vs_layer =
      vs && vs->info.writes_layer &&
      util_framebuffer_get_num_layers(&ctx->framebuffer) > 1;
   if (!ctx->stage[MESA_SHADER_GEOMETRY].shader &&
       !ctx->stage[MESA_SHADER_TESS_EVAL].shader && !vs_layer)
      return false;

   /* The draw module tessellates only with LLVM: without it, tessellation
    * runs on the GPU (agx_m5_gpu_tess()). */
   if (ctx->stage[MESA_SHADER_TESS_EVAL].shader &&
       agx_m5_gpu_tess(ctx))
      return false;

   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s) {
      struct agx_uncompiled_shader *so = ctx->stage[s].shader;
      if (so && so->info.nr_bindful_images)
         return false;
   }

   return ctx->stage[MESA_SHADER_VERTEX].shader && ctx->attributes;
}


/* The draw module takes 32-bit strides and offsets, and samples with signed
 * 32-bit offsets: a copy or buffer texture larger than llvmpipe's limit for
 * the same sampling code (2 GiB) does not fit, so the draw goes without it. */
#define AGX_SWTNL_MAX_TEXTURE_BYTES (1ull << 31)

static bool
swtnl_map_texture(struct agx_context *ctx, mesa_shader_stage s, unsigned i,
                  struct pipe_sampler_view *view, struct swtnl_texture *out)
{
   struct pipe_context *pctx = &ctx->base;
   struct pipe_resource *tex = view->texture;
   uint32_t row_stride[PIPE_MAX_TEXTURE_LEVELS] = {0};
   uint32_t img_stride[PIPE_MAX_TEXTURE_LEVELS] = {0};
   uint32_t mip_offsets[PIPE_MAX_TEXTURE_LEVELS] = {0};

   /* Multisampled views are not copied: the draw goes without them. */
   if (tex->nr_samples > 1)
      return false;

   const enum pipe_format format = view->format;
   const unsigned bs = util_format_get_blocksize(format);

   if (tex->target == PIPE_BUFFER || view->target == PIPE_BUFFER) {
      if (!bs)
         return false;
      const unsigned avail =
         agx_buffer_range_in_resource(tex, view->u.buf.offset, view->u.buf.size);
      /* In elements, clamped to the GPU path's texture-buffer limit so
       * textureSize() matches across CPU and GPU paths, and sized by what the
       * shader can sample rather than the whole backing buffer. */
      const unsigned elems = agx_texture_buffer_size_el(format, avail);
      const uint64_t copy_size = (uint64_t)elems * bs;
      if (copy_size > AGX_SWTNL_MAX_TEXTURE_BYTES)
         return false;
      uint8_t *data = calloc(1, (size_t)copy_size + 64);
      if (!data)
         return false;
      if (copy_size) {
         struct pipe_transfer *xfer = NULL;
         const void *map =
            pipe_buffer_map_range(pctx, tex, view->u.buf.offset, copy_size,
                                  PIPE_MAP_READ, &xfer);
         if (!map) {
            free(data);
            return false;
         }
         memcpy(data, map, copy_size);
         pipe_buffer_unmap(pctx, xfer);
      }
      row_stride[0] = copy_size;
      img_stride[0] = copy_size;
      draw_set_mapped_texture(ctx->swtnl_draw, s, i, elems, 1, 1, 0, 0, 1, 0,
                              data, row_stride, img_stride, mip_offsets);
      out->data = data;
      out->target = PIPE_BUFFER;
      out->format = format;
      out->buf_elements = elems;
      out->swizzle[0] = view->swizzle_r;
      out->swizzle[1] = view->swizzle_g;
      out->swizzle[2] = view->swizzle_b;
      out->swizzle[3] = view->swizzle_a;
      out->bound = true;
      return true;
   }

   const unsigned first_level = view->u.tex.first_level;
   const unsigned last_level = view->u.tex.last_level;
   const bool array = tex->target == PIPE_TEXTURE_1D_ARRAY ||
                      tex->target == PIPE_TEXTURE_2D_ARRAY ||
                      tex->target == PIPE_TEXTURE_CUBE ||
                      tex->target == PIPE_TEXTURE_CUBE_ARRAY;
   const unsigned first_layer = array ? view->u.tex.first_layer : 0;
   const unsigned layers =
      array ? view->u.tex.last_layer - view->u.tex.first_layer + 1 : 1;

   uint64_t size = 0;
   for (unsigned l = first_level; l <= last_level; ++l) {
      const unsigned w = u_minify(tex->width0, l), h = u_minify(tex->height0, l);
      const unsigned d = tex->target == PIPE_TEXTURE_3D ? u_minify(tex->depth0, l)
                                                         : layers;
      const uint64_t row_B = util_format_get_stride(format, w);
      const uint64_t img_B = row_B * util_format_get_nblocksy(format, h);

      size += img_B * d;
      if (size > AGX_SWTNL_MAX_TEXTURE_BYTES)
         return false;

      mip_offsets[l] = size - img_B * d;
      row_stride[l] = row_B;
      img_stride[l] = img_B;
   }

   uint8_t *data = calloc(1, size + 64);
   if (!data)
      return false;

   for (unsigned l = first_level; l <= last_level; ++l) {
      const unsigned w = u_minify(tex->width0, l), h = u_minify(tex->height0, l);
      const unsigned d = tex->target == PIPE_TEXTURE_3D ? u_minify(tex->depth0, l)
                                                         : layers;
      const struct pipe_box box = {
         .x = 0, .y = 0, .z = tex->target == PIPE_TEXTURE_3D ? 0 : first_layer,
         .width = w, .height = h, .depth = d,
      };
      struct pipe_transfer *transfer = NULL;
      const uint8_t *map =
         pctx->texture_map(pctx, tex, l, PIPE_MAP_READ, &box, &transfer);
      if (!map) {
         free(data);
         return false;
      }
      const unsigned rows = util_format_get_nblocksy(format, h);
      const unsigned row_bytes = util_format_get_nblocksx(format, w) * bs;
      for (unsigned z = 0; z < d; ++z) {
         for (unsigned y = 0; y < rows; ++y) {
            memcpy(data + mip_offsets[l] + (size_t)img_stride[l] * z +
                      (size_t)row_stride[l] * y,
                   map + (size_t)transfer->layer_stride * z +
                      (size_t)transfer->stride * y,
                   row_bytes);
         }
      }
      pctx->texture_unmap(pctx, transfer);
   }

   const unsigned depth = tex->target == PIPE_TEXTURE_3D ? tex->depth0 : layers;
   draw_set_mapped_texture(ctx->swtnl_draw, s, i, tex->width0, tex->height0,
                           depth, first_level, last_level, 1, 0, data,
                           row_stride, img_stride, mip_offsets);
   out->data = data;
   out->target = view->target;
   out->format = format;
   out->width0 = tex->width0;
   out->height0 = tex->height0;
   out->depth0 = tex->depth0;
   out->first_level = first_level;
   out->last_level = last_level;
   out->layers = layers;
   memcpy(out->row_stride, row_stride, sizeof(row_stride));
   memcpy(out->img_stride, img_stride, sizeof(img_stride));
   memcpy(out->mip_offsets, mip_offsets, sizeof(mip_offsets));
   out->swizzle[0] = view->swizzle_r;
   out->swizzle[1] = view->swizzle_g;
   out->swizzle[2] = view->swizzle_b;
   out->swizzle[3] = view->swizzle_a;
   out->bound = true;
   return true;
}

static const void *
map_buffer(struct agx_context *ctx, struct pipe_resource *prsc, unsigned usage,
           struct pipe_transfer **transfer)
{
   return pipe_buffer_map_range(&ctx->base, prsc, 0, prsc->width0, usage,
                                transfer);
}

static enum mesa_prim
agx_swtnl_output_prim(struct agx_context *ctx, enum mesa_prim mode)
{
   struct agx_uncompiled_shader *gs = ctx->stage[MESA_SHADER_GEOMETRY].shader;
   struct agx_uncompiled_shader *tes = ctx->stage[MESA_SHADER_TESS_EVAL].shader;

   if (gs)
      return u_reduced_prim(gs->gs_mode);
   if (tes) {
      if (tes->tess.point_mode)
         return MESA_PRIM_POINTS;
      return tes->tess.primitive == TESS_PRIMITIVE_ISOLINES ? MESA_PRIM_LINES
                                                            : MESA_PRIM_TRIANGLES;
   }
   return u_reduced_prim(mode);
}

void
agx_swtnl_draw(struct agx_context *ctx, const struct pipe_draw_info *info,
               const struct pipe_draw_start_count_bias *draw_info)
{
   if (!agx_swtnl_init(ctx))
      return;

   struct draw_context *draw = ctx->swtnl_draw;
   struct agx_swtnl_render *r = agx_swtnl_render(ctx->swtnl_render);
   struct pipe_transfer *transfers[MESA_SHADER_FRAGMENT][PIPE_MAX_CONSTANT_BUFFERS] = {0};
   struct pipe_transfer *ssbo_transfers[MESA_SHADER_FRAGMENT][PIPE_MAX_SHADER_BUFFERS] = {0};
   uint8_t *ssbo_shadow[MESA_SHADER_FRAGMENT][PIPE_MAX_SHADER_BUFFERS] = {0};
   uint8_t *ssbo_mapped[MESA_SHADER_FRAGMENT][PIPE_MAX_SHADER_BUFFERS] = {0};
   unsigned ssbo_size[MESA_SHADER_FRAGMENT][PIPE_MAX_SHADER_BUFFERS] = {0};
   unsigned cb_masks[MESA_SHADER_FRAGMENT] = {0};
   struct pipe_transfer *vb_transfers[PIPE_MAX_ATTRIBS] = {0};
   struct pipe_transfer *so_transfers[PIPE_MAX_SO_BUFFERS] = {0};
   struct pipe_transfer *ib_transfer = NULL;
   struct draw_so_target targets[PIPE_MAX_SO_BUFFERS];
   struct draw_so_target *target_ptrs[PIPE_MAX_SO_BUFFERS] = {0};
   bool mapped = true;

   struct agx_uncompiled_shader *vs = ctx->stage[MESA_SHADER_VERTEX].shader;
   struct agx_uncompiled_shader *tcs = ctx->stage[MESA_SHADER_TESS_CTRL].shader;
   struct agx_uncompiled_shader *tes = ctx->stage[MESA_SHADER_TESS_EVAL].shader;
   struct agx_uncompiled_shader *gs = ctx->stage[MESA_SHADER_GEOMETRY].shader;

   void *dvs = agx_swtnl_shader(ctx, vs);
   void *dtcs = tcs ? agx_swtnl_shader(ctx, tcs) : NULL;
   void *dtes = tes ? agx_swtnl_shader(ctx, tes) : NULL;
   void *dgs = gs ? agx_swtnl_shader(ctx, gs) : NULL;
   if (!dvs || (tcs && !dtcs) || (tes && !dtes) || (gs && !dgs)) {
      mesa_loge("asahi: G17G CPU vertex stages: a shader did not translate");
      return;
   }

   /* The draw module rasterizes nothing: it decomposes primitives, keeping
    * the provoking vertex, and the GPU does everything after that. */
   struct pipe_rasterizer_state rast = ctx->rast->base;
   rast.cull_face = PIPE_FACE_NONE;
   rast.fill_front = rast.fill_back = PIPE_POLYGON_MODE_FILL;
   rast.offset_point = rast.offset_line = rast.offset_tri = false;
   rast.poly_stipple_enable = rast.line_stipple_enable = false;
   rast.point_smooth = rast.line_smooth = rast.poly_smooth = false;
   rast.point_quad_rasterization = false;
   rast.sprite_coord_enable = 0;
   rast.flatshade = false;
   rast.light_twoside = false;
   rast.clip_plane_enable = 0;
   rast.rasterizer_discard = false;
   draw_set_rasterizer_state(draw, &rast, NULL);

   draw_bind_vertex_shader(draw, dvs);
   draw_bind_tess_ctrl_shader(draw, dtcs);
   draw_bind_tess_eval_shader(draw, dtes);
   draw_bind_geometry_shader(draw, dgs);
   if (tes)
      draw_set_tess_state(draw, ctx->default_outer_level,
                          ctx->default_inner_level);

   /* What the fragment shader reads, from the last stage's outputs. */
   struct agx_uncompiled_shader *fs = ctx->stage[MESA_SHADER_FRAGMENT].shader;
   const uint64_t slots =
      (fs ? fs->info.inputs_read : 0) & BITFIELD64_RANGE(VARYING_SLOT_VAR0, 32);
   const bool points =
      agx_swtnl_output_prim(ctx, info->mode) == MESA_PRIM_POINTS;

   draw_prepare_shader_outputs(draw);
   memset(&r->vinfo, 0, sizeof(r->vinfo));
   int pos = draw_find_shader_output(draw, TGSI_SEMANTIC_POSITION, 0);
   draw_emit_vertex_attr(&r->vinfo, EMIT_4F, MAX2(pos, 0));
   u_foreach_bit64(slot, slots) {
      unsigned name, index;
      tgsi_get_gl_varying_semantic(slot, true, &name, &index);
      int out = draw_find_shader_output(draw, name, index);
      draw_emit_vertex_attr(&r->vinfo, EMIT_4F, MAX2(out, 0));
   }
   if (points) {
      int psiz = draw_find_shader_output(draw, TGSI_SEMANTIC_PSIZE, 0);
      if (psiz >= 0)
         draw_emit_vertex_attr(&r->vinfo, EMIT_1F, psiz);
      else
         draw_emit_vertex_attr(&r->vinfo, EMIT_1F_PSIZE, 0);
   }
   draw_compute_vertex_size(&r->vinfo);
   /* Every attribute but the point size is a vec4 at 16 * i. */
   r->slots = slots;
   r->psiz = points;
   r->layer_attrib = -1;
   if (ctx->g17g_layered_fb) {
      int layer = draw_find_shader_output(draw, TGSI_SEMANTIC_LAYER, 0);
      if (layer >= 0) {
         r->layer_attrib = r->vinfo.num_attribs;
         r->layer_offset_B = r->vinfo.size * 4;
         draw_emit_vertex_attr(&r->vinfo, EMIT_4F, layer);
         draw_compute_vertex_size(&r->vinfo);
      }
   }

   draw_set_vertex_elements(draw, ctx->attributes->num_attribs,
                            ctx->attributes->pipe);
   const unsigned nr_vbs = util_last_bit(ctx->vb_mask);
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
         size = ptr ? vb->buffer.resource->width0 : 0;
         mapped &= ptr != NULL;
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
         available = indices ? info->index.resource->width0 : 0;
         mapped &= indices != NULL;
      }
      draw_set_indexes(draw, indices, info->index_size, available);
   }

   if (!draw->llvm) {
      for (unsigned t = 0; t < 2; ++t) {
         memset(r->tgsi_buffers[t].ptrs, 0, sizeof(r->tgsi_buffers[t].ptrs));
         memset(r->tgsi_buffers[t].sizes, 0, sizeof(r->tgsi_buffers[t].sizes));
         r->tgsi_samplers[t].textures = NULL;
         r->tgsi_samplers[t].nr_textures = 0;
         memset(r->tgsi_samplers[t].samplers, 0,
                sizeof(r->tgsi_samplers[t].samplers));
         r->tgsi_samplers[t].nr_samplers = 0;
      }
   }

   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s) {
      struct agx_stage *stage = &ctx->stage[s];
      if (!stage->shader)
         continue;

      const int tgsi_idx = s == MESA_SHADER_VERTEX ? 0
                         : s == MESA_SHADER_GEOMETRY ? 1 : -1;

      cb_masks[s] = stage->cb_mask;
      u_foreach_bit(i, cb_masks[s]) {
         mapped &= agx_cpu_bind_constant_buffer(ctx, draw, s, i,
                                                &transfers[s][i]);
      }

      /* Storage buffers: the executor's tessellation and geometry stages
       * read their inputs and write their results through them. */
      u_foreach_bit(i, stage->ssbo_mask) {
         /* The range bound, clamped to the buffer by
          * agx_set_shader_buffers(). */
         const struct pipe_shader_buffer *sb = &stage->ssbo[i];
         const unsigned size = sb->buffer ? sb->buffer_size : 0;
         uint8_t *ptr = NULL;

         if (size) {
            ptr = (uint8_t *)map_buffer(ctx, sb->buffer,
                                        PIPE_MAP_READ | PIPE_MAP_WRITE,
                                        &ssbo_transfers[s][i]);
            mapped &= ptr != NULL;
            if (ptr)
               ptr += sb->buffer_offset;
         }
         /* The executor's atomics (atomic counters, SSBO atomics) need
          * cacheable memory: on the write-combined BO mapping they raised
          * SIGBUS (GLES31 opaque_type_indexing.atomic_counter.*_geometry).
          * Run the stages on a shadow copy and write it back after. */
         if (ptr) {
            ssbo_shadow[s][i] = malloc(size);
            if (ssbo_shadow[s][i]) {
               memcpy(ssbo_shadow[s][i], ptr, size);
               ssbo_mapped[s][i] = ptr;
               ssbo_size[s][i] = size;
            } else {
               mapped = false;
            }
            ptr = ssbo_shadow[s][i];
         }
         draw_set_mapped_shader_buffer(draw, s, i, ptr, ptr ? size : 0);
         if (!draw->llvm && tgsi_idx >= 0) {
            r->tgsi_buffers[tgsi_idx].ptrs[i] = ptr;
            r->tgsi_buffers[tgsi_idx].sizes[i] = ptr ? size : 0;
         }
      }
   }

   unsigned mapped_views[MESA_SHADER_FRAGMENT] = {0};
   for (unsigned st = 0; st < MESA_SHADER_FRAGMENT; ++st) {
      struct agx_stage *stage = &ctx->stage[st];
      if (!stage->shader)
         continue;

      struct pipe_sampler_view *views[PIPE_MAX_SHADER_SAMPLER_VIEWS] = {0};
      struct pipe_sampler_state *samplers[PIPE_MAX_SAMPLERS] = {0};
      /* Only the views the shader can sample, as the GPU path binds them: a
       * view left bound beyond those is neither copied nor able to leave
       * the draw out. */
      const unsigned nr_views =
         MIN3(stage->texture_count, stage->shader->info.nr_bindful_textures,
              PIPE_MAX_SHADER_SAMPLER_VIEWS);
      const unsigned nr_samplers = MIN2(stage->sampler_count, PIPE_MAX_SAMPLERS);
      mapped_views[st] = nr_views;
      if (nr_views)
         memset(r->textures[st], 0, nr_views * sizeof(r->textures[st][0]));
      for (unsigned i = 0; i < nr_views; ++i)
         views[i] = stage->textures[i] ? &stage->textures[i]->base : NULL;
      for (unsigned i = 0; i < nr_samplers; ++i)
         samplers[i] = stage->samplers[i] ? &stage->samplers[i]->base : NULL;

      draw_set_sampler_views(draw, st, views, nr_views);
      draw_set_samplers(draw, st, samplers, nr_samplers);
      for (unsigned i = 0; i < nr_views; ++i) {
         if (views[i])
            mapped &= swtnl_map_texture(ctx, st, i, views[i], &r->textures[st][i]);
      }

      const int tgsi_idx = st == MESA_SHADER_VERTEX ? 0
                         : st == MESA_SHADER_GEOMETRY ? 1 : -1;
      if (!draw->llvm && tgsi_idx >= 0) {
         r->tgsi_samplers[tgsi_idx].textures = r->textures[st];
         r->tgsi_samplers[tgsi_idx].nr_textures = nr_views;
         for (unsigned i = 0; i < nr_samplers; ++i)
            r->tgsi_samplers[tgsi_idx].samplers[i] = samplers[i];
         r->tgsi_samplers[tgsi_idx].nr_samplers = nr_samplers;
      }
   }

   const unsigned num_targets = ctx->streamout.num_targets;
   struct agx_uncompiled_shader *last = gs ?: tes ?: vs;
   for (unsigned i = 0; i < num_targets && last->has_xfb_info; ++i) {
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
      mapped &= targets[i].mapping != NULL;
      atgt->stride = last->xfb_strides[i] * 4;
      target_ptrs[i] = &targets[i];
   }
   draw_set_mapped_so_targets(draw, last->has_xfb_info ? num_targets : 0,
                              target_ptrs);
   draw_collect_primitives_generated(draw, true);

   memset(r->written, 0, sizeof(r->written));
   memset(r->generated, 0, sizeof(r->generated));

   util_dynarray_clear(&r->acc_vertices);
   util_dynarray_clear(&r->acc_indices);
   r->acc_count = r->cur_base = 0;

   /* A buffer or texture that could not be mapped or copied leaves the draw
    * out. */
   if (mapped) {
      draw_vbo(draw, info, 0, NULL, draw_info, 1, ctx->patch_vertices);
      draw_flush(draw);
      agx_swtnl_flush(r);
   } else {
      mesa_loge_once("asahi: CPU vertex stages: a buffer or texture could "
                     "not be mapped, draw skipped");
   }

   for (unsigned i = 0; i < num_targets; ++i) {
      if (!target_ptrs[i])
         continue;

      uint32_t offset_B = targets[i].internal_offset;
      tc_buffer_write(&ctx->base,
                      agx_so_target(ctx->streamout.targets[i])->offset, 0, 4,
                      &offset_B);
   }

   draw_set_mapped_so_targets(draw, 0, NULL);
   if (!draw->llvm) {
      for (unsigned t = 0; t < 2; ++t) {
         memset(r->tgsi_buffers[t].ptrs, 0, sizeof(r->tgsi_buffers[t].ptrs));
         memset(r->tgsi_buffers[t].sizes, 0, sizeof(r->tgsi_buffers[t].sizes));
         r->tgsi_samplers[t].textures = NULL;
         r->tgsi_samplers[t].nr_textures = 0;
         memset(r->tgsi_samplers[t].samplers, 0,
                sizeof(r->tgsi_samplers[t].samplers));
         r->tgsi_samplers[t].nr_samplers = 0;
      }
   }
   for (unsigned st = 0; st < MESA_SHADER_FRAGMENT; ++st) {
      draw_set_sampler_views(draw, st, NULL, 0);
      draw_set_samplers(draw, st, NULL, 0);
      for (unsigned i = 0; i < mapped_views[st]; ++i) {
         struct swtnl_texture *t = &r->textures[st][i];
         uint32_t strides[PIPE_MAX_TEXTURE_LEVELS] = {0};

         if (t->bound) {
            draw_set_mapped_texture(draw, st, i, 0, 0, 0, 0, 0, 0, 0, NULL,
                                    strides, strides, strides);
         }
         free(t->data);
         t->data = NULL;
         t->bound = false;
      }
   }
   for (unsigned i = 0; i < nr_vbs; ++i)
      draw_set_mapped_vertex_buffer(draw, i, NULL, 0);
   if (indices)
      draw_set_indexes(draw, NULL, 0, 0);
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s) {
      u_foreach_bit(i, cb_masks[s])
         draw_set_mapped_constant_buffer(draw, s, i, NULL, 0);
      for (unsigned i = 0; i < PIPE_MAX_CONSTANT_BUFFERS; ++i) {
         if (transfers[s][i])
            pipe_buffer_unmap(&ctx->base, transfers[s][i]);
      }
      for (unsigned i = 0; i < PIPE_MAX_SHADER_BUFFERS; ++i) {
         if (ssbo_shadow[s][i]) {
            memcpy(ssbo_mapped[s][i], ssbo_shadow[s][i], ssbo_size[s][i]);
            free(ssbo_shadow[s][i]);
         }
         if (ssbo_transfers[s][i]) {
            draw_set_mapped_shader_buffer(draw, s, i, NULL, 0);
            pipe_buffer_unmap(&ctx->base, ssbo_transfers[s][i]);
         }
      }
   }
   for (unsigned i = 0; i < ARRAY_SIZE(vb_transfers); ++i) {
      if (vb_transfers[i])
         pipe_buffer_unmap(&ctx->base, vb_transfers[i]);
   }
   for (unsigned i = 0; i < ARRAY_SIZE(so_transfers); ++i) {
      if (so_transfers[i])
         pipe_buffer_unmap(&ctx->base, so_transfers[i]);
   }
   if (ib_transfer)
      pipe_buffer_unmap(&ctx->base, ib_transfer);

   if (ctx->active_queries) {
      agx_query_increment_cpu(ctx, ctx->prims_generated[0], r->generated[0]);
      for (unsigned s = 0; s < ARRAY_SIZE(ctx->tf_prims_generated); ++s) {
         agx_query_increment_cpu(ctx, ctx->tf_prims_generated[s], r->written[s]);
         if (r->written[s] < r->generated[s] && last->has_xfb_info) {
            agx_query_increment_cpu(ctx, ctx->tf_overflow[s], 1);
            agx_query_increment_cpu(ctx, ctx->tf_any_overflow, 1);
         }
      }
   }
}
