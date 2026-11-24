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

#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_serialize.h"
#include "gallium/auxiliary/nir/pipe_nir.h"
#include "draw/draw_context.h"
#include "draw/draw_nir.h"
#include "draw/draw_private.h"
#include "draw/draw_vbuf.h"
#include "draw/draw_vertex.h"
#include "nir/nir_to_tgsi.h"
#include "tgsi/tgsi_from_mesa.h"
#include "util/half_float.h"
#include "util/hash_table.h"
#include "util/u_inlines.h"
#include "util/u_memory.h"
#include "util/u_prim.h"
#include "util/u_dynarray.h"
#include "util/u_framebuffer.h"
#include "util/u_upload_mgr.h"
#include "agx_state.h"

struct agx_swtnl_render {
   struct vbuf_render base;
   struct agx_context *ctx;
   struct vertex_info vinfo;

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
 */
bool
agx_swtnl_applies(struct agx_context *ctx)
{
   if (!agx_chip_g17_m5(agx_device(ctx->base.screen)->chip) ||
       ctx->swtnl_inner ||
       (!ctx->stage[MESA_SHADER_GEOMETRY].shader &&
        !ctx->stage[MESA_SHADER_TESS_EVAL].shader))
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


/* Textures of the CPU stages: a linear copy of every level and layer the
 * view covers, read through the transfer path (which detiles), in the
 * layout the draw module samples from. */
struct swtnl_texture {
   void *data;
};

static bool
swtnl_map_texture(struct agx_context *ctx, mesa_shader_stage s, unsigned i,
                  struct pipe_sampler_view *view, struct swtnl_texture *out)
{
   struct pipe_context *pctx = &ctx->base;
   struct pipe_resource *tex = view->texture;
   uint32_t row_stride[PIPE_MAX_TEXTURE_LEVELS] = {0};
   uint32_t img_stride[PIPE_MAX_TEXTURE_LEVELS] = {0};
   uint32_t mip_offsets[PIPE_MAX_TEXTURE_LEVELS] = {0};

   if (tex->target == PIPE_BUFFER || tex->nr_samples > 1)
      return false;

   const unsigned first_level = view->u.tex.first_level;
   const unsigned last_level = view->u.tex.last_level;
   const bool array = tex->target == PIPE_TEXTURE_1D_ARRAY ||
                      tex->target == PIPE_TEXTURE_2D_ARRAY ||
                      tex->target == PIPE_TEXTURE_CUBE ||
                      tex->target == PIPE_TEXTURE_CUBE_ARRAY;
   const unsigned first_layer = array ? view->u.tex.first_layer : 0;
   const unsigned layers =
      array ? view->u.tex.last_layer - view->u.tex.first_layer + 1 : 1;
   const enum pipe_format format = view->format;
   const unsigned bs = util_format_get_blocksize(format);

   size_t size = 0;
   for (unsigned l = first_level; l <= last_level; ++l) {
      const unsigned w = u_minify(tex->width0, l), h = u_minify(tex->height0, l);
      const unsigned d = tex->target == PIPE_TEXTURE_3D ? u_minify(tex->depth0, l)
                                                         : layers;
      mip_offsets[l] = size;
      row_stride[l] = util_format_get_stride(format, w);
      img_stride[l] = row_stride[l] * util_format_get_nblocksy(format, h);
      size += (size_t)img_stride[l] * d;
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
   struct pipe_transfer *vb_transfers[PIPE_MAX_ATTRIBS] = {0};
   struct pipe_transfer *so_transfers[PIPE_MAX_SO_BUFFERS] = {0};
   struct pipe_transfer *ib_transfer = NULL;
   struct draw_so_target targets[PIPE_MAX_SO_BUFFERS];
   struct draw_so_target *target_ptrs[PIPE_MAX_SO_BUFFERS] = {0};

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

   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s) {
      struct agx_stage *stage = &ctx->stage[s];
      if (!stage->shader)
         continue;

      u_foreach_bit(i, stage->cb_mask) {
         const struct pipe_constant_buffer *cb = &stage->cb[i];
         const uint8_t *ptr = cb->user_buffer;

         if (!ptr && cb->buffer) {
            ptr = map_buffer(ctx, cb->buffer, PIPE_MAP_READ, &transfers[s][i]);
            if (ptr)
               ptr += cb->buffer_offset;
         }
         draw_set_mapped_constant_buffer(draw, s, i, ptr, cb->buffer_size);
      }

      /* Storage buffers: the executor's tessellation and geometry stages
       * read their inputs and write their results through them. */
      u_foreach_bit(i, stage->ssbo_mask) {
         const struct pipe_shader_buffer *sb = &stage->ssbo[i];
         uint8_t *ptr = NULL;

         if (sb->buffer) {
            ptr = (uint8_t *)map_buffer(ctx, sb->buffer,
                                        PIPE_MAP_READ | PIPE_MAP_WRITE,
                                        &ssbo_transfers[s][i]);
            if (ptr)
               ptr += sb->buffer_offset;
         }
         /* The executor's atomics (atomic counters, SSBO atomics) need
          * cacheable memory: on the write-combined BO mapping they raised
          * SIGBUS (GLES31 opaque_type_indexing.atomic_counter.*_geometry).
          * Run the stages on a shadow copy and write it back after. */
         if (ptr && sb->buffer_size) {
            ssbo_shadow[s][i] = malloc(sb->buffer_size);
            if (ssbo_shadow[s][i]) {
               memcpy(ssbo_shadow[s][i], ptr, sb->buffer_size);
               ssbo_mapped[s][i] = ptr;
               ssbo_size[s][i] = sb->buffer_size;
               ptr = ssbo_shadow[s][i];
            }
         }
         draw_set_mapped_shader_buffer(draw, s, i, ptr, sb->buffer_size);
      }
   }

   struct swtnl_texture textures[MESA_SHADER_FRAGMENT][PIPE_MAX_SHADER_SAMPLER_VIEWS] = {0};
   for (unsigned st = 0; st < MESA_SHADER_FRAGMENT; ++st) {
      struct agx_stage *stage = &ctx->stage[st];
      if (!stage->shader)
         continue;

      struct pipe_sampler_view *views[PIPE_MAX_SHADER_SAMPLER_VIEWS] = {0};
      struct pipe_sampler_state *samplers[PIPE_MAX_SAMPLERS] = {0};
      const unsigned nr_views =
         MIN2(stage->texture_count, PIPE_MAX_SHADER_SAMPLER_VIEWS);
      const unsigned nr_samplers = MIN2(stage->sampler_count, PIPE_MAX_SAMPLERS);
      for (unsigned i = 0; i < nr_views; ++i)
         views[i] = stage->textures[i] ? &stage->textures[i]->base : NULL;
      for (unsigned i = 0; i < nr_samplers; ++i)
         samplers[i] = stage->samplers[i] ? &stage->samplers[i]->base : NULL;

      draw_set_sampler_views(draw, st, views, nr_views);
      draw_set_samplers(draw, st, samplers, nr_samplers);
      for (unsigned i = 0; i < nr_views; ++i) {
         if (views[i])
            swtnl_map_texture(ctx, st, i, views[i], &textures[st][i]);
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

   draw_vbo(draw, info, 0, NULL, draw_info, 1, ctx->patch_vertices);
   draw_flush(draw);
   agx_swtnl_flush(r);

   for (unsigned i = 0; i < num_targets; ++i) {
      if (!target_ptrs[i])
         continue;

      uint32_t offset_B = targets[i].internal_offset;
      tc_buffer_write(&ctx->base,
                      agx_so_target(ctx->streamout.targets[i])->offset, 0, 4,
                      &offset_B);
   }

   draw_set_mapped_so_targets(draw, 0, NULL);
   for (unsigned st = 0; st < MESA_SHADER_FRAGMENT; ++st) {
      draw_set_sampler_views(draw, st, NULL, 0);
      draw_set_samplers(draw, st, NULL, 0);
      for (unsigned i = 0; i < PIPE_MAX_SHADER_SAMPLER_VIEWS; ++i)
         free(textures[st][i].data);
   }
   for (unsigned i = 0; i < nr_vbs; ++i)
      draw_set_mapped_vertex_buffer(draw, i, NULL, 0);
   if (indices)
      draw_set_indexes(draw, NULL, 0, 0);
   for (unsigned s = 0; s < MESA_SHADER_FRAGMENT; ++s) {
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
