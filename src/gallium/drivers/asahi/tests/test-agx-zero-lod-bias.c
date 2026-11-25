/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

/* Verify that zero sampler LOD bias folding only touches direct fixed-slot
 * nir_texop_lod_bias queries after binding lowering (leaving offset, handle,
 * deref, and backend-source queries untouched even when sampler_index == 0),
 * and that rebinding a sampler between zero and nonzero bias at unchanged draw
 * state switches fragment variants cleanly on both G14G and G17P. */
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir.h"
#include "compiler/glsl_types.h"
#include "compiler/nir/nir_builder.h"
#include "pipe/p_context.h"
#include "pipe/p_screen.h"
#include "pipe/p_state.h"
#include "util/driconf.h"
#include "util/half_float.h"
#include "util/hash_table.h"
#include "util/u_inlines.h"
#include "util/u_sampler.h"
#include "util/xmlconfig.h"
#include "agx_device.h"
#include "agx_native_descriptor.h"
#include "agx_public.h"
#include "agx_state.h"

#define CHECK(expr)                                                            \
   do {                                                                        \
      if (!(expr)) {                                                           \
         fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #expr);            \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static nir_tex_instr *
emit_lod_bias_query(nir_builder *b, unsigned sampler_index,
                    bool has_src, nir_tex_src_type src_type, nir_def *src_def)
{
   nir_tex_instr *q = nir_tex_instr_create(b->shader, has_src ? 1 : 0);
   q->op = nir_texop_lod_bias;
   q->sampler_dim = GLSL_SAMPLER_DIM_2D;
   q->dest_type = nir_type_float16;
   q->texture_index = sampler_index;
   q->sampler_index = sampler_index;
   if (has_src)
      q->src[0] = nir_tex_src_for_ssa(src_type, src_def);
   nir_def_init(&q->instr, &q->def, 1, 16);
   nir_builder_instr_insert(b, &q->instr);
   return q;
}

static unsigned
count_tex_op(nir_shader *nir, nir_texop op)
{
   unsigned count = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex &&
                nir_instr_as_tex(instr)->op == op)
               count++;
         }
      }
   }
   return count;
}

/* Pair a direct slot-0 lod_bias query with every non-direct source shape
 * (offsets, handles, derefs, backend sources) all carrying sampler_index == 0,
 * plus an out-of-range direct slot 32. Only the direct slot-0 query may be
 * recorded and folded. */
static void
check_direct_and_nondirect_pair(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "zero lod bias direct vs non-direct");

   nir_def *fc = nir_load_frag_coord(&b);
   nir_def *coord = nir_trim_vector(&b, fc, 2);
   nir_def *dyn_idx = nir_f2u32(&b, nir_channel(&b, fc, 0));
   nir_def *dyn_handle = nir_u2u64(&b, dyn_idx);

   nir_variable *samp_var = nir_variable_create(
      b.shader, nir_var_uniform,
      glsl_sampler_type(GLSL_SAMPLER_DIM_2D, false, false, GLSL_TYPE_FLOAT),
      "s_deref");
   nir_def *deref = &nir_build_deref_var(&b, samp_var)->def;

   /* Direct slot 0 feeding txb -> must fold to constant 0 and txb -> tex. */
   nir_tex_instr *direct_q = emit_lod_bias_query(&b, 0, false, 0, NULL);
   nir_tex_instr *sample = nir_tex_instr_create(b.shader, 2);
   sample->op = nir_texop_txb;
   sample->sampler_dim = GLSL_SAMPLER_DIM_2D;
   sample->coord_components = 2;
   sample->dest_type = nir_type_float32;
   sample->texture_index = 0;
   sample->sampler_index = 0;
   sample->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   sample->src[1] = nir_tex_src_for_ssa(nir_tex_src_bias, &direct_q->def);
   nir_def_init(&sample->instr, &sample->def, 4, 32);
   nir_builder_instr_insert(&b, &sample->instr);

   /* Non-direct queries with sampler_index == 0 and out-of-range slot 32. */
   nir_def *acc = nir_channel(&b, &sample->def, 0);
   const struct {
      nir_tex_src_type type;
      nir_def *def;
   } nondirect[] = {
      {nir_tex_src_sampler_offset, dyn_idx},
      {nir_tex_src_texture_offset, dyn_idx},
      {nir_tex_src_sampler_handle, dyn_handle},
      {nir_tex_src_texture_handle, dyn_handle},
      {nir_tex_src_sampler_deref, deref},
      {nir_tex_src_texture_deref, deref},
      {nir_tex_src_backend1, dyn_idx},
      {nir_tex_src_backend2, dyn_idx},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(nondirect); ++i) {
      nir_tex_instr *q =
         emit_lod_bias_query(&b, 0, true, nondirect[i].type, nondirect[i].def);
      acc = nir_fadd(&b, acc, nir_f2f32(&b, &q->def));
   }
   nir_tex_instr *oob_q = emit_lod_bias_query(&b, 32, false, 0, NULL);
   acc = nir_fadd(&b, acc, nir_f2f32(&b, &oob_q->def));

   nir_store_output(&b, nir_replicate(&b, acc, 4), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);
   nir_validate_shader(b.shader, "zero lod bias pair before fold");

   CHECK(count_tex_op(b.shader, nir_texop_lod_bias) == 10);
   CHECK(count_tex_op(b.shader, nir_texop_txb) == 1);
   CHECK(count_tex_op(b.shader, nir_texop_tex) == 0);
   CHECK(agx_lod_bias_samplers(b.shader) == BITFIELD_BIT(0));

   CHECK(agx_nir_fold_zero_lod_bias(b.shader, BITFIELD_BIT(0)));
   NIR_PASS(_, b.shader, nir_opt_constant_folding);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "zero lod bias pair after fold");

   /* Direct slot 0 folded and txb(0) collapsed back to implicit-level tex;
    * all 8 non-direct queries and slot 32 remain untouched. */
   CHECK(count_tex_op(b.shader, nir_texop_lod_bias) == 9);
   CHECK(count_tex_op(b.shader, nir_texop_txb) == 0);
   CHECK(count_tex_op(b.shader, nir_texop_tex) == 1);
   CHECK(agx_lod_bias_samplers(b.shader) == 0);
   CHECK(!agx_nir_fold_zero_lod_bias(b.shader, ~0u));

   ralloc_free(b.shader);
}

/* Build a fragment shader mixing a direct slot-0 sample, a dynamically indexed
 * sampler array spanning past the bindful sampler bank (forcing binding
 * lowering to rewrite offsets into handles with sampler_index == 0), and an
 * optional bindless-handle sample. */
static nir_shader *
build_mixed_fs(bool include_bindless)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "mixed direct and dynamic fs");

   nir_def *fc = nir_load_frag_coord(&b);
   nir_def *coord = nir_trim_vector(&b, fc, 2);
   nir_def *dyn_idx = nir_iand_imm(&b, nir_f2u32(&b, nir_channel(&b, fc, 0)), 1);

   /* Direct slot 0. */
   nir_tex_instr *t_direct = nir_tex_instr_create(b.shader, 1);
   t_direct->op = nir_texop_tex;
   t_direct->sampler_dim = GLSL_SAMPLER_DIM_2D;
   t_direct->coord_components = 2;
   t_direct->dest_type = nir_type_float32;
   t_direct->texture_index = 0;
   t_direct->sampler_index = 0;
   t_direct->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   nir_def_init(&t_direct->instr, &t_direct->def, 4, 32);
   nir_builder_instr_insert(&b, &t_direct->instr);

   /* Dynamic array starting at slot 1, with samplers_used reaching slot 17 so
    * agx_nir_lower_bindings lowers both texture_offset and sampler_offset to
    * handles on both AGX2 (bindful=16) and AGX3 (bindful=15). */
   nir_tex_instr *t_dyn = nir_tex_instr_create(b.shader, 3);
   t_dyn->op = nir_texop_tex;
   t_dyn->sampler_dim = GLSL_SAMPLER_DIM_2D;
   t_dyn->coord_components = 2;
   t_dyn->dest_type = nir_type_float32;
   t_dyn->texture_index = 1;
   t_dyn->sampler_index = 1;
   t_dyn->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
   t_dyn->src[1] = nir_tex_src_for_ssa(nir_tex_src_texture_offset, dyn_idx);
   t_dyn->src[2] = nir_tex_src_for_ssa(nir_tex_src_sampler_offset, dyn_idx);
   nir_def_init(&t_dyn->instr, &t_dyn->def, 4, 32);
   nir_builder_instr_insert(&b, &t_dyn->instr);

   nir_def *color = nir_fadd(&b, &t_direct->def, &t_dyn->def);

   if (include_bindless) {
      nir_def *handle = nir_u2u64(&b, nir_iadd_imm(&b, dyn_idx, 100));
      nir_tex_instr *t_bindless = nir_tex_instr_create(b.shader, 3);
      t_bindless->op = nir_texop_tex;
      t_bindless->sampler_dim = GLSL_SAMPLER_DIM_2D;
      t_bindless->coord_components = 2;
      t_bindless->dest_type = nir_type_float32;
      t_bindless->texture_index = 0;
      t_bindless->sampler_index = 0;
      t_bindless->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord, coord);
      t_bindless->src[1] =
         nir_tex_src_for_ssa(nir_tex_src_texture_handle, handle);
      t_bindless->src[2] =
         nir_tex_src_for_ssa(nir_tex_src_sampler_handle, handle);
      nir_def_init(&t_bindless->instr, &t_bindless->def, 4, 32);
      nir_builder_instr_insert(&b, &t_bindless->instr);
      color = nir_fadd(&b, color, &t_bindless->def);
   }

   nir_store_output(&b, color, nir_imm_int(&b, 0), .base = 0, .write_mask = 0xf,
                    .io_semantics.location = FRAG_RESULT_DATA0);

   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   for (unsigned s = 0; s < 18; ++s) {
      BITSET_SET(b.shader->info.textures_used, s);
      BITSET_SET(b.shader->info.samplers_used, s);
   }
   b.shader->info.num_textures = 18;
   return b.shader;
}

/* Prove the direct-slot invariant across the actual early texture lowering +
 * agx_nir_lower_bindings (+ agx_nir_rebind_g17p_texture_crawls on AGX3) pass
 * sequence for both AGX2 (G14G) and AGX3 (G17P). */
static void
check_post_binding_lowering(bool agx3)
{
   nir_shader *nir = build_mixed_fs(true);

   if (agx3)
      agx_nir_lower_g17p_texture_early(nir, true, false);
   else
      agx_nir_lower_texture_early(nir, true);

   bool uses_bindless_samplers = false;
   struct agx_lower_bindings_options opts = {
      .uses_bindless_samplers = &uses_bindless_samplers,
      .g17p_plain_txf_crawl = agx3,
      .g17p_native_sampler_handles = agx3,
      .g17p_bindless_storage = agx3,
      .image_records = agx3 ? 3 : 2,
      .bindful_sampler_count =
         agx3 ? AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS : 16,
   };
   agx_nir_lower_bindings(nir, &opts);
   if (agx3)
      agx_nir_rebind_g17p_texture_crawls(nir);
   nir_validate_shader(nir, "post binding lowering");

   CHECK(uses_bindless_samplers);
   CHECK(count_tex_op(nir, nir_texop_lod_bias) == 3);
   /* Binding lowering rewrote the dynamic array's offsets into handles with
    * sampler_index == 0, and left the bindless handles intact with
    * sampler_index == 0. Only the direct slot-0 query has zero sources. */
   CHECK(agx_lod_bias_samplers(nir) == BITFIELD_BIT(0));

   CHECK(agx_nir_fold_zero_lod_bias(nir, BITFIELD_BIT(0)));
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_dce);
   nir_validate_shader(nir, "post binding lowering after fold");

   CHECK(count_tex_op(nir, nir_texop_lod_bias) == 2);
   CHECK(count_tex_op(nir, nir_texop_tex) == 1);
   CHECK(count_tex_op(nir, nir_texop_txb) == 2);
   CHECK(agx_lod_bias_samplers(nir) == 0);
   CHECK(!agx_nir_fold_zero_lod_bias(nir, ~0u));

   ralloc_free(nir);
}

static const struct asahi_fs_shader_key *
find_fs_key(struct agx_uncompiled_shader *so, struct agx_compiled_shader *cs)
{
   hash_table_foreach(so->variants, ent) {
      if (ent->data == cs)
         return (const struct asahi_fs_shader_key *)ent->key;
   }
   return NULL;
}

static void *
create_vs(struct pipe_context *pctx)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "zero lod bias vs");
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .base = 0, .write_mask = 0xf,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_shader_gather_info(b.shader, nir_shader_get_entrypoint(b.shader));
   struct pipe_shader_state state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = b.shader,
   };
   return pctx->create_vs_state(pctx, &state);
}

static void *
create_sampler(struct pipe_context *pctx, float lod_bias)
{
   struct pipe_sampler_state state = {
      .wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE,
      .min_img_filter = PIPE_TEX_FILTER_NEAREST,
      .mag_img_filter = PIPE_TEX_FILTER_NEAREST,
      .min_mip_filter = PIPE_TEX_MIPFILTER_NEAREST,
      .max_lod = 4.0f,
      .lod_bias = lod_bias,
   };
   return pctx->create_sampler_state(pctx, &state);
}

static void
draw_tri(struct pipe_context *pctx)
{
   struct pipe_draw_info info = {
      .mode = MESA_PRIM_TRIANGLES,
      .instance_count = 1,
   };
   struct pipe_draw_start_count_bias d = {.count = 3};
   pctx->draw_vbo(pctx, &info, 0, NULL, &d, 1);
}

/* Under the no-op DRM shim (G14G and G17P), verify variant keying and
 * sampler rebind from +0 -> -0 -> nonzero -> +0 at unchanged draw state. */
static void
check_rebind_and_draw(struct pipe_screen *screen)
{
   struct pipe_context *pctx = screen->context_create(screen, NULL, 0);
   CHECK(pctx);
   struct agx_context *ctx = agx_context(pctx);
   CHECK(ctx->support_lod_bias);

   void *vs = create_vs(pctx);
   struct pipe_shader_state fs_state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_mixed_fs(false),
   };
   struct agx_uncompiled_shader *fs_so = pctx->create_fs_state(pctx, &fs_state);
   CHECK(vs && fs_so);
   agx_shader_wait_ready(fs_so);
   CHECK(fs_so->info.lod_bias_samplers == BITFIELD_BIT(0));

   struct pipe_rasterizer_state rast = {
      .depth_clip_near = true,
      .depth_clip_far = true,
      .point_size = 1,
      .line_width = 1,
   };
   struct pipe_blend_state blend = {.rt[0].colormask = 0xf};
   struct pipe_depth_stencil_alpha_state depth = {0};
   void *cso_rast = pctx->create_rasterizer_state(pctx, &rast);
   void *cso_blend = pctx->create_blend_state(pctx, &blend);
   void *cso_depth = pctx->create_depth_stencil_alpha_state(pctx, &depth);
   void *cso_ve = pctx->create_vertex_elements_state(pctx, 0, NULL);
   CHECK(cso_rast && cso_blend && cso_depth && cso_ve);

   struct pipe_resource rt_templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 16,
      .height0 = 16,
      .depth0 = 1,
      .array_size = 1,
      .bind = PIPE_BIND_RENDER_TARGET,
   };
   struct pipe_resource *rt = screen->resource_create(screen, &rt_templ);
   struct pipe_resource tex_templ = {
      .target = PIPE_TEXTURE_2D,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .width0 = 16,
      .height0 = 16,
      .depth0 = 1,
      .array_size = 1,
      .last_level = 4,
      .bind = PIPE_BIND_SAMPLER_VIEW,
   };
   struct pipe_resource *tex = screen->resource_create(screen, &tex_templ);
   CHECK(rt && tex);

   struct pipe_sampler_view sv_templ;
   u_sampler_view_default_template(&sv_templ, tex, tex->format);
   struct pipe_sampler_view *views[18];
   for (unsigned i = 0; i < ARRAY_SIZE(views); ++i) {
      views[i] = pctx->create_sampler_view(pctx, tex, &sv_templ);
      CHECK(views[i]);
   }

   void *samp_zero = create_sampler(pctx, 0.0f);
   void *samp_neg_zero = create_sampler(pctx, -0.0f);
   void *samp_nonzero = create_sampler(pctx, 2.0f);
   CHECK(samp_zero && samp_neg_zero && samp_nonzero);

   void *samplers[18];
   samplers[0] = samp_zero;
   for (unsigned i = 1; i < ARRAY_SIZE(samplers); ++i)
      samplers[i] = samp_nonzero;

   pctx->bind_vs_state(pctx, vs);
   pctx->bind_fs_state(pctx, fs_so);
   pctx->bind_rasterizer_state(pctx, cso_rast);
   pctx->bind_blend_state(pctx, cso_blend);
   pctx->bind_depth_stencil_alpha_state(pctx, cso_depth);
   pctx->bind_vertex_elements_state(pctx, cso_ve);
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, ARRAY_SIZE(views), 0,
                           views);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0,
                             ARRAY_SIZE(samplers), samplers);

   struct pipe_framebuffer_state fb = {
      .width = 16,
      .height = 16,
      .layers = 1,
      .nr_cbufs = 1,
      .cbufs[0] = {.texture = rt, .format = rt->format},
   };
   struct pipe_viewport_state vp = {
      .scale = {8, 8, 0.5},
      .translate = {8, 8, 0.5},
   };
   pctx->set_framebuffer_state(pctx, &fb);
   pctx->set_viewport_states(pctx, 0, 1, &vp);
   pctx->set_sample_mask(pctx, ~0u);

   /* 1. Direct slot 0 has +0.0 bias, dynamic slots 1..17 have +2.0 bias. */
   draw_tri(pctx);
   struct agx_compiled_shader *fs_zero = ctx->fs;
   CHECK(fs_zero && !fs_zero->failed && ctx->batch);
   const struct asahi_fs_shader_key *k_zero = find_fs_key(fs_so, fs_zero);
   CHECK(k_zero && k_zero->zero_lod_bias == BITFIELD_BIT(0));
   CHECK(ctx->batch->stage_uniforms[MESA_SHADER_FRAGMENT].lod_bias[1] ==
         _mesa_float_to_half(2.0f));

   /* 2. Rebind slot 0 to -0.0 at unchanged draw state: reuses zero variant. */
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 1, &samp_neg_zero);
   CHECK(ctx->dirty & AGX_DIRTY_FS_PROG);
   draw_tri(pctx);
   CHECK(ctx->fs == fs_zero);

   /* 3. Rebind slot 0 from zero to nonzero (+2.0) at unchanged draw state:
    * must dirty FS and switch to the general unfolded variant. */
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 1, &samp_nonzero);
   CHECK(ctx->dirty & AGX_DIRTY_FS_PROG);
   draw_tri(pctx);
   struct agx_compiled_shader *fs_nonzero = ctx->fs;
   CHECK(fs_nonzero && !fs_nonzero->failed && fs_nonzero != fs_zero);
   const struct asahi_fs_shader_key *k_nonzero = find_fs_key(fs_so, fs_nonzero);
   CHECK(k_nonzero && k_nonzero->zero_lod_bias == 0);
   CHECK(ctx->batch->stage_uniforms[MESA_SHADER_FRAGMENT].lod_bias[0] ==
         _mesa_float_to_half(2.0f));

   /* 4. Rebind slot 0 back to +0.0 at unchanged draw state: reuses fs_zero. */
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 1, &samp_zero);
   CHECK(ctx->dirty & AGX_DIRTY_FS_PROG);
   draw_tri(pctx);
   CHECK(ctx->fs == fs_zero);

   pctx->bind_vs_state(pctx, NULL);
   pctx->bind_fs_state(pctx, NULL);
   pctx->bind_rasterizer_state(pctx, NULL);
   pctx->bind_blend_state(pctx, NULL);
   pctx->bind_depth_stencil_alpha_state(pctx, NULL);
   pctx->bind_vertex_elements_state(pctx, NULL);
   pctx->bind_sampler_states(pctx, MESA_SHADER_FRAGMENT, 0, 0, NULL);
   pctx->set_sampler_views(pctx, MESA_SHADER_FRAGMENT, 0, 0, ARRAY_SIZE(views),
                           NULL);
   for (unsigned i = 0; i < ARRAY_SIZE(views); ++i)
      pipe_sampler_view_reference(&views[i], NULL);
   pctx->delete_sampler_state(pctx, samp_zero);
   pctx->delete_sampler_state(pctx, samp_neg_zero);
   pctx->delete_sampler_state(pctx, samp_nonzero);
   pctx->delete_vs_state(pctx, vs);
   pctx->delete_fs_state(pctx, fs_so);
   pctx->delete_rasterizer_state(pctx, cso_rast);
   pctx->delete_blend_state(pctx, cso_blend);
   pctx->delete_depth_stencil_alpha_state(pctx, cso_depth);
   pctx->delete_vertex_elements_state(pctx, cso_ve);
   pipe_resource_reference(&rt, NULL);
   pipe_resource_reference(&tex, NULL);
   pctx->destroy(pctx);

   /* Contexts without GL LOD bias (PIPE_CONTEXT_NO_LOD_BIAS) emit no queries. */
   struct pipe_context *no_bias =
      screen->context_create(screen, NULL, PIPE_CONTEXT_NO_LOD_BIAS);
   CHECK(no_bias && !agx_context(no_bias)->support_lod_bias);
   struct pipe_shader_state fs_no_bias_state = {
      .type = PIPE_SHADER_IR_NIR,
      .ir.nir = build_mixed_fs(false),
   };
   struct agx_uncompiled_shader *fs_no_bias =
      no_bias->create_fs_state(no_bias, &fs_no_bias_state);
   CHECK(fs_no_bias);
   agx_shader_wait_ready(fs_no_bias);
   CHECK(fs_no_bias->info.lod_bias_samplers == 0);
   no_bias->delete_fs_state(no_bias, fs_no_bias);
   no_bias->destroy(no_bias);
}

int
main(int argc, char **argv)
{
   bool nir_only = argc > 1 && strcmp(argv[1], "--nir-only") == 0;

   glsl_type_singleton_init_or_ref();
   check_direct_and_nondirect_pair();
   check_post_binding_lowering(false);
   check_post_binding_lowering(true);

   if (nir_only) {
      glsl_type_singleton_decref();
      printf("nir: zero lod bias ok\n");
      return 0;
   }

   const char *chip = getenv("ASAHI_SHIM_CHIP");
   CHECK(chip);

   int fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
   CHECK(fd >= 0);
   static const driOptionDescription descriptions[] = {
#include "driinfo_asahi.h"
   };
   driOptionCache info, options;
   driParseOptionInfo(&info, descriptions, ARRAY_SIZE(descriptions));
   struct pipe_screen_config config = {
      .options = &options,
      .options_info = &info,
   };
   struct pipe_screen *screen = agx_screen_create(fd, NULL, &config);
   CHECK(screen);

   check_rebind_and_draw(screen);

   screen->destroy(screen);
   glsl_type_singleton_decref();
   printf("%s: zero lod bias ok\n", chip);
   return 0;
}
