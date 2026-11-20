/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "hk_nir_lower_gs.h"
#include "asahi/compiler/agx_compile.h"
#include "asahi/compiler/agx_nir_gen1_ray_query.h"
#include "nir_builder.h"
#include "poly/nir/poly_nir.h"
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return false; \
} } while (0)

static bool
check_variant(nir_shader *nir, bool raster, bool require_query, bool instances)
{
   if (!nir)
      return !require_query;
   nir_validate_shader(nir, "GS native query split");
   unsigned intersects = 0, context_footers = 0, as_loads = 0, table_stores = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         CHECK(!nir_intrinsic_is_ray_query(intr->intrinsic));
         if (intr->intrinsic == nir_intrinsic_intersect_ray_gen1_agx)
            ++intersects;
         if (intr->intrinsic == nir_intrinsic_store_scs_agx &&
             nir_intrinsic_base(intr) == 0x6068) {
            CHECK(intr->src[0].ssa->num_components == (raster ? 4 : 2));
            ++context_footers;
         }
         if (intr->intrinsic == nir_intrinsic_load_global_constant &&
             intr->def.bit_size == 64 && nir_src_is_const(intr->src[0]) &&
             nir_src_as_uint(intr->src[0]) == 0x10003000000ull)
            ++as_loads;
         if (intr->intrinsic == nir_intrinsic_store_global &&
             intr->src[0].ssa->bit_size == 64)
            ++table_stores;
      }
   }
   CHECK(nir->info.ray_queries == 0);
   if (require_query) {
      CHECK(intersects && context_footers && as_loads);
      if (!instances)
         CHECK(table_stores);
   }
   return true;
}

static bool
check_split(bool dynamic_count, bool instances)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS AS descriptor query");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   nir_variable *query = nir_local_variable_create(
      b.impl, glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   /* The bare 64-bit address produced by AS descriptor lowering, loaded at
    * execution time so it observes descriptor updates between draws. */
   nir_def *as = nir_load_global_constant(&b, 1, 64,
      nir_imm_int64(&b, 0x10003000000ull), .align_mul = 8);
   nir_rq_initialize(&b, rq, as, nir_imm_int(&b, 1), nir_imm_int(&b, 255),
      nir_imm_vec3(&b, 0, 0, -1), nir_imm_float(&b, 0),
      nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   nir_push_loop(&b);
   nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
   nir_pop_loop(&b, NULL);
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = true);
   if (dynamic_count)
      nir_push_if(&b, nir_ine_imm(&b, type, 0));
   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, type),
      nir_imm_float(&b, 0), nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);
   if (dynamic_count)
      nir_pop_if(&b, NULL);
   nir_shader_gather_info(b.shader, b.impl);
   CHECK(b.shader->info.ray_queries == 1);
   agx_preprocess_nir_g17p(b.shader);
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = instances ? 118 : 124,
      .bvh_uniform = 96,
      .status_uniform = 128,
      .context_uniform = 160,
      .as_table_uniform = 164,
      .as_table_first = 1,
      .launch_header = instances ? 0x210000 : 0x110000,
      .instance_traversal = instances,
      .candidates = instances,
      .compute_kernel = true,
      .graphics_stage = true,
   };
   nir_shader *count = NULL, *rast = NULL, *pre_gs = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre_gs, &info, &layout,
                         POLY_GS_SIDE_EFFECTS_REPLAY, NULL));
   CHECK(rast && rast->info.stage == MESA_SHADER_VERTEX);
   CHECK(!!count == dynamic_count);
   CHECK(check_variant(rast, true, true, instances));
   CHECK(check_variant(b.shader, false, dynamic_count, instances));
   CHECK(check_variant(count, false, dynamic_count, instances));
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre_gs);
   printf("GS query split: dynamic=%u instances=%u PASS\n", dynamic_count, instances);
   return true;
}

/* CTS elects one invocation with CAS, then increments only on a successful
 * query. The geometry prepass must not claim that election. */
static bool
check_atomic_guard(bool native, bool instances,
                   enum poly_gs_side_effects side_effects)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS query atomic guard");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 3;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   nir_variable *query = nir_local_variable_create(
      b.impl, glsl_uint64_t_type(), "query");
   query->data.ray_query = true;
   nir_def *rq = &nir_build_deref_var(&b, query)->def;
   nir_def *addr = nir_imm_int64(&b, 0x10004000000ull);
   nir_def *flag = nir_global_atomic_swap(&b, 32, addr,
      nir_imm_int(&b, 0), nir_imm_int(&b, 1),
      .atomic_op = nir_atomic_op_cmpxchg);
   nir_push_if(&b, nir_ieq_imm(&b, flag, 0));
   nir_def *as = nir_load_global_constant(&b, 1, 64,
      nir_imm_int64(&b, 0x10003000000ull), .align_mul = 8);
   nir_rq_initialize(&b, rq, as, nir_imm_int(&b, instances ? 0 : 1),
      nir_imm_int(&b, 255),
      nir_imm_vec3(&b, 0, 0, 0), nir_imm_float(&b, 1),
      nir_imm_vec3(&b, 0, 0, 1), nir_imm_float(&b, 10));
   nir_variable *found = nir_local_variable_create(b.impl, glsl_bool_type(), "found");
   nir_store_var(&b, found, nir_imm_false(&b), 1);
   nir_push_loop(&b);
   nir_break_if(&b, nir_inot(&b, nir_rq_proceed(&b, 1, rq)));
   nir_def *type = nir_rq_load(&b, 1, 32, rq,
      .ray_query_value = nir_ray_query_value_intersection_type,
      .committed = !instances);
   nir_store_var(&b, found, nir_ior(&b, nir_load_var(&b, found),
      nir_ine_imm(&b, type, 0)), 1);
   nir_pop_loop(&b, NULL);
   nir_push_if(&b, nir_load_var(&b, found));
   nir_global_atomic(&b, 32, addr, nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);
   nir_pop_if(&b, NULL);
   nir_pop_if(&b, NULL);
   for (unsigned i = 0; i < 3; ++i) {
      nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
         .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics.location = VARYING_SLOT_POS);
      nir_emit_vertex(&b, 0);
   }
   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);
   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   const struct agx_gen1_ray_query_layout layout = {
      .payload_uniform = instances ? 118 : 124,
      .bvh_uniform = 96,
      .status_uniform = 128,
      .context_uniform = 160,
      .as_table_uniform = 164,
      .as_table_first = 1,
      .launch_header = instances ? 0x210000 : 0x110000,
      .instance_traversal = instances,
      .candidates = instances,
      .compute_kernel = true,
      .graphics_stage = true,
   };
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info,
                         native ? &layout : NULL, side_effects,
                         NULL));
   CHECK(!count);
   unsigned atomics = 0, queries = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            atomics += op == nir_intrinsic_global_atomic ||
                       op == nir_intrinsic_global_atomic_swap;
            queries += nir_intrinsic_is_ray_query(op) ||
                       op == nir_intrinsic_intersect_ray_gen1_agx;
         }
      }
   }
   if (native) {
      CHECK(atomics == 0 && queries == 0);
   } else {
      /* The default/software profile must retain the default Poly side-effect
       * rules. Its queries have not yet been lowered when Poly runs. */
      CHECK(atomics == 1 && queries > 0);
   }
   unsigned swaps = 0, adds = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(rast)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic == nir_intrinsic_global_atomic_swap) {
            CHECK(nir_intrinsic_atomic_op(intr) == nir_atomic_op_cmpxchg);
            ++swaps;
         } else if (intr->intrinsic == nir_intrinsic_global_atomic) {
            CHECK(nir_intrinsic_atomic_op(intr) == nir_atomic_op_iadd);
            ++adds;
         }
      }
   }
   CHECK(swaps == 1 && adds == 1);
   nir_validate_shader(b.shader, "GS prepass query policy");
   nir_validate_shader(rast, "GS replay retains query election and success");
   if (native)
      CHECK(check_variant(rast, true, true, instances));
   printf("GS query election: native=%u instances=%u PASS\n", native, instances);
   ralloc_free(b.shader); ralloc_free(count); ralloc_free(rast); ralloc_free(pre);
   return true;
}

static unsigned
count_intrinsics(nir_shader *nir, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic &&
             nir_instr_as_intrinsic(instr)->intrinsic == op)
            ++n;
      }
   }
   return n;
}

static bool
expr_uses_const_u64(nir_def *def, uint64_t val, unsigned depth)
{
   if (depth > 8)
      return false;
   nir_instr *parent = nir_def_instr(def);
   if (parent->type == nir_instr_type_load_const) {
      nir_load_const_instr *lc = nir_instr_as_load_const(parent);
      return lc->def.bit_size == 64 && lc->value[0].u64 == val;
   }
   if (parent->type == nir_instr_type_alu) {
      nir_alu_instr *alu = nir_instr_as_alu(parent);
      for (unsigned i = 0; i < nir_op_infos[alu->op].num_inputs; ++i) {
         if (expr_uses_const_u64(alu->src[i].src.ssa, val, depth + 1))
            return true;
      }
   }
   return false;
}

static unsigned
count_global_ops_at(nir_shader *nir, nir_intrinsic_op op, uint64_t base_addr)
{
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != op)
            continue;
         unsigned addr_src = op == nir_intrinsic_store_global ? 1 : 0;
         if (expr_uses_const_u64(intr->src[addr_src].ssa, base_addr, 0))
            ++n;
      }
   }
   return n;
}

static bool
has_loop(nir_shader *nir)
{
   nir_foreach_block(block, nir_shader_get_entrypoint(nir)) {
      for (nir_cf_node *node = block->cf_node.parent; node; node = node->parent) {
         if (node->type == nir_cf_node_loop)
            return true;
      }
   }
   return false;
}

/* With main_side_effects, API stores, unused atomics, and loops that only feed
 * them stay in the main kernel and are stripped from count and raster
 * variants, and a 0-vertex GS emits no dummy primitive. */
static bool
check_main_side_effects(bool emit_vertices, bool dynamic_emit)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS main side effects");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 3;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   if (emit_vertices)
      b.shader->info.outputs_written = VARYING_BIT_POS;

   const uint64_t api_base = 0x10004000000ull;
   const uint64_t atomic_addr = 0x10005000000ull;
   nir_variable *iter = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_store_var(&b, iter, nir_imm_int(&b, 0), 1);
   nir_def *base = nir_imm_int64(&b, api_base);
   nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iter);
      nir_break_if(&b, nir_uge_imm(&b, i, 1024));
      nir_def *addr = nir_iadd(&b, base, nir_u2u64(&b, nir_ishl_imm(&b, i, 2)));
      nir_store_global(&b, i, addr, .write_mask = 1, .align_mul = 4);
      nir_store_var(&b, iter, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   /* Unused atomic: runs once in main, stripped from count and rast. */
   nir_global_atomic(&b, 32, nir_imm_int64(&b, atomic_addr),
                     nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);

   if (emit_vertices) {
      if (dynamic_emit) {
         nir_def *cond = nir_ine_imm(&b, nir_load_primitive_id(&b), 0);
         nir_push_if(&b, cond);
      }
      for (unsigned i = 0; i < 3; ++i) {
         nir_def *pos = nir_load_per_vertex_input(
            &b, 4, 32, nir_imm_int(&b, i), nir_imm_int(&b, 0),
            .io_semantics.location = VARYING_SLOT_POS,
            .io_semantics.num_slots = 1, .dest_type = nir_type_float32);
         nir_store_output(&b, pos, nir_imm_int(&b, 0),
                          .write_mask = 15, .src_type = nir_type_float32,
                          .io_semantics.location = VARYING_SLOT_POS);
         nir_emit_vertex(&b, 0);
      }
      if (dynamic_emit)
         nir_pop_if(&b, NULL);
   }

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(rast);
   CHECK(!!count == (emit_vertices && dynamic_emit));
   nir_validate_shader(b.shader, "GS main side effects");
   nir_validate_shader(rast, "GS rast stripped side effects");
   if (count)
      nir_validate_shader(count, "GS count stripped side effects");

   CHECK(count_global_ops_at(b.shader, nir_intrinsic_store_global, api_base) == 1);
   CHECK(count_global_ops_at(b.shader, nir_intrinsic_global_atomic, atomic_addr) == 1);
   CHECK(has_loop(b.shader));

   CHECK(count_global_ops_at(rast, nir_intrinsic_store_global, api_base) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_global_atomic, atomic_addr) == 0);
   CHECK(!has_loop(rast));

   if (count) {
      CHECK(count_global_ops_at(count, nir_intrinsic_store_global, api_base) == 0);
      CHECK(count_global_ops_at(count, nir_intrinsic_global_atomic, atomic_addr) == 0);
      CHECK(!has_loop(count));
   }

   /* A loop before emission forces dynamic indexing (3 vertices + restart). */
   CHECK(info.max_indices == (emit_vertices ? 4 : 0));

   printf("GS main side effects: emit=%u dynamic=%u PASS\n",
          emit_vertices, dynamic_emit);
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under MAIN_IF_INDEPENDENT, an output-feeding atomic followed by an unused
 * atomic must fall back to replay so neither atomic is duplicated or reordered
 * between main and rast. */
static bool
check_independent_fallback_atomic_reorder(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS atomic reorder fallback");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;

   const uint64_t slot_addr = 0x10006000000ull;
   nir_def *addr = nir_imm_int64(&b, slot_addr);
   nir_def *old = nir_global_atomic(&b, 32, addr, nir_imm_int(&b, 1),
                                    .atomic_op = nir_atomic_op_iadd);
   nir_global_atomic(&b, 32, addr, nir_imm_int(&b, 7),
                     .atomic_op = nir_atomic_op_xchg);
   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, old), nir_imm_float(&b, 0),
                                 nir_imm_float(&b, 0), nir_imm_float(&b, 1)),
                    nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(!count && rast);
   nir_validate_shader(b.shader, "GS atomic reorder fallback main");
   nir_validate_shader(rast, "GS atomic reorder fallback rast");

   /* Both atomics remain in rast in source order; main drops both. */
   CHECK(count_intrinsics(b.shader, nir_intrinsic_global_atomic) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_global_atomic, slot_addr) == 2);

   printf("GS independent fallback atomic reorder: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under MAIN_IF_INDEPENDENT, any replay read of global, image, or texture
 * memory alongside an API write must fall back to replay so replay cannot
 * observe post-main writes (including lowered buffer-texel 2D fetches and
 * lowered image atomics). */
static bool
check_independent_fallback_mutable_read(unsigned kind)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS mutable read fallback");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;

   const uint64_t write_addr = 0x10007000000ull;
   nir_def *read_val = NULL;
   if (kind == 3) {
      /* Lowered UBO read (load_global_constant) that may alias a dependent
       * SSBO store to the same address. */
      read_val = nir_load_global_constant(&b, 1, 32,
                                          nir_imm_int64(&b, write_addr),
                                          .align_mul = 4);
      nir_store_global(&b, nir_iadd_imm(&b, read_val, 1),
                       nir_imm_int64(&b, write_addr), .write_mask = 1,
                       .align_mul = 4);
   } else {
      nir_store_global(&b, nir_imm_int(&b, 42), nir_imm_int64(&b, write_addr),
                       .write_mask = 1, .align_mul = 4);

      if (kind == 0) {
         read_val = nir_load_global(&b, 1, 32, nir_imm_int64(&b, write_addr),
                                    .align_mul = 4);
      } else if (kind == 1) {
         nir_def *v = nir_bindless_image_load(
            &b, 4, 32, nir_imm_int64(&b, 0), nir_imm_ivec4(&b, 0, 0, 0, 0),
            nir_imm_int(&b, 0), nir_imm_int(&b, 0),
            .image_dim = GLSL_SAMPLER_DIM_2D, .dest_type = nir_type_uint32);
         read_val = nir_channel(&b, v, 0);
      } else {
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
         tex->op = nir_texop_txf;
         tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
         tex->dest_type = nir_type_uint32;
         tex->coord_components = 2;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_coord,
                                           nir_imm_ivec2(&b, 0, 0));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         read_val = nir_channel(&b, &tex->def, 0);
      }
   }

   nir_store_output(&b, nir_vec4(&b, nir_u2f32(&b, read_val),
                                 nir_imm_float(&b, 0), nir_imm_float(&b, 0),
                                 nir_imm_float(&b, 1)),
                    nir_imm_int(&b, 0), .write_mask = 15,
                    .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(!count && rast);
   nir_validate_shader(b.shader, "GS mutable read fallback main");
   nir_validate_shader(rast, "GS mutable read fallback rast");

   CHECK(count_global_ops_at(b.shader, nir_intrinsic_store_global, write_addr) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_store_global, write_addr) == 1);

   printf("GS independent fallback mutable read kind=%u: PASS\n", kind);
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under POLY_GS_SIDE_EFFECTS_MAIN (M5), an unused atomic alongside a separate
 * SSBO read controlling emission keeps executing once in main, never repeated
 * in count or per-vertex rast. */
static bool
check_m5_unused_atomic_with_ssbo_read(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS M5 atomic plus SSBO read");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_TRIANGLE_STRIP;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 3;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;

   const uint64_t counter_addr = 0x10008000000ull;
   const uint64_t ssbo_addr = 0x10008100000ull;
   nir_global_atomic(&b, 32, nir_imm_int64(&b, counter_addr),
                     nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);
   nir_def *flag = nir_load_global(&b, 1, 32, nir_imm_int64(&b, ssbo_addr),
                                   .align_mul = 4);
   nir_push_if(&b, nir_ine_imm(&b, flag, 0));
   for (unsigned i = 0; i < 3; ++i) {
      nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                       .write_mask = 15, .src_type = nir_type_float32,
                       .io_semantics.location = VARYING_SLOT_POS);
      nir_emit_vertex(&b, 0);
   }
   nir_pop_if(&b, NULL);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN, NULL));
   CHECK(count && rast);
   nir_validate_shader(b.shader, "GS M5 atomic + SSBO read main");
   nir_validate_shader(count, "GS M5 atomic + SSBO read count");
   nir_validate_shader(rast, "GS M5 atomic + SSBO read rast");

   CHECK(count_global_ops_at(b.shader, nir_intrinsic_global_atomic, counter_addr) == 1);
   CHECK(count_global_ops_at(count, nir_intrinsic_global_atomic, counter_addr) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_global_atomic, counter_addr) == 0);

   printf("GS M5 atomic + SSBO read: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Stripping an unused atomic in replay must also strip fence_mem_to_tex_agx
 * when no atomics remain, so a loop is not kept alive for the fence alone. */
static bool
check_main_side_effects_fence_strip(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS fence_mem_to_tex strip");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;

   const uint64_t atomic_addr = 0x10009000000ull;
   nir_variable *iter = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_store_var(&b, iter, nir_imm_int(&b, 0), 1);
   nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iter);
      nir_break_if(&b, nir_uge_imm(&b, i, 64));
      nir_global_atomic(&b, 32, nir_imm_int64(&b, atomic_addr),
                        nir_imm_int(&b, 1), .atomic_op = nir_atomic_op_iadd);
      nir_fence_mem_to_tex_agx(&b);
      nir_store_var(&b, iter, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(rast);
   nir_validate_shader(b.shader, "GS fence strip main");
   nir_validate_shader(rast, "GS fence strip rast");

   CHECK(count_intrinsics(b.shader, nir_intrinsic_fence_mem_to_tex_agx) == 1);
   CHECK(has_loop(b.shader));
   CHECK(count_intrinsics(rast, nir_intrinsic_fence_mem_to_tex_agx) == 0);
   CHECK(!has_loop(rast));

   printf("GS main side effects fence strip: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under MAIN_IF_INDEPENDENT, if replay retains a subgroup operation (e.g.
 * subgroupElect() feeding both a store and an output), fall back to replay so
 * both use the same subgroup execution. */
static bool
check_independent_fallback_subgroup(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS subgroup fallback");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_VAR(0);

   const uint64_t write_addr = 0x1000a000000ull;
   nir_def *elected = nir_b2i32(&b, nir_elect(&b, 1));
   nir_store_global(&b, elected, nir_imm_int64(&b, write_addr),
                    .write_mask = 1, .align_mul = 4);
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_store_output(&b, elected, nir_imm_int(&b, 0),
                    .write_mask = 1, .src_type = nir_type_uint32,
                    .io_semantics.location = VARYING_SLOT_VAR0);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(!count && rast);
   nir_validate_shader(b.shader, "GS subgroup fallback main");
   nir_validate_shader(rast, "GS subgroup fallback rast");

   CHECK(count_global_ops_at(b.shader, nir_intrinsic_store_global, write_addr) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_store_global, write_addr) == 1);

   printf("GS independent fallback subgroup: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under MAIN_IF_INDEPENDENT, a surviving memory barrier in replay (e.g.
 * memoryBarrierImage() in a store loop) triggers fallback. */
static bool
check_independent_fallback_barrier(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS barrier fallback");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS;

   const uint64_t write_addr = 0x1000b000000ull;
   nir_variable *iter = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_store_var(&b, iter, nir_imm_int(&b, 0), 1);
   nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iter);
      nir_break_if(&b, nir_uge_imm(&b, i, 1024));
      nir_store_global(&b, i, nir_imm_int64(&b, write_addr),
                       .write_mask = 1, .align_mul = 4);
      nir_barrier(&b, .memory_scope = SCOPE_DEVICE,
                  .memory_semantics = NIR_MEMORY_ACQ_REL,
                  .memory_modes = nir_var_image);
      nir_store_var(&b, iter, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_g17p(b.shader);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(rast);
   nir_validate_shader(b.shader, "GS barrier fallback main");
   nir_validate_shader(rast, "GS barrier fallback rast");

   CHECK(count_global_ops_at(b.shader, nir_intrinsic_store_global, write_addr) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_store_global, write_addr) == 1);

   printf("GS independent fallback barrier: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under MAIN_IF_INDEPENDENT, a clustered boolean reduction lowered through the
 * G16G preprocessing path becomes quad_ballot_agx; if it survives in replay it
 * must also trigger fallback so main and rast cannot observe different quads. */
static bool
check_independent_fallback_quad_ballot(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS quad_ballot_agx fallback");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 1;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_VAR(0);

   const uint64_t write_addr = 0x1000c000000ull;
   nir_def *cond = nir_ine_imm(&b, nir_load_primitive_id(&b), 0);
   nir_def *reduced = nir_b2i32(
      &b, nir_reduce(&b, cond, .reduction_op = nir_op_ior, .cluster_size = 4));
   nir_store_global(&b, reduced, nir_imm_int64(&b, write_addr),
                    .write_mask = 1, .align_mul = 4);
   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_store_output(&b, reduced, nir_imm_int(&b, 0),
                    .write_mask = 1, .src_type = nir_type_uint32,
                    .io_semantics.location = VARYING_SLOT_VAR0);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G16G);
   CHECK(count_intrinsics(b.shader, nir_intrinsic_quad_ballot_agx) == 1);

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(!count && rast);
   nir_validate_shader(b.shader, "GS quad_ballot_agx fallback main");
   nir_validate_shader(rast, "GS quad_ballot_agx fallback rast");

   CHECK(count_global_ops_at(b.shader, nir_intrinsic_store_global, write_addr) == 0);
   CHECK(count_global_ops_at(rast, nir_intrinsic_store_global, write_addr) == 1);

   printf("GS independent fallback quad_ballot_agx: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

/* Under MAIN_IF_INDEPENDENT, the probe must see the same post-lower_gs_inputs
 * loop body as the real raster variant, so a loop that fails to unroll after
 * input lowering and keeps an atomic use in raster never retains the atomic in
 * both main and raster. */
static bool
check_independent_fallback_unroll_mismatch(void)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_GEOMETRY, &agx_nir_options, "GS unroll mismatch fallback");
   b.shader->info.io_lowered = true;
   b.shader->info.gs.input_primitive = MESA_PRIM_TRIANGLES;
   b.shader->info.gs.output_primitive = MESA_PRIM_POINTS;
   b.shader->info.gs.vertices_in = 3;
   b.shader->info.gs.vertices_out = 1;
   b.shader->info.gs.invocations = 1;
   b.shader->info.gs.active_stream_mask = 1;
   b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_VAR(0);

   const uint64_t atomic_addr = 0x1000d000000ull;
   nir_def *old = nir_global_atomic(&b, 32, nir_imm_int64(&b, atomic_addr),
                                    nir_imm_int(&b, 1),
                                    .atomic_op = nir_atomic_op_iadd);

   /* Bound the loop by load_invocation_id + 32 so agx_preprocess_nir_native()
    * keeps it rolled, while poly_nir_lower_gs_with_options() folds
    * invocation_id to 0 before opt_lowered_gs(). After lower_gs_inputs()
    * expands the per-vertex load, the loop is too large to unroll, so the
    * unreachable reference to the atomic result stays live in rast and must
    * force fallback. */
   nir_variable *iter = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *limit = nir_iadd_imm(&b, nir_load_invocation_id(&b), 32);
   nir_store_var(&b, iter, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iter);
      nir_break_if(&b, nir_uge(&b, i, limit));
      nir_def *v = nir_load_per_vertex_input(
         &b, 1, 32, nir_iand_imm(&b, i, 1), nir_imm_int(&b, 0),
         .io_semantics.location = VARYING_SLOT_VAR0,
         .io_semantics.num_slots = 1, .dest_type = nir_type_uint32);
      nir_def *cur = nir_iadd(&b, nir_load_var(&b, acc), v);
      /* Unreachable iteration referencing the atomic result. */
      cur = nir_bcsel(&b, nir_ieq_imm(&b, i, 99), old, cur);
      nir_store_var(&b, acc, cur, 1);
      nir_store_var(&b, iter, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, NULL);

   nir_store_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), nir_imm_int(&b, 0),
                    .write_mask = 15, .src_type = nir_type_float32,
                    .io_semantics.location = VARYING_SLOT_POS);
   nir_store_output(&b, nir_load_var(&b, acc), nir_imm_int(&b, 0),
                    .write_mask = 1, .src_type = nir_type_uint32,
                    .io_semantics.location = VARYING_SLOT_VAR0);
   nir_emit_vertex(&b, 0);

   nir_shader_gather_info(b.shader, b.impl);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G16G);
   CHECK(has_loop(b.shader));

   nir_shader *count = NULL, *rast = NULL, *pre = NULL;
   struct poly_gs_info info;
   CHECK(hk_nir_lower_gs(b.shader, &count, &rast, &pre, &info, NULL,
                         POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT, NULL));
   CHECK(!count && rast);
   nir_validate_shader(b.shader, "GS unroll mismatch main");
   nir_validate_shader(rast, "GS unroll mismatch rast");

   unsigned main_atomics =
      count_global_ops_at(b.shader, nir_intrinsic_global_atomic, atomic_addr);
   unsigned rast_atomics =
      count_global_ops_at(rast, nir_intrinsic_global_atomic, atomic_addr);
   CHECK(has_loop(rast));
   CHECK(main_atomics == 0 && rast_atomics == 1);

   printf("GS independent fallback unroll mismatch: PASS\n");
   ralloc_free(b.shader);
   ralloc_free(count);
   ralloc_free(rast);
   ralloc_free(pre);
   return true;
}

int
main(void)
{
   glsl_type_singleton_init_or_ref();
   bool valid = check_atomic_guard(false, false, POLY_GS_SIDE_EFFECTS_REPLAY);
   valid &= check_atomic_guard(true, false, POLY_GS_SIDE_EFFECTS_REPLAY);
   valid &= check_atomic_guard(true, true, POLY_GS_SIDE_EFFECTS_REPLAY);
   valid &= check_atomic_guard(true, false, POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT);
   valid &= check_atomic_guard(true, true, POLY_GS_SIDE_EFFECTS_MAIN_IF_INDEPENDENT);
   for (unsigned dynamic = 0; dynamic < 2; ++dynamic)
      for (unsigned instances = 0; instances < 2; ++instances)
         valid &= check_split(dynamic, instances);
   valid &= check_main_side_effects(false, false);
   valid &= check_main_side_effects(true, false);
   valid &= check_main_side_effects(true, true);
   valid &= check_independent_fallback_atomic_reorder();
   for (unsigned kind = 0; kind < 4; ++kind)
      valid &= check_independent_fallback_mutable_read(kind);
   valid &= check_m5_unused_atomic_with_ssbo_read();
   valid &= check_main_side_effects_fence_strip();
   valid &= check_independent_fallback_subgroup();
   valid &= check_independent_fallback_barrier();
   valid &= check_independent_fallback_quad_ballot();
   valid &= check_independent_fallback_unroll_mismatch();
   glsl_type_singleton_decref();
   return valid ? 0 : 1;
}
