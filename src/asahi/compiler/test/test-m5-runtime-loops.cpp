/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_constant_expressions.h"
extern "C" {
#include "agx_nir.h"
}

static nir_builder
compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static unsigned
count_loops(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type == nir_cf_node_loop &&
             nir_loop_first_block(nir_cf_node_as_loop(parent)) == block)
            n++;
      }
   }
   return n;
}

/* A storage buffer address as the descriptor lowering builds it: the
 * descriptor's address (its .xy) from a buffer binding descriptor load
 * (ACCESS_BUFFER_BINDING_AGX) plus a 32-bit offset. */
static nir_def *
buffer_address(nir_builder *b, nir_def *byte_offset, bool vec_pack = false)
{
   nir_def *set = nir_load_preamble(b, 1, 64, .base = 4);
   nir_def *desc = nir_load_global_constant(
      b, 4, 32, set,
      .access = (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                      ACCESS_BUFFER_BINDING_AGX),
      .align_mul = 16);
   nir_def *base = vec_pack
                      ? nir_pack_64_2x32(b, nir_trim_vector(b, desc, 2))
                      : nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                               nir_channel(b, desc, 1));
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

/* A physical storage buffer pointer, a 64-bit value of its own, plus a 32-bit
 * offset the shader computed: the offset may wrap into another allocation. */
static nir_def *
physical_address(nir_builder *b, nir_def *byte_offset)
{
   nir_def *base = nir_load_push_constant(b, 1, 64, nir_imm_int(b, 0), .range = 32);
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

enum packed_uvec4_source {
   UVEC4_PUSH_CONSTANT,
   UVEC4_UBO_LOAD,
   UVEC4_SSBO_LOAD,
   UVEC4_SPECULATIVE_ROOT_LOAD,
};

/* A pointer packed from the .xy of a loaded uvec4 (not a buffer binding
 * descriptor): must be refused just like a direct 64-bit physical pointer. */
static nir_def *
packed_uvec4_address(nir_builder *b, nir_def *byte_offset,
                     packed_uvec4_source src, bool vec_pack)
{
   nir_def *v = NULL;
   switch (src) {
   case UVEC4_PUSH_CONSTANT:
      v = nir_load_push_constant(b, 4, 32, nir_imm_int(b, 16), .range = 32);
      break;
   case UVEC4_UBO_LOAD:
      v = nir_load_global_constant(
         b, 4, 32, buffer_address(b, nir_imm_int(b, 0)),
         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE |
                                         ACCESS_CAN_REORDER),
         .align_mul = 16);
      break;
   case UVEC4_SSBO_LOAD:
      v = nir_load_global(b, 4, 32, buffer_address(b, nir_imm_int(b, 0)),
                          .align_mul = 16);
      break;
   case UVEC4_SPECULATIVE_ROOT_LOAD:
      v = nir_load_global_constant(
         b, 4, 32, nir_load_preamble(b, 1, 64, .base = 0),
         .access = ACCESS_CAN_SPECULATE, .align_mul = 16);
      break;
   }
   nir_def *base = vec_pack
                      ? nir_pack_64_2x32(b, nir_trim_vector(b, v, 2))
                      : nir_pack_64_2x32_split(b, nir_channel(b, v, 0),
                                               nir_channel(b, v, 1));
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

enum loop_shape { TOP_TESTED, BOTTOM_TESTED };

/* for (i = lo; i <= hi; i++) sum += buf[i], optionally with a barrier. */
static nir_shader *
counted_loop(loop_shape shape, nir_def **hi_out, bool barrier, int64_t hi_const)
{
   nir_builder b = compute_shader("runtime-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *lo = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *hi = hi_const == INT64_MAX
                    ? nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32)
                    : nir_imm_int(&b, (uint32_t)hi_const);
   *hi_out = hi;
   nir_store_var(&b, iv, lo, 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      if (shape == TOP_TESTED) {
         nir_break_if(&b, nir_ilt(&b, hi, i));
      }
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, i, 4)),
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .align_mul = 4);
      if (barrier)
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_def *next = nir_iadd_imm(&b, i, 1);
      nir_store_var(&b, iv, next, 1);
      if (shape == BOTTOM_TESTED) {
         nir_break_if(&b, nir_ilt(&b, hi, next));
      }
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);
   return b.shader;
}

TEST(M5RuntimeLoops, TopTestedUnrolls)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, false, INT64_MAX);
   ASSERT_EQ(count_loops(s), 1u);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   nir_validate_shader(s, "top-tested unroll");
   EXPECT_EQ(count_loops(s), 2u);
   /* Both loops are marked; a second run leaves them alone. */
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256));
   ralloc_free(s);
}

TEST(M5RuntimeLoops, BottomTestedUnrolls)
{
   nir_def *hi;
   nir_shader *s = counted_loop(BOTTOM_TESTED, &hi, false, INT64_MAX);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   nir_validate_shader(s, "bottom-tested unroll");
   EXPECT_EQ(count_loops(s), 2u);
   ralloc_free(s);
}

TEST(M5RuntimeLoops, BarrierRefused)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, true, INT64_MAX);
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256));
   EXPECT_EQ(count_loops(s), 1u);
   ralloc_free(s);
}

/* for (i = 0; i < n; i++) acc += op, with op what the M5 subgroup lowering
 * leaves of subgroupElect() (active-lane index == 0), of
 * subgroupBallotBitCount(subgroupBallot(true)) (the active-lane count), or of
 * subgroupQuadAny() (a quad ballot), or, as a control, a plain compare. */
enum lane_op { LANE_NONE, LANE_ELECT, LANE_BALLOT_COUNT, LANE_QUAD_VOTE };

static nir_shader *
lane_state_loop(lane_op op)
{
   nir_builder b = compute_shader("lane-state-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_def *lane_id = nir_load_local_invocation_index(&b);
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_uge(&b, i, n));
      nir_def *cond = nir_ult(&b, i, lane_id);
      nir_def *v = NULL;
      switch (op) {
      case LANE_NONE:
         v = nir_b2i32(&b, cond);
         break;
      case LANE_ELECT:
         v = nir_b2i32(&b, nir_ieq_imm(&b, nir_load_active_subgroup_invocation_agx(&b, 16), 0));
         break;
      case LANE_BALLOT_COUNT:
         v = nir_load_active_subgroup_count_agx(&b, 32);
         break;
      case LANE_QUAD_VOTE:
         v = nir_b2i32(&b, nir_ine_imm(&b, nir_quad_ballot_agx(&b, 16, cond), 0));
         break;
      }
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc), buffer_address(&b, nir_imm_int(&b, 0)),
                    .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "lane-state loop before");
   return b.shader;
}

/* These intrinsics carry no subgroup flag, but their result depends on which
 * invocations run them together: a main loop and a remainder run by
 * different lanes at once would elect once in each, or count and vote over
 * only part of the lanes. The loop is left alone; the control unrolls. */
TEST(M5RuntimeLoops, LaneStateRefused)
{
   nir_shader *s = lane_state_loop(LANE_NONE);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   nir_validate_shader(s, "lane-state control after");
   EXPECT_EQ(count_loops(s), 2u);
   ralloc_free(s);

   for (lane_op op : {LANE_ELECT, LANE_BALLOT_COUNT, LANE_QUAD_VOTE}) {
      s = lane_state_loop(op);
      EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256)) << "op " << op;
      EXPECT_EQ(count_loops(s), 1u) << "op " << op;
      ralloc_free(s);
   }
}

TEST(M5RuntimeLoops, BudgetRefused)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, false, INT64_MAX);
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 8));
   ralloc_free(s);
}

/* while (++i < n) acc += buf[i]: the back-edge value of i is defined in the
 * header, before the exit test. Every copy of the body must advance it. */
TEST(M5RuntimeLoops, HeaderDefinedBackEdge)
{
   nir_builder b = compute_shader("header-back-edge");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_store_var(&b, iv, nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *next = nir_iadd_imm(&b, nir_load_var(&b, iv), 1);
      nir_store_var(&b, iv, next, 1);
      nir_break_if(&b, nir_ige(&b, next, n));
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, next, 4)),
                                   .align_mul = 4);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   nir_shader *s = b.shader;
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   ASSERT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   NIR_PASS(_, s, nir_opt_algebraic);
   NIR_PASS(_, s, nir_opt_constant_folding);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "header-defined back edge");

   /* In the main (first) loop, i advances by 4 per iteration. */
   bool checked = false;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (checked || parent->type != nir_cf_node_loop ||
             nir_loop_first_block(nir_cf_node_as_loop(parent)) != block)
            continue;
         nir_block *latch = nir_loop_last_block(nir_cf_node_as_loop(parent));
         nir_foreach_phi(phi, block) {
            nir_phi_src *back = nir_phi_get_src_from_block(phi, latch);
            nir_scalar u = nir_scalar_resolved(back->src.ssa, 0);
            if (!nir_scalar_is_alu(u) || nir_scalar_alu_op(u) != nir_op_iadd)
               continue;
            for (unsigned k = 0; k < 2; ++k) {
               nir_scalar c = nir_scalar_chase_alu_src(u, k);
               nir_scalar o = nir_scalar_chase_alu_src(u, 1 - k);
               if (o.def == &phi->def && nir_scalar_is_const(c)) {
                  EXPECT_EQ(nir_scalar_as_uint(c), 4u);
                  checked = true;
               }
            }
         }
      }
   }
   EXPECT_TRUE(checked);
   ralloc_free(s);
}

/* The guard of the main loop compares the induction variable against a
 * select of a loop-invariant bound. With a constant bound it folds: check
 * that "i < G" proves i + 3 <= hi without wrapping, and that a bound too
 * close to INT32_MIN disables the main loop. */
static uint32_t
folded_guard(loop_shape shape, int64_t hi_const)
{
   nir_def *hi;
   nir_shader *s = counted_loop(shape, &hi, false, hi_const);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256));
   NIR_PASS(_, s, nir_opt_constant_folding);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "folded guard");
   /* The first loop's first compare is the guard; read its constant. */
   uint32_t g = 0xdeadbeef;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type != nir_cf_node_loop)
            continue;
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_ilt && nir_src_is_const(alu->src[1].src)) {
               g = nir_src_as_uint(alu->src[1].src);
               goto done;
            }
         }
      }
   }
done:
   ralloc_free(s);
   return g;
}

TEST(M5RuntimeLoops, GuardBounds)
{
   /* Top-tested i <= hi: four iterations run while i + 3 <= hi, i < hi - 2. */
   EXPECT_EQ(folded_guard(TOP_TESTED, 100), 98u);
   EXPECT_EQ(folded_guard(TOP_TESTED, 0x7fffffff), 0x7ffffffdu);
   /* hi - 3 would wrap: never. */
   EXPECT_EQ(folded_guard(TOP_TESTED, INT32_MIN + 2), (uint32_t)INT32_MIN);
   EXPECT_EQ(folded_guard(TOP_TESTED, INT32_MIN + 3), (uint32_t)INT32_MIN + 1);
}

static nir_intrinsic_instr *
first_intrinsic(nir_shader *s, nir_intrinsic_op op, unsigned skip)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op && skip-- == 0)
               return nir_instr_as_intrinsic(instr);
         }
      }
   }
   return NULL;
}

/* Four consecutive words read in one block become one vector load. */
TEST(M5MergeWords, FourWords)
{
   for (bool vec_pack : {false, true}) {
      nir_builder b = compute_shader("merge-words");
      nir_def *i = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
      nir_def *base = nir_amul_imm(&b, i, 4);
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *v = nir_load_global(&b, 1, 32,
                                      buffer_address(&b, nir_iadd_imm(&b, base, 4 * k), vec_pack),
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .align_mul = 4);
         sum = nir_iadd(&b, sum, v);
      }
      nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
      NIR_PASS(_, b.shader, nir_opt_cse);
      EXPECT_TRUE(agx_nir_merge_global_words(b.shader)) << "vec_pack " << vec_pack;
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "merge words");
      nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_global, 0);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(load->def.num_components, 4u);
      EXPECT_EQ(first_intrinsic(b.shader, nir_intrinsic_load_global, 1), nullptr);
      ralloc_free(b.shader);
   }
}

#define RO_ACCESS ((gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER))

/* Words at i, i + 4 and at x, x + 4, where x is the word read at i: merging
 * the first pair replaces x, from which the second pair's address is built.
 * The second merge must take its offset from the live address. */
TEST(M5MergeWords, OffsetFromMergedWord)
{
   nir_builder b = compute_shader("merge-chained");
   nir_def *i = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *x = nir_load_global(&b, 1, 32, buffer_address(&b, i), .access = RO_ACCESS,
                                .align_mul = 4);
   nir_def *y = nir_load_global(&b, 1, 32, buffer_address(&b, nir_iadd_imm(&b, i, 4)),
                                .access = RO_ACCESS, .align_mul = 4);
   nir_def *z = nir_load_global(&b, 1, 32, buffer_address(&b, x), .access = RO_ACCESS,
                                .align_mul = 4);
   nir_def *w = nir_load_global(&b, 1, 32, buffer_address(&b, nir_iadd_imm(&b, x, 4)),
                                .access = RO_ACCESS, .align_mul = 4);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, x, y), nir_iadd(&b, z, w));
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   nir_validate_shader(b.shader, "merge chained words");
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge chained words after dce");
   nir_intrinsic_instr *first = first_intrinsic(b.shader, nir_intrinsic_load_global, 0);
   nir_intrinsic_instr *second = first_intrinsic(b.shader, nir_intrinsic_load_global, 1);
   ASSERT_NE(first, nullptr);
   ASSERT_NE(second, nullptr);
   EXPECT_EQ(first->def.num_components, 2u);
   EXPECT_EQ(second->def.num_components, 2u);
   ralloc_free(b.shader);
}

/* A physical pointer plus x and x + 4: with x = 0xfffffffc the reads are at
 * pointer + 0xfffffffc and pointer + 0, which may both be valid; a vector
 * load would read pointer + 2^32. Not merged. */
TEST(M5MergeWords, PhysicalPointerRefused)
{
   nir_builder b = compute_shader("merge-physical");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 2; ++k)
      sum = nir_iadd(&b, sum,
                     nir_load_global(&b, 1, 32, physical_address(&b, nir_iadd_imm(&b, x, 4 * k)),
                                     .access = RO_ACCESS, .align_mul = 4));
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_FALSE(agx_nir_merge_global_words(b.shader));
   ralloc_free(b.shader);
}

/* A pointer packed from the .xy of a loaded uvec4 (UBO, SSBO, push constant,
 * or lowered root load) has the same shape as a descriptor pack but is not a
 * buffer binding descriptor: must not be merged. */
TEST(M5MergeWords, PackedUvec4Refused)
{
   for (packed_uvec4_source src : {UVEC4_PUSH_CONSTANT, UVEC4_UBO_LOAD,
                                   UVEC4_SSBO_LOAD, UVEC4_SPECULATIVE_ROOT_LOAD}) {
      for (bool vec_pack : {false, true}) {
         nir_builder b = compute_shader("merge-packed-uvec4");
         nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
         nir_def *sum = nir_imm_int(&b, 0);
         for (unsigned k = 0; k < 2; ++k) {
            nir_def *addr = packed_uvec4_address(&b, nir_iadd_imm(&b, x, 4 * k), src, vec_pack);
            sum = nir_iadd(&b, sum,
                           nir_load_global(&b, 1, 32, addr, .access = RO_ACCESS, .align_mul = 4));
         }
         nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
         NIR_PASS(_, b.shader, nir_opt_cse);
         EXPECT_FALSE(agx_nir_merge_global_words(b.shader))
            << "src " << src << " vec_pack " << vec_pack;
         ralloc_free(b.shader);
      }
   }
}

