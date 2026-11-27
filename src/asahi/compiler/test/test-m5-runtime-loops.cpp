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
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
   nir_validate_shader(s, "top-tested unroll");
   EXPECT_EQ(count_loops(s), 2u);
   /* Both loops are marked; a second run leaves them alone. */
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
   ralloc_free(s);
}

/* A second counted loop bounded by the count the first one leaves (its
 * induction variable after it, as a geometry shader's vertex loop leaves the
 * number of vertices): unrolling the first turns that header phi into a
 * register, and the second loop's guard must read the value that replaces
 * it. */
TEST(M5RuntimeLoops, LaterBoundFromEarlierLoop)
{
   nir_builder b = compute_shader("chained-bounds");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *kv = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *first = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, i, 4)),
                                   .align_mul = 4);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, first);
   nir_def *count = nir_load_var(&b, iv);
   nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
   nir_loop *second = nir_push_loop(&b);
   {
      nir_def *k = nir_load_var(&b, kv);
      nir_break_if(&b, nir_ige(&b, k, count));
      nir_def *v = nir_load_global(
         &b, 1, 32,
         buffer_address(&b, nir_iadd_imm(&b, nir_amul_imm(&b, k, 4), 256)),
         .align_mul = 4);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
   }
   nir_pop_loop(&b, second);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   nir_shader *s = b.shader;
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   ASSERT_EQ(count_loops(s), 2u);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
   nir_validate_shader(s, "bound from an earlier loop");
   /* Both loops unrolled: a main loop in front of each. */
   EXPECT_EQ(count_loops(s), 4u);
   ralloc_free(s);
}

static unsigned
loops_inside_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type == nir_cf_node_loop &&
             nir_loop_first_block(nir_cf_node_as_loop(parent)) == block &&
             parent->parent->type == nir_cf_node_if)
            n++;
      }
   }
   return n;
}

/* The rotated loop's only live-out (the sum) is a back-edge value: the
 * original loop moves under an if, so the main loop needs only 4 iterations
 * left. */
TEST(M5RuntimeLoops, BottomTestedUnrolls)
{
   nir_def *hi;
   nir_shader *s = counted_loop(BOTTOM_TESTED, &hi, false, INT64_MAX);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
   nir_validate_shader(s, "bottom-tested unroll");
   EXPECT_EQ(count_loops(s), 2u);
   EXPECT_EQ(loops_inside_ifs(s), 1u);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "bottom-tested unroll after dce");
   ralloc_free(s);
}

TEST(M5RuntimeLoops, BarrierRefused)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, true, INT64_MAX);
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
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
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
   nir_validate_shader(s, "lane-state control after");
   EXPECT_EQ(count_loops(s), 2u);
   ralloc_free(s);

   for (lane_op op : {LANE_ELECT, LANE_BALLOT_COUNT, LANE_QUAD_VOTE}) {
      s = lane_state_loop(op);
      EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL)) << "op " << op;
      EXPECT_EQ(count_loops(s), 1u) << "op " << op;
      ralloc_free(s);
   }
}

TEST(M5RuntimeLoops, BudgetRefused)
{
   nir_def *hi;
   nir_shader *s = counted_loop(TOP_TESTED, &hi, false, INT64_MAX);
   EXPECT_FALSE(agx_nir_unroll_runtime_loops(s, 4, 8, NULL));
   ralloc_free(s);
}

/* load_const and undef instructions are cheap and should not inflate the
 * unroll instruction budget. */
TEST(M5RuntimeLoops, ConstAndUndefExcludedFromBudget)
{
   nir_builder b = compute_shader("const-undef-budget");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, n));
      /* Add many distinct load_const and undef instructions inside the loop
       * body; they should not count toward the unroll budget. */
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, i, 4)),
                                   .align_mul = 4);
      nir_def *u0 = nir_undef(&b, 1, 32);
      nir_def *u1 = nir_undef(&b, 1, 32);
      (void)u0;
      (void)u1;
      for (unsigned k = 0; k < 24; ++k)
         (void)nir_imm_int(&b, 0x11 + k);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   nir_shader *s = b.shader;
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_copy_prop);
   /* Keep load_const/undef in place (no DCE) to verify count_unrollable skips
    * them under a budget (72) that fits factor 4 for the 17 non-const/non-undef
    * instructions (17 * 4 = 68 <= 72) while refusing even factor 2 if the 26+
    * const/undef ops are counted ((17 + 26) * 2 = 86 > 72). */
   unsigned widest = 0;
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 72, &widest));
   nir_validate_shader(s, "const-undef-budget after");
   EXPECT_EQ(widest, 4u);
   EXPECT_EQ(count_loops(s), 2u);
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
   ASSERT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
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
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
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

/* for (i = local_invocation_id.x; i < n; i += 32) acc += buf[i]: a loop
 * striped over a 32-wide workgroup with a constant bound, which no
 * invocation runs more than ceil(n / 32) times. With `pc_start` the start is
 * a push constant instead, which may be negative. */
static nir_shader *
striped_loop(uint32_t n, bool pc_start)
{
   nir_builder b = compute_shader("striped-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_int_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *start =
      pc_start ? nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32)
               : nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_store_var(&b, iv, start, 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, nir_imm_int(&b, n)));
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, i, 4)),
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .align_mul = 4);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, 32), 1);
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

/* The iterations per test the pass gives a striped loop at factor 8, or 0
 * when it leaves the loop as it is. */
static unsigned
striped_factor(uint32_t n, bool pc_start = false)
{
   nir_shader *s = striped_loop(n, pc_start);
   unsigned widest = 0;
   const bool progress = agx_nir_unroll_runtime_loops(s, 8, 320, &widest);
   nir_validate_shader(s, "striped loop");
   EXPECT_EQ(count_loops(s), progress ? 2u : 1u);
   ralloc_free(s);
   return progress ? widest : 0;
}

/* for (i = 0; i < n; i += step) acc += buf[i], the step a push constant: a
 * loop that steps by a runtime value. Returns the iterations per test the
 * pass gives it at factor 8, or 0 when it leaves the loop as it is. */
static unsigned
runtime_step_factor(uint32_t n)
{
   nir_builder b = compute_shader("runtime-step-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_int_type(), "i");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *step = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, nir_ige(&b, i, nir_imm_int(&b, n)));
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, i, 4)),
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .align_mul = 4);
      nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
      nir_store_var(&b, iv, nir_iadd(&b, i, step), 1);
   }
   nir_pop_loop(&b, loop);
   nir_store_global(&b, nir_load_var(&b, acc),
                    buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   nir_shader *s = b.shader;
   NIR_PASS(_, s, nir_lower_vars_to_ssa);
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_cse);
   NIR_PASS(_, s, nir_opt_dce);
   unsigned widest = 0;
   const bool progress = agx_nir_unroll_runtime_loops(s, 8, 320, &widest);
   nir_validate_shader(s, "runtime-step loop");
   ralloc_free(s);
   return progress ? widest : 0;
}

TEST(M5RuntimeLoops, TripBoundLimitsFactor)
{
   /* At most two iterations, like a halo fill: any main loop could run at
    * most once, so the loop stays as it is. */
   EXPECT_EQ(striped_factor(64), 0u);
   EXPECT_EQ(striped_factor(33), 0u);
   /* At most 4, 15 and 16 iterations. */
   EXPECT_EQ(striped_factor(128), 2u);
   EXPECT_EQ(striped_factor(32 * 15), 4u);
   EXPECT_EQ(striped_factor(32 * 16), 8u);
   /* A start that may be negative gives no bound. */
   EXPECT_EQ(striped_factor(64, true), 8u);
   /* Neither does a step only known at run time, whatever the bound. */
   EXPECT_EQ(runtime_step_factor(4096), 8u);
   EXPECT_EQ(runtime_step_factor(64), 8u);
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

/* Byte reads at X, X + 1 and X + 2 that execute together share X. */
TEST(M5FoldOffsets, ByteSiblings)
{
   nir_builder b = compute_shader("byte-siblings");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 3; ++k) {
      nir_def *v = nir_load_global(&b, 1, 8, buffer_address(&b, nir_iadd_imm(&b, x, k)),
                                   .align_mul = 1);
      sum = nir_iadd(&b, sum, nir_u2u32(&b, v));
   }
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_fold_m5_offsets(b.shader, false));
   nir_validate_shader(b.shader, "byte siblings");
   nir_def *index = NULL;
   for (unsigned k = 0; k < 3; ++k) {
      nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_agx, k);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(nir_intrinsic_base(load) >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT, k);
      EXPECT_FALSE(nir_intrinsic_sign_extend(load));
      if (index)
         EXPECT_EQ(load->src[1].ssa, index);
      index = load->src[1].ssa;
   }
   ralloc_free(b.shader);
}

/* Speculative byte siblings are refused without soft faults, and folded with
 * sign-extended shared indices under soft faults so an executed lane cannot
 * wrap past 2^32 even if an earlier sibling was hoisted from a guard. */
TEST(M5FoldOffsets, SoftFaultSpeculativeByteSiblings)
{
   nir_builder b = compute_shader("byte-siblings-spec");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 3; ++k) {
      nir_def *v = nir_load_global(
         &b, 1, 8, buffer_address(&b, nir_iadd_imm(&b, x, k)),
         .access = (gl_access_qualifier)ACCESS_CAN_SPECULATE, .align_mul = 1);
      sum = nir_iadd(&b, sum, nir_u2u32(&b, v));
   }
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);

   EXPECT_FALSE(agx_nir_fold_m5_offsets(b.shader, false));
   EXPECT_TRUE(agx_nir_fold_m5_offsets(b.shader, true));
   nir_validate_shader(b.shader, "soft-fault byte siblings");
   nir_def *index = NULL;
   for (unsigned k = 0; k < 3; ++k) {
      nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_agx, k);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(nir_intrinsic_base(load) >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT, k);
      EXPECT_TRUE(nir_intrinsic_sign_extend(load));
      if (index)
         EXPECT_EQ(load->src[1].ssa, index);
      index = load->src[1].ssa;
   }
   ralloc_free(b.shader);
}

/* A lone byte read at X + 1 cannot prove X + 1 does not wrap: unchanged. */
TEST(M5FoldOffsets, LoneByteKept)
{
   nir_builder b = compute_shader("lone-byte");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *v = nir_load_global(&b, 1, 8, buffer_address(&b, nir_iadd_imm(&b, x, 1)),
                                .align_mul = 1);
   nir_store_global(&b, nir_u2u32(&b, v), buffer_address(&b, nir_imm_int(&b, 64)),
                    .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_FALSE(agx_nir_fold_m5_offsets(b.shader, false));
   ralloc_free(b.shader);
}

/* Byte siblings at a physical pointer plus X + c: X + c may wrap into another
 * allocation there, so the shared index X + c0 is not provably the same
 * address. Left alone. */
TEST(M5FoldOffsets, PhysicalPointerRefused)
{
   nir_builder b = compute_shader("fold-physical");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 1; k <= 3; ++k) {
      nir_def *v = nir_load_global(&b, 1, 8, physical_address(&b, nir_iadd_imm(&b, x, k)),
                                   .align_mul = 1);
      sum = nir_iadd(&b, nir_imul_imm(&b, sum, 3), nir_u2u32(&b, v));
   }
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_FALSE(agx_nir_fold_m5_offsets(b.shader, false));
   ralloc_free(b.shader);
}

/* A pointer packed from the .xy of a loaded uvec4 (UBO, SSBO, push constant,
 * or lowered root load) is not a buffer binding descriptor: both byte siblings
 * and array-element folds must refuse it. */
TEST(M5FoldOffsets, PackedUvec4Refused)
{
   for (packed_uvec4_source src : {UVEC4_PUSH_CONSTANT, UVEC4_UBO_LOAD,
                                   UVEC4_SSBO_LOAD, UVEC4_SPECULATIVE_ROOT_LOAD}) {
      for (bool vec_pack : {false, true}) {
         /* Byte siblings */
         {
            nir_builder b = compute_shader("fold-packed-uvec4-bytes");
            nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
            nir_def *sum = nir_imm_int(&b, 0);
            for (unsigned k = 1; k <= 3; ++k) {
               nir_def *addr = packed_uvec4_address(&b, nir_iadd_imm(&b, x, k), src, vec_pack);
               nir_def *v = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
               sum = nir_iadd(&b, nir_imul_imm(&b, sum, 3), nir_u2u32(&b, v));
            }
            nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
            NIR_PASS(_, b.shader, nir_opt_cse);
            EXPECT_FALSE(agx_nir_fold_m5_offsets(b.shader, false))
               << "bytes src " << src << " vec_pack " << vec_pack;
            ralloc_free(b.shader);
         }
         /* Array element */
         {
            nir_builder b = compute_shader("fold-packed-uvec4-array");
            nir_def *i = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
            nir_def *off = nir_amul_imm(&b, nir_iadd_imm(&b, i, 3), 4);
            nir_def *addr = packed_uvec4_address(&b, off, src, vec_pack);
            nir_def *v = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
            nir_store_global(&b, v, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
            NIR_PASS(_, b.shader, nir_opt_cse);
            EXPECT_FALSE(agx_nir_fold_m5_offsets(b.shader, false))
               << "array src " << src << " vec_pack " << vec_pack;
            ralloc_free(b.shader);
         }
      }
   }
}

/* buf[i + 3] of 32-bit elements indexes with i and a 12-byte offset. */
TEST(M5FoldOffsets, ArrayElement)
{
   nir_builder b = compute_shader("array-element");
   nir_def *i = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *v = nir_load_global(&b, 1, 32,
                                buffer_address(&b, nir_amul_imm(&b, nir_iadd_imm(&b, i, 3), 4)),
                                .align_mul = 4);
   nir_store_global(&b, v, buffer_address(&b, nir_imm_int(&b, 64)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_fold_m5_offsets(b.shader, false));
   nir_validate_shader(b.shader, "array element");
   nir_intrinsic_instr *load = first_intrinsic(b.shader, nir_intrinsic_load_agx, 0);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->src[1].ssa, i);
   EXPECT_EQ(nir_intrinsic_base(load) >> AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT, 12u);
   EXPECT_TRUE(nir_intrinsic_sign_extend(load));
   ralloc_free(b.shader);
}

/* |a - b| of unsigned bytes in 16-bit arithmetic: evaluate the rewritten
 * expression for every operand pair. */
static uint32_t
eval_scalar(nir_scalar s, uint8_t a, uint8_t b, nir_def *la, nir_def *lb)
{
   if (s.def == la)
      return a;
   if (s.def == lb)
      return b;
   if (nir_scalar_is_const(s))
      return nir_scalar_as_uint(s);
   EXPECT_TRUE(nir_scalar_is_alu(s));
   nir_alu_instr *alu = nir_def_as_alu(s.def);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value src[4][NIR_MAX_VEC_COMPONENTS] = {};
   nir_const_value *srcs[4];
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar in = nir_scalar_chase_alu_src(s, i);
      uint64_t v = eval_scalar(in, a, b, la, lb);
      src[i][0] = nir_const_value_for_uint(v, in.def->bit_size);
      srcs[i] = src[i];
   }
   nir_const_value dest[NIR_MAX_VEC_COMPONENTS] = {};
   uint16_t poison = 0;
   nir_eval_const_opcode(alu->op, dest, &poison, 1, alu->def.bit_size, srcs, 0);
   return nir_const_value_as_uint(dest[0], alu->def.bit_size);
}

TEST(M5U8AbsDiff, Exhaustive)
{
   for (unsigned form = 0; form < 2; ++form) {
      nir_builder b = compute_shader("u8-absdiff");
      nir_def *addr = buffer_address(&b, nir_imm_int(&b, 0));
      nir_def *la = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
      nir_def *lb = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 1), .align_mul = 1);
      nir_def *d = nir_isub(&b, nir_u2u32(&b, la), nir_u2u32(&b, lb));
      nir_def *r = form == 0 ? nir_imax(&b, d, nir_ineg(&b, d)) : nir_iabs(&b, d);
      nir_intrinsic_instr *store =
         nir_store_global(&b, r, nir_iadd_imm(&b, addr, 4), .align_mul = 4);
      NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_opt_u8_absdiff(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8 absdiff");
      nir_scalar out = nir_scalar_resolved(store->src[0].ssa, 0);
      EXPECT_NE(nir_scalar_alu_op(out), nir_op_imax);
      for (unsigned x = 0; x < 256; ++x) {
         for (unsigned y = 0; y < 256; ++y) {
            uint32_t got = eval_scalar(out, x, y, la, lb);
            ASSERT_EQ(got, (uint32_t)abs((int)x - (int)y)) << x << " " << y;
         }
      }
      ralloc_free(b.shader);
   }
}

/* A delta with another use keeps its 32-bit form. */
TEST(M5U8AbsDiff, OtherUseRefused)
{
   nir_builder b = compute_shader("u8-absdiff-uses");
   nir_def *addr = buffer_address(&b, nir_imm_int(&b, 0));
   nir_def *la = nir_load_global(&b, 1, 8, addr, .align_mul = 1);
   nir_def *lb = nir_load_global(&b, 1, 8, nir_iadd_imm(&b, addr, 1), .align_mul = 1);
   nir_def *d = nir_isub(&b, nir_u2u32(&b, la), nir_u2u32(&b, lb));
   nir_store_global(&b, nir_imax(&b, d, nir_ineg(&b, d)), addr, .align_mul = 4);
   nir_store_global(&b, d, nir_iadd_imm(&b, addr, 8), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_FALSE(agx_nir_opt_u8_absdiff(b.shader));
   ralloc_free(b.shader);
}

/* Written-out population counts become one native count. */
static nir_def *
pc_level(nir_builder *b, nir_def *v, uint32_t m, unsigned s)
{
   return nir_iadd(b, nir_iand_imm(b, v, m), nir_iand_imm(b, nir_ushr_imm(b, v, s), m));
}

TEST(M5PopcountIdiom, TreeAndMultiplyForms)
{
   for (unsigned form = 0; form < 2; ++form) {
      nir_builder b = compute_shader("popcount-idiom");
      nir_def *addr = buffer_address(&b, nir_imm_int(&b, 0));
      nir_def *x = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_def *r;
      if (form == 0) {
         nir_def *v = pc_level(&b, x, 0x55555555, 1);
         v = pc_level(&b, v, 0x33333333, 2);
         v = pc_level(&b, v, 0x0f0f0f0f, 4);
         v = pc_level(&b, v, 0x00ff00ff, 8);
         r = pc_level(&b, v, 0x0000ffff, 16);
      } else {
         nir_def *v = nir_isub(&b, x, nir_iand_imm(&b, nir_ushr_imm(&b, x, 1), 0x55555555));
         v = pc_level(&b, v, 0x33333333, 2);
         v = nir_iand_imm(&b, nir_iadd(&b, v, nir_ushr_imm(&b, v, 4)), 0x0f0f0f0f);
         r = nir_ushr_imm(&b, nir_imul_imm(&b, v, 0x01010101), 24);
      }
      nir_intrinsic_instr *store =
         nir_store_global(&b, r, nir_iadd_imm(&b, addr, 4), .align_mul = 4);
      EXPECT_TRUE(agx_nir_opt_popcount_idiom(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "popcount idiom");
      nir_scalar out = nir_scalar_resolved(store->src[0].ssa, 0);
      ASSERT_TRUE(nir_scalar_is_alu(out));
      EXPECT_EQ(nir_scalar_alu_op(out), nir_op_bit_count);
      /* The original expression's value equals the count everywhere. */
      uint32_t v = 0x9e3779b9;
      for (unsigned i = 0; i < 100000; ++i) {
         v = v * 1664525u + 1013904223u;
         uint32_t t = v - ((v >> 1) & 0x55555555u);
         t = (t & 0x33333333u) + ((t >> 2) & 0x33333333u);
         t = (((t + (t >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
         ASSERT_EQ(t, (uint32_t)__builtin_popcount(v));
      }
      ralloc_free(b.shader);
   }
}

/* Workgroups of one subgroup: shared-memory and execution-only barriers
 * narrow to subgroup scope; device-memory barriers and larger workgroups
 * keep workgroup scope. */
static nir_intrinsic_instr *
barrier_in(nir_builder *b, nir_variable_mode modes, mesa_scope memory)
{
   nir_barrier(b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = memory,
               .memory_semantics = memory == SCOPE_NONE ? (nir_memory_semantics)0
                                                         : NIR_MEMORY_ACQ_REL,
               .memory_modes = modes);
   return nir_instr_as_intrinsic(nir_block_last_instr(nir_cursor_current_block(b->cursor)));
}

TEST(M5NarrowBarriers, SingleSubgroupWorkgroup)
{
   for (unsigned width : {32u, 8u, 64u}) {
      nir_builder b = compute_shader("narrow-barriers");
      b.shader->info.workgroup_size[0] = width;
      nir_intrinsic_instr *shared = barrier_in(&b, nir_var_mem_shared, SCOPE_WORKGROUP);
      nir_intrinsic_instr *exec = barrier_in(&b, (nir_variable_mode)0, SCOPE_NONE);
      nir_intrinsic_instr *device =
         barrier_in(&b, (nir_variable_mode)(nir_var_mem_shared | nir_var_mem_global),
                    SCOPE_WORKGROUP);
      const bool narrows = width <= 32;
      EXPECT_EQ(agx_nir_narrow_single_subgroup_barriers(b.shader), narrows);
      nir_validate_shader(b.shader, "narrow barriers");
      const mesa_scope expect = narrows ? SCOPE_SUBGROUP : SCOPE_WORKGROUP;
      EXPECT_EQ(nir_intrinsic_execution_scope(shared), expect);
      EXPECT_EQ(nir_intrinsic_memory_scope(shared), expect);
      EXPECT_EQ(nir_intrinsic_execution_scope(exec), expect);
      EXPECT_EQ(nir_intrinsic_memory_scope(exec), SCOPE_NONE);
      EXPECT_EQ(nir_intrinsic_execution_scope(device), SCOPE_WORKGROUP);
      EXPECT_EQ(nir_intrinsic_memory_scope(device), SCOPE_WORKGROUP);
      ralloc_free(b.shader);
   }
}

/* SPIR-V often emits OpMemoryBarrier(Device, WorkgroupMemory) beside a
 * workgroup control barrier. Shared memory is private to one workgroup, so
 * a single-subgroup workgroup still narrows SCOPE_DEVICE to SCOPE_SUBGROUP
 * when only nir_var_mem_shared is ordered, while keeping SCOPE_DEVICE when
 * global memory is also ordered. */
TEST(M5NarrowBarriers, SingleSubgroupDeviceScopeSharedOnly)
{
   for (unsigned width : {32u, 16u, 64u}) {
      nir_builder b = compute_shader("narrow-device-shared");
      b.shader->info.workgroup_size[0] = width;
      nir_intrinsic_instr *shared_dev =
         barrier_in(&b, nir_var_mem_shared, SCOPE_DEVICE);
      nir_intrinsic_instr *global_dev =
         barrier_in(&b, (nir_variable_mode)(nir_var_mem_shared | nir_var_mem_global),
                    SCOPE_DEVICE);
      const bool narrows = width <= 32;
      EXPECT_EQ(agx_nir_narrow_single_subgroup_barriers(b.shader), narrows);
      nir_validate_shader(b.shader, "narrow device shared barriers");
      EXPECT_EQ(nir_intrinsic_execution_scope(shared_dev),
                narrows ? SCOPE_SUBGROUP : SCOPE_WORKGROUP);
      EXPECT_EQ(nir_intrinsic_memory_scope(shared_dev),
                narrows ? SCOPE_SUBGROUP : SCOPE_DEVICE);
      EXPECT_EQ(nir_intrinsic_execution_scope(global_dev), SCOPE_WORKGROUP);
      EXPECT_EQ(nir_intrinsic_memory_scope(global_dev), SCOPE_DEVICE);
      ralloc_free(b.shader);
   }
}

/* Rotated loops for the exact-tail path; shape selects the live-outs. */
enum tail_shape {
   TAIL_INVARIANT_FROM_OUTER_PHI, /* t = w * 3 in the loop, w an if/else phi */
   TAIL_SWAP,                     /* a = b; b = b + x; a used after */
   TAIL_SHARED_BACK_EDGE,         /* i and k both advance to i + 1 */
};

static nir_shader *
rotated_loop(tail_shape shape)
{
   nir_builder b = compute_shader("rotated-tail");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
   nir_variable *kv = nir_local_variable_create(b.impl, glsl_uint_type(), "k");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_variable *av = nir_local_variable_create(b.impl, glsl_uint_type(), "a");
   nir_variable *wv = nir_local_variable_create(b.impl, glsl_uint_type(), "w");
   nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
   nir_def *sel = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 32);
   nir_push_if(&b, nir_ine_imm(&b, sel, 0));
   nir_store_var(&b, wv, nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 32), 1);
   nir_push_else(&b, NULL);
   nir_store_var(&b, wv, nir_imm_int(&b, 7), 1);
   nir_pop_if(&b, NULL);
   nir_def *w = nir_load_var(&b, wv);
   nir_store_var(&b, iv, sel, 1);
   nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, av, w, 1);
   nir_def *t = NULL;
   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      nir_def *index = shape == TAIL_SHARED_BACK_EDGE ? nir_load_var(&b, kv) : i;
      nir_def *v = nir_load_global(&b, 1, 32,
                                   buffer_address(&b, nir_amul_imm(&b, index, 4)),
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .align_mul = 4);
      t = nir_imul_imm(&b, w, 3);
      nir_def *sum = nir_iadd(&b, nir_load_var(&b, acc), v);
      if (shape == TAIL_SWAP) {
         nir_store_var(&b, av, sum, 1);
         nir_store_var(&b, acc, nir_iadd(&b, sum, v), 1);
      } else {
         nir_store_var(&b, acc, sum, 1);
      }
      nir_def *next = nir_iadd_imm(&b, i, 1);
      nir_store_var(&b, iv, next, 1);
      nir_store_var(&b, kv, next, 1);
      nir_break_if(&b, nir_ilt(&b, n, next));
   }
   nir_pop_loop(&b, loop);
   nir_def *result = nir_load_var(&b, acc);
   if (shape == TAIL_INVARIANT_FROM_OUTER_PHI)
      result = nir_iadd(&b, result, t);
   if (shape == TAIL_SWAP)
      result = nir_iadd(&b, result, nir_load_var(&b, av));
   nir_store_global(&b, result, buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   return b.shader;
}

/* Each shape unrolls and stays valid SSA; the invariant live-out is
 * recomputed from the if/else phi without cloning anything before the loop,
 * and the swap shape (a header phi live after the loop) keeps the plain
 * form. Two loops in a row: the second's bound is the first's live-out. */
TEST(M5RuntimeLoops, ExactTailShapes)
{
   for (tail_shape shape : {TAIL_INVARIANT_FROM_OUTER_PHI, TAIL_SWAP, TAIL_SHARED_BACK_EDGE}) {
      nir_shader *s = rotated_loop(shape);
      nir_validate_shader(s, "rotated before");
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 4, 256, NULL));
      nir_validate_shader(s, "rotated unrolled");
      EXPECT_EQ(count_loops(s), 2u);
      EXPECT_EQ(loops_inside_ifs(s), shape == TAIL_SWAP ? 0u : 1u);
      NIR_PASS(_, s, nir_opt_dce);
      nir_validate_shader(s, "rotated after dce");
      ralloc_free(s);
   }
}

/* A 64-wide tree reduction: s[t] = v; barrier; for off = 32, 16:
 * if (t < off) s[t] += s[t + off]; barrier; then t == 0 reads s[0]. The
 * first barrier orders every invocation's store: kept. The one after the
 * off = 32 step sits between two subgroup-0 steps: narrowed. The last one
 * has only t == 0 after it: narrowed. With a guard t < 48 nothing after the
 * first barrier is limited to subgroup 0 and the steps keep their barriers. */
static nir_shader *
reduction(unsigned first_off)
{
   nir_builder b = compute_shader("reduction");
   b.shader->info.workgroup_size[0] = 64;
   b.shader->info.shared_size = 256;
   nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));
   nir_store_shared(&b, t, addr, .align_mul = 4);
   nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   for (unsigned off : {first_off, 16u}) {
      nir_push_if(&b, nir_ult_imm(&b, t, off));
      nir_def *other = nir_load_shared(&b, 1, 32, nir_u2u16(&b, nir_ishl_imm(&b, nir_iadd_imm(&b, t, off), 2)),
                                       .align_mul = 4);
      nir_def *mine = nir_load_shared(&b, 1, 32, addr, .align_mul = 4);
      nir_store_shared(&b, nir_iadd(&b, mine, other), addr, .align_mul = 4);
      nir_pop_if(&b, NULL);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
   }
   nir_push_if(&b, nir_ieq_imm(&b, t, 0));
   nir_def *sum = nir_load_shared(&b, 1, 32, nir_imm_intN_t(&b, 0, 16), .align_mul = 4);
   nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   nir_pop_if(&b, NULL);
   return b.shader;
}

static void
barrier_scopes(nir_shader *s, mesa_scope *out, unsigned n)
{
   unsigned i = 0;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_barrier && i < n)
               out[i++] = nir_intrinsic_execution_scope(nir_instr_as_intrinsic(instr));
         }
      }
   }
   ASSERT_EQ(i, n);
}

TEST(M5NarrowBarriers, Subgroup0ReductionTail)
{
   nir_shader *s = reduction(32);
   EXPECT_TRUE(agx_nir_narrow_subgroup0_barriers(s));
   nir_validate_shader(s, "reduction");
   mesa_scope scopes[3];
   barrier_scopes(s, scopes, 3);
   EXPECT_EQ(scopes[0], SCOPE_WORKGROUP);
   EXPECT_EQ(scopes[1], SCOPE_SUBGROUP);
   EXPECT_EQ(scopes[2], SCOPE_SUBGROUP);
   ralloc_free(s);

   s = reduction(48);
   agx_nir_narrow_subgroup0_barriers(s);
   barrier_scopes(s, scopes, 3);
   EXPECT_EQ(scopes[0], SCOPE_WORKGROUP);
   EXPECT_EQ(scopes[1], SCOPE_WORKGROUP);
   /* Between the t < 16 step and the t == 0 read: both subgroup 0. */
   EXPECT_EQ(scopes[2], SCOPE_SUBGROUP);
   ralloc_free(s);
}

/* A buffer flag read before a shared-memory barrier and written after it, by
 * a 64-wide workgroup with no shared traffic around the barrier: the
 * rendezvous orders the read before the write across subgroups, so it keeps
 * workgroup scope. A read-only (reorderable) read cannot observe the write:
 * then the barrier is narrowed. */
TEST(M5NarrowBarriers, Subgroup0OtherMemoryOrdered)
{
   for (bool readonly : {false, true}) {
      nir_builder b = compute_shader("flag");
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.shared_size = 4;
      nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *flag =
         nir_load_global(&b, 1, 32, buffer_address(&b, nir_imm_int(&b, 0)), .access = readonly ? RO_ACCESS : (gl_access_qualifier)0,
                         .align_mul = 4);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
      nir_push_if(&b, nir_ieq_imm(&b, t, 63));
      nir_store_global(&b, nir_imm_int(&b, 1), buffer_address(&b, nir_imm_int(&b, 0)),
                       .align_mul = 4);
      nir_pop_if(&b, NULL);
      nir_store_global(&b, flag, buffer_address(&b, nir_ishl_imm(&b, nir_iadd_imm(&b, t, 1), 2)),
                       .align_mul = 4);
      agx_nir_narrow_subgroup0_barriers(b.shader);
      nir_validate_shader(b.shader, "flag barrier");
      mesa_scope scope;
      barrier_scopes(b.shader, &scope, 1);
      EXPECT_EQ(scope, readonly ? SCOPE_SUBGROUP : SCOPE_WORKGROUP) << "readonly " << readonly;
      ralloc_free(b.shader);
   }
}

/* Shapes the pass must leave alone or survive: a buffer fence relying on the
 * control barrier's rendezvous, a subgroup barrier next to the workgroup one,
 * and a guard condition that is a unary op (b2b1 of a shared bool). */
TEST(M5NarrowBarriers, Subgroup0Refusals)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      nir_builder b = compute_shader("refusals");
      b.shader->info.workgroup_size[0] = 64;
      b.shader->info.shared_size = 256;
      nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *addr = nir_u2u16(&b, nir_ishl_imm(&b, t, 2));
      if (shape == 2) {
         nir_def *flag = nir_b2b1(&b, nir_load_shared(&b, 1, 32, nir_imm_intN_t(&b, 252, 16),
                                                      .align_mul = 4));
         nir_push_if(&b, flag);
         nir_store_shared(&b, t, addr, .align_mul = 4);
         nir_pop_if(&b, NULL);
      } else {
         nir_store_shared(&b, t, addr, .align_mul = 4);
      }
      if (shape == 0)
         nir_barrier(&b, .execution_scope = SCOPE_NONE, .memory_scope = SCOPE_DEVICE,
                     .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_ssbo);
      if (shape == 1)
         nir_barrier(&b, .execution_scope = SCOPE_SUBGROUP, .memory_scope = SCOPE_SUBGROUP,
                     .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
      nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
                  .memory_semantics = NIR_MEMORY_ACQ_REL, .memory_modes = nir_var_mem_shared);
      nir_def *other = nir_load_shared(&b, 1, 32,
                                       nir_u2u16(&b, nir_ishl_imm(&b, nir_isub(&b, nir_imm_int(&b, 63), t), 2)),
                                       .align_mul = 4);
      nir_store_global(&b, other, buffer_address(&b, nir_ishl_imm(&b, t, 2)), .align_mul = 4);
      nir_validate_shader(b.shader, "refusals before");
      if (shape < 2) {
         EXPECT_FALSE(agx_nir_narrow_subgroup0_barriers(b.shader));
      } else {
         agx_nir_narrow_subgroup0_barriers(b.shader);
      }
      nir_validate_shader(b.shader, "refusals after");
      /* The workgroup barrier orders a store every invocation made: kept. */
      unsigned workgroup = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_barrier &&
                   nir_intrinsic_execution_scope(nir_instr_as_intrinsic(instr)) == SCOPE_WORKGROUP)
                  ++workgroup;
            }
         }
      }
      EXPECT_EQ(workgroup, 1u) << "shape " << shape;
      ralloc_free(b.shader);
   }
}

/* Rows at (t + c) * stride + y with a uniform stride: three rows in one
 * block share t * stride + y and add a uniform c * stride each; a lone row
 * and rows in different blocks are left alone. Values: exhaustive over a few
 * wrapping inputs on the rewritten expression. */
TEST(M5UniformStride, RowsShareTheirBase)
{
   for (unsigned shape = 0; shape < 3; ++shape) {
      nir_builder b = compute_shader("uniform-stride");
      nir_def *t = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
      nir_def *y = nir_channel(&b, nir_load_local_invocation_id(&b), 1);
      const unsigned rows = shape == 1 ? 1 : 3;
      for (unsigned c = 1; c <= rows; ++c) {
         if (shape == 2)
            nir_push_if(&b, nir_ine_imm(&b, stride, c));
         nir_def *index = nir_iadd(&b, nir_imul(&b, nir_iadd_imm(&b, t, c), stride), y);
         nir_store_global(&b, index, buffer_address(&b, nir_ishl_imm(&b, index, 2)), .align_mul = 4);
         if (shape == 2)
            nir_pop_if(&b, NULL);
      }
      const bool expect = shape == 0;
      EXPECT_EQ(agx_nir_opt_uniform_stride(b.shader), expect) << "shape " << shape;
      nir_validate_shader(b.shader, "uniform stride");
      ralloc_free(b.shader);
   }

   /* A later group whose x is a row of an earlier group (a 3D stencil's
    * ((z + dz) * H + y) * W + x): both rewrite, the second from the first's
    * new result (validation catches a use of the removed row). */
   {
      nir_builder b = compute_shader("uniform-stride-nested");
      nir_def *id = nir_load_local_invocation_id(&b);
      nir_def *z = nir_channel(&b, id, 0), *y = nir_channel(&b, id, 1), *x = nir_channel(&b, id, 2);
      nir_def *H = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
      nir_def *W = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
      nir_def *r1 = nir_iadd(&b, nir_imul(&b, nir_iadd_imm(&b, z, 1), H), y);
      nir_def *r2 = nir_iadd(&b, nir_imul(&b, nir_iadd_imm(&b, z, 2), H), y);
      nir_def *q1 = nir_iadd(&b, nir_imul(&b, nir_iadd_imm(&b, r1, 1), W), x);
      nir_def *q2 = nir_iadd(&b, nir_imul(&b, nir_iadd_imm(&b, r1, 2), W), x);
      nir_def *sum = nir_iadd(&b, nir_iadd(&b, r2, q1), q2);
      nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
      EXPECT_TRUE(agx_nir_opt_uniform_stride(b.shader));
      nir_validate_shader(b.shader, "uniform stride nested");
      ralloc_free(b.shader);
   }

   /* (t + c) * s + y == (t * s + y) + c * s in 32-bit wrapping arithmetic. */
   static const uint32_t v[] = {0, 1, 7, 0x7fffffffu, 0x80000000u, 0xfffffff9u, 0xdeadbeefu};
   for (uint32_t t : v)
      for (uint32_t s : v)
         for (uint32_t y : v)
            for (uint32_t c = 1; c < 9; ++c)
               ASSERT_EQ((uint32_t)((t + c) * s + y), (uint32_t)((t * s + y) + c * s));
}

/* Every loop-header phi's sources are defined values (an undef one means a
 * copy starts from an unwritten register). */
static void
expect_no_undef_header_phis(nir_shader *s)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_cf_node *parent = block->cf_node.parent;
         if (parent->type != nir_cf_node_loop ||
             nir_loop_first_block(nir_cf_node_as_loop(parent)) != block)
            continue;
         nir_foreach_phi(phi, block) {
            nir_foreach_phi_src(src, phi)
               EXPECT_NE(nir_def_instr(src->src.ssa)->type, nir_instr_type_undef);
         }
      }
   }
}

enum box_variant {
   BOX_PLAIN,
   BOX_BARRIER,
   BOX_LANE_STATE,
   BOX_INTRINSIC_LIVE_OUT,
   BOX_BOTTOM_TESTED,  /* the outer loop of BOX_INTRINSIC_LIVE_OUT, no live-out */
   BOX_INNER_IF,       /* inner body ends in an if, increment defined before it */
   BOX_SHARED_ATOMIC,  /* relaxed atomic shared load in the outer loop */
   BOX_MAX_RECONVERGE, /* shader compiled with maximal reconvergence */
   BOX_UBO_RADIUS,     /* the inner window read per row from a uniform buffer */
   BOX_SSBO_RADIUS,    /* ... from a buffer the shader may write */
};

/* A box filter: for (y = -r; y <= r; y++) for (x = c - r; x <= c + r; x++)
 * sum += buf[x]. The inner trip count does not change across the outer loop,
 * so the outer loop is unswitched on it: then the unrolled inner loop (main
 * and remainder), else the original; 5 loops in all, valid SSA. */
static nir_shader *
box_filter(box_variant variant)
{
   const bool barrier = variant == BOX_BARRIER;
   const bool bottom = variant == BOX_INTRINSIC_LIVE_OUT || variant == BOX_BOTTOM_TESTED;
   nir_builder b = compute_shader("box");
   b.shader->info.shared_size = 4;
   b.shader->info.maximally_reconverges = variant == BOX_MAX_RECONVERGE;
   nir_variable *yv = nir_local_variable_create(b.impl, glsl_int_type(), "y");
   nir_variable *xv = nir_local_variable_create(b.impl, glsl_int_type(), "x");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_def *r = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
   nir_def *c = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_store_var(&b, yv, nir_ineg(&b, r), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_def *live_out = NULL;
   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *y = nir_load_var(&b, yv);
      if (!bottom)
         nir_break_if(&b, nir_ilt(&b, r, y));
      /* The window as a depthwise convolution reads it, inside the loop: a
       * uniform-buffer read may run before the loop; a buffer read may not. */
      nir_def *w = r;
      if (variant == BOX_UBO_RADIUS)
         w = nir_load_global_constant(&b, 1, 32, buffer_address(&b, nir_imm_int(&b, 8)),
                                      .access = ACCESS_CAN_SPECULATE, .align_mul = 4);
      else if (variant == BOX_SSBO_RADIUS)
         w = nir_load_global(&b, 1, 32, buffer_address(&b, nir_imm_int(&b, 8)),
                             .align_mul = 4);
      nir_store_var(&b, xv, nir_isub(&b, c, w), 1);
      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, xv);
         nir_break_if(&b, nir_ilt(&b, nir_iadd(&b, c, w), x));
         if (variant == BOX_INNER_IF) {
            /* buf[x + 1] shares x + 1 with the increment, which is then
             * defined before the if the body ends in. */
            nir_def *x1 = nir_iadd_imm(&b, x, 1);
            nir_def *v = nir_load_global(&b, 1, 32,
                                         buffer_address(&b, nir_amul_imm(&b, x1, 4)),
                                         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                         .align_mul = 4);
            nir_push_if(&b, nir_ult(&b, nir_imm_int(&b, 7), v));
            nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
            nir_pop_if(&b, NULL);
            nir_store_var(&b, xv, x1, 1);
         } else {
            nir_def *v = nir_load_global(&b, 1, 32,
                                         buffer_address(&b, nir_amul_imm(&b, x, 4)),
                                         .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                         .align_mul = 4);
            nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), v), 1);
            nir_store_var(&b, xv, nir_iadd_imm(&b, x, 1), 1);
         }
      }
      nir_pop_loop(&b, inner);
      if (barrier)
         nir_barrier(&b, .execution_scope = SCOPE_WORKGROUP);
      if (variant == BOX_LANE_STATE) {
         nir_def *lane = nir_load_active_subgroup_invocation_agx(&b, 32);
         nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), lane), 1);
      }
      if (variant == BOX_SHARED_ATOMIC) {
         /* OpAtomicLoad on Workgroup storage: only ACCESS_ATOMIC is set. */
         nir_def *flag = nir_load_shared(&b, 1, 32, nir_imm_int(&b, 0), .access = ACCESS_ATOMIC,
                                         .align_mul = 4);
         nir_store_var(&b, acc, nir_iadd(&b, nir_load_var(&b, acc), flag), 1);
      }
      nir_def *next = nir_iadd_imm(&b, y, 1);
      nir_store_var(&b, yv, next, 1);
      if (bottom) {
         /* Bottom-tested, optionally with an intrinsic result of the body
          * used after the loop. */
         if (variant == BOX_INTRINSIC_LIVE_OUT)
            live_out = nir_load_local_invocation_index(&b);
         nir_break_if(&b, nir_ilt(&b, r, next));
      }
   }
   nir_pop_loop(&b, outer);
   nir_def *result = nir_load_var(&b, acc);
   if (live_out)
      result = nir_iadd(&b, result, live_out);
   nir_store_global(&b, result, buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "box before");
   return b.shader;
}

TEST(M5RuntimeLoops, NestedWindowUnswitches)
{
   /* Split, and the first copy's inner loop unrolled (main loop and
    * remainder): 2 + 3 loops. With the inner body ending in an if, the
    * increment's register write must stay in the latch, or the rebuilt
    * back-edge value is a phi and that inner loop is no longer unrolled. */
   for (box_variant v : {BOX_PLAIN, BOX_BOTTOM_TESTED, BOX_INNER_IF, BOX_UBO_RADIUS}) {
      nir_shader *s = box_filter(v);
      ASSERT_EQ(count_loops(s), 2u);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(s, 8, 320, NULL)) << "variant " << v;
      nir_validate_shader(s, "box after");
      EXPECT_EQ(count_loops(s), 5u) << "variant " << v;
      EXPECT_EQ(loops_inside_ifs(s), 2u) << "variant " << v; /* the two outer copies */
      expect_no_undef_header_phis(s);
      NIR_PASS(_, s, nir_opt_dce);
      nir_validate_shader(s, "box after dce");
      ralloc_free(s);
   }

   /* Not split: a barrier or a lane-state operation (the condition may
    * differ between invocations), an atomic access (a lane may wait on
    * another through memory), maximal reconvergence, an intrinsic result
    * used after the outer loop (it would become a phi of the two copies'
    * values; the backend reads some intrinsics, such as bindless handles,
    * from their definition), or a window read from memory the shader may
    * write (it cannot be read before the loop). */
   for (box_variant v : {BOX_BARRIER, BOX_LANE_STATE, BOX_SHARED_ATOMIC, BOX_MAX_RECONVERGE,
                         BOX_INTRINSIC_LIVE_OUT, BOX_SSBO_RADIUS}) {
      nir_shader *s = box_filter(v);
      agx_nir_unroll_runtime_loops(s, 8, 320, NULL);
      nir_validate_shader(s, "box refused after");
      EXPECT_EQ(loops_inside_ifs(s), 0u) << "variant " << v;
      expect_no_undef_header_phis(s);
      ralloc_free(s);
   }
}

/* Under soft_fault on AGX3C, non-robust shaders (checked_offsets == false)
 * merge consecutive buffer words into a single vector load and fold constant
 * byte offsets, while robust shaders (checked_offsets == true) leave wrapping
 * reads separate. */
TEST(M5MergeWords, SoftFaultRespectsCheckedOffsets)
{
   auto compile_and_count_loads = [](bool checked_offsets, unsigned &num_loads,
                                     unsigned &max_comps) {
      nir_builder b = compute_shader(checked_offsets ? "sf-checked" : "sf-unchecked");
      nir_def *idx = nir_load_preamble(&b, 1, 32, .base = 2);
      nir_def *off = nir_ishl_imm(&b, idx, 2);
      nir_def *sum = nir_imm_int(&b, 0);
      for (unsigned k = 0; k < 4; ++k) {
         nir_def *v = nir_load_global(
            &b, 1, 32, buffer_address(&b, nir_iadd_imm(&b, off, 4 * k)),
            .access = RO_ACCESS, .align_mul = 4);
         sum = nir_iadd(&b, sum, v);
      }
      nir_store_global(&b, sum, buffer_address(&b, nir_imm_int(&b, 64)),
                       .align_mul = 4);
      agx_preprocess_nir(b.shader);
      struct agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17S;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.soft_fault = true;
      key.checked_offsets = checked_offsets;
      key.reserved_preamble = 8;
      key.no_preamble = true;
      struct agx_shader_part out = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));
      num_loads = 0;
      max_comps = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic == nir_intrinsic_load_agx &&
                   !(nir_intrinsic_access(intr) & ACCESS_BUFFER_BINDING_AGX)) {
                  num_loads++;
                  max_comps = MAX2(max_comps, intr->def.num_components);
               }
            }
         }
      }
      free(out.binary);
      ralloc_free(b.shader);
   };

   unsigned unchecked_loads = 0, unchecked_comps = 0;
   compile_and_count_loads(false, unchecked_loads, unchecked_comps);
   EXPECT_EQ(unchecked_loads, 1u);
   EXPECT_EQ(unchecked_comps, 4u);

   unsigned checked_loads = 0, checked_comps = 0;
   compile_and_count_loads(true, checked_loads, checked_comps);
   EXPECT_EQ(checked_loads, 4u);
   EXPECT_EQ(checked_comps, 1u);
}

TEST(M5RuntimeLoops, RuntimeStepReductionUnrolls)
{
   /* Bottom-tested signed and top-tested unsigned read-only strided reduction
    * loops unroll by 8; a strided loop that writes memory is left alone. */
   for (unsigned mode = 0; mode < 3; ++mode) {
      const bool bottom = (mode == 0);
      const bool is_unsigned = (mode == 1);
      const bool has_store = (mode == 2);

      nir_builder b = compute_shader("runtime-step");
      nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "off");
      nir_variable *acc = nir_local_variable_create(b.impl, glsl_float_type(), "sum");
      nir_def *step = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
      nir_def *bound = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 32);
      nir_def *idx = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc, nir_imm_float(&b, 0.0f), 1);

      nir_loop *lp = nir_push_loop(&b);
      {
         nir_def *off = nir_load_var(&b, iv);
         if (!bottom) {
            nir_def *done = is_unsigned ? nir_uge(&b, off, bound) : nir_ige(&b, off, bound);
            nir_break_if(&b, done);
         }
         nir_def *addr = buffer_address(&b, nir_ishl_imm(&b, nir_iadd(&b, off, idx), 2));
         nir_def *v = nir_load_global(&b, 1, 32, addr,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .align_mul = 4);
         if (has_store) {
            nir_store_global(&b, v, addr, .align_mul = 4);
         } else {
            nir_store_var(&b, acc, nir_fadd(&b, nir_load_var(&b, acc), v), 1);
         }
         nir_def *next = nir_iadd(&b, off, step);
         nir_store_var(&b, iv, next, 1);
         if (bottom) {
            nir_def *done = is_unsigned ? nir_uge(&b, next, bound) : nir_ige(&b, next, bound);
            nir_break_if(&b, done);
         }
      }
      nir_pop_loop(&b, lp);
      nir_store_global(&b, nir_load_var(&b, acc), buffer_address(&b, nir_imm_int(&b, 0)), .align_mul = 4);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "runtime-step before");

      unsigned widest = 0;
      const bool progress = agx_nir_unroll_runtime_loops(b.shader, 8, 320, &widest);
      nir_validate_shader(b.shader, "runtime-step after");
      if (has_store) {
         EXPECT_FALSE(progress);
         EXPECT_EQ(count_loops(b.shader), 1u);
      } else {
         EXPECT_TRUE(progress) << "mode " << mode;
         EXPECT_EQ(widest, 8u) << "mode " << mode;
         EXPECT_EQ(count_loops(b.shader), 2u) << "mode " << mode;
         expect_no_undef_header_phis(b.shader);
      }
      ralloc_free(b.shader);
   }

   /* Exhaustive boundary check of signed and unsigned runtime-step guards. */
   const uint32_t F = 8;
   static const int32_t s_vals[] = {
      INT32_MIN, INT32_MIN + 7, INT32_MIN + 8, -100, -1, 0, 1, 7, 8, 9, 320,
      INT32_MAX / 8, INT32_MAX / 8 + 1, INT32_MAX - 8, INT32_MAX - 1, INT32_MAX,
   };
   for (int32_t step : s_vals) {
      for (int32_t bound : s_vals) {
         const uint32_t span = (uint32_t)step * F;
         const bool valid = (step > 0) && (step <= INT32_MAX / (int32_t)F) &&
                            (bound >= (int32_t)(span + (uint32_t)INT32_MIN));
         const int32_t g = valid ? (int32_t)((uint32_t)bound - span) : INT32_MIN;
         for (int32_t iv : s_vals) {
            if (iv < g) {
               ASSERT_TRUE(valid);
               int64_t end = (int64_t)iv + (int64_t)F * (int64_t)step;
               ASSERT_LT(end, (int64_t)bound);
               ASSERT_LE(end, (int64_t)INT32_MAX);
            }
         }
      }
   }
   static const uint32_t u_vals[] = {
      0u, 1u, 7u, 8u, 9u, 320u, UINT32_MAX / 8u, UINT32_MAX / 8u + 1u,
      UINT32_MAX - 8u, UINT32_MAX - 1u, UINT32_MAX,
   };
   for (uint32_t step : u_vals) {
      for (uint32_t bound : u_vals) {
         const uint32_t span = step * F;
         const bool valid = (step != 0u) && (step <= UINT32_MAX / F) && (bound >= span);
         const uint32_t g = valid ? (bound - span) : 0u;
         for (uint32_t iv : u_vals) {
            if (iv < g) {
               ASSERT_TRUE(valid);
               uint64_t end = (uint64_t)iv + (uint64_t)F * (uint64_t)step;
               ASSERT_LT(end, (uint64_t)bound);
               ASSERT_LE(end, (uint64_t)UINT32_MAX);
            }
         }
      }
   }
}

TEST(M5RuntimeLoops, SharedStepOverflowSkipsUnroll)
{
   /* A stride-32 loop indexing 4-byte shared elements when shared_size == 128
    * cannot run two in-bounds iterations; skip dead runtime unrolling. When
    * shared_size == 1024 (room for 8 iterations), unrolling proceeds. */
   for (unsigned sh_bytes : {128u, 1024u}) {
      nir_builder b = compute_shader("shared-step-overflow");
      b.shader->info.shared_size = sh_bytes;
      nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 32);
      nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_store_var(&b, iv, lid, 1);

      nir_loop *lp = nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, iv);
         nir_break_if(&b, nir_ige(&b, i, n));
         nir_def *v = nir_load_global(&b, 1, 32, buffer_address(&b, nir_ishl_imm(&b, i, 2)),
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .align_mul = 4);
         nir_store_shared(&b, v, nir_ishl_imm(&b, i, 2), .write_mask = 1, .align_mul = 4);
         nir_store_var(&b, iv, nir_iadd_imm(&b, i, 32), 1);
      }
      nir_pop_loop(&b, lp);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "shared-step-overflow before");

      const bool progress = agx_nir_unroll_runtime_loops(b.shader, 8, 320, NULL);
      nir_validate_shader(b.shader, "shared-step-overflow after");
      EXPECT_EQ(progress, sh_bytes > 128u) << "sh_bytes=" << sh_bytes;
      EXPECT_EQ(count_loops(b.shader), sh_bytes > 128u ? 2u : 1u);
      ralloc_free(b.shader);
   }
}
