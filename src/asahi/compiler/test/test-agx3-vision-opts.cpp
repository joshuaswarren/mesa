/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include <cmath>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_format.h"
extern "C" {
#include "agx_nir.h"
}

static nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

static nir_def *
make_buffer_addr(nir_builder *b, nir_def *base_u64, nir_def *byte_offset)
{
   return nir_iadd(b, base_u64, nir_u2u64(b, byte_offset));
}

static nir_def *
make_desc_base(nir_builder *b, int preamble_base)
{
   nir_def *set = nir_load_preamble(b, 1, 64);
   nir_intrinsic_set_base(nir_def_as_intrinsic(set), preamble_base);
   nir_def *desc = nir_load_global_constant(b, 4, 32, set);
   nir_intrinsic_instr *load = nir_def_as_intrinsic(desc);
   nir_intrinsic_set_access(load, (gl_access_qualifier)(ACCESS_CAN_SPECULATE |
                                                        ACCESS_BUFFER_BINDING_AGX));
   nir_intrinsic_set_align(load, 16, 0);
   return nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                 nir_channel(b, desc, 1));
}

static nir_def *
load_push_word(nir_builder *b, unsigned offset, unsigned range)
{
   nir_def *v = nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset));
   nir_intrinsic_set_range(nir_def_as_intrinsic(v), range);
   return v;
}

static nir_def *
load_readonly_byte(nir_builder *b, nir_def *addr)
{
   nir_def *v = nir_load_global(b, 1, 8, addr);
   nir_intrinsic_set_access(nir_def_as_intrinsic(v),
                            (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER));
   return v;
}

static nir_def *
load_u32_agx(nir_builder *b, nir_def *base, nir_def *index,
             gl_access_qualifier access = (gl_access_qualifier)0)
{
   nir_def *v = nir_load_agx(b, 1, 32, base, index);
   nir_intrinsic_set_format(nir_def_as_intrinsic(v), PIPE_FORMAT_R32_UINT);
   nir_intrinsic_set_access(nir_def_as_intrinsic(v), access);
   return v;
}

static void
store_u32_agx(nir_builder *b, nir_def *v, nir_def *base, nir_def *index)
{
   nir_intrinsic_set_format(nir_store_agx(b, v, base, index), PIPE_FORMAT_R32_UINT);
}

static unsigned
count_intrinsics(nir_shader *shader, nir_intrinsic_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic == op)
               n++;
         }
      }
   }
   return n;
}

static unsigned
count_alu_op(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

static unsigned
count_ifs(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_if) {
            nir_if *nif = nir_cf_node_as_if(block->cf_node.parent);
            if (nir_if_first_then_block(nif) == block)
               n++;
         }
      }
   }
   return n;
}

static unsigned
count_loops(nir_shader *shader)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         if (block->cf_node.parent->type == nir_cf_node_loop &&
             nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)) == block)
            n++;
      }
   }
   return n;
}

/* --- Commit 1: agx_nir_merge_global_words --- */

TEST(AGX3VisionOpts, RuntimeUnrollWorkgroupStride)
{
   struct {
      int step;
      bool nested;
      bool negated_add;
   } cases[] = {{1, true, false}, {32, false, false}, {32, true, false},
                {-32, true, false}, {-32, true, true}};
   for (auto c : cases) {
      int step = c.step;
      nir_builder b = make_compute_shader("workgroup-stride");
      nir_def *bound = load_push_word(&b, 0, 16);
      nir_def *lane = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_variable *iv = nir_local_variable_create(b.impl, glsl_int_type(), "i");
      nir_loop *outer = c.nested ? nir_push_loop(&b) : NULL;
      nir_def *start = lane;
      if (step < 0)
         start = c.negated_add ? nir_iadd(&b, bound, nir_ineg(&b, lane))
                               : nir_isub(&b, bound, lane);
      nir_store_var(&b, iv, start, 1);
      nir_def *base = make_desc_base(&b, 4);
      nir_loop *loop = nir_push_loop(&b);
      nir_def *i = nir_load_var(&b, iv);
      nir_break_if(&b, step > 0 ? nir_ige(&b, i, bound)
                                : nir_ilt(&b, i, nir_imm_int(&b, 0)));
      nir_store_global(&b, i, make_buffer_addr(&b, base, nir_ishl_imm(&b, i, 2)));
      nir_store_var(&b, iv, nir_iadd_imm(&b, i, step), 1);
      nir_pop_loop(&b, loop);
      if (outer) {
         nir_break_if(&b, nir_ieq_imm(&b, bound, 0));
         nir_pop_loop(&b, outer);
      }
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      EXPECT_EQ(agx_nir_unroll_runtime_loops(b.shader, 8, 320, NULL), step == 1 || !c.nested);
      nir_validate_shader(b.shader, "workgroup-stride");
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, MergeGlobalWordsNarrowU8AndSignedDuplicates)
{
   nir_builder b = make_compute_shader("merge-u8-signed");
   nir_def *base = make_desc_base(&b, 4);
   nir_def *idx = load_push_word(&b, 0, 16);
   nir_def *out = make_desc_base(&b, 8);

   /* Signed offsets {-4, 0, +4, +4} of 32-bit loads -> single vec3 load + CSE. */
   nir_def *l0 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, nir_iadd_imm(&b, idx, -4)));
   nir_def *l1 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, idx));
   nir_def *l2 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, nir_iadd_imm(&b, idx, 4)));
   nir_def *l3 = nir_load_global(&b, 1, 32, make_buffer_addr(&b, base, nir_iadd_imm(&b, idx, 4)));
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, l0, l1), nir_iadd(&b, l2, l3));
   nir_store_global(&b, sum, out);

   ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 4u);
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge signed duplicates");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 1u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, MergeGlobalWordsU8LoopFreeUnpackGate)
{
   for (bool free_unpack : {true, false}) {
      nir_builder b = make_compute_shader(free_unpack ? "merge-u8-free" : "merge-u8-nonfree");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *n = load_push_word(&b, 0, 16);
      nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "i");
      nir_variable *acc = nir_local_variable_create(b.impl, free_unpack ? glsl_float_type() : glsl_uint16_t_type(), "acc");
      nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, acc, free_unpack ? nir_imm_float(&b, 0.0f) : nir_imm_intN_t(&b, 0, 16), 1);

      nir_loop *loop = nir_push_loop(&b);
      {
         nir_def *i = nir_load_var(&b, iv);
         nir_break_if(&b, nir_ige(&b, i, n));
         nir_def *cur = nir_load_var(&b, acc);
         for (unsigned k = 0; k < 4; ++k) {
            nir_def *v = load_readonly_byte(&b, make_buffer_addr(&b, base, nir_iadd_imm(&b, i, k)));
            if (free_unpack)
               cur = nir_fadd(&b, cur, nir_u2f32(&b, v));
            else
               cur = nir_iadd(&b, cur, nir_ishl_imm(&b, nir_u2u16(&b, v), 1));
         }
         nir_store_var(&b, acc, cur, 1);
         nir_store_var(&b, iv, nir_iadd_imm(&b, i, 4), 1);
      }
      nir_pop_loop(&b, loop);
      nir_def *res = nir_load_var(&b, acc);
      nir_store_global(&b, free_unpack ? res : nir_u2u32(&b, res), out);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 4u);

      bool changed = agx_nir_merge_global_words(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "merge u8 loop gate");
      EXPECT_EQ(changed, free_unpack);
      EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global),
                free_unpack ? 1u : 4u);
      ralloc_free(b.shader);
   }
}

/* --- Commit 2: agx_nir_reassoc_const_offsets, u8_absdiff_fp, unroll_and_jam --- */

TEST(AGX3VisionOpts, ReassocConstOffsetsRespectsSingleUse)
{
   for (bool multi_use : {false, true}) {
      nir_builder b = make_compute_shader(multi_use ? "reassoc-multi" : "reassoc-single");
      nir_def *a = load_push_word(&b, 0, 16);
      nir_def *c = load_push_word(&b, 4, 16);
      nir_def *out = make_desc_base(&b, 4);

      nir_def *ap4 = nir_iadd_imm(&b, a, 4);
      nir_def *cp8 = nir_iadd_imm(&b, c, 8);
      nir_def *sum = nir_iadd(&b, ap4, cp8);
      nir_store_global(&b, sum, out);
      if (multi_use) {
         nir_store_global(&b, ap4, make_buffer_addr(&b, out, nir_imm_int(&b, 4)));
         nir_store_global(&b, cp8, make_buffer_addr(&b, out, nir_imm_int(&b, 8)));
      }

      bool changed = agx_nir_reassoc_const_offsets(b.shader);
      nir_validate_shader(b.shader, "reassoc");
      EXPECT_EQ(changed, !multi_use);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, U8AbsdiffFpTransformsReusedAndIsBitExact)
{
   for (bool reused : {true, false}) {
      nir_builder b = make_compute_shader(reused ? "absdiff-reused" : "absdiff-unique");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);

      nir_def *a[4], *c[4];
      for (unsigned i = 0; i < 4; ++i) {
         a[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, i)));
         c[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, 16 + i)));
      }

      /* Chain 1: sum_i |a[i] - c[i]|. Chain 2 (when reused): sum_i |a[i] - c[(i+1)&3]|. */
      nir_def *acc1 = nir_imm_int(&b, 0);
      for (unsigned i = 0; i < 4; ++i) {
         nir_def *d = nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[i]));
         acc1 = nir_iadd(&b, acc1, nir_iabs(&b, d));
      }
      nir_store_global(&b, acc1, out);

      if (reused) {
         nir_def *acc2 = nir_imm_int(&b, 0);
         for (unsigned i = 0; i < 4; ++i) {
            nir_def *d = nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[(i + 1) & 3]));
            acc2 = nir_iadd(&b, acc2, nir_iabs(&b, d));
         }
         nir_store_global(&b, acc2, make_buffer_addr(&b, out, nir_imm_int(&b, 4)));
      }

      bool changed = agx_nir_opt_u8_absdiff_fp(b.shader);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "u8 absdiff fp");
      EXPECT_EQ(changed, reused);
      if (reused) {
         EXPECT_GT(count_alu_op(b.shader, nir_op_u2f32), 0u);
         EXPECT_GT(count_alu_op(b.shader, nir_op_f2u32), 0u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_iabs), 0u);
      }
      ralloc_free(b.shader);
   }

   /* Run the actual pass on known u8 values, then replace loads with constants
    * and fold with nir_opt_constant_folding to verify the rewritten NIR produces
    * the exact integer sum. */
   {
      nir_builder b = make_compute_shader("absdiff-eval");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      const uint8_t av[4] = {0, 255, 13, 200};
      const uint8_t cv[4] = {255, 1, 200, 13};
      nir_def *a[4], *c[4];
      for (unsigned i = 0; i < 4; ++i) {
         a[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, i)));
         c[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, 16 + i)));
      }
      nir_def *acc1 = nir_imm_int(&b, 0);
      nir_def *acc2 = nir_imm_int(&b, 0);
      for (unsigned i = 0; i < 4; ++i) {
         acc1 = nir_iadd(&b, acc1, nir_iabs(&b, nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[i]))));
         acc2 = nir_iadd(&b, acc2, nir_iabs(&b, nir_isub(&b, nir_u2u32(&b, a[i]), nir_u2u32(&b, c[(i + 1) & 3]))));
      }
      nir_intrinsic_instr *st1 = nir_store_global(&b, acc1, out);
      nir_intrinsic_instr *st2 = nir_store_global(&b, acc2, make_buffer_addr(&b, out, nir_imm_int(&b, 4)));

      ASSERT_TRUE(agx_nir_opt_u8_absdiff_fp(b.shader));
      nir_validate_shader(b.shader, "u8 absdiff fp eval");
      /* Substitute constant u8 values into the rewritten NIR and constant-fold. */
      b.cursor = nir_before_impl(b.impl);
      for (unsigned i = 0; i < 4; ++i) {
         nir_def_replace(a[i], nir_imm_intN_t(&b, av[i], 8));
         nir_def_replace(c[i], nir_imm_intN_t(&b, cv[i], 8));
      }
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      uint32_t exp1 = 0, exp2 = 0;
      for (unsigned i = 0; i < 4; ++i) {
         exp1 += (uint32_t)std::abs((int)av[i] - (int)cv[i]);
         exp2 += (uint32_t)std::abs((int)av[i] - (int)cv[(i + 1) & 3]);
      }
      ASSERT_TRUE(nir_src_is_const(st1->src[0]));
      ASSERT_TRUE(nir_src_is_const(st2->src[0]));
      EXPECT_EQ(nir_src_as_uint(st1->src[0]), exp1);
      EXPECT_EQ(nir_src_as_uint(st2->src[0]), exp2);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, UnrollAndJam2DNestedLoopAndBarrierRejection)
{
   for (bool with_barrier : {false, true}) {
      nir_builder b = make_compute_shader(with_barrier ? "jam-barrier" : "jam-2d");
      nir_def *left_base = make_desc_base(&b, 4);
      nir_def *right_base = make_desc_base(&b, 8);
      nir_def *out_base = make_desc_base(&b, 12);
      nir_def *max_d = load_push_word(&b, 0, 32);
      nir_def *width = load_push_word(&b, 4, 32);
      nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

      nir_variable *dv = nir_local_variable_create(b.impl, glsl_int_type(), "d");
      nir_variable *best_cost = nir_local_variable_create(b.impl, glsl_uint_type(), "best_cost");
      nir_variable *best_d = nir_local_variable_create(b.impl, glsl_int_type(), "best_d");
      nir_store_var(&b, dv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, best_cost, nir_imm_int(&b, 0xffffffffu), 1);
      nir_store_var(&b, best_d, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *d = nir_load_var(&b, dv);

         if (with_barrier)
            nir_intrinsic_set_execution_scope(nir_barrier(&b), SCOPE_WORKGROUP);

         nir_variable *kv = nir_local_variable_create(b.impl, glsl_int_type(), "k");
         nir_variable *sad = nir_local_variable_create(b.impl, glsl_uint_type(), "sad");
         nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
         nir_store_var(&b, sad, nir_imm_int(&b, 0), 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *k = nir_load_var(&b, kv);
            nir_break_if(&b, nir_ige(&b, k, width));
            nir_def *l = load_readonly_byte(&b, make_buffer_addr(&b, left_base, nir_iadd(&b, x, k)));
            nir_def *r = load_readonly_byte(&b, make_buffer_addr(&b, right_base, nir_isub(&b, nir_iadd(&b, x, k), d)));
            nir_def *diff = nir_iabs(&b, nir_isub(&b, nir_u2u32(&b, l), nir_u2u32(&b, r)));
            nir_store_var(&b, sad, nir_iadd(&b, nir_load_var(&b, sad), diff), 1);
            nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
         }
         nir_pop_loop(&b, inner);

         nir_def *cost = nir_load_var(&b, sad);
         nir_def *cur_best = nir_load_var(&b, best_cost);
         nir_def *better = nir_ult(&b, cost, cur_best);
         nir_store_var(&b, best_cost, nir_bcsel(&b, better, cost, cur_best), 1);
         nir_store_var(&b, best_d, nir_bcsel(&b, better, d, nir_load_var(&b, best_d)), 1);
         nir_def *next_d = nir_iadd_imm(&b, d, 1);
         nir_store_var(&b, dv, next_d, 1);
         nir_break_if(&b, nir_ige(&b, next_d, max_d));
      }
      nir_pop_loop(&b, outer);

      nir_store_global(&b, nir_load_var(&b, best_d), out_base);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_copy_prop);
      NIR_PASS(_, b.shader, nir_opt_cse);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_EQ(count_loops(b.shader), 2u);

      unsigned widest = 0;
      bool jammed = agx_nir_unroll_and_jam(b.shader, 8, &widest);
      nir_validate_shader(b.shader, "unroll and jam");
      EXPECT_EQ(jammed, !with_barrier);
      if (!with_barrier) {
         EXPECT_GT(count_loops(b.shader), 2u);
         EXPECT_GE(widest, 2u);
      }
      ralloc_free(b.shader);
   }
}

/* --- Commit 3: guard_selected_loads, forward_merge_loads, fuse_selects --- */

TEST(AGX3VisionOpts, GuardSelectedLoadsRequiresReadonlyChain)
{
   struct {
      bool readonly;
      bool chain;
      bool expect_guard;
   } cases[] = {{true, true, true}, {false, true, false}, {true, false, false}};
   for (auto test : cases) {
      nir_builder b = make_compute_shader(test.readonly ? "guard-ro" : "guard-rw");
      nir_def *base_ptr = make_desc_base(&b, 4);
      nir_def *out_ptr = make_desc_base(&b, 8);
      nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
      nir_def *center = load_push_word(&b, 0, 16);

      enum gl_access_qualifier acc =
         test.readonly ? (enum gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER)
                  : (enum gl_access_qualifier)0;
      nir_def *n0 = load_u32_agx(&b, base_ptr, nir_iadd_imm(&b, idx, -1), acc);
      nir_def *n1 = load_u32_agx(&b, base_ptr, nir_iadd_imm(&b, idx, 1), acc);
      nir_def *c0 = nir_uge(&b, center, n0);
      nir_def *c1 = nir_uge(&b, center, n1);
      nir_def *zero = nir_imm_int(&b, 0);
      nir_def *res = test.chain
                        ? nir_bcsel(&b, c1, nir_bcsel(&b, c0, center, zero), zero)
                        : nir_bcsel(&b, nir_iand(&b, c0, c1), center, zero);
      store_u32_agx(&b, res, out_ptr, idx);

      ASSERT_EQ(count_ifs(b.shader), 0u);
      bool changed = agx_nir_opt_guard_selected_loads(b.shader);
      nir_validate_shader(b.shader, "guard selected loads");
      EXPECT_EQ(changed, test.expect_guard);
      EXPECT_EQ(count_ifs(b.shader), test.expect_guard ? 1u : 0u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, ForwardMergeLoadsEliminatesRoundTrip)
{
   nir_builder b = make_compute_shader("forward-merge");
   nir_def *ptr = make_desc_base(&b, 4);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *cond = nir_ine_imm(&b, idx, 0);
   nir_def *va = load_push_word(&b, 0, 16);
   nir_def *vb = load_push_word(&b, 4, 16);

   nir_push_if(&b, cond);
   store_u32_agx(&b, va, ptr, idx);
   nir_push_else(&b, NULL);
   store_u32_agx(&b, vb, ptr, idx);
   nir_pop_if(&b, NULL);

   nir_def *reread = load_u32_agx(&b, ptr, idx);
   store_u32_agx(&b, reread, out, idx);

   ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
   EXPECT_TRUE(agx_nir_opt_forward_merge_loads(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "forward merge loads");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 0u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, FuseSelectsCollapsesSharedPredicateMultiUse)
{
   nir_builder b = make_compute_shader("fuse-selects-shared");
   nir_def *out = make_desc_base(&b, 4);
   nir_def *p0 = load_push_word(&b, 0, 32);
   nir_def *p1 = load_push_word(&b, 4, 32);
   nir_def *v0 = load_push_word(&b, 8, 32);
   nir_def *v1 = load_push_word(&b, 12, 32);
   nir_def *v2 = load_push_word(&b, 16, 32);

   nir_def *a = nir_ine_imm(&b, p0, 0);
   nir_def *c = nir_ine_imm(&b, p1, 0);
   nir_def *both = nir_ior(&b, a, c);
   /* Two selects both collapse `both` so `ior` becomes completely dead. */
   nir_def *r0 = nir_bcsel(&b, both, nir_bcsel(&b, a, v0, v1), v2);
   nir_def *r1 = nir_bcsel(&b, both, nir_bcsel(&b, a, v1, v2), v0);
   nir_store_global(&b, nir_iadd(&b, r0, r1), out);

   ASSERT_EQ(count_alu_op(b.shader, nir_op_ior), 1u);
   EXPECT_TRUE(agx_nir_fuse_selects(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "fuse selects shared");
   EXPECT_EQ(count_alu_op(b.shader, nir_op_ior), 0u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, MergeGlobalWordsSkipsRobustBoundedNegativeOffset)
{
   /* Soft-fault robust 32-bit storage loads with negative lowest offset
    * {X-4, X, X+4}: at X = 0, X-4 wraps to 0xFFFFFFFC so merging into one
    * vec3 load at base + u2u64(X-4) would read channel 1 from base + 2^32. */
   for (bool is_u8 : {false, true}) {
      nir_builder b = make_compute_shader(is_u8 ? "robust-neg-u8" : "robust-neg-u32");
      nir_def *base = make_desc_base(&b, 4);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *idx = load_push_word(&b, 0, 16);
      nir_def *bound = load_push_word(&b, 4, 16);
      const unsigned bits = is_u8 ? 8 : 32;
      const unsigned stride = bits / 8;

      nir_def *vals[3];
      for (int k = -1; k <= 1; ++k) {
         nir_def *off = k == 0 ? idx : nir_iadd_imm(&b, idx, (uint32_t)(k * (int)stride));
         nir_def *ld = nir_load_global(&b, 1, bits, make_buffer_addr(&b, base, off));
         nir_def *end = nir_uadd_sat(&b, off, nir_imm_int(&b, stride));
         if (is_u8) {
            vals[k + 1] = nir_u2u32(&b, nir_bcsel(&b, nir_uge(&b, bound, end), ld,
                                                  nir_imm_intN_t(&b, 0, 8)));
         } else {
            vals[k + 1] = nir_bounds_agx(&b, ld, end, bound);
         }
      }
      nir_def *sum = nir_iadd(&b, nir_iadd(&b, vals[0], vals[1]), vals[2]);
      nir_store_global(&b, sum, out);

      ASSERT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 3u);
      EXPECT_FALSE(agx_nir_merge_global_words(b.shader));
      nir_validate_shader(b.shader, "merge robust negative offset");
      EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 3u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, UnrollAndJamRejectsOuterReductionUsedInInnerLoop)
{
   nir_builder b = make_compute_shader("jam-red-in-inner");
   nir_def *left_base = make_desc_base(&b, 4);
   nir_def *right_base = make_desc_base(&b, 8);
   nir_def *out_base = make_desc_base(&b, 12);
   nir_def *max_d = load_push_word(&b, 0, 32);
   nir_def *width = load_push_word(&b, 4, 32);
   nir_def *x = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_variable *dv = nir_local_variable_create(b.impl, glsl_int_type(), "d");
   nir_variable *acc = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, dv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 7), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *d = nir_load_var(&b, dv);
      nir_def *cur_acc = nir_load_var(&b, acc);

      nir_variable *kv = nir_local_variable_create(b.impl, glsl_int_type(), "k");
      nir_variable *sad = nir_local_variable_create(b.impl, glsl_uint_type(), "sad");
      nir_store_var(&b, kv, nir_imm_int(&b, 0), 1);
      nir_store_var(&b, sad, nir_imm_int(&b, 0), 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *k = nir_load_var(&b, kv);
         nir_break_if(&b, nir_ige(&b, k, width));
         nir_def *l = load_readonly_byte(&b, make_buffer_addr(&b, left_base, nir_iadd(&b, x, k)));
         nir_def *r = load_readonly_byte(&b, make_buffer_addr(&b, right_base, nir_isub(&b, nir_iadd(&b, x, k), d)));
         /* Cross-lane dependency: inner loop reads outer reduction phi cur_acc. */
         nir_def *term = nir_iadd(&b, nir_u2u32(&b, l), nir_iadd(&b, nir_u2u32(&b, r), cur_acc));
         nir_store_var(&b, sad, nir_iadd(&b, nir_load_var(&b, sad), term), 1);
         nir_store_var(&b, kv, nir_iadd_imm(&b, k, 1), 1);
      }
      nir_pop_loop(&b, inner);

      nir_store_var(&b, acc, nir_iadd(&b, cur_acc, nir_load_var(&b, sad)), 1);
      nir_def *next_d = nir_iadd_imm(&b, d, 1);
      nir_store_var(&b, dv, next_d, 1);
      nir_break_if(&b, nir_ige(&b, next_d, max_d));
   }
   nir_pop_loop(&b, outer);

   nir_store_global(&b, nir_load_var(&b, acc), out_base);
   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);

   unsigned widest = 0;
   EXPECT_FALSE(agx_nir_unroll_and_jam(b.shader, 8, &widest));
   nir_validate_shader(b.shader, "unroll and jam reject inner red");
   EXPECT_EQ(count_loops(b.shader), 2u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, ForwardMergeLoadsChecksSignExtend)
{
   nir_builder b = make_compute_shader("forward-merge-sext");
   nir_def *ptr = make_desc_base(&b, 4);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *idx = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *cond = nir_ine_imm(&b, idx, 0);
   nir_def *va = load_push_word(&b, 0, 16);
   nir_def *vb = load_push_word(&b, 4, 16);

   nir_push_if(&b, cond);
   store_u32_agx(&b, va, ptr, idx);
   nir_push_else(&b, NULL);
   store_u32_agx(&b, vb, ptr, idx);
   nir_pop_if(&b, NULL);

   /* Load uses sign_extend=true while stores use sign_extend=false -> must not forward. */
   nir_def *reread = load_u32_agx(&b, ptr, idx);
   nir_intrinsic_set_sign_extend(nir_def_as_intrinsic(reread), true);
   store_u32_agx(&b, reread, out, idx);

   EXPECT_FALSE(agx_nir_opt_forward_merge_loads(b.shader));
   nir_validate_shader(b.shader, "forward merge sign_extend");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, NarrowU8FaddTreeTransformsAndEvaluatesExact)
{
   nir_builder b = make_compute_shader("narrow-u8-fadd");
   nir_def *base = make_desc_base(&b, 4);
   nir_def *out = make_desc_base(&b, 8);
   const uint8_t vals[4] = {10, 200, 55, 255};
   nir_def *lds[4];
   nir_def *fsum = nir_imm_float(&b, 0.0f);
   for (unsigned i = 0; i < 4; ++i) {
      lds[i] = nir_load_global(&b, 1, 8, make_buffer_addr(&b, base, nir_imm_int(&b, i)));
      nir_def *f = nir_u2f32(&b, lds[i]);
      fsum = (i == 0) ? f : nir_fadd(&b, fsum, f);
   }
   nir_intrinsic_instr *st = nir_store_global(&b, fsum, out);

   ASSERT_EQ(count_alu_op(b.shader, nir_op_u2f32), 4u);
   ASSERT_EQ(count_alu_op(b.shader, nir_op_fadd), 3u);
   EXPECT_TRUE(agx_nir_opt_narrow_u8_fadd_tree(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "narrow u8 fadd tree");
   EXPECT_EQ(count_alu_op(b.shader, nir_op_u2f32), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_i2f32), 1u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_fadd), 0u);

   b.cursor = nir_before_impl(b.impl);
   for (unsigned i = 0; i < 4; ++i)
      nir_def_replace(lds[i], nir_imm_intN_t(&b, vals[i], 8));
   NIR_PASS(_, b.shader, nir_opt_constant_folding);
   NIR_PASS(_, b.shader, nir_opt_dce);
   ASSERT_TRUE(nir_src_is_const(st->src[0]));
   EXPECT_FLOAT_EQ(nir_src_as_float(st->src[0]), 10.0f + 200.0f + 55.0f + 255.0f);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, FoldConstSelectAluEvaluatesExact)
{
   for (int cond_val : {0, 1}) {
      nir_builder b = make_compute_shader("fold-const-sel");
      nir_def *out = make_desc_base(&b, 4);
      nir_def *p = load_push_word(&b, 0, 16);
      nir_def *cond = nir_ine_imm(&b, p, 0);
      nir_def *sel = nir_bcsel(&b, cond, nir_imm_int(&b, 10), nir_imm_int(&b, 20));
      nir_def *res = nir_iadd(&b, sel, nir_imm_int(&b, 7));
      nir_intrinsic_instr *st = nir_store_global(&b, res, out);

      ASSERT_EQ(count_alu_op(b.shader, nir_op_iadd), 1u);
      EXPECT_TRUE(agx_nir_fold_const_select_alu(b.shader));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "fold const select alu");
      EXPECT_EQ(count_alu_op(b.shader, nir_op_iadd), 0u);

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(p, nir_imm_int(&b, cond_val));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_TRUE(nir_src_is_const(st->src[0]));
      EXPECT_EQ(nir_src_as_uint(st->src[0]), cond_val ? 17u : 27u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, FuseSelectsEvaluatesExactOnAllBooleanInputs)
{
   for (unsigned mask = 0; mask < 4; ++mask) {
      const bool av = mask & 1;
      const bool cv = (mask >> 1) & 1;
      nir_builder b = make_compute_shader("fuse-selects-eval");
      nir_def *out = make_desc_base(&b, 4);
      nir_def *pa = load_push_word(&b, 0, 16);
      nir_def *pc = load_push_word(&b, 4, 16);
      nir_def *a = nir_ine_imm(&b, pa, 0);
      nir_def *c = nir_ine_imm(&b, pc, 0);
      nir_def *r1 = nir_bcsel(&b, nir_ior(&b, a, c),
                              nir_bcsel(&b, a, nir_imm_int(&b, 11), nir_imm_int(&b, 22)),
                              nir_imm_int(&b, 33));
      nir_def *r2 = nir_bcsel(&b, a,
                              nir_bcsel(&b, a, nir_imm_int(&b, 44), nir_imm_int(&b, 55)),
                              nir_imm_int(&b, 66));
      nir_intrinsic_instr *st1 = nir_store_global(&b, r1, out);
      nir_intrinsic_instr *st2 = nir_store_global(&b, r2, make_buffer_addr(&b, out, nir_imm_int(&b, 4)));

      EXPECT_TRUE(agx_nir_fuse_selects(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "fuse selects eval");

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(pa, nir_imm_int(&b, av ? 1 : 0));
      nir_def_replace(pc, nir_imm_int(&b, cv ? 1 : 0));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      NIR_PASS(_, b.shader, nir_opt_dce);
      ASSERT_TRUE(nir_src_is_const(st1->src[0]));
      ASSERT_TRUE(nir_src_is_const(st2->src[0]));
      EXPECT_EQ(nir_src_as_uint(st1->src[0]), av ? 11u : (cv ? 22u : 33u));
      EXPECT_EQ(nir_src_as_uint(st2->src[0]), av ? 44u : 66u);
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchExactThreeAndFour)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      /* mode 0: rad=2 (trip == 3, straight-line 3x, 0 inner loops)
       * mode 1: rad=4 (trip == 5, straight-line 4x + 1-iter remainder)
       * mode 2: rad=0 (trip == 1, scalar fallback)
       * mode 3: x0=INT32_MAX-2, bound=INT32_MAX (REL_LE wrap guard rejects exact-3) */
      nir_builder b = make_compute_shader("small-window-unswitch");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xhi = nir_iadd(&b, x0, rad);
      nir_def *yhi = nir_iadd(&b, y0, rad);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, y_var, y0, 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var,
                          nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      for (unsigned r = 0; r < 4; ++r) {
         NIR_PASS(_, b.shader, nir_opt_copy_prop);
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         NIR_PASS(_, b.shader, nir_opt_algebraic);
         NIR_PASS(_, b.shader, nir_opt_dead_cf);
         NIR_PASS(_, b.shader, nir_opt_dce);
      }
      nir_validate_shader(b.shader, "small window unswitch");

      b.cursor = nir_before_impl(b.impl);
      if (mode == 0) {
         nir_def_replace(rad, nir_imm_int(&b, 2));
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 1) {
         nir_def_replace(rad, nir_imm_int(&b, 4));
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 2) {
         nir_def_replace(rad, nir_imm_int(&b, 0));
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else {
         nir_def_replace(rad, nir_imm_int(&b, 2));
         nir_def_replace(x0, nir_imm_int(&b, INT32_MAX - 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      }
      for (unsigned r = 0; r < 6; ++r) {
         NIR_PASS(_, b.shader, nir_opt_copy_prop);
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         NIR_PASS(_, b.shader, nir_opt_algebraic);
         NIR_PASS(_, b.shader, nir_opt_dead_cf);
         NIR_PASS(_, b.shader, nir_opt_dce);
      }
      nir_validate_shader(b.shader, "small window unswitch specialized");

      /* Compute maximum loop nesting depth and loop count after specialization. */
      unsigned loops = 0, max_depth = 0;
      nir_foreach_block(block, b.impl) {
         unsigned depth = 0;
         for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent)
            if (n->type == nir_cf_node_loop)
               depth++;
         if (depth > max_depth)
            max_depth = depth;
         if (block->cf_node.parent->type == nir_cf_node_loop &&
             block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
            loops++;
      }

      if (mode == 0) {
         /* Exact 3-tap inner loop is completely eliminated (max depth 1), so
          * the outer loop becomes innermost and is eligible for runtime unroll
          * with 3 straight-line loads per outer iteration. */
         EXPECT_EQ(max_depth, 1u);
         EXPECT_GE(count_intrinsics(b.shader, nir_intrinsic_load_agx), 3u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx) % 3u, 0u);
      } else if (mode == 1) {
         /* 5-tap: 4 straight-line loads + 1 in compact remainder loop. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 5u);
      } else if (mode == 2) {
         /* 1-tap: scalar fallback outer + inner loop. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      } else {
         /* bound == INT32_MAX: exact-3 no-wrap guard rejects straight-line removal. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchSkipsWide32BitLoop)
{
   nir_builder b = make_compute_shader("small-window-skip-32bit");
   nir_def *base = make_desc_base(&b, 0);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
   nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
   nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
   nir_def *xhi = nir_iadd(&b, x0, rad);
   nir_def *yhi = nir_iadd(&b, y0, rad);

   nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
   nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, y_var, y0, 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *y = nir_load_var(&b, y_var);
      nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, yex);

      nir_def *row_off = nir_imul(&b, y, stride);
      nir_store_var(&b, x_var, x0, 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, xex);

         nir_def *idx = nir_iadd(&b, row_off, x);
         nir_def *v = nir_load_agx(&b, 1, 32, base, idx,
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .format = PIPE_FORMAT_R32_UINT);
         nir_store_var(&b, acc_var, nir_iadd(&b, nir_load_var(&b, acc_var), v), 1);
         nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
      }
      nir_pop_loop(&b, inner);
      nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
   }
   nir_pop_loop(&b, outer);
   nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
   nir_validate_shader(b.shader, "small window skip 32bit");

   /* Standard 2-arm unswitch only (main 8x + scalar tail = 5 loops total). */
   unsigned loops = 0;
   nir_foreach_block(block, b.impl) {
      if (block->cf_node.parent->type == nir_cf_node_loop &&
          block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
         loops++;
   }
   EXPECT_EQ(loops, 5u);
   ralloc_free(b.shader);
}


static void
count_loop_shape(nir_function_impl *impl, unsigned *out_loops,
                 unsigned *out_max_depth)
{
   unsigned loops = 0, max_depth = 0;
   nir_foreach_block(block, impl) {
      unsigned depth = 0;
      for (nir_cf_node *n = block->cf_node.parent; n; n = n->parent)
         if (n->type == nir_cf_node_loop)
            depth++;
      if (depth > max_depth)
         max_depth = depth;
      if (block->cf_node.parent->type == nir_cf_node_loop &&
          block == nir_loop_first_block(nir_cf_node_as_loop(block->cf_node.parent)))
         loops++;
   }
   *out_loops = loops;
   *out_max_depth = max_depth;
}

static void
fold_after_specialize(nir_shader *shader, const char *when)
{
   for (unsigned r = 0; r < 6; ++r) {
      NIR_PASS(_, shader, nir_opt_copy_prop);
      NIR_PASS(_, shader, nir_opt_constant_folding);
      NIR_PASS(_, shader, nir_opt_algebraic);
      NIR_PASS(_, shader, nir_opt_dead_cf);
      NIR_PASS(_, shader, nir_opt_dce);
   }
   nir_validate_shader(shader, when);
}

TEST(AGX3VisionOpts, SmallWindowUnswitchDecreasing)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      /* mode 0: strict > (REL_GT), x0=10, xlo=7 (trip == 3 -> exact-3)
       * mode 1: strict > (REL_GT), x0=10, xlo=5 (trip == 5 -> 4+1)
       * mode 2: strict > (REL_GT), x0=10, xlo=9 (trip == 1 -> scalar)
       * mode 3: inclusive >= (REL_GE), x0=10, xlo=8 (trip == 3 -> exact-3)
       * mode 4: inclusive >= (REL_GE), x0=INT32_MIN+2, xlo=INT32_MIN (wrap guard) */
      const bool inclusive = (mode >= 3);
      nir_builder b = make_compute_shader("small-window-decreasing");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *span = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xlo = nir_isub(&b, x0, span);
      nir_def *yhi = nir_iadd(&b, y0, span);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, y_var, y0, 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_def *exit_cond = inclusive ? nir_ilt(&b, x, xlo)
                                           : nir_ige(&b, xlo, x);
            nir_if *xex = nir_push_if(&b, exit_cond);
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var,
                          nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, -1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "small window decreasing");

      b.cursor = nir_before_impl(b.impl);
      if (mode == 0) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 3));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 1) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 5));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 2) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 1));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 3) {
         nir_def_replace(x0, nir_imm_int(&b, 10));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else {
         nir_def_replace(x0, nir_imm_int(&b, INT32_MIN + 2));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      }
      fold_after_specialize(b.shader, "small window decreasing specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      if (mode == 0 || mode == 3) {
         EXPECT_EQ(max_depth, 1u);
         EXPECT_GE(count_intrinsics(b.shader, nir_intrinsic_load_agx), 3u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx) % 3u, 0u);
      } else if (mode == 1) {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 5u);
      } else {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchUnsigned)
{
   for (unsigned mode = 0; mode < 6; ++mode) {
      /* mode 0: unsigned <= (REL_LE), x0=0x80000000u, span=2 (trip == 3 -> exact-3)
       * mode 1: unsigned <= (REL_LE), x0=0x80000000u, span=4 (trip == 5 -> 4+1)
       * mode 2: unsigned <= (REL_LE), x0=UINT32_MAX-2, span=2 (wrap guard)
       * mode 3: unsigned >= (REL_GE), x0=0x80000005u, span=2 (trip == 3 -> exact-3)
       * mode 4: unsigned >= (REL_GE), x0=2u, span=2 (bound==0 wrap guard)
       * mode 5: unsigned <= (REL_LE), x0=0x80000000u, span=0 (trip == 1 -> scalar) */
      const bool dec = (mode == 3 || mode == 4);
      nir_builder b = make_compute_shader("small-window-unsigned");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *out = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *span = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xbound = dec ? nir_isub(&b, x0, span) : nir_iadd(&b, x0, span);
      nir_def *yhi = nir_iadd(&b, y0, span);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_uint_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_uint_type(), "x");
      nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
      nir_store_var(&b, y_var, y0, 1);
      nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ult(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_def *exit_cond = dec ? nir_ult(&b, x, xbound)
                                     : nir_ult(&b, xbound, x);
            nir_if *xex = nir_push_if(&b, exit_cond);
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, acc_var,
                          nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, dec ? -1 : 1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);
      nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "small window unsigned");

      b.cursor = nir_before_impl(b.impl);
      if (mode == 0) {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000000u));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 1) {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000000u));
         nir_def_replace(span, nir_imm_int(&b, 4));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 2) {
         nir_def_replace(x0, nir_imm_int(&b, (int)(UINT32_MAX - 2u)));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 3) {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000005u));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else if (mode == 4) {
         nir_def_replace(x0, nir_imm_int(&b, 2));
         nir_def_replace(span, nir_imm_int(&b, 2));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      } else {
         nir_def_replace(x0, nir_imm_int(&b, (int)0x80000000u));
         nir_def_replace(span, nir_imm_int(&b, 0));
         nir_def_replace(y0, nir_imm_int(&b, 0));
      }
      fold_after_specialize(b.shader, "small window unsigned specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      if (mode == 0 || mode == 3) {
         EXPECT_EQ(max_depth, 1u);
         EXPECT_GE(count_intrinsics(b.shader, nir_intrinsic_load_agx), 3u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx) % 3u, 0u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_store_global), 1u);
      } else if (mode == 1) {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 5u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_store_global), 1u);
      } else if (mode == 5) {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_store_global), 1u);
      } else if (mode == 2) {
         /* bound == UINT32_MAX: exact-3 unsigned no-wrap guard rejects peeling. */
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 1u);
      } else {
         /* bound == 0u (REL_GE): exact-3 unsigned no-wrap guard sends to the
          * unchanged copy, where nir_opt_algebraic folds ult(x, 0) -> false. */
         EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_agx), 0u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, SmallWindowUnswitchExactThreeLiveOutIV)
{
   nir_builder b = make_compute_shader("small-window-live-out-iv");
   nir_def *base = make_desc_base(&b, 0);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *out_iv = make_desc_base(&b, 16);
   nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
   nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
   nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
   nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
   nir_def *xhi = nir_iadd(&b, x0, rad);
   nir_def *yhi = nir_iadd(&b, y0, rad);

   nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
   nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
   nir_variable *acc_var = nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_store_var(&b, y_var, y0, 1);
   nir_store_var(&b, acc_var, nir_imm_int(&b, 0), 1);

   nir_loop *outer = nir_push_loop(&b);
   {
      nir_def *y = nir_load_var(&b, y_var);
      nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, yex);

      nir_def *row_off = nir_imul(&b, y, stride);
      nir_store_var(&b, x_var, x0, 1);

      nir_loop *inner = nir_push_loop(&b);
      {
         nir_def *x = nir_load_var(&b, x_var);
         nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, xex);

         nir_def *idx = nir_iadd(&b, row_off, x);
         nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                   .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                   .format = PIPE_FORMAT_R8_UINT);
         nir_store_var(&b, acc_var,
                       nir_iadd(&b, nir_load_var(&b, acc_var), nir_u2u32(&b, v)), 1);
         nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
      }
      nir_pop_loop(&b, inner);

      /* Read the inner induction variable after the loop exits (should equal
       * x0 + 3 on the exact-3 path because of the final exiting header run). */
      nir_store_agx(&b, nir_load_var(&b, x_var), out_iv, y,
                    .access = ACCESS_NON_READABLE,
                    .format = PIPE_FORMAT_R32_UINT);
      nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
   }
   nir_pop_loop(&b, outer);
   nir_store_global(&b, nir_load_var(&b, acc_var), out, .align_mul = 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_dce);
   EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
   fold_after_specialize(b.shader, "small window live-out iv");

   b.cursor = nir_before_impl(b.impl);
   nir_def_replace(x0, nir_imm_int(&b, 10));
   nir_def_replace(rad, nir_imm_int(&b, 2));
   nir_def_replace(y0, nir_imm_int(&b, 0));
   fold_after_specialize(b.shader, "small window live-out iv specialized");

   unsigned loops = 0, max_depth = 0;
   count_loop_shape(b.impl, &loops, &max_depth);
   EXPECT_EQ(max_depth, 1u);

   unsigned iv_stores = 0;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
         if (intr->intrinsic != nir_intrinsic_store_agx)
            continue;
         ASSERT_TRUE(nir_src_is_const(intr->src[0]));
         EXPECT_EQ(nir_src_as_uint(intr->src[0]), 13u);
         iv_stores++;
      }
   }
   EXPECT_GE(iv_stores, 1u);
   ralloc_free(b.shader);
}

TEST(AGX3VisionOpts, SmallWindowUnswitchInnerStore)
{
   for (unsigned mode = 0; mode < 2; ++mode) {
      /* mode 0: rad=2 (trip == 3 -> exact 3 stores, exiting header runs no 4th store)
       * mode 1: rad=4 (trip == 5 -> 4 peeled stores + 1 in remainder loop) */
      nir_builder b = make_compute_shader("small-window-inner-store");
      nir_def *base = make_desc_base(&b, 0);
      nir_def *dst = make_desc_base(&b, 8);
      nir_def *x0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0), .range = 16);
      nir_def *y0 = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4), .range = 16);
      nir_def *rad = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 8), .range = 16);
      nir_def *stride = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 12), .range = 16);
      nir_def *xhi = nir_iadd(&b, x0, rad);
      nir_def *yhi = nir_iadd(&b, y0, rad);

      nir_variable *y_var = nir_local_variable_create(b.impl, glsl_int_type(), "y");
      nir_variable *x_var = nir_local_variable_create(b.impl, glsl_int_type(), "x");
      nir_store_var(&b, y_var, y0, 1);

      nir_loop *outer = nir_push_loop(&b);
      {
         nir_def *y = nir_load_var(&b, y_var);
         nir_if *yex = nir_push_if(&b, nir_ilt(&b, yhi, y));
         nir_jump(&b, nir_jump_break);
         nir_pop_if(&b, yex);

         nir_def *row_off = nir_imul(&b, y, stride);
         nir_store_var(&b, x_var, x0, 1);

         nir_loop *inner = nir_push_loop(&b);
         {
            nir_def *x = nir_load_var(&b, x_var);
            nir_if *xex = nir_push_if(&b, nir_ilt(&b, xhi, x));
            nir_jump(&b, nir_jump_break);
            nir_pop_if(&b, xex);

            nir_def *idx = nir_iadd(&b, row_off, x);
            nir_def *v = nir_load_agx(&b, 1, 8, base, idx,
                                      .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                                      .format = PIPE_FORMAT_R8_UINT);
            nir_store_agx(&b, v, dst, idx,
                          .access = ACCESS_NON_READABLE,
                          .format = PIPE_FORMAT_R8_UINT);
            nir_store_var(&b, x_var, nir_iadd_imm(&b, x, 1), 1);
         }
         nir_pop_loop(&b, inner);
         nir_store_var(&b, y_var, nir_iadd_imm(&b, y, 1), 1);
      }
      nir_pop_loop(&b, outer);

      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      EXPECT_TRUE(agx_nir_unroll_runtime_loops(b.shader, 8, 96, NULL));
      fold_after_specialize(b.shader, "small window inner store");

      b.cursor = nir_before_impl(b.impl);
      nir_def_replace(x0, nir_imm_int(&b, 10));
      nir_def_replace(rad, nir_imm_int(&b, mode == 0 ? 2 : 4));
      nir_def_replace(y0, nir_imm_int(&b, 0));
      fold_after_specialize(b.shader, "small window inner store specialized");

      unsigned loops = 0, max_depth = 0;
      count_loop_shape(b.impl, &loops, &max_depth);
      const unsigned n_loads = count_intrinsics(b.shader, nir_intrinsic_load_agx);
      const unsigned n_stores = count_intrinsics(b.shader, nir_intrinsic_store_agx);
      if (mode == 0) {
         EXPECT_EQ(max_depth, 1u);
         EXPECT_GE(n_stores, 3u);
         EXPECT_EQ(n_stores % 3u, 0u);
         EXPECT_EQ(n_stores, n_loads);
      } else {
         EXPECT_EQ(max_depth, 2u);
         EXPECT_EQ(loops, 2u);
         EXPECT_EQ(n_loads, 5u);
         EXPECT_EQ(n_stores, 5u);
      }
      ralloc_free(b.shader);
   }
}

TEST(AGX3VisionOpts, FoldW8WordUnpackThreeComponentsAGX3B)
{
   nir_builder b = make_compute_shader("fold-w8-word-unpack-3comp");
   nir_def *base = make_desc_base(&b, 0);
   nir_def *out = make_desc_base(&b, 8);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *v = nir_load_agx(&b, 3, 8, base, gid,
                             .access = (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER),
                             .format = PIPE_FORMAT_R8_UINT);
   nir_def *sum = nir_iadd(&b, nir_iadd(&b, nir_u2u32(&b, nir_channel(&b, v, 0)),
                                        nir_u2u32(&b, nir_channel(&b, v, 1))),
                           nir_u2u32(&b, nir_channel(&b, v, 2)));
   nir_store_agx(&b, sum, out, gid, .access = ACCESS_NON_READABLE,
                 .format = PIPE_FORMAT_R32_UINT);

   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17P;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.no_preamble = true;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_GT(part.info.main_size, 0u);
   EXPECT_EQ(part.info.scratch_size, 0u);
   EXPECT_EQ(part.info.stats.spills, 0u);
   free(part.binary);
   ralloc_free(b.shader);
}

static unsigned
compute_loads_before_first_use(enum agx_chip chip, unsigned extra_live_32,
                               bool allocate = false)
{
   void *mem = ralloc_context(NULL);
   struct agx_shader_key key = {};
   key.dev.encoding = agx_chip_isa_encoding(chip);
   key.dev.chip = chip;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   body->loop_header = true;
   agx_block_add_successor(entry, body);
   agx_block_add_successor(body, latch);
   agx_block_add_successor(latch, body);
   agx_block_add_successor(body, exit);

   agx_index base = agx_uniform(4, AGX_SIZE_64);
   agx_index off = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));
   agx_index held[AGX3B_NUM_REGS / 2];
   assert(extra_live_32 <= ARRAY_SIZE(held));
   for (unsigned i = 0; i < extra_live_32; ++i)
      held[i] = agx_mov_imm(b, 32, 100 + i);

   /* Emit 8 independent device_loads in pressure order (load, immediate use). */
   b->cursor = agx_after_block(body);
   for (unsigned i = 0; i < 8; ++i) {
      agx_index addr_off = agx_iadd(b, off, agx_immediate(i * 4), 0);
      agx_index val = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *ld = agx_device_load_to(
         b, val, base, addr_off, AGX_FORMAT_I32, 0x1, 0, 0);
      ld->memory_can_reorder = true;
      acc = agx_fadd(b, acc, val);
   }
   for (unsigned i = 0; i < extra_live_32; ++i)
      acc = agx_iadd(b, acc, held[i], 0);
   agx_device_store(b, acc, agx_uniform(8, AGX_SIZE_64), off, AGX_FORMAT_I32,
                    0x1, 0, 0);

   agx_ilp_schedule(b->shader);

   if (allocate) {
      agx_ra(b->shader);
      EXPECT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(b->shader->scratch_size_B, 0u);
      EXPECT_EQ(agx_occupancy_for_register_count(b->shader->max_reg + 1)
                   .max_threads, 1024u);
   }
   unsigned loads_before_first_fadd = 0;
   agx_foreach_instr_in_block(body, I) {
      if (I->op == AGX_OPCODE_FADD)
         break;
      if (I->op == AGX_OPCODE_DEVICE_LOAD) {
         if (allocate)
            EXPECT_EQ(I->dest[0].type, AGX_INDEX_REGISTER);
         ++loads_before_first_fadd;
      }
   }
   ralloc_free(mem);
   return loads_before_first_fadd;
}

TEST(AGX3VisionOpts, Agx3bComputeDeviceLoadIlpWideWindow)
{
   EXPECT_EQ(compute_loads_before_first_use(AGX_CHIP_G17P, 0), 8u);
   EXPECT_EQ(compute_loads_before_first_use(AGX_CHIP_G17P, AGX3B_NUM_REGS / 2),
             1u);
   const unsigned m5_loads = compute_loads_before_first_use(AGX_CHIP_G17S, 0);
   EXPECT_GE(m5_loads, 2u);
   EXPECT_LE(m5_loads, 4u);
}

TEST(AGX3VisionOpts, Agx3bDeviceLoadWindowAfterRegisterAllocation)
{
   /* Holding one more 32-bit value across the same eight loads leaves room
    * for six ahead of the first use. Check both allocated schedules, including
    * their scratch usage and register-count tier. */
   EXPECT_EQ(compute_loads_before_first_use(AGX_CHIP_G16G, 38, true), 8u);
   EXPECT_EQ(compute_loads_before_first_use(AGX_CHIP_G16G, 39, true), 6u);
}
