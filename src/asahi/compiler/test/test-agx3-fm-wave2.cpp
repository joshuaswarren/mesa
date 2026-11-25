/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include <vector>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compiler.h"
#include "agx_nir.h"
}

namespace {

nir_builder
make_compute_shader(const char *name)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   return b;
}

nir_def *
load_preamble_ex(nir_builder *b, unsigned comps, unsigned bits, unsigned base)
{
   nir_def *d = nir_load_preamble(b, comps, bits);
   nir_intrinsic_set_base(nir_def_as_intrinsic(d), base);
   return d;
}

nir_def *
load_push_u32(nir_builder *b, unsigned offset, unsigned range = 32)
{
   nir_def *d = nir_load_push_constant(b, 1, 32, nir_imm_int(b, offset));
   nir_intrinsic_set_range(nir_def_as_intrinsic(d), range);
   return d;
}

nir_def *
load_global_ex(nir_builder *b, unsigned comps, unsigned bits, nir_def *addr,
               unsigned align_mul,
               gl_access_qualifier access = (gl_access_qualifier)0)
{
   nir_def *d = nir_load_global(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(d);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return d;
}

nir_def *
load_global_const_ex(nir_builder *b, unsigned comps, unsigned bits,
                     nir_def *addr, unsigned align_mul,
                     gl_access_qualifier access)
{
   nir_def *d = nir_load_global_constant(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(d);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return d;
}

void
store_global_ex(nir_builder *b, nir_def *val, nir_def *addr, unsigned align_mul)
{
   nir_intrinsic_instr *intr = nir_store_global(b, val, addr);
   nir_intrinsic_set_align(intr, align_mul, 0);
}

void
emit_workgroup_barrier(nir_builder *b)
{
   nir_intrinsic_instr *intr = nir_barrier(b);
   nir_intrinsic_set_execution_scope(intr, SCOPE_WORKGROUP);
}

nir_def *
make_buffer_addr(nir_builder *b, unsigned preamble_base, nir_def *byte_offset)
{
   nir_def *set = load_preamble_ex(b, 1, 64, preamble_base);
   nir_def *desc = load_global_const_ex(
      b, 4, 32, set, 16,
      (gl_access_qualifier)(ACCESS_CAN_SPECULATE | ACCESS_BUFFER_BINDING_AGX));
   nir_def *base = nir_pack_64_2x32_split(b, nir_channel(b, desc, 0),
                                          nir_channel(b, desc, 1));
   return nir_iadd(b, base, nir_u2u64(b, byte_offset));
}

unsigned
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

unsigned
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

unsigned
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

/* Helper building a counted [0, trips) loop with >= 20 body instructions,
 * global loads, and an optional monotonic 32-bit bit-pack accumulator. */
nir_shader *
build_const_bitpack_loop(unsigned trips, bool bottom_tested, bool with_load,
                         bool with_barrier, bool read_d_in_loop)
{
   nir_builder b = make_compute_shader("const-bitpack-loop");
   nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(), "iv");
   nir_variable *acc =
      nir_local_variable_create(b.impl, glsl_uint_type(), "acc");
   nir_variable *extra =
      nir_local_variable_create(b.impl, glsl_uint_type(), "extra");

   nir_def *c0 = load_push_u32(&b, 0, 32);
   nir_def *c1 = load_push_u32(&b, 4, 32);
   nir_def *stride = load_push_u32(&b, 8, 32);

   nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, acc, nir_imm_int(&b, 0), 1);
   nir_store_var(&b, extra, nir_imm_int(&b, 0), 1);

   const gl_access_qualifier ro =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);

   nir_loop *loop = nir_push_loop(&b);
   {
      nir_def *i = nir_load_var(&b, iv);
      if (!bottom_tested)
         nir_break_if(&b, nir_ige(&b, i, nir_imm_int(&b, trips)));

      nir_def *a = NULL;
      nir_def *bv = NULL;
      if (with_load) {
         nir_def *pat = load_global_ex(
            &b, 4, 32, make_buffer_addr(&b, 4, nir_ishl_imm(&b, i, 4)), 16, ro);
         nir_def *x0 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 0), c0),
                                nir_imul(&b, nir_channel(&b, pat, 1), c1));
         nir_def *y0 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 0), c1),
                                nir_imul(&b, nir_channel(&b, pat, 1), c0));
         nir_def *x1 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 2), c0),
                                nir_imul(&b, nir_channel(&b, pat, 3), c1));
         nir_def *y1 = nir_iadd(&b, nir_imul(&b, nir_channel(&b, pat, 2), c1),
                                nir_imul(&b, nir_channel(&b, pat, 3), c0));
         nir_def *o0 = nir_iadd(&b, x0, nir_imul(&b, y0, stride));
         nir_def *o1 = nir_iadd(&b, x1, nir_imul(&b, y1, stride));
         a = load_global_ex(
            &b, 1, 32, make_buffer_addr(&b, 8, nir_ishl_imm(&b, o0, 2)), 4, ro);
         bv = load_global_ex(
            &b, 1, 32, make_buffer_addr(&b, 8, nir_ishl_imm(&b, o1, 2)), 4, ro);
      } else {
         nir_def *v = nir_iadd(&b, i, c0);
         for (unsigned k = 0; k < 20; ++k)
            v = nir_ixor(&b, nir_iadd(&b, v, c1), nir_imul(&b, v, stride));
         a = v;
         bv = c1;
      }

      if (with_barrier)
         emit_workgroup_barrier(&b);

      nir_def *d = nir_load_var(&b, acc);
      if (read_d_in_loop)
         nir_store_var(&b, extra, nir_iadd(&b, nir_load_var(&b, extra), d), 1);

      nir_def *mask =
         nir_ishl(&b, nir_imm_int(&b, 1), nir_iand_imm(&b, i, 31));
      nir_def *d_next =
         nir_bcsel(&b, nir_ilt(&b, a, bv), nir_ior(&b, d, mask), d);
      nir_store_var(&b, acc, d_next, 1);

      nir_def *next = nir_iadd_imm(&b, i, 1);
      nir_store_var(&b, iv, next, 1);
      if (bottom_tested)
         nir_break_if(&b, nir_ige(&b, next, nir_imm_int(&b, trips)));
   }
   nir_pop_loop(&b, loop);

   nir_def *res = nir_iadd(&b, nir_load_var(&b, acc), nir_load_var(&b, extra));
   store_global_ex(&b, res, make_buffer_addr(&b, 12, nir_imm_int(&b, 0)), 4);

   NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
   NIR_PASS(_, b.shader, nir_opt_copy_prop);
   NIR_PASS(_, b.shader, nir_opt_cse);
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "const-bitpack setup");
   return b.shader;
}

} /* namespace */

/* Bottom-tested rotated loop (do { ... ++iv; } while (iv < n)) must lower
 * idiv/imod and udiv/umod into a running (q, r) recurrence just like a
 * top-tested loop, while refusing step != 1. */
TEST(FmWave2IvDivMod, BottomTestedLoop)
{
   for (int mode = 0; mode < 4; ++mode) {
      /* mode 0: signed guarded D > 0, step 1
       * mode 1: signed unknown-sign D, step 1
       * mode 2: unsigned udiv/umod, step 1
       * mode 3: signed step 2 (must not match) */
      nir_builder b = make_compute_shader("iv-divmod-bottom");
      nir_def *out = make_buffer_addr(&b, 8, nir_imm_int(&b, 0));
      nir_def *n = load_push_u32(&b, 0, 16);
      nir_def *d = load_push_u32(&b, 4, 16);

      nir_if *guard = NULL;
      if (mode == 0)
         guard = nir_push_if(&b, nir_ilt(&b, nir_imm_int(&b, 0), d));

      nir_def *iv_init = nir_imm_int(&b, 0);
      nir_def *acc_init = nir_imm_int(&b, 0);

      nir_loop *loop = nir_push_loop(&b);
      nir_block *pre = nir_loop_predecessor_block(loop);
      nir_block *header = nir_loop_first_block(loop);

      nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
      nir_phi_instr_add_src(iv_phi, pre, iv_init);

      nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
      nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
      nir_phi_instr_add_src(acc_phi, pre, acc_init);

      nir_def *q = mode == 2 ? nir_udiv(&b, &iv_phi->def, d)
                             : nir_idiv(&b, &iv_phi->def, d);
      nir_def *r = mode == 2 ? nir_umod(&b, &iv_phi->def, d)
                             : nir_imod(&b, &iv_phi->def, d);
      nir_def *sum = nir_iadd(&b, &acc_phi->def, nir_iadd(&b, q, r));
      nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, mode == 3 ? 2 : 1);

      nir_def *done =
         mode == 2 ? nir_uge(&b, iv_next, n) : nir_ige(&b, iv_next, n);
      nir_push_if(&b, done);
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      nir_block *latch = nir_loop_last_block(loop);
      nir_phi_instr_add_src(iv_phi, latch, iv_next);
      nir_phi_instr_add_src(acc_phi, latch, sum);
      nir_instr_insert(nir_before_block(header), &iv_phi->instr);
      nir_instr_insert(nir_before_block(header), &acc_phi->instr);
      nir_pop_loop(&b, loop);

      store_global_ex(&b, sum, out, 4);
      if (guard)
         nir_pop_if(&b, guard);
      nir_validate_shader(b.shader, "iv-divmod-bottom setup");

      bool changed = agx_nir_opt_iv_divmod(b.shader);
      nir_validate_shader(b.shader, "iv-divmod-bottom lowered");
      if (mode < 3) {
         EXPECT_TRUE(changed) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_udiv), 0u) << "mode " << mode;
         EXPECT_EQ(count_alu_op(b.shader, nir_op_umod), 0u) << "mode " << mode;
         if (mode == 0)
            EXPECT_EQ(count_alu_op(b.shader, nir_op_ineg), 0u);
      } else {
         EXPECT_FALSE(changed);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 1u);
         EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 1u);
      }
      ralloc_free(b.shader);
   }
}

/* Push-constant divisor and bound (`bs * bs`) loaded inside the loop (before
 * LICM) must still match, hoist the divisor load to the preheader, and share
 * a single (q, r) recurrence across idiv and imod. */
TEST(FmWave2IvDivMod, InLoopPushConstantDivisorAndBound)
{
   nir_builder b = make_compute_shader("iv-divmod-inloop-pc");
   nir_def *out = make_buffer_addr(&b, 8, nir_imm_int(&b, 0));
   nir_def *zero = nir_imm_int(&b, 0);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *pre = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);

   nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
   nir_phi_instr_add_src(iv_phi, pre, zero);

   nir_phi_instr *acc_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&acc_phi->instr, &acc_phi->def, 1, 32);
   nir_phi_instr_add_src(acc_phi, pre, zero);

   nir_def *bs0 = load_push_u32(&b, 20, 32);
   nir_def *bs1 = load_push_u32(&b, 20, 32);
   nir_def *bound = nir_imul(&b, bs0, bs1);
   nir_push_if(&b, nir_ige(&b, &iv_phi->def, bound));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *d_q = load_push_u32(&b, 20, 32);
   nir_def *d_r = load_push_u32(&b, 20, 32);
   nir_def *q = nir_idiv(&b, &iv_phi->def, d_q);
   nir_def *r = nir_imod(&b, &iv_phi->def, d_r);
   nir_def *sum = nir_iadd(&b, &acc_phi->def, nir_iadd(&b, q, r));
   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 1);

   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_instr_add_src(iv_phi, latch, iv_next);
   nir_phi_instr_add_src(acc_phi, latch, sum);
   nir_instr_insert(nir_before_block(header), &iv_phi->instr);
   nir_instr_insert(nir_before_block(header), &acc_phi->instr);
   nir_pop_loop(&b, loop);

   store_global_ex(&b, &acc_phi->def, out, 4);
   nir_validate_shader(b.shader, "iv-divmod-inloop-pc setup");

   EXPECT_TRUE(agx_nir_opt_iv_divmod(b.shader));
   nir_validate_shader(b.shader, "iv-divmod-inloop-pc lowered");
   EXPECT_EQ(count_alu_op(b.shader, nir_op_idiv), 0u);
   EXPECT_EQ(count_alu_op(b.shader, nir_op_imod), 0u);
   ralloc_free(b.shader);
}

TEST(FmWave2ConstLoopUnroll, TopTested32TripBitpack8x)
{
   nir_shader *s = build_const_bitpack_loop(32, false, true, false, false);
   ASSERT_EQ(count_loops(s), 1u);
   unsigned widest = 0;
   EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "const-bitpack 8x");

   /* Exact divisor: still one loop (no remainder loop) and widest == 8. */
   EXPECT_EQ(count_loops(s), 1u);
   EXPECT_EQ(widest, 8u);
   /* Bit-pack funnel-shift strength reduction: one ushr by 8 and one ishl by
    * 24 for the chunk merge; no variable 1u << (iv + k) shifts remain. */
   EXPECT_EQ(count_alu_op(s, nir_op_ushr), 1u);
   /* Running again is a no-op (partially_unrolled is set). */
   EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
   ralloc_free(s);
}

TEST(FmWave2ConstLoopUnroll, BottomTested32TripBitpack4x)
{
   nir_shader *s = build_const_bitpack_loop(32, true, true, false, false);
   ASSERT_EQ(count_loops(s), 1u);
   unsigned widest = 0;
   EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, 4, &widest));
   NIR_PASS(_, s, nir_opt_copy_prop);
   NIR_PASS(_, s, nir_opt_dce);
   nir_validate_shader(s, "const-bitpack bottom 4x");

   EXPECT_EQ(count_loops(s), 1u);
   EXPECT_EQ(widest, 4u);
   EXPECT_EQ(count_alu_op(s, nir_op_ushr), 1u);
   ralloc_free(s);
}

TEST(FmWave2ConstLoopUnroll, RefusalsAndEscapingAccumulator)
{
   /* Non-divisible trip count (30 is divisible by neither 8 nor 4). */
   {
      nir_shader *s = build_const_bitpack_loop(30, false, true, false, false);
      unsigned widest = 0;
      EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
      EXPECT_EQ(widest, 0u);
      ralloc_free(s);
   }

   /* Pure-ALU loop without memory loads is left alone. */
   {
      nir_shader *s = build_const_bitpack_loop(32, false, false, false, false);
      EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, NULL));
      ralloc_free(s);
   }

   /* Loop with a workgroup barrier is refused. */
   {
      nir_shader *s = build_const_bitpack_loop(32, false, true, true, false);
      EXPECT_FALSE(agx_nir_opt_const_loop_unroll(s, 8, NULL));
      ralloc_free(s);
   }

   /* When `d` is read inside the loop outside the bit-pack chain, unrolling
    * still succeeds, but bit-pack funnel-shift strength reduction is skipped
    * so intermediate values of `d` stay exact. */
   {
      nir_shader *s = build_const_bitpack_loop(32, false, true, false, true);
      unsigned widest = 0;
      EXPECT_TRUE(agx_nir_opt_const_loop_unroll(s, 8, &widest));
      NIR_PASS(_, s, nir_opt_dce);
      nir_validate_shader(s, "const-bitpack escaping d");
      EXPECT_EQ(widest, 8u);
      EXPECT_EQ(count_alu_op(s, nir_op_ushr), 0u);
      ralloc_free(s);
   }
}

/* Five additive terms in a byte-load address (as in a 2D neighborhood loop
 * with border + row*width + col + k) must parse in agx_nir_linear and merge
 * into one 4-byte vector load. */
TEST(FmWave2MergeBytes, FiveTermAffineAddress)
{
   nir_builder b = make_compute_shader("merge-5term-u8");
   nir_def *t0 = load_push_u32(&b, 0, 32);
   nir_def *t1 = load_push_u32(&b, 4, 32);
   nir_def *t2 = load_push_u32(&b, 8, 32);
   nir_def *t3 = load_push_u32(&b, 12, 32);
   nir_def *t4 = load_push_u32(&b, 16, 32);

   nir_def *base_off =
      nir_iadd(&b, nir_iadd(&b, nir_iadd(&b, t0, t1), nir_iadd(&b, t2, t3)),
               t4);
   const gl_access_qualifier ro =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
   nir_def *sum = nir_imm_int(&b, 0);
   for (unsigned k = 0; k < 4; ++k) {
      nir_def *v = load_global_ex(
         &b, 1, 8, make_buffer_addr(&b, 4, nir_iadd_imm(&b, base_off, k)), 1,
         ro);
      sum = nir_iadd(&b, sum, nir_u2u32(&b, v));
   }
   store_global_ex(&b, sum, make_buffer_addr(&b, 8, nir_imm_int(&b, 0)), 4);

   NIR_PASS(_, b.shader, nir_opt_cse);
   EXPECT_TRUE(agx_nir_merge_global_words(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "merge 5-term u8");
   EXPECT_EQ(count_intrinsics(b.shader, nir_intrinsic_load_global), 1u);
   ralloc_free(b.shader);
}

/* In a two-level indirect load chain (p0 -> addr0 -> img0; p1 -> addr1 -> img1)
 * inside a compute block, the ILP scheduler must issue independent outer load
 * p1 ahead of addr0 (which stalls on p0) instead of serializing by source
 * order. */
TEST(FmWave2IlpSchedule, TwoLevelIndirectLoadInterleave)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *blk = agx_start_block(b->shader);
   b->cursor = agx_after_block(blk);

   agx_index base0 = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index base1 = agx_preload(b, agx_register(6, AGX_SIZE_64));

   agx_index p0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load_p0 = agx_device_load_to(
      b, p0, base0, agx_immediate(0), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index a0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *alu_a0 = agx_iadd_to(b, a0, p0, agx_immediate(1), 0);
   agx_index v0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v0, base1, a0, AGX_FORMAT_I32, 0x1, 0, false);

   agx_index p1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load_p1 = agx_device_load_to(
      b, p1, base0, agx_immediate(4), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index a1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, a1, p1, agx_immediate(1), 0);
   agx_index v1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v1, base1, a1, AGX_FORMAT_I32, 0x1, 0, false);

   agx_unit_test(b, agx_iadd(b, v0, v1, 0));

   agx_ilp_schedule(b->shader);

   int pos_p0 = -1, pos_p1 = -1, pos_a0 = -1, idx = 0;
   agx_foreach_instr_in_block(blk, I) {
      if (I == load_p0)
         pos_p0 = idx;
      else if (I == load_p1)
         pos_p1 = idx;
      else if (I == alu_a0)
         pos_a0 = idx;
      idx++;
   }
   ASSERT_GE(pos_p0, 0);
   ASSERT_GE(pos_p1, 0);
   ASSERT_GE(pos_a0, 0);
   EXPECT_LT(pos_p1, pos_a0);
   ralloc_free(mem);
}

/* A device store between two indirect load chains must still serialize later
 * loads after the store (preserving RAW/WAR ordering). */
TEST(FmWave2IlpSchedule, StoreStillSerializesLaterLoads)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *blk = agx_start_block(b->shader);
   b->cursor = agx_after_block(blk);

   agx_index base0 = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index base1 = agx_preload(b, agx_register(6, AGX_SIZE_64));

   agx_index p0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, p0, base0, agx_immediate(0), AGX_FORMAT_I32, 0x1, 0,
                      false);
   agx_index a0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, a0, p0, agx_immediate(1), 0);
   agx_index v0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v0, base1, a0, AGX_FORMAT_I32, 0x1, 0, false);

   agx_instr *st = agx_device_store(b, v0, base0, agx_immediate(4),
                                    AGX_FORMAT_I32, 0x1, 0, false);

   agx_index p1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load_p1 = agx_device_load_to(
      b, p1, base0, agx_immediate(4), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index a1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_iadd_to(b, a1, p1, agx_immediate(1), 0);
   agx_index v1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v1, base1, a1, AGX_FORMAT_I32, 0x1, 0, false);

   agx_unit_test(b, agx_iadd(b, v0, v1, 0));

   agx_ilp_schedule(b->shader);

   int pos_st = -1, pos_p1 = -1, idx = 0;
   agx_foreach_instr_in_block(blk, I) {
      if (I == st)
         pos_st = idx;
      else if (I == load_p1)
         pos_p1 = idx;
      idx++;
   }
   ASSERT_GE(pos_st, 0);
   ASSERT_GE(pos_p1, 0);
   EXPECT_GT(pos_p1, pos_st);
   ralloc_free(mem);
}

/* A compute loop block with two device loads (such as an 8x-unrolled byte loop
 * vectorized into two u8x4 loads) must hoist the second load ahead of the
 * first load's ALU consumers. */
TEST(FmWave2IlpSchedule, TwoLoadComputeLoopHoists)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.has_scratch = true;

   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = rzalloc(mem, agx_shader_info);
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   body->loop_header = true;
   agx_block_add_successor(entry, body);
   agx_block_add_successor(body, body);
   agx_block_add_successor(body, exit);

   b->cursor = agx_after_block(entry);
   agx_index base = agx_preload(b, agx_register(2, AGX_SIZE_64));
   agx_index acc = agx_preload(b, agx_register(6, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index v0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v0, base, agx_immediate(0), AGX_FORMAT_I32, 0x1, 0,
                      false);
   agx_index s0 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *add0 = agx_iadd_to(b, s0, acc, v0, 0);

   agx_index v1 = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load1 = agx_device_load_to(
      b, v1, base, agx_immediate(4), AGX_FORMAT_I32, 0x1, 0, false);
   agx_index s1 = agx_iadd(b, s0, v1, 0);
   agx_unit_test(b, s1);

   agx_ilp_schedule(b->shader);

   int pos_add0 = -1, pos_load1 = -1, idx = 0;
   agx_foreach_instr_in_block(body, I) {
      if (I == add0)
         pos_add0 = idx;
      else if (I == load1)
         pos_load1 = idx;
      idx++;
   }
   ASSERT_GE(pos_add0, 0);
   ASSERT_GE(pos_load1, 0);
   EXPECT_LT(pos_load1, pos_add0);
   ralloc_free(mem);
}

/* An 8-step unrolled Abel prefix-sum chain (`w += s; s += v_k`) collapses into
 * a balanced tree with `s_phi << 3` and loop-carried depth 1-2. */
TEST(FmWave2AbelChain, BalancedTree8x)
{
   nir_builder b = make_compute_shader("abel-tree-8x");
   nir_def *n = load_push_u32(&b, 0, 16);
   nir_def *zero = nir_imm_int(&b, 0);
   const gl_access_qualifier ro =
      (gl_access_qualifier)(ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);

   nir_loop *loop = nir_push_loop(&b);
   nir_block *pre = nir_loop_predecessor_block(loop);
   nir_block *header = nir_loop_first_block(loop);

   nir_phi_instr *iv_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&iv_phi->instr, &iv_phi->def, 1, 32);
   nir_phi_instr_add_src(iv_phi, pre, zero);

   nir_phi_instr *w_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&w_phi->instr, &w_phi->def, 1, 32);
   nir_phi_instr_add_src(w_phi, pre, zero);

   nir_phi_instr *s_phi = nir_phi_instr_create(b.shader);
   nir_def_init(&s_phi->instr, &s_phi->def, 1, 32);
   nir_phi_instr_add_src(s_phi, pre, zero);

   nir_push_if(&b, nir_ige(&b, &iv_phi->def, n));
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *w = &w_phi->def;
   nir_def *s = &s_phi->def;
   for (unsigned k = 0; k < 8; ++k) {
      nir_def *v = nir_u2u32(
         &b, load_global_ex(
                &b, 1, 8, make_buffer_addr(&b, 4, nir_iadd_imm(&b, &iv_phi->def, k)),
                1, ro));
      w = nir_iadd(&b, w, s);
      s = nir_iadd(&b, s, v);
   }
   nir_def *iv_next = nir_iadd_imm(&b, &iv_phi->def, 8);

   nir_block *latch = nir_loop_last_block(loop);
   nir_phi_instr_add_src(iv_phi, latch, iv_next);
   nir_phi_instr_add_src(w_phi, latch, w);
   nir_phi_instr_add_src(s_phi, latch, s);
   nir_instr_insert(nir_before_block(header), &iv_phi->instr);
   nir_instr_insert(nir_before_block(header), &w_phi->instr);
   nir_instr_insert(nir_before_block(header), &s_phi->instr);
   nir_pop_loop(&b, loop);

   store_global_ex(&b, nir_iadd(&b, &w_phi->def, &s_phi->def),
                   make_buffer_addr(&b, 8, zero), 4);
   nir_validate_shader(b.shader, "abel-tree-8x setup");

   EXPECT_TRUE(agx_nir_opt_factor_loop_mac(b.shader));
   NIR_PASS(_, b.shader, nir_opt_dce);
   nir_validate_shader(b.shader, "abel-tree-8x lowered");
   /* Emits 4 shifts: a01<<1, b45<<1, sa<<2, s_phi<<3. */
   EXPECT_GE(count_alu_op(b.shader, nir_op_ishl), 4u);
   ralloc_free(b.shader);
}
