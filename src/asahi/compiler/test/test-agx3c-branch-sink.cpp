/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compile.h"
#include "agx_nir.h"
}

namespace {

static struct agx_shader_key
agx3c_key(bool emit_agx3d = false)
{
   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.emit_agx3d = emit_agx3d;
   key.reserved_preamble = 16;
   return key;
}

static struct agx_shader_key
agx3b_key(enum agx_chip chip)
{
   struct agx_shader_key key = {};
   key.dev.chip = chip;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.reserved_preamble = 16;
   return key;
}

static nir_def *
make_load_global(nir_builder *b, unsigned comps, unsigned bits, nir_def *addr,
                 unsigned align_mul, gl_access_qualifier access)
{
   nir_def *def = nir_load_global(b, comps, bits, addr);
   nir_intrinsic_instr *intr = nir_def_as_intrinsic(def);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_access(intr, access);
   return def;
}

static void
make_store_global(nir_builder *b, nir_def *val, nir_def *addr,
                  unsigned align_mul)
{
   nir_intrinsic_instr *intr = nir_store_global(b, val, addr);
   nir_intrinsic_set_align(intr, align_mul, 0);
   nir_intrinsic_set_write_mask(intr, nir_component_mask(val->num_components));
}

static unsigned
count_ifs(const nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_cf_node *next = nir_cf_node_next(&block->cf_node);
         if (next && next->type == nir_cf_node_if)
            n++;
      }
   }
   return n;
}

static bool
instr_inside_if(const nir_instr *instr)
{
   for (nir_cf_node *p = instr->block->cf_node.parent; p; p = p->parent) {
      if (p->type == nir_cf_node_if)
         return true;
   }
   return false;
}

static bool
has_op(const nir_shader *nir, nir_op op)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               return true;
         }
      }
   }
   return false;
}

static bool
has_op_inside_if(const nir_shader *nir, nir_op op)
{
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op && instr_inside_if(instr))
               return true;
         }
      }
   }
   return false;
}

static unsigned
count_load_agx_inside_if(const nir_shader *nir)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic &&
                nir_instr_as_intrinsic(instr)->intrinsic ==
                   nir_intrinsic_load_agx &&
                instr_inside_if(instr))
               n++;
         }
      }
   }
   return n;
}

static nir_shader *
build_frsq_branch_shader(unsigned extra_alu_ops)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "frsq-branch");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *in_addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *out_addr = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *off = nir_u2u64(&b, nir_ishl_imm(&b, tid, 2));

   nir_def *x = make_load_global(&b, 1, 32, nir_iadd(&b, in_addr, off), 4,
                                 (gl_access_qualifier)0);
   nir_def *cond = nir_flt(&b, x, nir_imm_float(&b, 1.0f));

   nir_push_if(&b, cond);
   nir_def *acc = nir_frsq(&b, x);
   for (unsigned i = 0; i < extra_alu_ops; ++i) {
      /* Alternate fmax/fmul so neither constant folding nor fmul+fadd fusion
       * collapses pairs before nir_opt_peephole_select counts them. */
      nir_def *c = nir_imm_float(&b, 1.0f + 0.01f * (float)(i + 1));
      acc = (i & 1) ? nir_fmax(&b, acc, c) : nir_fmul(&b, acc, c);
   }
   nir_pop_if(&b, NULL);

   nir_def *res = nir_if_phi(&b, acc, x);
   make_store_global(&b, res, nir_iadd(&b, out_addr, off), 4);
   return b.shader;
}

TEST(Agx3cBranchSink, ExpensiveLimitPreservesLargeFrsqBranch)
{
   /* 26 non-movelike ops + frsq exceeds expensive_limit (24) while staying well
    * under limit (64): must stay a branch on AGX3C. */
   nir_shader *large = build_frsq_branch_shader(26);
   agx_preprocess_nir_m5(large);
   EXPECT_GE(count_ifs(large), 1u);
   EXPECT_TRUE(has_op_inside_if(large, nir_op_frsq));

   struct agx_shader_key key = agx3c_key();
   struct agx_shader_part out = {};
   ASSERT_TRUE(agx_compile_shader_nir(large, &key, &out));
   EXPECT_GE(count_ifs(large), 1u);
   EXPECT_TRUE(has_op_inside_if(large, nir_op_frsq));
   free(out.binary);
   ralloc_free(large);

   /* Short arm (4 ops + frsq <= 24) still flattens cleanly. */
   nir_shader *small = build_frsq_branch_shader(4);
   agx_preprocess_nir_m5(small);
   EXPECT_EQ(count_ifs(small), 0u);
   ralloc_free(small);
}

TEST(Agx3cBranchSink, SinkConditionalLoadsMovesNonReorderableLoadIntoIf)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "sink-cond-load");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
   nir_def *off = nir_u2u64(&b, nir_ishl_imm(&b, tid, 3));

   nir_def *pos = make_load_global(&b, 2, 32, nir_iadd(&b, pos_base, off), 8,
                                   (gl_access_qualifier)0);
   nir_def *vel = make_load_global(&b, 2, 32, nir_iadd(&b, vel_base, off), 8,
                                   (gl_access_qualifier)0);

   nir_def *dx = nir_channel(&b, pos, 0);
   nir_def *dy = nir_channel(&b, pos, 1);
   nir_def *dist2 = nir_fadd(&b, nir_fmul(&b, dx, dx), nir_fmul(&b, dy, dy));
   nir_def *cond = nir_flt(&b, dist2, nir_imm_float(&b, 4.0f));
   nir_def *zero = nir_imm_float(&b, 0.0f);

   nir_push_if(&b, cond);
   nir_def *inv_len = nir_frsq(&b, dist2);
   nir_def *scaled = nir_fmul(&b, nir_channel(&b, vel, 0), inv_len);
   nir_pop_if(&b, NULL);

   nir_def *res = nir_if_phi(&b, scaled, zero);
   make_store_global(&b, res, nir_iadd(&b, out_base, off), 4);

   EXPECT_TRUE(agx_nir_opt_sink_conditional_loads(b.shader));
   nir_validate_shader(b.shader, "after sink conditional loads");

   EXPECT_FALSE(instr_inside_if(nir_def_instr(pos)));
   EXPECT_TRUE(instr_inside_if(nir_def_instr(vel)));
   ralloc_free(b.shader);
}

TEST(Agx3cBranchSink, SinkConditionalLoadsRespectsHazardsAndLoops)
{
   auto make_base = [&](bool store_before_if, bool barrier_before_if,
                        gl_access_qualifier access, bool use_in_loop,
                        bool same_level_merge) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "sink-neg");
      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *in_base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *off = nir_u2u64(&b, nir_ishl_imm(&b, tid, 2));
      nir_def *addr = nir_iadd(&b, in_base, off);
      nir_def *val = make_load_global(&b, 1, 32, addr, 4, access);

      if (store_before_if)
         make_store_global(&b, nir_imm_int(&b, 1), nir_iadd(&b, out_base, off), 4);
      if (barrier_before_if) {
         nir_intrinsic_instr *bar = nir_barrier(&b);
         nir_intrinsic_set_execution_scope(bar, SCOPE_WORKGROUP);
         nir_intrinsic_set_memory_scope(bar, SCOPE_WORKGROUP);
         nir_intrinsic_set_memory_semantics(bar, NIR_MEMORY_ACQ_REL);
         nir_intrinsic_set_memory_modes(bar, nir_var_mem_ssbo);
      }

      if (use_in_loop) {
         nir_loop *lp = nir_push_loop(&b);
         make_store_global(&b, val, nir_iadd(&b, out_base, off), 4);
         nir_jump(&b, nir_jump_break);
         nir_pop_loop(&b, lp);
      } else if (same_level_merge) {
         nir_push_if(&b, nir_ine_imm(&b, tid, 0));
         nir_pop_if(&b, NULL);
         make_store_global(&b, val, nir_iadd(&b, out_base, off), 4);
      } else {
         nir_push_if(&b, nir_ine_imm(&b, tid, 0));
         make_store_global(&b, val, nir_iadd(&b, out_base, off), 4);
         nir_pop_if(&b, NULL);
      }

      EXPECT_FALSE(agx_nir_opt_sink_conditional_loads(b.shader));
      nir_validate_shader(b.shader, "negative control");
      ralloc_free(b.shader);
   };

   make_base(true, false, (gl_access_qualifier)0, false, false);
   make_base(false, true, (gl_access_qualifier)0, false, false);
   make_base(false, false, ACCESS_VOLATILE, false, false);
   make_base(false, false, ACCESS_COHERENT, false, false);
   make_base(false, false, (gl_access_qualifier)0, true, false);
   make_base(false, false, (gl_access_qualifier)0, false, true);
}

TEST(Agx3cBranchSink, EndToEndParticleShapePreservesIfAndSinksLoad)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "particle-shape");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
   nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
   nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

   nir_def *off = nir_u2u64(&b, nir_ishl_imm(&b, tid, 3));
   nir_def *pos = make_load_global(&b, 2, 32, nir_iadd(&b, pos_base, off), 8,
                                   (gl_access_qualifier)0);
   nir_def *vel = make_load_global(&b, 2, 32, nir_iadd(&b, vel_base, off), 8,
                                   (gl_access_qualifier)0);

   nir_def *dx = nir_channel(&b, pos, 0);
   nir_def *dy = nir_channel(&b, pos, 1);
   nir_def *dist2 = nir_fadd(&b, nir_fmul(&b, dx, dx), nir_fmul(&b, dy, dy));
   nir_def *sep = nir_fsub(&b, dist2, r2);
   nir_def *zero = nir_imm_float(&b, 0.0f);
   nir_def *collide = nir_flt(&b, sep, zero);

   nir_push_if(&b, collide);
   nir_def *inv_len = nir_frsq(&b, dist2);
   nir_def *nx = nir_fmul(&b, dx, inv_len);
   nir_def *ny = nir_fmul(&b, dy, inv_len);
   nir_def *dot_v = nir_fadd(&b, nir_fmul(&b, nir_channel(&b, vel, 0), nx),
                             nir_fmul(&b, nir_channel(&b, vel, 1), ny));
   nir_def *imp = nir_fmul(&b, dot_v, nir_imm_float(&b, -0.312f));
   nir_def *fx = nir_fmul(&b, imp, nx);
   nir_def *fy = nir_fmul(&b, imp, ny);
   nir_pop_if(&b, NULL);

   nir_def *out_x = nir_if_phi(&b, fx, zero);
   nir_def *out_y = nir_if_phi(&b, fy, zero);
   make_store_global(&b, nir_vec2(&b, out_x, out_y),
                     nir_iadd(&b, out_base, off), 8);

   agx_preprocess_nir_m5(b.shader);

   struct agx_shader_key key = agx3c_key();
   struct agx_shader_part out = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));
   EXPECT_EQ(out.info.stats.spills, 0u);
   EXPECT_GE(count_ifs(b.shader), 1u);
   EXPECT_TRUE(has_op_inside_if(b.shader, nir_op_frsq));
   EXPECT_EQ(count_load_agx_inside_if(b.shader), 1u);

   free(out.binary);
   ralloc_free(b.shader);
}

TEST(Agx3cBranchSink, JmpNoneWeightsSfuWithoutJumpingCheapBoundsChecks)
{
   void *mem_ctx = ralloc_context(NULL);
   struct agx_shader_key key = agx3c_key();

   auto count_jmps = [&](bool include_rsqrt, unsigned loads, unsigned alus) {
      agx_builder *b = agx_test_builder(mem_ctx);
      b->shader->key = &key;
      agx_block *b0 = agx_start_block(b->shader);
      agx_block *b1 = agx_test_block(b->shader);
      agx_block *b2 = agx_test_block(b->shader);

      b->cursor = agx_after_block(b0);
      agx_if_fcmp(b, agx_zero(), agx_zero(), 1, AGX_FCOND_LT, false, b2);

      b->cursor = agx_after_block(b1);
      for (unsigned i = 0; i < loads; ++i) {
         agx_device_load_to(b, agx_register(2 * i, AGX_SIZE_32),
                            agx_uniform(0, AGX_SIZE_64),
                            agx_register(16, AGX_SIZE_32), AGX_FORMAT_I32, 0x3,
                            1, false);
      }
      if (include_rsqrt)
         agx_rsqrt_to(b, agx_register(8, AGX_SIZE_32),
                      agx_register(0, AGX_SIZE_32));
      for (unsigned i = 0; i < alus; ++i)
         agx_fadd_to(b, agx_register(10, AGX_SIZE_32),
                     agx_register(10, AGX_SIZE_32),
                     agx_register(8, AGX_SIZE_32));

      b->cursor = agx_after_block(b2);
      agx_pop_exec(b, 1);

      agx_opt_jmp_none(b->shader);

      unsigned jmps = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_JMP_EXEC_NONE)
            jmps++;
      }
      return jmps;
   };

   /* Particle collision body (1 load + rsqrt + 14 ALU = 40 >= 38) skips. */
   EXPECT_EQ(count_jmps(true, 1, 14), 1u);
   /* Cheap bounds check (2 loads + 5 ALU = 25 < 38) does not insert a jump. */
   EXPECT_EQ(count_jmps(false, 2, 5), 0u);

   ralloc_free(mem_ctx);
}

TEST(Agx3cBranchSink, Agx3bEndToEndParticleShapePreservesIfAndSinksLoad)
{
   for (enum agx_chip chip : {AGX_CHIP_G15G, AGX_CHIP_G16G, AGX_CHIP_G17P}) {
      nir_shader *large = build_frsq_branch_shader(26);
      agx_preprocess_nir_native(large, chip);
      EXPECT_GE(count_ifs(large), 1u);
      EXPECT_TRUE(has_op_inside_if(large, nir_op_frsq));

      struct agx_shader_key key = agx3b_key(chip);
      struct agx_shader_part out = {};
      ASSERT_TRUE(agx_compile_shader_nir(large, &key, &out));
      EXPECT_GE(count_ifs(large), 1u);
      EXPECT_TRUE(has_op_inside_if(large, nir_op_frsq));
      free(out.binary);
      ralloc_free(large);

      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "agx3b-particle-shape");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
      nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

      nir_def *off = nir_u2u64(&b, nir_ishl_imm(&b, tid, 3));
      nir_def *pos = make_load_global(&b, 2, 32, nir_iadd(&b, pos_base, off), 8,
                                      (gl_access_qualifier)0);
      nir_def *vel = make_load_global(&b, 2, 32, nir_iadd(&b, vel_base, off), 8,
                                      (gl_access_qualifier)0);

      nir_def *dx = nir_channel(&b, pos, 0);
      nir_def *dy = nir_channel(&b, pos, 1);
      nir_def *dist2 = nir_fadd(&b, nir_fmul(&b, dx, dx), nir_fmul(&b, dy, dy));
      nir_def *sep = nir_fsub(&b, dist2, r2);
      nir_def *zero = nir_imm_float(&b, 0.0f);
      nir_def *collide = nir_flt(&b, sep, zero);

      nir_push_if(&b, collide);
      nir_def *inv_len = nir_frsq(&b, dist2);
      nir_def *nx = nir_fmul(&b, dx, inv_len);
      nir_def *ny = nir_fmul(&b, dy, inv_len);
      nir_def *dot_v = nir_fadd(&b, nir_fmul(&b, nir_channel(&b, vel, 0), nx),
                                nir_fmul(&b, nir_channel(&b, vel, 1), ny));
      nir_def *imp = nir_fmul(&b, dot_v, nir_imm_float(&b, -0.312f));
      nir_def *fx = nir_fmul(&b, imp, nx);
      nir_def *fy = nir_fmul(&b, imp, ny);
      nir_pop_if(&b, NULL);

      nir_def *out_x = nir_if_phi(&b, fx, zero);
      nir_def *out_y = nir_if_phi(&b, fy, zero);
      make_store_global(&b, nir_vec2(&b, out_x, out_y),
                        nir_iadd(&b, out_base, off), 8);

      agx_preprocess_nir_native(b.shader, chip);

      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_EQ(part.info.stats.spills, 0u);
      EXPECT_GE(count_ifs(b.shader), 1u);
      EXPECT_TRUE(has_op_inside_if(b.shader, nir_op_frsq));
      EXPECT_EQ(count_load_agx_inside_if(b.shader), 1u);

      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(Agx3cBranchSink, GalliumPreprocessPreservesAgx3bAndFlattensAgx2)
{
   for (enum agx_chip chip : {AGX_CHIP_G15G, AGX_CHIP_G16G, AGX_CHIP_G17P}) {
      const enum agx_isa_encoding encoding = agx_chip_isa_encoding(chip);
      nir_shader *large = build_frsq_branch_shader(26);
      agx_preprocess_nir_for_encoding(large, encoding);
      EXPECT_GE(count_ifs(large), 1u);
      EXPECT_TRUE(has_op_inside_if(large, nir_op_frsq));

      struct agx_shader_key key = agx3b_key(chip);
      struct agx_shader_part out = {};
      ASSERT_TRUE(agx_compile_shader_nir(large, &key, &out));
      EXPECT_GE(count_ifs(large), 1u);
      EXPECT_TRUE(has_op_inside_if(large, nir_op_frsq));
      free(out.binary);
      ralloc_free(large);

      nir_shader *small = build_frsq_branch_shader(4);
      agx_preprocess_nir_for_encoding(small, encoding);
      EXPECT_EQ(count_ifs(small), 0u);
      ralloc_free(small);

      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "gallium-agx3b-particle-shape");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *tid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *pos_base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *vel_base = agx_test_load_preamble(&b, 1, 64, 4);
      nir_def *out_base = agx_test_load_preamble(&b, 1, 64, 8);
      nir_def *r2 = agx_test_load_preamble(&b, 1, 32, 12);

      nir_def *off = nir_u2u64(&b, nir_ishl_imm(&b, tid, 3));
      nir_def *pos = make_load_global(&b, 2, 32, nir_iadd(&b, pos_base, off), 8,
                                      (gl_access_qualifier)0);
      nir_def *vel = make_load_global(&b, 2, 32, nir_iadd(&b, vel_base, off), 8,
                                      (gl_access_qualifier)0);

      nir_def *dx = nir_channel(&b, pos, 0);
      nir_def *dy = nir_channel(&b, pos, 1);
      nir_def *dist2 = nir_fadd(&b, nir_fmul(&b, dx, dx), nir_fmul(&b, dy, dy));
      nir_def *sep = nir_fsub(&b, dist2, r2);
      nir_def *zero = nir_imm_float(&b, 0.0f);
      nir_def *collide = nir_flt(&b, sep, zero);

      nir_push_if(&b, collide);
      nir_def *inv_len = nir_frsq(&b, dist2);
      nir_def *nx = nir_fmul(&b, dx, inv_len);
      nir_def *ny = nir_fmul(&b, dy, inv_len);
      nir_def *dot_v = nir_fadd(&b, nir_fmul(&b, nir_channel(&b, vel, 0), nx),
                                nir_fmul(&b, nir_channel(&b, vel, 1), ny));
      nir_def *imp = nir_fmul(&b, dot_v, nir_imm_float(&b, -0.312f));
      nir_def *fx = nir_fmul(&b, imp, nx);
      nir_def *fy = nir_fmul(&b, imp, ny);
      nir_pop_if(&b, NULL);

      nir_def *out_x = nir_if_phi(&b, fx, zero);
      nir_def *out_y = nir_if_phi(&b, fy, zero);
      make_store_global(&b, nir_vec2(&b, out_x, out_y),
                        nir_iadd(&b, out_base, off), 8);

      agx_preprocess_nir_for_encoding(b.shader, encoding);

      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_EQ(part.info.stats.spills, 0u);
      EXPECT_GE(count_ifs(b.shader), 1u);
      EXPECT_TRUE(has_op_inside_if(b.shader, nir_op_frsq));
      EXPECT_EQ(count_load_agx_inside_if(b.shader), 1u);

      free(part.binary);
      ralloc_free(b.shader);
   }

   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G15P}) {
      nir_shader *large = build_frsq_branch_shader(26);
      agx_preprocess_nir_for_encoding(large, agx_chip_isa_encoding(chip));
      EXPECT_EQ(count_ifs(large), 0u);
      ralloc_free(large);
   }

   nir_shader *generic = build_frsq_branch_shader(26);
   agx_preprocess_nir(generic);
   EXPECT_EQ(count_ifs(generic), 0u);
   ralloc_free(generic);

   auto build_d3d_f2i_shader = []() {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "d3d-f2i-gate");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      nir_def *out = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *a = agx_test_load_preamble(&b, 1, 32, 4);
      nir_def *nan = nir_fneu(&b, a, a);
      nir_def *lo = nir_flt(&b, a, nir_imm_float(&b, -2147483648.0f));
      nir_def *hi = nir_flt(&b, nir_imm_float(&b, 2147483520.0f), a);
      nir_def *val = nir_bcsel(
         &b, nan, nir_imm_int(&b, 0),
         nir_bcsel(&b, lo, nir_imm_int(&b, INT32_MIN),
                   nir_bcsel(&b, hi, nir_imm_int(&b, INT32_MAX),
                             nir_f2i32(&b, a))));
      make_store_global(&b, val, out, 4);
      return b.shader;
   };

   nir_shader *gallium_d3d = build_d3d_f2i_shader();
   agx_preprocess_nir_for_encoding(gallium_d3d, AGX_ISA_ENCODING_AGX3B);
   EXPECT_FALSE(has_op(gallium_d3d, nir_op_f2i32_sat));
   ralloc_free(gallium_d3d);

   nir_shader *native_d3d = build_d3d_f2i_shader();
   agx_preprocess_nir_native(native_d3d, AGX_CHIP_G16G);
   EXPECT_TRUE(has_op(native_d3d, nir_op_f2i32_sat));
   ralloc_free(native_d3d);

   nir_shader *m5_d3d = build_d3d_f2i_shader();
   agx_preprocess_nir_m5(m5_d3d);
   EXPECT_TRUE(has_op(m5_d3d, nir_op_f2i32_sat));
   ralloc_free(m5_d3d);
}

} /* namespace */
