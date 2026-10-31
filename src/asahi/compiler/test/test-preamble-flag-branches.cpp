/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* A constant-program flag that the main program branches on and selects on.
 * After nir_opt_peephole_select (late), G17P exports it once, as the half 1.0
 * or 0.0 in a word with zeroes on top: selects compare the half against zero,
 * branches the word. agx_nir_opt_preamble() itself leaves such a flag alone.
 */
struct flag_result {
   unsigned flag_loads;     /* 1-bit load_preamble left in main */
   bool branch_on_word;     /* if (ine(load_preamble 32-bit B, 0)) */
   bool select_on_half;     /* bcsel(fneu(load_preamble 16-bit B, 0), ..) */
   bool store_half_word;    /* store_preamble(u2u32(bcsel(f, 1.0h, 0.0h))) */
};

static nir_def *
skip_mov(nir_def *def)
{
   nir_alu_instr *alu = nir_def_as_alu_or_null(def);
   return alu && alu->op == nir_op_mov ? alu->src[0].src.ssa : def;
}

static bool
is_load_preamble(nir_def *def, unsigned bit_size, unsigned *base)
{
   nir_intrinsic_instr *I = nir_def_as_intrinsic_or_null(skip_mov(def));
   if (!I || I->intrinsic != nir_intrinsic_load_preamble ||
       I->def.bit_size != bit_size)
      return false;
   *base = nir_intrinsic_base(I);
   return true;
}

static flag_result
run_flag(bool late)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "preamble flag branches");

   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *flag = nir_ine_imm(
      &b,
      agx_test_load_global_constant(&b, 1, 32, nir_iadd_imm(&b, addr, 64), 4,
                                    (gl_access_qualifier)0),
      0);

   nir_def *id = nir_u2f32(&b, nir_load_sample_id(&b));
   nir_push_if(&b, flag);
   agx_test_store_output(&b, nir_vec4(&b, id, id, id, id), nir_imm_int(&b, 0),
                         nir_type_float32, FRAG_RESULT_DATA1);
   nir_pop_if(&b, NULL);
   nir_def *sel = nir_bcsel(&b, flag, id, nir_fadd_imm(&b, id, 1.0));
   agx_test_store_output(&b, nir_vec4(&b, sel, sel, sel, sel),
                         nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);

   nir_validate_shader(b.shader, "before preamble");
   unsigned sizes[] = {4, 0, 0};
   EXPECT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, false, NULL));
   if (late) {
      EXPECT_TRUE(agx_nir_export_select_flags_as_halves(b.shader, true));
   }
   nir_validate_shader(b.shader, "after preamble");

   flag_result r = {};
   unsigned word_base = ~0u, half_base = ~1u, store_base = ~2u;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_if *nif = nir_block_get_following_if(block);
         if (nif && !impl->function->is_preamble) {
            nir_alu_instr *ine = nir_def_as_alu_or_null(nif->condition.ssa);
            r.branch_on_word =
               ine && ine->op == nir_op_ine &&
               nir_src_is_const(ine->src[1].src) &&
               nir_src_as_uint(ine->src[1].src) == 0 &&
               is_load_preamble(ine->src[0].src.ssa, 32, &word_base);
         }

         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               nir_alu_instr *cmp =
                  nir_def_as_alu_or_null(alu->src[0].src.ssa);
               if (!impl->function->is_preamble && alu->op == nir_op_bcsel &&
                   cmp && cmp->op == nir_op_fneu &&
                   is_load_preamble(cmp->src[0].src.ssa, 16, &half_base))
                  r.select_on_half = true;
               continue;
            }
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (!impl->function->is_preamble &&
                I->intrinsic == nir_intrinsic_load_preamble &&
                I->def.bit_size == 1)
               r.flag_loads++;
            if (impl->function->is_preamble &&
                I->intrinsic == nir_intrinsic_store_preamble &&
                I->src[0].ssa->bit_size == 32) {
               nir_alu_instr *u2u = nir_def_as_alu_or_null(I->src[0].ssa);
               nir_alu_instr *sel =
                  u2u && u2u->op == nir_op_u2u32
                     ? nir_def_as_alu_or_null(u2u->src[0].src.ssa)
                     : NULL;
               if (sel && sel->op == nir_op_bcsel &&
                   sel->def.bit_size == 16 &&
                   nir_src_is_const(sel->src[1].src) &&
                   nir_src_as_uint(sel->src[1].src) == 0x3c00 &&
                   nir_src_is_const(sel->src[2].src) &&
                   nir_src_as_uint(sel->src[2].src) == 0) {
                  r.store_half_word = true;
                  store_base = nir_intrinsic_base(I);
               }
            }
         }
      }
   }

   if (r.branch_on_word) {
      EXPECT_EQ(word_base, store_base);
   }
   if (r.select_on_half) {
      EXPECT_EQ(half_base, store_base);
   }
   ralloc_free(b.shader);
   return r;
}

class PreambleFlagBranches : public testing::Test {
 protected:
   PreambleFlagBranches()
   {
      glsl_type_singleton_init_or_ref();
   }
   ~PreambleFlagBranches()
   {
      glsl_type_singleton_decref();
   }
};

TEST_F(PreambleFlagBranches, G17PBranchesOnTheWordAndSelectsOnTheHalf)
{
   flag_result r = run_flag(true);
   EXPECT_EQ(r.flag_loads, 0u);
   EXPECT_TRUE(r.branch_on_word);
   EXPECT_TRUE(r.select_on_half);
   EXPECT_TRUE(r.store_half_word);
}

/* At preamble time a branch use keeps the flag as it is. */
TEST_F(PreambleFlagBranches, PreambleTimeKeepsABranchedFlag)
{
   flag_result r = run_flag(false);
   EXPECT_GT(r.flag_loads, 0u);
   EXPECT_FALSE(r.branch_on_word);
}
