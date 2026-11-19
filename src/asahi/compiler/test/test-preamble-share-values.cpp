/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

/* An expensive uniform expression repeated in both arms of a non-uniform if:
 * neither copy dominates the other, so CSE keeps both. nir_opt_preamble must
 * give the equal values one slot, and every main-shader use must read it.
 */
#define SPEC ((gl_access_qualifier)(ACCESS_CAN_SPECULATE | ACCESS_CAN_REORDER))

struct share_result {
   unsigned main_loads;
   unsigned distinct_main_bases;
   unsigned stores;
   unsigned used_halves;
};

static share_result
run_share(bool uniform_if, unsigned arms_sharing)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "preamble share values");

   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *cond = uniform_if
                      ? nir_ine_imm(&b,
                                    agx_test_load_global_constant(
                                       &b, 1, 32, nir_iadd_imm(&b, addr, 64), 4,
                                       (gl_access_qualifier)0),
                                    0)
                      : nir_ine_imm(&b, nir_load_sample_id(&b), 0);

   nir_def *values[2];
   nir_push_if(&b, cond);
   {
      nir_def *x = agx_test_load_global_constant(
         &b, 1, 32, nir_iadd_imm(&b, addr, 16), 4, SPEC);
      values[0] = nir_fmul_imm(&b, nir_fsqrt(&b, x), 3.0);
   }
   nir_push_else(&b, NULL);
   {
      /* arms_sharing == 2: the same expression; otherwise a different load */
      nir_def *x = agx_test_load_global_constant(
         &b, 1, 32, nir_iadd_imm(&b, addr, arms_sharing == 2 ? 16 : 32), 4,
         SPEC);
      values[1] = nir_fmul_imm(&b, nir_fsqrt(&b, x), 3.0);
   }
   nir_pop_if(&b, NULL);
   nir_def *phi = nir_if_phi(&b, values[0], values[1]);
   agx_test_store_output(&b, nir_vec4(&b, phi, phi, phi, phi),
                         nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);

   nir_validate_shader(b.shader, "before preamble");
   unsigned sizes[] = {4, 0, 0};
   EXPECT_TRUE(agx_nir_opt_preamble(b.shader, sizes, true, false, NULL));
   nir_validate_shader(b.shader, "after preamble");

   share_result r = {};
   r.used_halves = sizes[0] - 4;
   unsigned bases[8];
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *I = nir_instr_as_intrinsic(instr);
            if (impl->function->is_preamble &&
                I->intrinsic == nir_intrinsic_store_preamble)
               r.stores++;
            if (!impl->function->is_preamble &&
                I->intrinsic == nir_intrinsic_load_preamble &&
                nir_intrinsic_base(I) >= 4 && I->def.bit_size == 32) {
               unsigned base = nir_intrinsic_base(I);
               bool seen = false;
               for (unsigned i = 0; i < r.distinct_main_bases; ++i)
                  seen |= bases[i] == base;
               if (!seen && r.distinct_main_bases < 8)
                  bases[r.distinct_main_bases++] = base;
               r.main_loads++;
            }
         }
      }
   }
   ralloc_free(b.shader);
   return r;
}

TEST(PreambleShareValues, EqualArmsShareOneSlot)
{
   share_result r = run_share(false, 2);
   EXPECT_EQ(r.main_loads, 2u);
   EXPECT_EQ(r.distinct_main_bases, 1u);
   EXPECT_EQ(r.used_halves, 2u);
   /* Both copies are speculated to the top of the preamble: one store. */
   EXPECT_EQ(r.stores, 1u);
}

TEST(PreambleShareValues, DifferentArmsKeepTwoSlots)
{
   share_result r = run_share(false, 1);
   EXPECT_EQ(r.main_loads, 2u);
   EXPECT_EQ(r.distinct_main_bases, 2u);
   EXPECT_EQ(r.used_halves, 4u);
   EXPECT_EQ(r.stores, 2u);
}

TEST(PreambleShareValues, UniformIfSharesSlot)
{
   share_result r = run_share(true, 2);
   EXPECT_EQ(r.distinct_main_bases, 1u);
   EXPECT_GE(r.stores, 1u);
}
