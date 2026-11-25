/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"

extern "C" {
#include "agx_nir.h"
}

static nir_def *
pc_level(nir_builder *b, nir_def *v, uint32_t m, unsigned s)
{
   return nir_iadd(b, nir_iand_imm(b, v, m),
                   nir_iand_imm(b, nir_ushr_imm(b, v, s), m));
}

TEST(PopcountIdiom, TreeAndMultiplyForms)
{
   for (unsigned form = 0; form < 3; ++form) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "popcount-idiom");
      nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
      nir_intrinsic_set_range(nir_def_as_intrinsic(x), 4);
      nir_def *r;
      if (form == 0) {
         /* Full 5-level SWAR tree with final mask on both halves. */
         nir_def *v = pc_level(&b, x, 0x55555555, 1);
         v = pc_level(&b, v, 0x33333333, 2);
         v = pc_level(&b, v, 0x0f0f0f0f, 4);
         v = pc_level(&b, v, 0x00ff00ff, 8);
         r = pc_level(&b, v, 0x0000ffff, 16);
      } else if (form == 1) {
         /* Unmasked final right-shift form with commuted operands. */
         nir_def *v = pc_level(&b, x, 0x55555555, 1);
         v = pc_level(&b, v, 0x33333333, 2);
         v = pc_level(&b, v, 0x0f0f0f0f, 4);
         v = pc_level(&b, v, 0x00ff00ff, 8);
         r = nir_iadd(&b, nir_ushr_imm(&b, v, 16), nir_iand_imm(&b, v, 0xffff));
      } else {
         /* Multiply-by-0x01010101 form. */
         nir_def *v = nir_isub(&b, x, nir_iand_imm(&b, nir_ushr_imm(&b, x, 1), 0x55555555));
         v = pc_level(&b, v, 0x33333333, 2);
         v = nir_iand_imm(&b, nir_iadd(&b, nir_ushr_imm(&b, v, 4), v), 0x0f0f0f0f);
         r = nir_ushr_imm(&b, nir_imul_imm(&b, v, 0x01010101), 24);
      }
      nir_intrinsic_instr *store =
         nir_store_global(&b, r, nir_imm_int64(&b, 0));
      nir_intrinsic_set_align(store, 4, 0);

      EXPECT_TRUE(agx_nir_opt_popcount_idiom(b.shader));
      NIR_PASS(_, b.shader, nir_opt_dce);
      nir_validate_shader(b.shader, "popcount-idiom");

      nir_scalar out = nir_scalar_resolved(store->src[0].ssa, 0);
      ASSERT_TRUE(nir_scalar_is_alu(out));
      EXPECT_EQ(nir_scalar_alu_op(out), nir_op_bit_count);
      EXPECT_EQ(nir_scalar_chase_alu_src(out, 0).def, x);
      ralloc_free(b.shader);
   }

   /* Verify bit-exactness of the multiply idiom over 100k pseudo-random inputs. */
   uint32_t v = 0x9e3779b9u;
   for (unsigned i = 0; i < 100000; ++i) {
      v = v * 1664525u + 1013904223u;
      uint32_t t = v - ((v >> 1) & 0x55555555u);
      t = (t & 0x33333333u) + ((t >> 2) & 0x33333333u);
      t = (((t + (t >> 4)) & 0x0f0f0f0fu) * 0x01010101u) >> 24;
      ASSERT_EQ(t, (uint32_t)__builtin_popcount(v));
   }
}

TEST(PopcountIdiom, RejectsMismatchedMaskOrShift)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "popcount-negative");
   nir_def *x = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0));
   nir_intrinsic_set_range(nir_def_as_intrinsic(x), 4);

   /* Wrong mask at first level (0x55555554 instead of 0x55555555). */
   nir_def *v = pc_level(&b, x, 0x55555554, 1);
   v = pc_level(&b, v, 0x33333333, 2);
   v = pc_level(&b, v, 0x0f0f0f0f, 4);
   v = pc_level(&b, v, 0x00ff00ff, 8);
   nir_def *r1 = pc_level(&b, v, 0x0000ffff, 16);

   /* Wrong final shift in multiply form (25 instead of 24). */
   nir_def *u = nir_isub(&b, x, nir_iand_imm(&b, nir_ushr_imm(&b, x, 1), 0x55555555));
   u = pc_level(&b, u, 0x33333333, 2);
   u = nir_iand_imm(&b, nir_iadd(&b, u, nir_ushr_imm(&b, u, 4)), 0x0f0f0f0f);
   nir_def *r2 = nir_ushr_imm(&b, nir_imul_imm(&b, u, 0x01010101), 25);

   nir_intrinsic_set_align(
      nir_store_global(&b, nir_iadd(&b, r1, r2), nir_imm_int64(&b, 0)), 4, 0);

   EXPECT_FALSE(agx_nir_opt_popcount_idiom(b.shader));
   nir_validate_shader(b.shader, "popcount-negative");
   ralloc_free(b.shader);
}
