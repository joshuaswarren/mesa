/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "agx_nir.h"
}

static void
check_modifier(bool g17p, bool widen)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "uniform modifiers");
   nir_def *value = agx_test_load_preamble(&b, 1, widen ? 16 : 32, 4);
   value = widen ? nir_f2f32(&b, value) : nir_fabs(&b, value);
   nir_def *x = nir_channel(&b, nir_load_frag_coord(&b), 0);
   nir_def *result = nir_fmul(&b, x, value);
   agx_test_store_output(&b, nir_vec4(&b, result, result, result, result),
                         nir_imm_int(&b, 0), nir_type_float32,
                         FRAG_RESULT_DATA0);
   unsigned sizes[3] = {16, 0, 0};
   bool progress = agx_nir_opt_preamble(b.shader, sizes, g17p, false, NULL);
   const bool hoisted = g17p && !widen;
   if (hoisted) {
      EXPECT_TRUE(progress);
   }
   unsigned modifiers = 0, exported = 0;
   nir_function_impl *main = nir_shader_get_entrypoint(b.shader);
   nir_foreach_block(block, main) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_alu) {
            nir_op op = nir_instr_as_alu(instr)->op;
            modifiers += op == (widen ? nir_op_f2f32 : nir_op_fabs);
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            exported += intr->intrinsic == nir_intrinsic_load_preamble &&
                        nir_intrinsic_base(intr) >= 16;
         }
      }
   }
   EXPECT_EQ(modifiers, hoisted ? 0u : 1u);
   EXPECT_EQ(exported, hoisted ? 1u : 0u);
   if (hoisted) {
      EXPECT_EQ(sizes[0], 18u);
   }
   ralloc_free(b.shader);
}

TEST(PreambleModifiers, HoistsUniformAbsoluteValueOnAGX3B)
{
   check_modifier(true, false);
   check_modifier(false, false);
}

TEST(PreambleModifiers, PreservesFreeUniformHalfWidening)
{
   check_modifier(true, true);
   check_modifier(false, true);
}
