/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

class PreambleHalfLoopOperands : public testing::Test {
 protected:
   PreambleHalfLoopOperands()
   {
      glsl_type_singleton_init_or_ref();
   }
   ~PreambleHalfLoopOperands()
   {
      glsl_type_singleton_decref();
   }

   /* h = f2f16(push constant); in_loop: for (i < n) acc = fma(acc, h, acc)
    * in 16 bits, else acc = fma(acc, h, acc) once with no loop. */
   static nir_shader *build(bool in_loop)
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options,
                                                     "half-loop");
      nir_def *a = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 0),
                                          .range = 16);
      nir_def *h = nir_f2f16(&b, a);
      nir_variable *acc = nir_local_variable_create(b.impl,
                                                    glsl_float16_t_type(),
                                                    "acc");
      nir_store_var(&b, acc,
                    nir_u2f16(&b, nir_channel(&b,
                       nir_load_local_invocation_id(&b), 0)), 1);
      if (in_loop) {
         nir_variable *iv = nir_local_variable_create(b.impl, glsl_uint_type(),
                                                      "i");
         nir_def *n = nir_load_push_constant(&b, 1, 32, nir_imm_int(&b, 4),
                                             .range = 16);
         nir_store_var(&b, iv, nir_imm_int(&b, 0), 1);
         nir_loop *loop = nir_push_loop(&b);
         {
            nir_def *i = nir_load_var(&b, iv);
            nir_break_if(&b, nir_uge(&b, i, n));
            nir_def *x = nir_load_var(&b, acc);
            nir_store_var(&b, acc, nir_ffma(&b, x, h, x), 1);
            nir_store_var(&b, iv, nir_iadd_imm(&b, i, 1), 1);
         }
         nir_pop_loop(&b, loop);
      } else {
         nir_def *x = nir_load_var(&b, acc);
         nir_store_var(&b, acc, nir_ffma(&b, x, h, x), 1);
      }
      nir_store_global(&b, nir_f2f32(&b, nir_load_var(&b, acc)),
                       nir_imm_int64(&b, 0x1000), .align_mul = 4);
      NIR_PASS(_, b.shader, nir_lower_vars_to_ssa);
      NIR_PASS(_, b.shader, nir_opt_dce);
      return b.shader;
   }

   /* Whether the main shader still converts the push constant itself, from
    * the push constant or from its preamble copy. */
   static bool main_converts(nir_shader *s)
   {
      nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == nir_op_f2f16)
               return true;
         }
      }
      return false;
   }

   static bool run(nir_shader *s, bool g17p, bool agx3c)
   {
      unsigned sizes[] = {1, 1000, 1000};
      return agx_nir_opt_preamble(s, sizes, g17p, agx3c, NULL);
   }
};

/* AGX3B: the 16-bit value the loop multiplies by stays in the shader, read
 * from a register; only its 32-bit input comes from the preamble. */
TEST_F(PreambleHalfLoopOperands, Agx3bKeepsLoopOperandInShader)
{
   nir_shader *s = build(true);
   run(s, true, false);
   nir_validate_shader(s, "half loop operand");
   EXPECT_TRUE(main_converts(s));
   ralloc_free(s);
}

/* Outside any loop the conversion still moves to the preamble. */
TEST_F(PreambleHalfLoopOperands, Agx3bHoistsOperandOutsideLoops)
{
   nir_shader *s = build(false);
   run(s, true, false);
   nir_validate_shader(s, "half operand, no loop");
   EXPECT_FALSE(main_converts(s));
   ralloc_free(s);
}

/* AGX3C is unchanged. */
TEST_F(PreambleHalfLoopOperands, Agx3cHoistsLoopOperand)
{
   nir_shader *s = build(true);
   run(s, true, true);
   nir_validate_shader(s, "half loop operand, agx3c");
   EXPECT_FALSE(main_converts(s));
   ralloc_free(s);
}
