/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"

extern "C" {
#include "agx_nir.h"
}

#include <gtest/gtest.h>

/*
 * Behavioral pins for the float-control keying in agx_nir_lower_math.c.
 *
 * The default path (no execution modes declared) pins the op histogram of
 * the exact division and reciprocal emissions, so an accidental change to
 * the sequences shows up here before it reaches hardware.
 *
 * The DenormFlushToZero sin test pins the one keyed change: a zero-comparing
 * subnormal argument must produce a sign-preserving flushed zero instead of
 * leaking the unflushed argument through the select.
 */

class LowerMathTest : public ::testing::Test {
 protected:
   LowerMathTest()
   {
      mem_ctx = ralloc_context(NULL);
      b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options,
                                         "lower math test");
   }

   ~LowerMathTest() override { ralloc_free(mem_ctx); }

   void *mem_ctx;
   nir_builder b;
   static const nir_shader_compiler_options options;

   void set_mode(unsigned mode)
   {
      b.shader->info.float_controls_execution_mode = mode;
   }

   unsigned count_alu(nir_op op)
   {
      unsigned n = 0;
      nir_foreach_function_impl(impl, b.shader) {
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

   unsigned count_iand_imm(uint32_t mask)
   {
      unsigned n = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_alu)
                  continue;

               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_iand &&
                   nir_src_is_const(alu->src[1].src) &&
                   nir_src_as_uint(alu->src[1].src) == mask)
                  n++;
            }
         }
      }
      return n;
   }
};

const nir_shader_compiler_options LowerMathTest::options = {};

TEST_F(LowerMathTest, DivDefaultFastSequencePinned)
{
   set_mode(0);
   nir_def *a = nir_undef(&b, 1, 32);
   nir_def *d = nir_undef(&b, 1, 32);
   nir_fdiv(&b, a, d);

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   EXPECT_EQ(count_alu(nir_op_fdiv), 0u);
   EXPECT_EQ(count_alu(nir_op_frcp), 1u);
   EXPECT_EQ(count_alu(nir_op_ffma), 4u);
   EXPECT_EQ(count_alu(nir_op_fmul), 1u);
   EXPECT_EQ(count_alu(nir_op_fneg), 2u);
   EXPECT_EQ(count_alu(nir_op_bcsel), 2u);
   EXPECT_EQ(count_alu(nir_op_fneu), 2u);
   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_alu(nir_op_ior), 1u);
   /* The mantissa-normalizing exact sequence is integer arithmetic; its
    * absence is the marker that the default emission is untouched. */
   EXPECT_EQ(count_alu(nir_op_iand), 0u);
   EXPECT_EQ(count_alu(nir_op_ushr), 0u);
   EXPECT_EQ(count_alu(nir_op_ishl), 0u);
}

TEST_F(LowerMathTest, DivFtzModeUnchanged)
{
   set_mode(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32);
   nir_def *a = nir_undef(&b, 1, 32);
   nir_def *d = nir_undef(&b, 1, 32);
   nir_fdiv(&b, a, d);

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   /* Hardware flushes arithmetic outputs, so the refined form already
    * satisfies DenormFlushToZero; no keyed emission is warranted. */
   EXPECT_EQ(count_alu(nir_op_iand), 0u);
   EXPECT_EQ(count_alu(nir_op_ushr), 0u);
   EXPECT_EQ(count_alu(nir_op_ishl), 0u);
   EXPECT_EQ(count_alu(nir_op_frcp), 1u);
}

TEST_F(LowerMathTest, FrcpDefaultSequencePinned)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_frcp(&b, x);

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   EXPECT_EQ(count_alu(nir_op_frcp), 1u);
   EXPECT_EQ(count_alu(nir_op_ffma), 2u);
   EXPECT_EQ(count_alu(nir_op_bcsel), 1u);
   EXPECT_EQ(count_alu(nir_op_fneu), 1u);
}

TEST_F(LowerMathTest, SinDefaultKeepsSelect)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_iand_imm(0x80000000), 0u);
}

TEST_F(LowerMathTest, SinFtzReturnsSignedZero)
{
   set_mode(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_iand_imm(0x80000000), 1u);
}

TEST_F(LowerMathTest, SinPreserveKeepsSelect)
{
   set_mode(FLOAT_CONTROLS_DENORM_PRESERVE_FP32);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_feq), 1u);
   EXPECT_EQ(count_iand_imm(0x80000000), 0u);
}

