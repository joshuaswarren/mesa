/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include "nir_builder.h"

extern "C" {
#include "agx_compiler.h"
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
      /* Driver-internal shaders take the exact forms on every op. */
      b.shader->info.internal = true;
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

TEST_F(LowerMathTest, SinCosPairSharesReduction)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);
   nir_fcos(&b, x);

   agx_nir_lower_sincos(b.shader);

   /* The pair runs one argument reduction and one polynomial set; the
    * Cody-Waite branch's fround_even is the counter for it.
    */
   EXPECT_EQ(count_alu(nir_op_fsin), 0u);
   EXPECT_EQ(count_alu(nir_op_fcos), 0u);
   EXPECT_EQ(count_alu(nir_op_fround_even), 1u);
   EXPECT_EQ(count_alu(nir_op_feq), 1u);
}

TEST_F(LowerMathTest, SinCosDistinctArgsReduceSeparately)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_def *y = nir_fadd(&b, x, nir_imm_float(&b, 1.0f));
   nir_fsin(&b, x);
   nir_fcos(&b, y);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_fround_even), 2u);
}

TEST_F(LowerMathTest, DivByOneUsesRcp)
{
   set_mode(0);
   nir_def *d = nir_undef(&b, 1, 32);
   nir_fdiv(&b, nir_imm_float(&b, 1.0f), d);

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   /* 1.0f/d degenerates to the reciprocal refinement: same histogram as
    * FrcpDefaultSequencePinned, not the four-fma Markstein tail. */
   EXPECT_EQ(count_alu(nir_op_frcp), 1u);
   EXPECT_EQ(count_alu(nir_op_ffma), 2u);
   EXPECT_EQ(count_alu(nir_op_bcsel), 1u);
   EXPECT_EQ(count_alu(nir_op_fneu), 1u);
   EXPECT_EQ(count_alu(nir_op_fmul), 0u);
}

TEST_F(LowerMathTest, DivVecPartialOneKeepsMarkstein)
{
   set_mode(0);
   nir_def *v = nir_undef(&b, 2, 32);
   nir_def *num = nir_imm_vec2(&b, 1.0f, 2.0f);
   nir_fdiv(&b, num, v);

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   /* Only lane 0's numerator is 1.0: the rcp shortcut must not fire, or
    * lane 1 would compute rcp(d) instead of 2.0/d. A vec ALU op feeding
    * the divide (nir_vec2) would not exercise the check at all, so this
    * must be a real constant. */
   EXPECT_EQ(count_alu(nir_op_ffma), 4u);
   EXPECT_EQ(count_alu(nir_op_frcp), 1u);
}

TEST_F(LowerMathTest, CosThenSinPairSharesReduction)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fcos(&b, x);
   nir_fsin(&b, x);

   agx_nir_lower_sincos(b.shader);

   /* Same one shared reduction with the order flipped; the fsin partner's
    * zero fixup is the one feq. */
   EXPECT_EQ(count_alu(nir_op_fround_even), 1u);
   EXPECT_EQ(count_alu(nir_op_feq), 1u);
}

TEST_F(LowerMathTest, SinCosSiblingIfSharesReduction)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_def *c = nir_undef(&b, 1, 32);

   /* The trig-wrapper shape: each op under its own range check. The shared
    * sequence must hoist above the first branch to dominate both. */
   nir_push_if(&b, nir_flt(&b, c, nir_imm_float(&b, 1.0f)));
   nir_fcos(&b, x);
   nir_push_else(&b, NULL);
   nir_pop_if(&b, NULL);
   nir_push_if(&b, nir_flt(&b, c, nir_imm_float(&b, 1.0f)));
   nir_fsin(&b, x);
   nir_push_else(&b, NULL);
   nir_pop_if(&b, NULL);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_fsin), 0u);
   EXPECT_EQ(count_alu(nir_op_fcos), 0u);
   EXPECT_EQ(count_alu(nir_op_fround_even), 1u);
   nir_validate_shader(b.shader, "after agx_nir_lower_sincos");
}

TEST_F(LowerMathTest, SinCosThenElseArmsShareAboveIf)
{
   set_mode(0);
   nir_def *x = nir_undef(&b, 1, 32);
   nir_def *c = nir_undef(&b, 1, 32);

   /* Opposite arms of one branch have no program order between them, but
    * the point above the if dominates both, so sharing is legal as long as
    * the sequence hoists. */
   nir_push_if(&b, nir_flt(&b, c, nir_imm_float(&b, 1.0f)));
   nir_fcos(&b, x);
   nir_push_else(&b, NULL);
   nir_fsin(&b, x);
   nir_pop_if(&b, NULL);

   agx_nir_lower_sincos(b.shader);

   EXPECT_EQ(count_alu(nir_op_fround_even), 1u);
   nir_validate_shader(b.shader, "after agx_nir_lower_sincos");
}

/* The switch is shader-level: driver-internal shaders take the exact
 * forms and nothing the app declares or sets per-op can reach them.
 * These tests build on the fixture's compute shader with internal
 * cleared. */

TEST_F(LowerMathTest, AppShaderMarkedOpsStayFast)
{
   /* SPIR-V maps NoContraction to the full nir_fp_exact mask and DXVK
    * stamps it on every op, so op marks must not open the gate. */
   b.shader->info.internal = false;
   nir_def *a = nir_undef(&b, 1, 32);
   nir_def *d = nir_undef(&b, 1, 32);
   nir_fdiv(&b, a, d);
   nir_def *e = nir_fdiv(&b, a, d);
   nir_instr_as_alu(nir_def_instr(e))->fp_math_ctrl |= nir_fp_exact;

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   EXPECT_EQ(count_alu(nir_op_fdiv), 0u);
   EXPECT_EQ(count_alu(nir_op_frcp), 2u);
   EXPECT_EQ(count_alu(nir_op_fmul), 2u);
   EXPECT_EQ(count_alu(nir_op_ffma), 0u);
}

TEST_F(LowerMathTest, AppShaderUnmarkedTrigStaysStock)
{
   b.shader->info.internal = false;
   nir_def *x = nir_undef(&b, 1, 32);
   nir_fsin(&b, x);

   /* The pass owns nothing here: no copy-prop/cse, no lowering. */
   EXPECT_FALSE(agx_nir_lower_sincos(b.shader));
   EXPECT_EQ(count_alu(nir_op_fsin), 1u);
}

TEST_F(LowerMathTest, DivPreserveModeLowersEmittedMul)
{
   /* The a * frcp(d) rewrite runs before denorm-preserve lowering, the
    * ordering .lower_fdiv had inside nir_opt_algebraic, so the emitted
    * multiply gets the preserve expansion under DenormPreserve fp32. */
   b.shader->info.internal = false;
   set_mode(FLOAT_CONTROLS_DENORM_PRESERVE_FP32);
   nir_def *a = nir_undef(&b, 1, 32);
   nir_def *d = nir_undef(&b, 1, 32);
   nir_fdiv(&b, a, d);

   ASSERT_TRUE(agx_nir_lower_fdiv_late(b.shader));
   ASSERT_TRUE(agx_nir_lower_denorm_preserve(b.shader));

   EXPECT_EQ(count_alu(nir_op_fdiv), 0u);
   EXPECT_EQ(count_alu(nir_op_frcp), 1u);
   /* The preserve expansion computes denormal products in integer; the
    * msb hunt in exact_mul marks it ran on the emitted fmul. */
   EXPECT_GT(count_alu(nir_op_ufind_msb), 0u);
   nir_validate_shader(b.shader, "after fdiv denorm preserve");
}

TEST_F(LowerMathTest, DivFallbackKeepsFpMathCtrl)
{
   /* The stock rewrite keeps the source op's float controls: a divide the
    * app marked NoContraction (which SPIR-V maps to full nir_fp_exact)
    * must not come back as a contractible fmul * frcp pair. */
   b.shader->info.internal = false;
   nir_def *a = nir_undef(&b, 1, 32);
   nir_def *d = nir_undef(&b, 1, 32);
   nir_def *e = nir_fdiv(&b, a, d);
   nir_instr_as_alu(nir_def_instr(e))->fp_math_ctrl |= nir_fp_exact;

   ASSERT_TRUE(agx_nir_lower_fdiv(b.shader));

   unsigned marked = 0, total = 0;
   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_alu)
               continue;
            nir_alu_instr *alu = nir_instr_as_alu(instr);
            if (alu->op == nir_op_fmul || alu->op == nir_op_frcp) {
               total++;
               if ((alu->fp_math_ctrl & nir_fp_exact) == nir_fp_exact)
                  marked++;
            }
         }
      }
   }
   EXPECT_EQ(total, 2u);
   EXPECT_EQ(marked, 2u);
}
