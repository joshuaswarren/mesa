/*
 * Copyright 2021 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx3_select_core.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <vector>
#include "util/u_debug.h"
#include <gtest/gtest.h>

static void
agx_optimize_and_dce(agx_context *ctx)
{
   agx_optimizer_backward(ctx);
   agx_optimizer_forward(ctx);
   agx_dce(ctx, true);
}

#define CASE(instr, expected, size, returns)                                   \
   INSTRUCTION_CASE(                                                           \
      {                                                                        \
         UNUSED agx_index out = agx_temp(b->shader, AGX_SIZE_##size);          \
         instr;                                                                \
         if (returns)                                                          \
            agx_unit_test(b, out);                                             \
      },                                                                       \
      {                                                                        \
         UNUSED agx_index out = agx_temp(b->shader, AGX_SIZE_##size);          \
         expected;                                                             \
         if (returns)                                                          \
            agx_unit_test(b, out);                                             \
      },                                                                       \
      agx_optimize_and_dce)

#define NEGCASE(instr, size) CASE(instr, instr, size, true)

#define CASE16(instr, expected) CASE(instr, expected, 16, true)
#define CASE32(instr, expected) CASE(instr, expected, 32, true)
#define CASE64(instr, expected) CASE(instr, expected, 64, true)

#define CASE_NO_RETURN(instr, expected)                                        \
   CASE(instr, expected, 32 /* irrelevant */, false)

#define NEGCASE16(instr) NEGCASE(instr, 16)
#define NEGCASE32(instr) NEGCASE(instr, 32)

static inline agx_index
agx_fmov(agx_builder *b, agx_index s0)
{
   agx_index tmp = agx_temp(b->shader, s0.size);
   agx_fmov_to(b, tmp, s0);
   return tmp;
}

class Optimizer : public testing::Test {
 protected:
   Optimizer()
   {
      mem_ctx = ralloc_context(NULL);

      dx = agx_register(0, AGX_SIZE_64);
      dz = agx_register(4, AGX_SIZE_64);

      wx = agx_register(0, AGX_SIZE_32);
      wy = agx_register(2, AGX_SIZE_32);
      wz = agx_register(4, AGX_SIZE_32);

      hx = agx_register(0, AGX_SIZE_16);
      hy = agx_register(1, AGX_SIZE_16);
      hz = agx_register(2, AGX_SIZE_16);
   }

   ~Optimizer()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;

   agx_index dx, dz, wx, wy, wz, hx, hy, hz;
};

TEST_F(Optimizer, FloatCopyprop)
{
   CASE32(agx_fadd_to(b, out, agx_abs(agx_fmov(b, wx)), wy),
          agx_fadd_to(b, out, agx_abs(wx), wy));

   CASE32(agx_fadd_to(b, out, agx_neg(agx_fmov(b, wx)), wy),
          agx_fadd_to(b, out, agx_neg(wx), wy));
}

TEST_F(Optimizer, FloatConversion)
{
   CASE32(
      {
         agx_index cvt = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, cvt, hx);
         agx_fadd_to(b, out, cvt, wy);
      },
      { agx_fadd_to(b, out, hx, wy); });

   CASE16(
      {
         agx_index sum = agx_temp(b->shader, AGX_SIZE_32);
         agx_fadd_to(b, sum, wx, wy);
         agx_fmov_to(b, out, sum);
      },
      { agx_fadd_to(b, out, wx, wy); });
}

TEST_F(Optimizer, G17PPreservesHalfNotAfterBitwise)
{
   for (unsigned table : {AGX3_SEL_TT_AND, AGX3_SEL_TT_OR, AGX3_SEL_TT_XOR}) {
      agx_builder *actual = agx_test_builder(mem_ctx);
      agx_builder *expected = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      for (agx_builder *b : {actual, expected}) {
         b->shader->key = &key;
         agx_index value = agx_bitop(b, hx, hy, table);
         agx_index inverted = agx_not(b, value);
         agx_unit_test(b, inverted);
      }
      agx_optimize_and_dce(actual->shader);
      ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
   }
}

/* AGX3 FMA has no form with two immediate multiplicands: the first constant
 * folds as the minifloat multiplier, the second stays in its register. */
TEST_F(Optimizer, AGX3FmaInlinesOneConstantMultiplicand)
{
   for (enum agx_isa_encoding encoding :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      agx_builder *actual = agx_test_builder(mem_ctx);
      agx_builder *expected = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = encoding;
      actual->shader->key = &key;
      expected->shader->key = &key;

      agx_index one = agx_mov_imm(actual, 16, 0x3c00);
      agx_index two = agx_mov_imm(actual, 16, 0x4000);
      agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
      agx_ffma_to(actual, out, one, two, wx);
      agx_unit_test(actual, out);

      agx_temp(expected->shader, AGX_SIZE_16); /* the dead mov of 1.0 */
      two = agx_mov_imm(expected, 16, 0x4000);
      out = agx_temp(expected->shader, AGX_SIZE_32);
      agx_ffma_to(expected, out, agx_immediate_f(1.0f), two, wx);
      agx_unit_test(expected, out);

      agx_optimize_and_dce(actual->shader);
      ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
   }
}

TEST_F(Optimizer, G17PPreservesNarrowingAfterAdd)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index sum = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fadd_to(actual, sum, wx, wy);
   agx_index narrowed = agx_temp(actual->shader, AGX_SIZE_16);
   agx_fmov_to(actual, narrowed, sum);
   agx_unit_test(actual, narrowed);

   sum = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fadd_to(expected, sum, wx, wy);
   narrowed = agx_temp(expected->shader, AGX_SIZE_16);
   agx_fmov_to(expected, narrowed, sum);
   agx_unit_test(expected, narrowed);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, G17PPreservesZeroExtensionsBeforeExtract)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   for (agx_builder *b : {actual, expected}) {
      b->shader->key = &key;
      agx_index lo = agx_temp(b->shader, AGX_SIZE_32);
      agx_index hi = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, lo, hx);
      agx_mov_to(b, hi, hy);
      agx_index out = agx_temp(b->shader, AGX_SIZE_32);
      agx_extr_to(b, out, lo, hi, agx_immediate(4), 0);
      agx_unit_test(b, out);
   }
   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, G17PPreservesHalfWidenBeforeFltnorm)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index lo = agx_temp(actual->shader, AGX_SIZE_32);
   agx_index hi = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fmov_to(actual, lo, hx);
   agx_fmov_to(actual, hi, hy);
   agx_index packed = agx_temp(actual->shader, AGX_SIZE_16);
   agx_fltnorm_agx3_to(actual, packed, lo, hi, AGX_FORMAT_U8NORM);
   agx_unit_test(actual, packed);

   lo = agx_temp(expected->shader, AGX_SIZE_32);
   hi = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fmov_to(expected, lo, hx);
   agx_fmov_to(expected, hi, hy);
   packed = agx_temp(expected->shader, AGX_SIZE_16);
   agx_fltnorm_agx3_to(expected, packed, lo, hi, AGX_FORMAT_U8NORM);
   agx_unit_test(expected, packed);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, G18GPreservesEpilogConversions)
{
   /* A G18G mediump fragment epilog widens half exports, blends in f32, then
    * packs normalized color. All three boundaries must keep their widths. */
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.emit_agx3d = true;
   for (agx_builder *b : {actual, expected}) {
      b->shader->key = &key;
      agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
      agx_fmov_to(b, wide, hx);
      agx_index sum = agx_fadd(b, wide, wy);
      agx_index narrow = agx_temp(b->shader, AGX_SIZE_16);
      agx_fmov_to(b, narrow, sum);
      agx_unit_test(b, narrow);
      agx_index packed = agx_temp(b->shader, AGX_SIZE_16);
      agx_fltnorm_agx3_to(b, packed, wide, sum, AGX_FORMAT_U8NORM);
      agx_unit_test(b, packed);
   }
   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/*
 * The projective multiply and reciprocal only have a plain 32-bit destination
 * and no saturating form. A mediump perspective varying narrows the product to
 * 16 bits and clamp() saturates it; both moves have to stay separate on every
 * AGX3 encoding (G17S refused `persp_mul_agx3 d16, r32, r32`).
 */
TEST_F(Optimizer, AGX3KeepsMovesAfterProjectiveMultiply)
{
   const struct {
      enum agx_isa_encoding encoding;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, false},
      {AGX_ISA_ENCODING_AGX3C, false},
      {AGX_ISA_ENCODING_AGX3C, true},
   };

   for (const auto &target : targets) {
      for (bool saturate : {false, true}) {
         agx_builder *actual = agx_test_builder(mem_ctx);
         agx_builder *expected = agx_test_builder(mem_ctx);
         struct agx_shader_key key = {};
         key.dev.encoding = target.encoding;
         key.dev.emit_agx3d = target.emit_agx3d;
         for (agx_builder *b : {actual, expected}) {
            b->shader->key = &key;
            agx_index rcp = agx_persp_rcp_agx3(b, wx, 3);
            agx_index product = agx_persp_mul_agx3(b, wy, rcp, 4);
            agx_index out =
               agx_temp(b->shader, saturate ? AGX_SIZE_32 : AGX_SIZE_16);
            agx_fmov_to(b, out, product)->saturate = saturate;
            agx_unit_test(b, out);
         }
         agx_optimize_and_dce(actual->shader);
         ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
      }
   }
}

/*
 * AGX3 has no select form without a register compare operand, so a select of
 * two immediates -- here fused out of a compare of two constants -- has to
 * fold to the chosen operand.
 */
TEST_F(Optimizer, G17PFoldsAConstantSelect)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index two = agx_mov_imm(actual, 32, 2);
   agx_index five = agx_mov_imm(actual, 32, 5);
   agx_index lt = agx_icmp(actual, two, five, AGX_ICOND_ULT, false);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_icmpsel_to(actual, out, lt, agx_zero(), wx, wy, AGX_ICOND_UEQ);
   agx_unit_test(actual, out);

   /* Burn the indices of the dead constants, compare and select; the chosen
    * operand then propagates into the use. */
   for (unsigned i = 0; i < 4; ++i)
      (void)agx_temp(expected->shader, AGX_SIZE_32);
   agx_unit_test(expected, wy);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/* A compare of two constants used as a value folds the same way, before it is
 * lowered to a select of two immediates.
 */
TEST_F(Optimizer, G17PFoldsAConstantCompare)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index zero = agx_mov_imm(actual, 16, 0);
   agx_index six = agx_mov_imm(actual, 16, 6);
   agx_index eq = agx_icmp(actual, zero, six, AGX_ICOND_UEQ, true);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_16);
   agx_iadd_to(actual, out, eq, hx, 0);
   agx_unit_test(actual, out);

   for (unsigned i = 0; i < 3; ++i)
      (void)agx_temp(expected->shader, AGX_SIZE_16);
   out = agx_temp(expected->shader, AGX_SIZE_16);
   agx_iadd_to(expected, out, agx_immediate(1), hx, 0);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/* The comparison width is unknown once both operands are immediates, so a
 * signed compare of a 16-bit immediate with its top bit set stays a select.
 */
TEST_F(Optimizer, G17PKeepsASignDependentConstantSelect)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_16);
   agx_icmpsel_to(actual, out, agx_abs(agx_immediate(0x8000)),
                  agx_immediate(1), hx, hy, AGX_ICOND_SLT);
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_16);
   agx_icmpsel_to(expected, out, agx_abs(agx_immediate(0x8000)),
                  agx_immediate(1), hx, hy, AGX_ICOND_SLT);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/*
 * A tied fcmpsel with a GTN or LTN condition is G17P's native minimum or
 * maximum, and Apple's encoding carries the constant operand as one minifloat
 * that serves as both the compared and the selected value. Fold the constant
 * into both slots: folding into only the comparison -- which is what the
 * generic rule does, because source 3 of a select is not a float source --
 * leaves the packer a mixed-width general select it has no encoding for, and
 * that is what made every max(x, 0.0) fail the shader closed.
 */
TEST_F(Optimizer, G17PFoldsBothHalvesOfATiedConstantFmax)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index minus_one = agx_mov_imm(actual, 32, 0xbf800000);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fcmpsel_to(actual, out, wx, minus_one, wx, minus_one,
                  AGX_FCOND_GTN);
   agx_unit_test(actual, out);

   const agx_index folded = agx_immediate_f(-1.0f);
   /* Burn the index the dead constant held, so only the folding differs. */
   (void)agx_temp(expected->shader, AGX_SIZE_32);
   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fcmpsel_to(expected, out, wx, folded, wx, folded, AGX_FCOND_GTN)
      ->fcmpsel_minifloat_select = true;
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, G17PVertexConstantWidth)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      agx_builder *b = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = mode == 3 ? AGX_ISA_ENCODING_AGX2 : AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      b->shader->stage = mode == 1 ? MESA_SHADER_FRAGMENT : MESA_SHADER_VERTEX;
      b->shader->is_preamble = mode == 2;
      agx_index constant = agx_mov_imm(b, 32, fui(1.0f / 1024.0f));
      agx_index out = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, out, wx, constant, wy);
      agx_unit_test(b, out);
      agx_opt_compact_constants(b->shader);
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_FFMA) {
            EXPECT_EQ(I->src[1].size, AGX_SIZE_16);
         }
         if (I->op == AGX_OPCODE_MOV_IMM) {
            EXPECT_EQ(I->imm, 0x1400u);
         }
      }
   }
}

/* The constant may be written on either side. Half-folding the first pair is
 * worse than the second: agx_opt_compact_constants then sees the surviving
 * reference as the constant's only float use and narrows it to 16 bits,
 * leaving a 32-bit maximum compared against a 16-bit register. */
TEST_F(Optimizer, G17PFoldsBothHalvesOfATiedConstantFmaxWrittenFirst)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index zero = agx_mov_imm(actual, 32, 0);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fcmpsel_to(actual, out, zero, wx, zero, wx, AGX_FCOND_GTN);
   agx_unit_test(actual, out);

   const agx_index folded = agx_immediate_f(0.0f);
   (void)agx_temp(expected->shader, AGX_SIZE_32);
   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fcmpsel_to(expected, out, folded, wx, folded, wx, AGX_FCOND_GTN)
      ->fcmpsel_minifloat_select = true;
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/* A constant that is not an exact minifloat has no immediate encoding, so the
 * tie stays in a register -- in both slots, never one. */
TEST_F(Optimizer, G17PPreservesTiedConstantFmaxThatIsNotAMinifloat)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   /* 0.1f is not representable as an AGX minifloat. */
   agx_index tenth = agx_mov_imm(actual, 32, 0x3dcccccd);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fcmpsel_to(actual, out, wx, tenth, wx, tenth, AGX_FCOND_GTN);
   agx_unit_test(actual, out);

   tenth = agx_mov_imm(expected, 32, 0x3dcccccd);
   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fcmpsel_to(expected, out, wx, tenth, wx, tenth, AGX_FCOND_GTN);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/*
 * A clamp against zero written as a compare and a select -- (0 < x) ? x : 0,
 * (0 > x) ? x : 0, (x < 0) ? x : 0 or (x > 0) ? x : 0 -- is exactly the tied
 * maximum (x GTN 0) ? x : 0 or minimum (x LTN 0) ? x : 0: a NaN or a zero of
 * either sign compares false in both forms. Fusion leaves the comparator zero
 * as a minifloat and the selected zero as a bit pattern; they are the same
 * +0.0, so the select becomes the tied form.
 */
struct ZeroClamp {
   bool zero_first;
   enum agx_fcond cond;
   enum agx_fcond tied;
};

static const ZeroClamp zero_clamps[] = {
   {true, AGX_FCOND_LT, AGX_FCOND_GTN},
   {true, AGX_FCOND_GT, AGX_FCOND_LTN},
   {false, AGX_FCOND_LT, AGX_FCOND_LTN},
   {false, AGX_FCOND_GT, AGX_FCOND_GTN},
};

static void
build_zero_clamp(agx_builder *b, agx_index x, const ZeroClamp &c,
                 uint32_t constant, bool abs_x)
{
   agx_index k = agx_mov_imm(b, 32, constant);
   agx_index xs = abs_x ? agx_abs(x) : x;
   agx_index cmp = c.zero_first ? agx_fcmp(b, k, xs, c.cond, false)
                                : agx_fcmp(b, xs, k, c.cond, false);
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   agx_icmpsel_to(b, out, cmp, agx_zero(), k, x, AGX_ICOND_UEQ);
   agx_unit_test(b, out);
}

TEST_F(Optimizer, G17PTiesAClampAgainstZero)
{
   for (const ZeroClamp &c : zero_clamps) {
      agx_builder *actual = agx_test_builder(mem_ctx);
      agx_builder *expected = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      actual->shader->key = &key;
      expected->shader->key = &key;

      agx_index x = agx_fadd(actual, wx, wy);
      build_zero_clamp(actual, x, c, 0, false);

      x = agx_fadd(expected, wx, wy);
      /* Burn the indices of the dead constant and compare. */
      (void)agx_temp(expected->shader, AGX_SIZE_32);
      (void)agx_temp(expected->shader, AGX_SIZE_16);
      agx_index out = agx_temp(expected->shader, AGX_SIZE_32);
      agx_fcmpsel_to(expected, out, x, agx_immediate(0), x, agx_immediate(0),
                     c.tied);
      agx_unit_test(expected, out);

      agx_optimize_and_dce(actual->shader);
      ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
   }
}

/* Only a zero is the same value in both roles, and a modified x is a
 * different value from the one selected; AGX2 has no tied form. */
TEST_F(Optimizer, G17PKeepsOtherClampsGeneral)
{
   for (const ZeroClamp &c : zero_clamps) {
      for (unsigned variant = 0; variant < 3; ++variant) {
         agx_builder *b = agx_test_builder(mem_ctx);
         struct agx_shader_key key = {};
         key.dev.encoding =
            variant == 2 ? AGX_ISA_ENCODING_AGX2 : AGX_ISA_ENCODING_AGX3B;
         b->shader->key = &key;

         agx_index x = agx_fadd(b, wx, wy);
         build_zero_clamp(b, x, c, variant == 0 ? 0x3f800000 : 0,
                          variant == 1);
         agx_optimize_and_dce(b->shader);

         unsigned selects = 0;
         agx_foreach_instr_global(b->shader, I) {
            if (I->op != AGX_OPCODE_FCMPSEL)
               continue;
            selects++;
            EXPECT_EQ(I->fcond, c.cond);
         }
         EXPECT_EQ(selects, 1u);
      }
   }
}

/* A general select is not a minimum: its selected operands are not tied to the
 * compared ones, so the compare-slot fold must stay compare-only. */
TEST_F(Optimizer, G17PDoesNotFoldTheSelectSlotOfAnUntiedFcmpsel)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index minus_one = agx_mov_imm(actual, 32, 0xbf800000);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fcmpsel_to(actual, out, wx, minus_one, wy, wz, AGX_FCOND_GTN);
   agx_unit_test(actual, out);

   (void)agx_temp(expected->shader, AGX_SIZE_32);
   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fcmpsel_to(expected, out, wx, agx_immediate_f(-1.0f), wy, wz,
                  AGX_FCOND_GTN);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, NativePreservesWideningBeforeWidthMatchedIntegerALU)
{
   for (enum agx_isa_encoding encoding :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = encoding;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index widened = agx_temp(actual->shader, AGX_SIZE_32);
   agx_mov_to(actual, widened, hx);
   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_xor_to(actual, out, widened, wy);
   agx_unit_test(actual, out);
   agx_index right_out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_or_to(actual, right_out, wy, widened);
   agx_unit_test(actual, right_out);
   agx_index selected = agx_temp(actual->shader, AGX_SIZE_32);
   agx_icmpsel_to(actual, selected, widened, wz, wx, wy, AGX_ICOND_ULT);
   agx_unit_test(actual, selected);

   widened = agx_temp(expected->shader, AGX_SIZE_32);
   agx_mov_to(expected, widened, hx);
   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_xor_to(expected, out, widened, wy);
   agx_unit_test(expected, out);
   right_out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_or_to(expected, right_out, wy, widened);
   agx_unit_test(expected, right_out);
   selected = agx_temp(expected->shader, AGX_SIZE_32);
   agx_icmpsel_to(expected, selected, widened, wz, wx, wy, AGX_ICOND_ULT);
   agx_unit_test(expected, selected);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
   }
}

TEST_F(Optimizer, FusedFABSNEG)
{
   CASE32(agx_fadd_to(b, out, agx_fmov(b, agx_abs(wx)), wy),
          agx_fadd_to(b, out, agx_abs(wx), wy));

   CASE32(agx_fmul_to(b, out, wx, agx_fmov(b, agx_neg(agx_abs(wx)))),
          agx_fmul_to(b, out, wx, agx_neg(agx_abs(wx))));
}

TEST_F(Optimizer, FusedFabsAbsorb)
{
   CASE32(agx_fadd_to(b, out, agx_abs(agx_fmov(b, agx_abs(wx))), wy),
          agx_fadd_to(b, out, agx_abs(wx), wy));
}

TEST_F(Optimizer, FusedFnegCancel)
{
   CASE32(agx_fmul_to(b, out, wx, agx_neg(agx_fmov(b, agx_neg(wx)))),
          agx_fmul_to(b, out, wx, wx));

   CASE32(agx_fmul_to(b, out, wx, agx_neg(agx_fmov(b, agx_neg(agx_abs(wx))))),
          agx_fmul_to(b, out, wx, agx_abs(wx)));
}

TEST_F(Optimizer, FusedNot)
{
   CASE32(agx_not_to(b, out, agx_and(b, wx, wx)), agx_nand_to(b, out, wx, wx));

   CASE32(agx_not_to(b, out, agx_or(b, wx, wx)), agx_nor_to(b, out, wx, wx));

   CASE32(agx_not_to(b, out, agx_xor(b, wx, wx)), agx_xnor_to(b, out, wx, wx));

   CASE32(agx_xor_to(b, out, agx_not(b, wx), agx_not(b, wx)),
          agx_xor_to(b, out, wx, wx));

   CASE32(agx_xor_to(b, out, agx_not(b, wx), wx), agx_xnor_to(b, out, wx, wx));

   CASE32(agx_xor_to(b, out, wx, agx_not(b, wx)), agx_xnor_to(b, out, wx, wx));

   CASE32(agx_nand_to(b, out, agx_not(b, wx), agx_not(b, wx)),
          agx_or_to(b, out, wx, wx));

   CASE32(agx_andn1_to(b, out, agx_not(b, wx), wx), agx_and_to(b, out, wx, wx));

   CASE32(agx_andn1_to(b, out, wx, agx_not(b, wx)), agx_nor_to(b, out, wx, wx));

   CASE32(agx_andn2_to(b, out, agx_not(b, wx), wx), agx_nor_to(b, out, wx, wx));

   CASE32(agx_andn2_to(b, out, wx, agx_not(b, wx)), agx_and_to(b, out, wx, wx));

   CASE32(agx_xor_to(b, out, agx_not(b, wx), agx_uniform(8, AGX_SIZE_32)),
          agx_xnor_to(b, out, wx, agx_uniform(8, AGX_SIZE_32)));

   CASE32(agx_or_to(b, out, agx_immediate(123), agx_not(b, wx)),
          agx_orn2_to(b, out, agx_immediate(123), wx));

   CASE32(agx_xor_to(b, out, wx, agx_not(b, wy)), agx_xnor_to(b, out, wx, wy));

   CASE32(agx_xor_to(b, out, wy, agx_not(b, wx)), agx_xnor_to(b, out, wy, wx));

   CASE32(agx_and_to(b, out, agx_not(b, wx), wy), agx_andn1_to(b, out, wx, wy));

   CASE32(agx_or_to(b, out, wx, agx_not(b, wy)), agx_orn2_to(b, out, wx, wy));
}

TEST_F(Optimizer, FmulFsatF2F16)
{
   CASE16(
      {
         agx_index tmp = agx_temp(b->shader, AGX_SIZE_32);
         agx_fmov_to(b, tmp, agx_fmul(b, wx, wy))->saturate = true;
         agx_fmov_to(b, out, tmp);
      },
      { agx_fmul_to(b, out, wx, wy)->saturate = true; });
}

TEST_F(Optimizer, G17PPreservesStandaloneSaturate)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_index product = agx_temp(actual->shader, AGX_SIZE_32);
   agx_fmul_to(actual, product, wx, wy);
   agx_instr *sat = agx_fmov_to(actual, out, product);
   sat->saturate = true;
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_32);
   product = agx_temp(expected->shader, AGX_SIZE_32);
   agx_fmul_to(expected, product, wx, wy);
   sat = agx_fmov_to(expected, out, product);
   sat->saturate = true;
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, FsatWithPhi)
{
   /*
    * Construct the loop:
    *
    * A:
    *   ...
    *
    * B:
    *    phi ..., u
    *    u = wx * phi
    *    out = fsat u
    *    --> B
    *
    * This example shows that phi sources are read at the end of the
    * predecessor, not at the start of the successor. If phis are not handled
    * properly, the fsat would be fused incorrectly.
    *
    * This reproduces an issue hit in a Control shader. Astonishingly, it is not
    * hit anywhere in CTS.
    */
   NEGCASE32({
      agx_block *A = agx_start_block(b->shader);
      agx_block *B = agx_test_block(b->shader);

      agx_block_add_successor(A, B);
      agx_block_add_successor(B, B);

      b->cursor = agx_after_block(B);
      agx_index u = agx_temp(b->shader, AGX_SIZE_32);

      agx_instr *phi = agx_phi_to(b, agx_temp(b->shader, AGX_SIZE_32), 2);
      phi->src[0] = wx;
      phi->src[1] = u;

      agx_fmul_to(b, u, wx, phi->dest[0]);
      agx_fmov_to(b, out, u)->saturate = true;
   });
}

TEST_F(Optimizer, Copyprop)
{
   CASE32(agx_fmul_to(b, out, wx, agx_mov(b, wy)), agx_fmul_to(b, out, wx, wy));
   CASE32(agx_fmul_to(b, out, agx_mov(b, wx), agx_mov(b, wy)),
          agx_fmul_to(b, out, wx, wy));
}

TEST_F(Optimizer, SourceZeroExtend)
{
   CASE32(
      {
         agx_index t = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_to(b, t, hy);
         agx_ffs_to(b, out, t);
      },
      agx_ffs_to(b, out, hy));
}

TEST_F(Optimizer, AddSourceZeroExtend)
{
   CASE32(
      {
         agx_index t = agx_temp(b->shader, AGX_SIZE_32);
         agx_mov_to(b, t, hy);
         agx_iadd_to(b, out, wx, t, 1);
      },
      agx_iadd_to(b, out, wx, agx_abs(hy), 1));
}

TEST_F(Optimizer, AddSourceSignExtend)
{
   CASE32(
      {
         agx_index t = agx_temp(b->shader, AGX_SIZE_32);
         agx_signext_to(b, t, hy);
         agx_iadd_to(b, out, wx, t, 1);
      },
      agx_iadd_to(b, out, wx, hy, 1));
}

TEST_F(Optimizer, ProductSourceSignExtend)
{
   /* A signed 64-bit product extends the half itself. */
   CASE64(
      {
         agx_index t = agx_temp(b->shader, AGX_SIZE_32);
         agx_signext_to(b, t, hy);
         agx_imad_to(b, out, t, wx, agx_zero(), 0);
      },
      agx_imad_to(b, out, hy, wx, agx_zero(), 0));

   /* An unsigned one zero-extends its sources (abs), so the sign-extended
    * word stays: the half in its place would lose bits 16..31. */
   NEGCASE(
      {
         agx_index t = agx_temp(b->shader, AGX_SIZE_32);
         agx_signext_to(b, t, hy);
         agx_imad_to(b, out, agx_abs(t), agx_abs(wx), agx_zero(), 0);
      },
      64);

   NEGCASE32({
      agx_index t = agx_temp(b->shader, AGX_SIZE_32);
      agx_signext_to(b, t, hy);
      agx_iadd_to(b, out, wx, agx_abs(t), 0);
   });
}

TEST_F(Optimizer, SubInlineImmediate)
{
   CASE16(agx_iadd_to(b, out, hx, agx_mov_imm(b, 16, -2), 0),
          agx_iadd_to(b, out, hx, agx_neg(agx_immediate(2)), 0));

   CASE32(agx_iadd_to(b, out, wx, agx_mov_imm(b, 32, -1), 0),
          agx_iadd_to(b, out, wx, agx_neg(agx_immediate(1)), 0));

   CASE64(agx_iadd_to(b, out, dx, agx_mov_imm(b, 64, -17), 0),
          agx_iadd_to(b, out, dx, agx_neg(agx_immediate(17)), 0));

   CASE16(agx_imad_to(b, out, hx, hy, agx_mov_imm(b, 16, -2), 0),
          agx_imad_to(b, out, hx, hy, agx_neg(agx_immediate(2)), 0));

   CASE32(agx_imad_to(b, out, wx, wy, agx_mov_imm(b, 32, -1), 0),
          agx_imad_to(b, out, wx, wy, agx_neg(agx_immediate(1)), 0));

   CASE64(agx_imad_to(b, out, dx, dz, agx_mov_imm(b, 64, -17), 0),
          agx_imad_to(b, out, dx, dz, agx_neg(agx_immediate(17)), 0));
}

TEST_F(Optimizer, InlineHazards)
{
   NEGCASE32({
      agx_index zero = agx_mov_imm(b, AGX_SIZE_32, 0);
      agx_instr *I = agx_collect_to(b, out, 4);

      I->src[0] = zero;
      I->src[1] = wy;
      I->src[2] = wz;
      I->src[3] = wz;
   });
}

TEST_F(Optimizer, CopypropRespectsAbsNeg)
{
   CASE32(agx_fadd_to(b, out, agx_abs(agx_mov(b, wx)), wy),
          agx_fadd_to(b, out, agx_abs(wx), wy));

   CASE32(agx_fadd_to(b, out, agx_neg(agx_mov(b, wx)), wy),
          agx_fadd_to(b, out, agx_neg(wx), wy));

   CASE32(agx_fadd_to(b, out, agx_neg(agx_abs(agx_mov(b, wx))), wy),
          agx_fadd_to(b, out, agx_neg(agx_abs(wx)), wy));
}

TEST_F(Optimizer, IntCopyprop)
{
   CASE32(agx_xor_to(b, out, agx_mov(b, wx), wy), agx_xor_to(b, out, wx, wy));
}

TEST_F(Optimizer, CopypropSplitMovedUniform64)
{
   CASE32(
      {
         /* emit_load_preamble puts in the move, so we do too */
         agx_index mov = agx_mov(b, agx_uniform(40, AGX_SIZE_64));
         agx_instr *spl = agx_split(b, 2, mov);
         spl->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
         spl->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
         agx_xor_to(b, out, spl->dest[0], spl->dest[1]);
      },
      {
         agx_xor_to(b, out, agx_uniform(40, AGX_SIZE_32),
                    agx_uniform(42, AGX_SIZE_32));
      });
}

TEST_F(Optimizer, IntCopypropDoesntConvert)
{
   NEGCASE32({
      agx_index cvt = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, cvt, hx);
      agx_fmul_to(b, out, cvt, wy);
   });
}

TEST_F(Optimizer, SkipPreloads)
{
   NEGCASE32({
      agx_index preload = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_xor_to(b, out, preload, wy);
   });
}

TEST_F(Optimizer, NoConversionsOn16BitALU)
{
   NEGCASE16({
      agx_index cvt = agx_temp(b->shader, AGX_SIZE_16);
      agx_fmov_to(b, cvt, wx);
      agx_fadd_to(b, out, cvt, hy);
   });

   NEGCASE32(agx_fmov_to(b, out, agx_fadd(b, hx, hy)));
}

TEST_F(Optimizer, BallotCondition)
{
   CASE32(agx_ballot_to(b, out, agx_icmp(b, wx, wy, AGX_ICOND_UEQ, true)),
          agx_icmp_ballot_to(b, out, wx, wy, AGX_ICOND_UEQ, true));

   CASE32(agx_ballot_to(b, out, agx_fcmp(b, wx, wy, AGX_FCOND_GE, false)),
          agx_fcmp_ballot_to(b, out, wx, wy, AGX_FCOND_GE, false));

   CASE32(agx_quad_ballot_to(b, out, agx_icmp(b, wx, wy, AGX_ICOND_UEQ, true)),
          agx_icmp_quad_ballot_to(b, out, wx, wy, AGX_ICOND_UEQ, true));

   CASE32(agx_quad_ballot_to(b, out, agx_fcmp(b, wx, wy, AGX_FCOND_GT, false)),
          agx_fcmp_quad_ballot_to(b, out, wx, wy, AGX_FCOND_GT, false));
}

TEST_F(Optimizer, G17PBallotOfTwoRegistersKeepsTheComparison)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_ballot_to(actual, out, agx_icmp(actual, hx, hy, AGX_ICOND_UEQ, true));
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_temp(expected->shader, AGX_SIZE_16);
   agx_index opposite = agx_temp(expected->shader, AGX_SIZE_16);
   agx_icmp_to(expected, opposite, hx, hy, AGX_ICOND_UEQ, false);
   agx_icmp_ballot_to(expected, out, opposite, agx_zero(), AGX_ICOND_UEQ,
                      false);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/* Astro Bot's mesh shader: ballot(x < 6), signed. AGX3B has no immediate
 * ballot form for a signed comparison against a nonzero constant. */
TEST_F(Optimizer, G17PBallotOfSignedCompareWithConstantKeepsTheComparison)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_ballot_to(actual, out,
                 agx_icmp(actual, wx, agx_immediate(6), AGX_ICOND_SLT, false));
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_temp(expected->shader, AGX_SIZE_16);
   agx_index opposite = agx_temp(expected->shader, AGX_SIZE_16);
   agx_icmp_to(expected, opposite, wx, agx_immediate(6), AGX_ICOND_SLT, true);
   agx_icmp_ballot_to(expected, out, opposite, agx_zero(), AGX_ICOND_UEQ,
                      false);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/* Astro Bot's fragment preambles: ballot(x != 0) of a 16-bit x. A 16-bit
 * ballot source has only the equal-to-zero form on AGX3B. */
TEST_F(Optimizer, G17PBallotOfHalfNotZeroKeepsTheComparison)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_ballot_to(actual, out, agx_icmp(actual, hx, agx_zero(), AGX_ICOND_UEQ, true));
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_temp(expected->shader, AGX_SIZE_16);
   agx_index opposite = agx_temp(expected->shader, AGX_SIZE_16);
   agx_icmp_to(expected, opposite, hx, agx_zero(), AGX_ICOND_UEQ, false);
   agx_icmp_ballot_to(expected, out, opposite, agx_zero(), AGX_ICOND_UEQ,
                      false);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, G17PBallotOfHalfPredicateWidens)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_index active = agx_temp(actual->shader, AGX_SIZE_16);
   agx_get_sr_coverage_to(actual, active, AGX_SR_IS_ACTIVE_THREAD);
   agx_ballot_to(actual, out, active);
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_32);
   active = agx_temp(expected->shader, AGX_SIZE_16);
   agx_get_sr_coverage_to(expected, active, AGX_SR_IS_ACTIVE_THREAD);
   agx_index wide = agx_temp(expected->shader, AGX_SIZE_32);
   agx_mov_to(expected, wide, active);
   agx_ballot_to(expected, out, wide);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

/* Astro Bot's fragment shaders with a constant program: ballot of a 16-bit
 * value the preamble computed, so a uniform by the time the optimizer runs. */
TEST_F(Optimizer, G17PBallotOfHalfUniformWidens)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_32);
   agx_ballot_to(actual, out, agx_uniform(6, AGX_SIZE_16));
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_32);
   agx_index wide = agx_temp(expected->shader, AGX_SIZE_32);
   agx_mov_to(expected, wide, agx_uniform(6, AGX_SIZE_16));
   agx_ballot_to(expected, out, wide);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, G17PQuadBallotOfConstantIsARegister)
{
   agx_builder *actual = agx_test_builder(mem_ctx);
   agx_builder *expected = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   actual->shader->key = &key;
   expected->shader->key = &key;

   agx_index out = agx_temp(actual->shader, AGX_SIZE_16);
   agx_quad_ballot_to(actual, out, agx_immediate(1));
   agx_unit_test(actual, out);

   out = agx_temp(expected->shader, AGX_SIZE_16);
   agx_index predicate = agx_temp(expected->shader, AGX_SIZE_16);
   agx_mov_imm_to(expected, predicate, 1);
   agx_quad_ballot_to(expected, out, predicate);
   agx_unit_test(expected, out);

   agx_optimize_and_dce(actual->shader);
   ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
}

TEST_F(Optimizer, BallotMultipleUses)
{
   CASE32(
      {
         agx_index cmp = agx_fcmp(b, wx, wy, AGX_FCOND_GT, false);
         agx_index ballot = agx_quad_ballot(b, cmp);
         agx_fadd_to(b, out, cmp, ballot);
      },
      {
         agx_index cmp = agx_fcmp(b, wx, wy, AGX_FCOND_GT, false);
         agx_index ballot =
            agx_fcmp_quad_ballot(b, wx, wy, AGX_FCOND_GT, false);
         agx_fadd_to(b, out, cmp, ballot);
      });
}

/*
 * We had a bug where the ballot optimization didn't check the agx_index's type
 * so would fuse constants with overlapping values. An unrelated common code
 * change surfaced this in CTS case:
 *
 *    dEQP-VK.subgroups.vote.frag_helper.subgroupallequal_bool_fragment
 *
 * We passed Vulkan CTS without hitting it though, hence the targeted test.
 */
TEST_F(Optimizer, BallotConstant)
{
   CASE32(
      {
         agx_index cmp = agx_fcmp(b, wx, wy, AGX_FCOND_GT, false);
         agx_index ballot = agx_quad_ballot(b, agx_immediate(cmp.value));
         agx_index ballot2 = agx_quad_ballot(b, cmp);
         agx_fadd_to(b, out, ballot, agx_fadd(b, ballot2, cmp));
      },
      {
         agx_index cmp = agx_fcmp(b, wx, wy, AGX_FCOND_GT, false);
         agx_index ballot = agx_quad_ballot(b, agx_immediate(cmp.value));
         agx_index ballot2 =
            agx_fcmp_quad_ballot(b, wx, wy, AGX_FCOND_GT, false);
         agx_fadd_to(b, out, ballot, agx_fadd(b, ballot2, cmp));
      });
}

TEST_F(Optimizer, IfCondition)
{
   CASE_NO_RETURN(agx_if_icmp(b, agx_icmp(b, wx, wy, AGX_ICOND_UEQ, true),
                              agx_zero(), 1, AGX_ICOND_UEQ, true, NULL),
                  agx_if_icmp(b, wx, wy, 1, AGX_ICOND_UEQ, true, NULL));

   CASE_NO_RETURN(agx_if_icmp(b, agx_fcmp(b, wx, wy, AGX_FCOND_EQ, true),
                              agx_zero(), 1, AGX_ICOND_UEQ, true, NULL),
                  agx_if_fcmp(b, wx, wy, 1, AGX_FCOND_EQ, true, NULL));

   CASE_NO_RETURN(agx_if_icmp(b, agx_fcmp(b, hx, hy, AGX_FCOND_LT, false),
                              agx_zero(), 1, AGX_ICOND_UEQ, true, NULL),
                  agx_if_fcmp(b, hx, hy, 1, AGX_FCOND_LT, false, NULL));
}

TEST_F(Optimizer, SelectCondition)
{
   CASE32(agx_icmpsel_to(b, out, agx_icmp(b, wx, wy, AGX_ICOND_UEQ, false),
                         agx_zero(), wz, wx, AGX_ICOND_UEQ),
          agx_icmpsel_to(b, out, wx, wy, wx, wz, AGX_ICOND_UEQ));

   CASE32(agx_icmpsel_to(b, out, agx_icmp(b, wx, wy, AGX_ICOND_UEQ, true),
                         agx_zero(), wz, wx, AGX_ICOND_UEQ),
          agx_icmpsel_to(b, out, wx, wy, wz, wx, AGX_ICOND_UEQ));

   CASE32(agx_icmpsel_to(b, out, agx_fcmp(b, wx, wy, AGX_FCOND_EQ, false),
                         agx_zero(), wz, wx, AGX_ICOND_UEQ),
          agx_fcmpsel_to(b, out, wx, wy, wx, wz, AGX_FCOND_EQ));

   CASE32(agx_icmpsel_to(b, out, agx_fcmp(b, wx, wy, AGX_FCOND_LT, true),
                         agx_zero(), wz, wx, AGX_ICOND_UEQ),
          agx_fcmpsel_to(b, out, wx, wy, wz, wx, AGX_FCOND_LT));
}

TEST_F(Optimizer, IfInverted)
{
   CASE_NO_RETURN(
      agx_if_icmp(b, agx_xor(b, hx, agx_immediate(1)), agx_zero(), 1,
                  AGX_ICOND_UEQ, true, NULL),
      agx_if_icmp(b, hx, agx_zero(), 1, AGX_ICOND_UEQ, false, NULL));

   CASE_NO_RETURN(agx_if_icmp(b, agx_xor(b, hx, agx_immediate(1)), agx_zero(),
                              1, AGX_ICOND_UEQ, false, NULL),
                  agx_if_icmp(b, hx, agx_zero(), 1, AGX_ICOND_UEQ, true, NULL));
}

TEST_F(Optimizer, IfInvertedCondition)
{
   CASE_NO_RETURN(
      agx_if_icmp(
         b,
         agx_xor(b, agx_icmp(b, wx, wy, AGX_ICOND_UEQ, true), agx_immediate(1)),
         agx_zero(), 1, AGX_ICOND_UEQ, true, NULL),
      agx_if_icmp(b, wx, wy, 1, AGX_ICOND_UEQ, false, NULL));

   CASE_NO_RETURN(
      agx_if_icmp(
         b,
         agx_xor(b, agx_fcmp(b, wx, wy, AGX_FCOND_EQ, true), agx_immediate(1)),
         agx_zero(), 1, AGX_ICOND_UEQ, true, NULL),
      agx_if_fcmp(b, wx, wy, 1, AGX_FCOND_EQ, false, NULL));

   CASE_NO_RETURN(
      agx_if_icmp(
         b,
         agx_xor(b, agx_fcmp(b, hx, hy, AGX_FCOND_LT, false), agx_immediate(1)),
         agx_zero(), 1, AGX_ICOND_UEQ, true, NULL),
      agx_if_fcmp(b, hx, hy, 1, AGX_FCOND_LT, true, NULL));
}

TEST_F(Optimizer, G17PTileSampleConstantsStayInRegisters)
{
   /* The packer's existing G17PMultisampleTileSampleOperandIsARegister
    * regression accepts registers and rejects these raw mask immediates.
    * Optimizing a valid MOV+store must not recreate a refused operand.
    */
   for (unsigned mask : {0u, 1u, 3u, 15u, 32u, 64u}) {
      SCOPED_TRACE(mask);
      agx_builder *b = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      agx_index coverage = agx_mov_imm(b, 16, mask);
      agx_instr *store = agx_st_tile(b, hx, coverage, agx_null(),
                                    AGX_FORMAT_I16, 1, 0, false, 0);
      agx_optimize_and_dce(b->shader);
      EXPECT_EQ(store->src[1].type, AGX_INDEX_NORMAL);
      EXPECT_EQ(store->src[1].size, AGX_SIZE_16);
      EXPECT_EQ(store->src[1].value, coverage.value);
      unsigned moves = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_MOV_IMM) {
            EXPECT_EQ(I->imm, mask);
            EXPECT_EQ(I->dest[0].value, coverage.value);
            moves++;
         }
      }
      EXPECT_EQ(moves, 1u);
   }
}

TEST_F(Optimizer, TileSampleImmediateControls)
{
   for (enum agx_isa_encoding enc : {AGX_ISA_ENCODING_AGX2,
                                      AGX_ISA_ENCODING_AGX3B,
                                      AGX_ISA_ENCODING_AGX3C}) {
      for (unsigned mask : {0u, 255u}) {
         if (enc == AGX_ISA_ENCODING_AGX3B && mask == 0)
            continue;
         SCOPED_TRACE(unsigned(enc));
         SCOPED_TRACE(mask);
         agx_builder *b = agx_test_builder(mem_ctx);
         struct agx_shader_key key = {};
         key.dev.encoding = enc;
         b->shader->key = &key;
         agx_instr *store = agx_st_tile(
            b, hx, agx_mov_imm(b, 16, mask), agx_null(),
            AGX_FORMAT_I16, 1, 0, false, 0);
         agx_optimize_and_dce(b->shader);
         EXPECT_EQ(store->src[1].type, AGX_INDEX_IMMEDIATE);
         EXPECT_EQ(store->src[1].value, mask);
      }
   }
}

TEST_F(Optimizer, G17PAbsorbsExclusiveSelectGuards)
{
   for (enum agx_size size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (unsigned target = 0; target < 8; ++target) {
         for (unsigned guard = 0; guard < 8; ++guard) {
            for (bool reverse : {false, true}) {
               agx_builder *actual = agx_test_builder(mem_ctx);
               agx_builder *expected = agx_test_builder(mem_ctx);
               struct agx_shader_key key = {};
               key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
               for (agx_builder *b : {actual, expected}) {
                  b->shader->key = &key;
                  agx_index variable = agx_preload(b, agx_register(0, size));
                  agx_index predicate = agx_icmp(
                     b, variable, agx_immediate(guard), AGX_ICOND_UEQ, false);
                  agx_index inner = agx_temp(b->shader, AGX_SIZE_32);
                  agx_icmpsel_to(b, inner, variable, agx_immediate(target), wy,
                                 wz, AGX_ICOND_UEQ);
                  agx_index out = agx_temp(b->shader, AGX_SIZE_32);
                  if (b == expected && guard == target) {
                     agx_unit_test(b, wz);
                     continue;
                  }
                  if (b == expected)
                     agx_icmpsel_to(b, out, variable, agx_immediate(target), wy,
                                    wz, AGX_ICOND_UEQ);
                  else if (b == expected && !reverse)
                     agx_icmpsel_to(b, out, variable, agx_immediate(guard), wz,
                                    inner, AGX_ICOND_UEQ);
                  else
                     agx_icmpsel_to(b, out, predicate, agx_zero(),
                                    reverse ? wz : inner, reverse ? inner : wz,
                                    reverse ? AGX_ICOND_UGT : AGX_ICOND_UEQ);
                  agx_unit_test(b, out);
               }
               agx_optimize_and_dce(actual->shader);
               agx_dce(expected->shader, true);
               ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
            }
         }
      }
   }
}

TEST_F(Optimizer, G17PAbsorbsCompoundSelectGuards)
{
   for (unsigned first = 0; first < 8; ++first) {
      for (unsigned second = 0; second < 8; ++second) {
         agx_builder *actual = agx_test_builder(mem_ctx);
         agx_builder *expected = agx_test_builder(mem_ctx);
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         for (agx_builder *b : {actual, expected}) {
            b->shader->key = &key;
            agx_index variable = agx_preload(b, wx);
            agx_index predicate = agx_temp(b->shader, AGX_SIZE_16);
            agx_icmp_to(b, predicate, variable, agx_immediate(first),
                        AGX_ICOND_UEQ, false);
            agx_index compound = agx_temp(b->shader, AGX_SIZE_16);
            agx_icmpsel_to(b, compound, variable, agx_immediate(second),
                           agx_immediate(1), predicate, AGX_ICOND_UEQ);
            agx_index inner = agx_temp(b->shader, AGX_SIZE_32);
            agx_icmpsel_to(b, inner, variable, agx_immediate(7), wy, wz,
                           AGX_ICOND_UEQ);
            agx_index out = agx_temp(b->shader, AGX_SIZE_32);
            if (b == expected && (first == 7 || second == 7)) {
               agx_unit_test(b, wz);
               continue;
            }
            if (b == expected)
               agx_icmpsel_to(b, out, variable, agx_immediate(7), wy, wz,
                              AGX_ICOND_UEQ);
            else
               agx_icmpsel_to(b, out, compound, agx_zero(), inner, wz,
                              AGX_ICOND_UEQ);
            agx_unit_test(b, out);
         }
         agx_optimize_and_dce(actual->shader);
         agx_dce(expected->shader, true);
         ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
      }
   }
}

TEST_F(Optimizer, G17PAbsorbsRangeSelectGuards)
{
   for (enum agx_icond cond :
        {AGX_ICOND_ULT, AGX_ICOND_UGT, AGX_ICOND_SLT, AGX_ICOND_SGT}) {
      for (unsigned target = 0; target < 8; ++target) {
         for (unsigned threshold = 0; threshold < 8; ++threshold) {
            const bool guarded =
               (cond == AGX_ICOND_ULT || cond == AGX_ICOND_SLT)
                  ? target < threshold
                  : target > threshold;
            agx_builder *actual = agx_test_builder(mem_ctx);
            agx_builder *expected = agx_test_builder(mem_ctx);
            struct agx_shader_key key = {};
            key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
            for (agx_builder *b : {actual, expected}) {
               b->shader->key = &key;
               agx_index variable = agx_preload(b, wx);
               agx_index inner = agx_temp(b->shader, AGX_SIZE_32);
               agx_icmpsel_to(b, inner, agx_immediate(target), variable, wy, wz,
                              AGX_ICOND_UEQ);
               agx_index out = agx_temp(b->shader, AGX_SIZE_32);
               if (b == expected && guarded) {
                  agx_unit_test(b, wz);
                  continue;
               }
               if (b == expected)
                  agx_icmpsel_to(b, out, variable, agx_immediate(target), wy,
                                 wz, AGX_ICOND_UEQ);
               else
                  agx_icmpsel_to(b, out, variable, agx_immediate(threshold), wz,
                                 inner, cond);
               agx_unit_test(b, out);
            }
            agx_optimize_and_dce(actual->shader);
            agx_dce(expected->shader, true);
            ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
         }
      }
   }
}

static uint32_t
select_test_run(agx_context *ctx, uint32_t input)
{
   std::vector<uint32_t> values(ctx->alloc);
   auto read = [&](agx_index index) {
      if (index.type == AGX_INDEX_IMMEDIATE)
         return index.value;
      if (index.type == AGX_INDEX_REGISTER)
         return index.value == 2 ? 0x12345678u : 0xabcdef01u;
      return values[index.value];
   };
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_PRELOAD) {
         values[I->dest[0].value] = input;
      } else if (I->op == AGX_OPCODE_ICMPSEL || I->op == AGX_OPCODE_ICMP) {
         uint32_t x = read(I->src[0]), y = read(I->src[1]);
         const bool half =
            I->src[0].size == AGX_SIZE_16 && I->src[1].size == AGX_SIZE_16;
         int64_t sx = half ? (int16_t)x : (int32_t)x;
         int64_t sy = half ? (int16_t)y : (int32_t)y;
         bool result = false;
         switch (I->icond) {
         case AGX_ICOND_UEQ:
         case AGX_ICOND_SEQ:
            result = x == y;
            break;
         case AGX_ICOND_ULT:
            result = x < y;
            break;
         case AGX_ICOND_UGT:
            result = x > y;
            break;
         case AGX_ICOND_SLT:
            result = sx < sy;
            break;
         case AGX_ICOND_SGT:
            result = sx > sy;
            break;
         default:
            ADD_FAILURE();
            break;
         }
         values[I->dest[0].value] = I->op == AGX_OPCODE_ICMP
                                       ? result != I->invert_cond
                                       : read(I->src[result ? 2 : 3]);
      } else if (I->op == AGX_OPCODE_UNIT_TEST) {
         return read(I->src[0]);
      } else {
         ADD_FAILURE() << I->op;
         return 0;
      }
   }
   ADD_FAILURE();
   return 0;
}

TEST_F(Optimizer, G17PRangeSelectImmediateBoundary)
{
   for (enum agx_size size : {AGX_SIZE_16, AGX_SIZE_32}) {
      agx_builder *before = agx_test_builder(mem_ctx);
      agx_builder *after = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      for (agx_builder *b : {before, after}) {
         b->shader->key = &key;
         agx_index variable = agx_preload(b, agx_register(0, size));
         agx_index inner = agx_temp(b->shader, AGX_SIZE_16);
         agx_icmpsel_to(b, inner, variable, agx_immediate(255),
                        agx_immediate(0), agx_immediate(1), AGX_ICOND_UEQ);
         agx_index out = agx_temp(b->shader, AGX_SIZE_16);
         agx_icmpsel_to(b, out, variable, agx_immediate(255),
                        agx_immediate(0), inner, AGX_ICOND_ULT);
         agx_unit_test(b, out);
      }
      agx_optimize_and_dce(after->shader);
      agx_foreach_instr_global(after->shader, I) {
         agx_foreach_src(I, s) {
            if (I->src[s].type == AGX_INDEX_IMMEDIATE) {
               ASSERT_LT(I->src[s].value, 256u);
            }
         }
      }
      for (uint32_t input : {0u, 254u, 255u, 256u, 257u, 65535u})
         ASSERT_EQ(select_test_run(before->shader, input),
                   select_test_run(after->shader, input)) << input;
   }
}

TEST_F(Optimizer, G17PRangeSelectSemantics)
{
   const enum agx_icond conditions[] = {
      AGX_ICOND_UEQ, AGX_ICOND_ULT, AGX_ICOND_UGT,
      AGX_ICOND_SEQ, AGX_ICOND_SLT, AGX_ICOND_SGT,
   };
   for (enum agx_size size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (enum agx_icond guard : conditions) {
         for (enum agx_icond inner_cond : conditions) {
            for (unsigned threshold : {6u, 0x7fffu, 0x8000u, 0xffffu}) {
               agx_builder *before = agx_test_builder(mem_ctx);
               agx_builder *after = agx_test_builder(mem_ctx);
               struct agx_shader_key key = {};
               key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
               for (agx_builder *b : {before, after}) {
                  b->shader->key = &key;
                  agx_index variable = agx_preload(b, agx_register(0, size));
                  agx_index predicate = agx_temp(b->shader, AGX_SIZE_16);
                  agx_icmp_to(b, predicate, variable, agx_immediate(threshold),
                              guard, false);
                  agx_index compound = agx_temp(b->shader, AGX_SIZE_16);
                  agx_icmpsel_to(b, compound, variable,
                                 agx_immediate(threshold - 2), agx_immediate(1),
                                 predicate, guard);
                  agx_index inner = agx_temp(b->shader, AGX_SIZE_32);
                  agx_icmpsel_to(b, inner, variable, agx_immediate(7), wy, wz,
                                 inner_cond);
                  agx_index out = agx_temp(b->shader, AGX_SIZE_32);
                  agx_icmpsel_to(b, out, compound, agx_zero(), inner, wz,
                                 AGX_ICOND_UEQ);
                  agx_unit_test(b, out);
               }
               agx_optimize_and_dce(after->shader);
               if (threshold == 6 && guard == AGX_ICOND_ULT &&
                   inner_cond == AGX_ICOND_ULT) {
                  unsigned selects = 0;
                  agx_foreach_instr_global(after->shader, I)
                     selects += I->op == AGX_OPCODE_ICMPSEL;
                  ASSERT_EQ(selects, 1u);
               }
               for (uint32_t input = 0; input < 0x10000; ++input)
                  ASSERT_EQ(select_test_run(before->shader, input),
                            select_test_run(after->shader, input))
                     << input;
               if (size == AGX_SIZE_32) {
                  for (uint32_t input : {0x10000u, 0x7fffffffu, 0x80000000u,
                                         0xfffffffeu, 0xffffffffu})
                     ASSERT_EQ(select_test_run(before->shader, input),
                               select_test_run(after->shader, input))
                        << input;
               }
            }
         }
      }
   }
}

TEST_F(Optimizer, G17PFusesConstantValuedSelectPredicates)
{
   for (unsigned yes : {0u, 1u, 255u, 65535u}) {
      for (unsigned no : {0u, 1u, 255u, 65535u}) {
         agx_builder *actual = agx_test_builder(mem_ctx);
         agx_builder *expected = agx_test_builder(mem_ctx);
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         for (agx_builder *b : {actual, expected}) {
            b->shader->key = &key;
            agx_index variable = agx_preload(b, wx);
            agx_index predicate = agx_temp(b->shader, AGX_SIZE_16);
            agx_icmpsel_to(b, predicate, variable, agx_immediate(7),
                           agx_immediate(yes), agx_immediate(no),
                           AGX_ICOND_ULT);
            agx_index out = agx_temp(b->shader, AGX_SIZE_32);
            if (b == expected && (yes == 0) == (no == 0)) {
               agx_unit_test(b, yes == 0 ? wy : wz);
               continue;
            }
            if (b == expected)
               agx_icmpsel_to(b, out, variable, agx_immediate(7),
                              yes == 0 ? wy : wz, no == 0 ? wy : wz,
                              AGX_ICOND_ULT);
            else
               agx_icmpsel_to(b, out, predicate, agx_zero(), wy, wz,
                              AGX_ICOND_UEQ);
            agx_unit_test(b, out);
         }
         agx_optimize_and_dce(actual->shader);
         agx_dce(expected->shader, true);
         ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
      }
   }
}

TEST_F(Optimizer, G17POuterInputProvesCompoundSelectGuard)
{
   for (enum agx_size size : {AGX_SIZE_16, AGX_SIZE_32}) {
      agx_builder *actual = agx_test_builder(mem_ctx);
      agx_builder *expected = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      for (agx_builder *b : {actual, expected}) {
         b->shader->key = &key;
         agx_index variable = agx_preload(b, agx_register(0, size));
         agx_index upper = agx_temp(b->shader, AGX_SIZE_16);
         agx_icmp_to(b, upper, variable, agx_immediate(6), AGX_ICOND_SLT,
                     false);
         agx_index interval = agx_temp(b->shader, AGX_SIZE_16);
         agx_icmpsel_to(b, interval, variable, agx_immediate(4), agx_zero(),
                        upper, AGX_ICOND_SLT);
         agx_index inner = agx_temp(b->shader, AGX_SIZE_32);
         agx_icmpsel_to(b, inner, interval, agx_zero(), wz, wy, AGX_ICOND_UEQ);
         agx_index out = agx_temp(b->shader, AGX_SIZE_32);
         if (b == expected)
            agx_icmpsel_to(b, out, variable, agx_immediate(5), wy, wz,
                           AGX_ICOND_UEQ);
         else
            agx_icmpsel_to(b, out, variable, agx_immediate(5), wz, inner,
                           AGX_ICOND_SLT);
         agx_unit_test(b, out);
      }
      agx_optimize_and_dce(actual->shader);
      agx_dce(expected->shader, true);
      ASSERT_SHADER_EQUAL(actual->shader, expected->shader);
   }
}

TEST_F(Optimizer, G17PSelectGuardImmediateBoundary)
{
   for (enum agx_size size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (unsigned threshold : {254u, 255u}) {
         agx_builder *b = agx_test_builder(mem_ctx);
         struct agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         b->shader->key = &key;
         agx_index variable = agx_preload(b, agx_register(0, size));
         agx_index predicate = agx_icmp(
            b, variable, agx_immediate(threshold), AGX_ICOND_UGT, false);
         agx_index out = agx_temp(b->shader, AGX_SIZE_16);
         agx_icmpsel_to(b, out, variable, agx_immediate(threshold), predicate,
                        agx_zero(), AGX_ICOND_UGT);
         agx_unit_test(b, out);
         agx_optimize_and_dce(b->shader);
         agx_foreach_instr_global(b->shader, I) {
            agx_foreach_src(I, s) {
               if (I->src[s].type == AGX_INDEX_IMMEDIATE) {
                  EXPECT_LT(I->src[s].value, 0x100u);
               }
            }
         }
         for (uint32_t input = threshold - 1; input <= threshold + 2; ++input)
            EXPECT_EQ(select_test_run(b->shader, input), input > threshold);
      }
   }
}

TEST_F(Optimizer, G17PCompoundInnerGuardSemantics)
{
   const enum agx_icond conditions[] = {
      AGX_ICOND_UEQ, AGX_ICOND_ULT, AGX_ICOND_UGT,
      AGX_ICOND_SEQ, AGX_ICOND_SLT, AGX_ICOND_SGT,
   };
   for (enum agx_size size : {AGX_SIZE_16, AGX_SIZE_32}) {
      for (enum agx_icond guard : conditions) {
         for (enum agx_icond inner_cond : conditions) {
            for (unsigned threshold : {6u, 0x7fffu, 0x8000u, 0xffffu}) {
               agx_builder *before = agx_test_builder(mem_ctx);
               agx_builder *after = agx_test_builder(mem_ctx);
               struct agx_shader_key key = {};
               key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
               for (agx_builder *b : {before, after}) {
                  b->shader->key = &key;
                  agx_index variable = agx_preload(b, agx_register(0, size));
                  agx_index predicate = agx_temp(b->shader, AGX_SIZE_16);
                  agx_icmp_to(b, predicate, variable, agx_immediate(threshold),
                              inner_cond, false);
                  agx_index compound = agx_temp(b->shader, AGX_SIZE_16);
                  agx_icmpsel_to(b, compound, variable,
                                 agx_immediate(threshold - 2), agx_zero(),
                                 predicate, inner_cond);
                  agx_index inner = agx_temp(b->shader, AGX_SIZE_32);
                  agx_icmpsel_to(b, inner, compound, agx_zero(), wz, wy,
                                 AGX_ICOND_UEQ);
                  agx_index out = agx_temp(b->shader, AGX_SIZE_32);
                  agx_icmpsel_to(b, out, variable, agx_immediate(7), wz, inner,
                                 guard);
                  agx_unit_test(b, out);
               }
               agx_optimize_and_dce(after->shader);
               for (uint32_t input = 0; input < 0x10000; ++input)
                  ASSERT_EQ(select_test_run(before->shader, input),
                            select_test_run(after->shader, input))
                     << input;
               if (size == AGX_SIZE_32) {
                  for (uint32_t input : {0x10000u, 0x7fffffffu, 0x80000000u,
                                         0xfffffffeu, 0xffffffffu})
                     ASSERT_EQ(select_test_run(before->shader, input),
                               select_test_run(after->shader, input))
                        << input;
               }
            }
         }
      }
   }
}

TEST_F(Optimizer, Agx3bFoldsUnmodifiedHalfGprIntoFfma)
{
   struct Case {
      const char *name;
      enum agx_isa_encoding enc;
      bool emit_agx3d;
      enum agx_opcode op;
      bool saturate;
      bool src_neg;
      bool has_imm;
      bool has_uniform;
      bool expect_folded;
   };
   const Case cases[] = {
      {"agx3b_ffma_plain", AGX_ISA_ENCODING_AGX3B, false, AGX_OPCODE_FFMA,
       false, false, false, false, true},
      {"agx3b_fadd_blocked", AGX_ISA_ENCODING_AGX3B, false, AGX_OPCODE_FADD,
       false, false, false, false, false},
      {"agx3b_fmul_blocked", AGX_ISA_ENCODING_AGX3B, false, AGX_OPCODE_FMUL,
       false, false, false, false, false},
      {"agx3b_ffma_saturate_blocked", AGX_ISA_ENCODING_AGX3B, false,
       AGX_OPCODE_FFMA, true, false, false, false, false},
      {"agx3b_ffma_neg_blocked", AGX_ISA_ENCODING_AGX3B, false, AGX_OPCODE_FFMA,
       false, true, false, false, false},
      {"agx3b_ffma_imm_blocked", AGX_ISA_ENCODING_AGX3B, false, AGX_OPCODE_FFMA,
       false, false, true, false, false},
      {"agx3b_ffma_uniform_blocked", AGX_ISA_ENCODING_AGX3B, false,
       AGX_OPCODE_FFMA, false, false, false, true, false},
      {"g18g_ffma_blocked", AGX_ISA_ENCODING_AGX3C, true, AGX_OPCODE_FFMA,
       false, false, false, false, false},
   };

   for (const Case &c : cases) {
      agx_builder *b = agx_test_builder(mem_ctx);
      struct agx_shader_key key = {};
      key.dev.encoding = c.enc;
      key.dev.emit_agx3d = c.emit_agx3d;
      b->shader->key = &key;

      agx_index half = agx_preload(b, hx);
      agx_index widened = agx_temp(b->shader, AGX_SIZE_32);
      agx_fmov_to(b, widened, half);

      agx_index s0 = widened;
      if (c.src_neg)
         s0 = agx_neg(s0);
      agx_index s1 = c.has_imm       ? agx_immediate_f(1.0f)
                     : c.has_uniform ? agx_uniform(0, AGX_SIZE_32)
                                     : agx_preload(b, wy);
      agx_index s2 = agx_preload(b, wz);
      agx_index out = agx_temp(b->shader, AGX_SIZE_32);

      agx_instr *I;
      if (c.op == AGX_OPCODE_FADD)
         I = agx_fadd_to(b, out, s0, s1);
      else if (c.op == AGX_OPCODE_FMUL)
         I = agx_fmul_to(b, out, s0, s1);
      else
         I = agx_ffma_to(b, out, s0, s1, s2);
      I->saturate = c.saturate;
      agx_unit_test(b, out);

      agx_optimize_and_dce(b->shader);

      EXPECT_EQ(I->src[0].size, c.expect_folded ? AGX_SIZE_16 : AGX_SIZE_32)
         << c.name;
   }
}

/* A constant index folds into a device load only while the index at its
 * scale plus the load's byte displacement is still a signed 16-bit offset.
 */
static enum agx_index_type
device_load_index_type(void *mem_ctx, uint32_t index, int16_t displacement)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   b->shader->key = &key;

   agx_index idx = agx_mov_imm(b, 32, index);
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *load = agx_device_load_to(b, out, agx_register(0, AGX_SIZE_64),
                                        idx, AGX_FORMAT_I32, 1, 0, false);
   load->memory_byte_offset = displacement;
   agx_unit_test(b, out);

   agx_optimizer_forward(b->shader);
   return load->src[1].type;
}

TEST_F(Optimizer, DeviceLoadImmediateIndexKeepsItsDisplacementEncodable)
{
   /* 32764 + (8 << 2) = 32796 does not fit, so the index stays a register. */
   EXPECT_EQ(device_load_index_type(mem_ctx, 8, 32764), AGX_INDEX_NORMAL);
   EXPECT_EQ(device_load_index_type(mem_ctx, 8, -32768), AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(device_load_index_type(mem_ctx, 0, 32764), AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(device_load_index_type(mem_ctx, 8, 0), AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(device_load_index_type(mem_ctx, 8, 32735), AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(device_load_index_type(mem_ctx, 8, 32736), AGX_INDEX_NORMAL);
}
