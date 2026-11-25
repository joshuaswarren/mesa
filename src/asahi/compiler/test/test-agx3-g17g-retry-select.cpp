/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

/*
 * The second M5 attempt (agx_lower_g17g_retry_float_operands) computes every
 * half-destination select at 32 bits and narrows the result. Its selected
 * operands must be widened with it, including a half uniform such as a
 * constant-program export: left in place, it was a 16-bit value selected into
 * a 32-bit destination, which the packer refused as
 * `fcmpsel d32, s0=32, s1=32, s2=32, s3=16`.
 */

static agx_instr *
def_of(agx_context *ctx, agx_index value)
{
   agx_foreach_instr_global(ctx, I) {
      agx_foreach_ssa_dest(I, d) {
         if (I->dest[d].value == value.value)
            return I;
      }
   }
   return NULL;
}

static agx_instr *
only_select(agx_context *ctx)
{
   agx_instr *select = NULL;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_FCMPSEL) {
         EXPECT_EQ(select, nullptr);
         select = I;
      }
   }
   return select;
}

class AGX3G17GRetrySelect : public testing::Test {
 protected:
   AGX3G17GRetrySelect()
   {
      mem = ralloc_context(NULL);
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->agx3_reg_float_consts = true;
   }

   ~AGX3G17GRetrySelect()
   {
      ralloc_free(mem);
   }

   /* Every selected register or uniform has the destination's width. */
   void expect_selected_width(const agx_instr *I)
   {
      for (unsigned s = 2; s < 4; ++s) {
         if (I->src[s].type != AGX_INDEX_IMMEDIATE)
            EXPECT_EQ(I->src[s].size, I->dest[0].size) << "source " << s;
      }
   }

   /* The selected value is the zero-extended half of uniform @u. */
   void expect_widened_uniform(agx_index selected, unsigned u)
   {
      ASSERT_EQ(selected.type, AGX_INDEX_NORMAL);
      ASSERT_EQ(selected.size, AGX_SIZE_32);
      agx_instr *wide = def_of(b->shader, selected);
      ASSERT_NE(wide, nullptr);
      ASSERT_EQ(wide->op, AGX_OPCODE_MOV);
      const agx_index half = wide->src[0];
      ASSERT_EQ(half.type, AGX_INDEX_NORMAL);
      ASSERT_EQ(half.size, AGX_SIZE_16);
      agx_instr *copy = def_of(b->shader, half);
      ASSERT_NE(copy, nullptr);
      ASSERT_EQ(copy->op, AGX_OPCODE_MOV);
      EXPECT_TRUE(agx_is_equiv(copy->src[0], agx_uniform(u, AGX_SIZE_16)));
      EXPECT_EQ(copy->src[0].size, AGX_SIZE_16);
   }

   /* The original half destination is the low half of the 32-bit result. */
   void expect_narrowed(const agx_instr *I, agx_index narrow)
   {
      EXPECT_EQ(I->dest[0].size, AGX_SIZE_32);
      bool found = false;
      agx_foreach_instr_global(b->shader, J) {
         if (J->op == AGX_OPCODE_SPLIT && J->nr_srcs == 1 &&
             agx_is_equiv(J->src[0], I->dest[0])) {
            EXPECT_TRUE(agx_is_equiv(J->dest[0], narrow));
            found = true;
         }
      }
      EXPECT_TRUE(found);
   }

   void *mem;
   agx_shader_key key = {};
   agx_builder *b;
};

TEST_F(AGX3G17GRetrySelect, HalfUniformAlternativeOfAHalfSelect)
{
   /* d16 = (a < c) ? x : u14l, all halves */
   const agx_index a = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index c = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index x = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index u = agx_uniform(14, AGX_SIZE_16);
   const agx_index d = agx_temp(b->shader, AGX_SIZE_16);
   agx_fcmpsel_to(b, d, a, c, x, u, AGX_FCOND_LT);

   EXPECT_TRUE(agx_lower_g17g_retry_float_operands(b->shader));

   agx_instr *I = only_select(b->shader);
   ASSERT_NE(I, nullptr);
   EXPECT_EQ(I->src[0].size, AGX_SIZE_32);
   EXPECT_EQ(I->src[1].size, AGX_SIZE_32);
   expect_selected_width(I);
   expect_widened_uniform(I->src[3], 14);
   expect_narrowed(I, d);
}

TEST_F(AGX3G17GRetrySelect, TiedHalfMaximumAgainstAUniform)
{
   /* d16 = max(x, u15h): the tied select of fmax(x, u). */
   const agx_index x = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index u = agx_uniform(15, AGX_SIZE_16);
   const agx_index d = agx_temp(b->shader, AGX_SIZE_16);
   agx_fcmpsel_to(b, d, x, u, x, u, AGX_FCOND_GTN);

   EXPECT_TRUE(agx_lower_g17g_retry_float_operands(b->shader));

   /* GTN is the negation of LE, with the selected values exchanged. */
   agx_instr *I = only_select(b->shader);
   ASSERT_NE(I, nullptr);
   EXPECT_EQ(I->fcond, AGX_FCOND_LE);
   expect_selected_width(I);
   expect_widened_uniform(I->src[2], 15);
   expect_narrowed(I, d);
}

TEST_F(AGX3G17GRetrySelect, FirstAttemptKeepsTheHalfSelect)
{
   b->shader->agx3_reg_float_consts = false;
   const agx_index a = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index c = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index x = agx_temp(b->shader, AGX_SIZE_16);
   const agx_index d = agx_temp(b->shader, AGX_SIZE_16);
   agx_fcmpsel_to(b, d, a, c, x, agx_uniform(14, AGX_SIZE_16),
                  AGX_FCOND_LT);

   EXPECT_FALSE(agx_lower_g17g_retry_float_operands(b->shader));

   agx_instr *I = only_select(b->shader);
   ASSERT_NE(I, nullptr);
   EXPECT_EQ(I->dest[0].size, AGX_SIZE_16);
   EXPECT_EQ(I->src[3].type, AGX_INDEX_UNIFORM);
   expect_selected_width(I);
}
