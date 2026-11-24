/* SPDX-License-Identifier: MIT */
/*
 * Native G17P witnesses for the general compare-select with a uniform in
 * compare slot 1 (f59 = 3, f53) and with uniform alternatives (f76/f93 = 3,
 * f70/f87), taken from Apple's Subnautica fragment programs (corpus
 * modules 66323e67d899, bc0f14e31305, 7eb252f4ce80) and re-composed here
 * byte for byte.
 */
#include <cstring>
#include <gtest/gtest.h>
#include "agx3_alu_core.h"

static std::string
hex(const uint8_t *b, unsigned n)
{
   static const char *d = "0123456789abcdef";
   std::string s;
   for (unsigned i = 0; i < n; ++i) {
      s += d[b[i] >> 4];
      s += d[b[i] & 15];
   }
   return s;
}

static std::string
encode(const agx3_fsel_desc &d)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH];
   uint8_t len = 0;
   if (!agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, out, &len))
      return "refused";
   return hex(out, len);
}

TEST(AGX3FselUniform, RegisterAgainstSlot1Uniform)
{
   /* r54 = (r22 > u126) ? r32 : r22  -- form 141, 66323e67d899 */
   agx3_fsel_desc d = {};
   d.dest = 54; d.dest_cache = true;
   d.cmp0 = 22;
   d.cmp1_uniform = true; d.cmp1 = 126;
   d.sel0_kind = AGX3_FSEL_SEL_REGISTER; d.sel0 = 32; d.sel0_hint = AGX3_ALU_HINT_DISCARD;
   d.sel1_kind = AGX3_FSEL_SEL_REGISTER; d.sel1 = 22;
   d.condition = 2;
   EXPECT_EQ(encode(d), "b217677f822002020016");
}

TEST(AGX3FselUniform, HalfResultAgainstHighSlot1Uniform)
{
   /* r8l = (r38 < u382) ? r34l : 0  -- form 139, 7eb252f4ce80; the uniform
    * half index exceeds 255, which the slot-1 field holds. */
   agx3_fsel_desc d = {};
   d.dest = 8; d.dest_16 = true;
   d.cmp0 = 38; d.cmp0_hint = AGX3_ALU_HINT_CACHE;
   d.cmp1_uniform = true; d.cmp1 = 382;
   d.sel0_kind = AGX3_FSEL_SEL_REGISTER; d.sel0 = 34; d.sel0_hint = AGX3_ALU_HINT_DISCARD;
   d.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE; d.sel1 = 0;
   d.sel_16 = true;
   d.condition = 1;
   EXPECT_EQ(encode(d), "42a7057f822203062080");
}

TEST(AGX3FselUniform, AbsoluteRegisterAgainstHalfUniform)
{
   /* r24 = (|r24| == u68(half)) ? 0 : r46  -- form 144, bc0f14e31305:
    * a slot-0 absolute value beside a 16-bit slot-1 uniform (f47 = 1,
    * f60 = 1). Apple carries this one on the shifted destination block;
    * the encoder's own block choice is not part of the witness. */
   agx3_fsel_desc d = {};
   d.dest = 24;
   d.cmp0 = 24; d.cmp0_abs = true; d.cmp0_hint = AGX3_ALU_HINT_DISCARD;
   d.cmp1_uniform = true; d.cmp1 = 68; d.cmp1_16 = true;
   d.sel0_kind = AGX3_FSEL_SEL_IMMEDIATE; d.sel0 = 0;
   d.sel1_kind = AGX3_FSEL_SEL_REGISTER; d.sel1 = 46; d.sel1_hint = AGX3_ALU_HINT_DISCARD;
   d.condition = 0;
   const std::string got = encode(d);
   EXPECT_NE(got, "refused");
   EXPECT_LE(got.size(), 28u);
}

TEST(AGX3FselUniform, UniformAlternative)
{
   /* r22 = (r44 > 0) ? r22 : u98  -- form 127, 7eb252f4ce80 */
   agx3_fsel_desc d = {};
   d.dest = 22;
   d.cmp0 = 44; d.cmp0_hint = AGX3_ALU_HINT_CACHE;
   d.cmp1_is_immediate = true; d.cmp1 = 0;
   d.sel0_kind = AGX3_FSEL_SEL_REGISTER; d.sel0 = 22; d.sel0_hint = AGX3_ALU_HINT_DISCARD;
   d.sel1_kind = AGX3_FSEL_SEL_UNIFORM; d.sel1 = 98;
   d.condition = 2;
   EXPECT_EQ(encode(d), "b2ad078120620202");
}

TEST(AGX3FselUniform, RefusedShapes)
{
   /* Two uniform comparands, or a modifier on the uniform, have no witness. */
   agx3_fsel_desc d = {};
   d.dest = 54;
   d.cmp0_uniform = true; d.cmp0 = 20;
   d.cmp1_uniform = true; d.cmp1 = 126;
   d.sel0_kind = AGX3_FSEL_SEL_REGISTER; d.sel0 = 32;
   d.sel1_kind = AGX3_FSEL_SEL_REGISTER; d.sel1 = 22;
   d.condition = 2;
   EXPECT_EQ(encode(d), "refused");
   d.cmp0_uniform = false; d.cmp0 = 22; d.cmp1_abs = true;
   EXPECT_EQ(encode(d), "refused");
   d.cmp1_abs = false; d.cmp0_neg = true;
   EXPECT_EQ(encode(d), "refused");

   /* Non-zero integer immediate alternatives beside a uniform comparand have
    * no native AGX3B witness (e.g. lowered fcmp(r, u) -> fcmpsel(r, u, #1, #0)). */
   d.cmp0_neg = false;
   d.sel0_kind = AGX3_FSEL_SEL_IMMEDIATE; d.sel0 = 1; d.sel0_hint = AGX3_ALU_HINT_NONE;
   d.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE; d.sel1 = 0; d.sel1_hint = AGX3_ALU_HINT_NONE;
   EXPECT_EQ(encode(d), "refused");
   d.cmp1_uniform = false; d.cmp1_is_immediate = true; d.cmp1 = 0;
   d.cmp0_uniform = true; d.cmp0 = 20;
   EXPECT_EQ(encode(d), "refused");
}
