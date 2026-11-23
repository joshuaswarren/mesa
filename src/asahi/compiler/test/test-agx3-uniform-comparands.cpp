/* SPDX-License-Identifier: MIT */
/*
 * Uniform comparands that an AGX3 comparison reads in place instead of
 * copying into a GPR first: a float comparison takes the uniform rules of the
 * fcmpsel it lowers to, and a comparison with a 16-bit result uses the 32-bit
 * native select and its low half. Each lowered form is packed, decoded back
 * to its fields by the table decoder (agx3_decode, the inverse of the
 * composer), and walked by the AGX3B length decoder that the native
 * instruction statistics use.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "util/bitset.h"
/* These headers have no C++ linkage of their own; their includes come first. */
extern "C" {
#include "asahi/isa/disasm.h"
#include "agx3_compose/agx3_compose.h"
#include "agx3_compose/agx3_decode.h"
}
#include "agx3_alu_core.h"
#include "agx3_isel_core.h"
#include "agx_test.h"

namespace {

class AGX3UniformComparands : public testing::Test {
 protected:
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};

   ~AGX3UniformComparands() { ralloc_free(mem); }

   agx_builder *builder(enum agx_isa_encoding encoding,
                        mesa_shader_stage stage = MESA_SHADER_FRAGMENT)
   {
      key = {};
      key.dev.encoding = encoding;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = stage;
      return b;
   }

   static unsigned uniform_copies(agx_context *ctx)
   {
      unsigned n = 0;
      agx_foreach_instr_global(ctx, I)
         n += I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM;
      return n;
   }

   static agx_instr *only(agx_context *ctx, enum agx_opcode op)
   {
      agx_instr *found = NULL;
      agx_foreach_instr_global(ctx, I) {
         if (I->op == op) {
            EXPECT_EQ(found, nullptr);
            found = I;
         }
      }
      return found;
   }

   /* Pack one post-RA instruction and decode it again. */
   std::vector<uint8_t> pack(agx_builder *b)
   {
      agx_stop(b);
      util_dynarray bin;
      util_dynarray_init(&bin, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bin));
      const uint8_t *d = static_cast<const uint8_t *>(bin.data);
      std::vector<uint8_t> v(d, d + bin.size);
      util_dynarray_fini(&bin);
      return v;
   }

   /* The fields `bytes` decode to as one `mnemonic`, by the AGX3B tables. */
   static bool decode(const uint8_t *bytes, unsigned length,
                      enum agx3_mnemonic mnemonic,
                      uint64_t fields[AGX3_MAX_FIELDS])
   {
      memset(fields, 0, AGX3_MAX_FIELDS * sizeof(*fields));
      return agx3_decode(&agx3b_table, bytes, length, mnemonic, fields) >= 0;
   }

   /* The length and name the instruction stream walker gives `bytes`. */
   static signed walk(const uint8_t *bytes, unsigned length,
                      std::string *name)
   {
      const char *display = NULL;
      const signed n = agx3b_instr_length(bytes, length, &display);
      *name = display ? display : "";
      name->erase(name->find_last_not_of(' ') + 1);
      return n;
   }

   static std::string hex(const std::vector<uint8_t> &code, unsigned n)
   {
      static const char *digits = "0123456789abcdef";
      std::string s;
      for (unsigned i = 0; i < n; ++i) {
         s += digits[code[i] >> 4];
         s += digits[code[i] & 15];
      }
      return s;
   }
};

} /* namespace */

/* fcmp(r, u) lowers to fcmpsel(r, u, #1, #0), the r < u form (f59 = 3), so
 * the uniform stays in slot 1 like the select's. */
TEST_F(AGX3UniformComparands, FloatCompareKeepsSlot1Uniform)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index r = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *cmp = agx_fcmp_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                agx_uniform(20, AGX_SIZE_32), AGX_FCOND_LT,
                                false);
   agx_unit_test(b, cmp->dest[0]);
   agx_lower_uniform_sources(b->shader);

   EXPECT_EQ(uniform_copies(b->shader), 0u);
   EXPECT_EQ(cmp->src[0].type, AGX_INDEX_NORMAL);
   EXPECT_EQ(cmp->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(cmp->src[1].value, 20u);
   EXPECT_EQ(cmp->fcond, AGX_FCOND_LT);
}

/* A uniform first comparand beside a register is exchanged into slot 1 with
 * the relation reversed, as for fcmpsel. */
TEST_F(AGX3UniformComparands, FloatCompareExchangesUniformFirstComparand)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
   agx_index r = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *cmp = agx_fcmp_to(b, agx_temp(b->shader, AGX_SIZE_16),
                                agx_uniform(300, AGX_SIZE_32), r,
                                AGX_FCOND_LT, true);
   agx_unit_test(b, cmp->dest[0]);
   agx_lower_uniform_sources(b->shader);

   /* Slot 1 holds half indices up to 511. */
   EXPECT_EQ(uniform_copies(b->shader), 0u);
   EXPECT_TRUE(agx_is_equiv(cmp->src[0], r));
   EXPECT_EQ(cmp->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(cmp->src[1].value, 300u);
   EXPECT_EQ(cmp->fcond, AGX_FCOND_GT);
   EXPECT_TRUE(cmp->invert_cond);
}

/* A uniform against an immediate zero is the constant-program flag form
 * (f42 = 3); its slot-0 field holds half indices below 256 only. */
TEST_F(AGX3UniformComparands, FloatCompareUniformFlagRange)
{
   for (unsigned value : {40u, 300u}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_instr *cmp = agx_fcmp_to(b, agx_temp(b->shader, AGX_SIZE_16),
                                   agx_uniform(value, AGX_SIZE_16),
                                   agx_immediate(0), AGX_FCOND_EQ, false);
      agx_unit_test(b, cmp->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(uniform_copies(b->shader), value < 256 ? 0u : 1u) << value;
   }
}

/* Modified uniforms and the M5 encoding keep their copies. */
TEST_F(AGX3UniformComparands, FloatCompareOutsideTheFormCopies)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *cmp = agx_fcmp_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                   agx_neg(agx_uniform(20, AGX_SIZE_32)),
                                   AGX_FCOND_LT, false);
      agx_unit_test(b, cmp->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(uniform_copies(b->shader), 1u);
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C);
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *cmp = agx_fcmp_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                   agx_uniform(20, AGX_SIZE_32),
                                   AGX_FCOND_LT, false);
      agx_unit_test(b, cmp->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(uniform_copies(b->shader), 1u);
   }
}

/* icmp(r, u) with a 16-bit result becomes the native 32-bit select reading
 * the uniform and the low half of its result. */
TEST_F(AGX3UniformComparands, HalfIntegerCompareUsesWideNativeSelect)
{
   for (bool invert : {false, true}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      const agx_index result = agx_temp(b->shader, AGX_SIZE_16);
      agx_icmp_to(b, result, r, agx_uniform(44, AGX_SIZE_32), AGX_ICOND_ULT,
                  invert);
      agx_unit_test(b, result);
      agx_lower_uniform_sources(b->shader);

      EXPECT_EQ(uniform_copies(b->shader), 0u);
      EXPECT_EQ(only(b->shader, AGX_OPCODE_ICMP), nullptr);
      agx_instr *sel = only(b->shader, AGX_OPCODE_ICMPSEL);
      agx_instr *split = only(b->shader, AGX_OPCODE_SPLIT);
      ASSERT_NE(sel, nullptr);
      ASSERT_NE(split, nullptr);
      EXPECT_EQ(sel->dest[0].size, AGX_SIZE_32);
      EXPECT_TRUE(agx_is_equiv(sel->src[0], r));
      EXPECT_EQ(sel->src[1].type, AGX_INDEX_UNIFORM);
      EXPECT_EQ(sel->src[1].value, 44u);
      EXPECT_EQ(sel->icond, AGX_ICOND_ULT);
      EXPECT_FALSE(sel->invert_cond);
      EXPECT_EQ(sel->src[2].type, AGX_INDEX_IMMEDIATE);
      EXPECT_EQ(sel->src[3].type, AGX_INDEX_IMMEDIATE);
      EXPECT_EQ(sel->src[2].value, invert ? 0u : 1u);
      EXPECT_EQ(sel->src[3].value, invert ? 1u : 0u);
      EXPECT_TRUE(agx_is_equiv(split->src[0], sel->dest[0]));
      EXPECT_TRUE(agx_is_equiv(split->dest[0], result));
   }
}

/* A 16-bit select of a widened result takes the 32-bit value, so a chain of
 * comparisons against uniforms keeps every uniform in place. */
TEST_F(AGX3UniformComparands, HalfSelectOfWidenedResultStaysWide)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
   agx_index x = agx_temp(b->shader, AGX_SIZE_32);
   agx_index y = agx_temp(b->shader, AGX_SIZE_32);
   const agx_index first = agx_temp(b->shader, AGX_SIZE_16);
   agx_icmp_to(b, first, x, agx_uniform(44, AGX_SIZE_32), AGX_ICOND_ULT,
               false);
   const agx_index second = agx_temp(b->shader, AGX_SIZE_16);
   agx_icmpsel_to(b, second, y, agx_uniform(40, AGX_SIZE_32), first,
                  agx_immediate(0), AGX_ICOND_ULT);
   agx_unit_test(b, second);
   agx_lower_uniform_sources(b->shader);

   EXPECT_EQ(uniform_copies(b->shader), 0u);
   std::vector<agx_instr *> sels;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_ICMPSEL)
         sels.push_back(I);
   }
   ASSERT_EQ(sels.size(), 2u);
   EXPECT_EQ(sels[1]->dest[0].size, AGX_SIZE_32);
   EXPECT_TRUE(agx_is_equiv(sels[1]->src[2], sels[0]->dest[0]));
   EXPECT_EQ(sels[1]->src[2].size, AGX_SIZE_32);
}

/* Shapes the wide select cannot carry keep the copy and the narrow form:
 * a half alternative of unknown upper bits, a 16-bit comparand, a uniform
 * beyond the low file, and constant programs. */
TEST_F(AGX3UniformComparands, HalfCompareOutsideTheWideFormCopies)
{
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_index half = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *sel = agx_icmpsel_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                      agx_uniform(40, AGX_SIZE_32), half,
                                      agx_immediate(0), AGX_ICOND_ULT);
      agx_unit_test(b, sel->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(uniform_copies(b->shader), 1u);
      EXPECT_EQ(sel->dest[0].size, AGX_SIZE_16);
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
      agx_index r = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *cmp = agx_icmp_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                   agx_uniform(41, AGX_SIZE_16),
                                   AGX_ICOND_UEQ, false);
      agx_unit_test(b, cmp->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(only(b->shader, AGX_OPCODE_ICMPSEL), nullptr);
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *cmp = agx_icmp_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                   agx_uniform(300, AGX_SIZE_32),
                                   AGX_ICOND_UEQ, false);
      agx_unit_test(b, cmp->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(uniform_copies(b->shader), 1u);
      EXPECT_EQ(only(b->shader, AGX_OPCODE_ICMPSEL), nullptr);
   }
   {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
      b->shader->is_preamble = true;
      agx_index r = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *cmp = agx_icmp_to(b, agx_temp(b->shader, AGX_SIZE_16), r,
                                   agx_uniform(44, AGX_SIZE_32),
                                   AGX_ICOND_UEQ, false);
      agx_unit_test(b, cmp->dest[0]);
      agx_lower_uniform_sources(b->shader);
      EXPECT_EQ(uniform_copies(b->shader), 1u);
      EXPECT_EQ(only(b->shader, AGX_OPCODE_ICMPSEL), nullptr);
   }
}

/* M5 queries its own encoder: a right uniform comparand is native there,
 * and a left one is exchanged into it. */
TEST_F(AGX3UniformComparands, M5HalfIntegerCompareUsesWideNativeSelect)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3C, MESA_SHADER_COMPUTE);
   agx_index r = agx_temp(b->shader, AGX_SIZE_32);
   const agx_index result = agx_temp(b->shader, AGX_SIZE_16);
   agx_icmp_to(b, result, agx_uniform(44, AGX_SIZE_32), r, AGX_ICOND_ULT,
               false);
   agx_unit_test(b, result);
   agx_lower_uniform_sources(b->shader);

   EXPECT_EQ(uniform_copies(b->shader), 0u);
   agx_instr *sel = only(b->shader, AGX_OPCODE_ICMPSEL);
   ASSERT_NE(sel, nullptr);
   EXPECT_TRUE(agx_is_equiv(sel->src[0], r));
   EXPECT_EQ(sel->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(sel->icond, AGX_ICOND_UGT);
}

/* The lowered comparison packs to the general select with a slot-1 uniform
 * and immediate alternatives, and decodes back to that form. */
TEST_F(AGX3UniformComparands, LoweredFloatComparePacksAndDecodes)
{
   for (bool half : {true, false}) {
      agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B);
      key.dev.chip = AGX_CHIP_G17P;
      agx_fcmpsel_to(b, agx_register(2, half ? AGX_SIZE_16 : AGX_SIZE_32),
                     agx_register(4, AGX_SIZE_32),
                     agx_uniform(20, AGX_SIZE_32), agx_immediate(1),
                     agx_immediate(0), AGX_FCOND_LT);
      const std::vector<uint8_t> code = pack(b);
      ASSERT_GT(code.size(), 4u);

      /* The same instruction from its field description. */
      struct agx3_fsel_desc d = {};
      d.dest = 2;
      d.dest_16 = half;
      d.cmp0 = 4;
      d.cmp1_uniform = true;
      d.cmp1 = 20;
      d.sel0_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel0 = 1;
      d.sel1_kind = AGX3_FSEL_SEL_IMMEDIATE;
      d.sel1 = 0;
      d.sel_16 = half;
      d.condition = AGX_FCOND_LT;
      uint8_t ref[AGX3_ALU_MAX_LENGTH], ref_len = 0;
      ASSERT_TRUE(agx3_alu_fcmp_select(AGX_ISA_ENCODING_AGX3B, &d, ref,
                                       &ref_len));
      ASSERT_LE(ref_len, code.size());
      EXPECT_EQ(memcmp(code.data(), ref, ref_len), 0);

      std::string name;
      EXPECT_EQ(walk(code.data(), code.size(), &name), ref_len);
      EXPECT_EQ(name, "fsel");

      uint64_t f[AGX3_MAX_FIELDS];
      ASSERT_TRUE(decode(code.data(), ref_len, AGX3_MN_fsel, f))
         << hex(code, ref_len);
      EXPECT_EQ(f[3], 2u);               /* destination half */
      EXPECT_EQ(f[11], half ? 1u : 2u);  /* its width */
      EXPECT_EQ(f[35], 4u);              /* register comparand */
      EXPECT_EQ(f[42], 2u);              /* ... of register kind */
      EXPECT_EQ(f[59], 3u);              /* uniform comparand kind */
      EXPECT_EQ(f[53], 20u);             /* ... its half index */
      EXPECT_EQ(f[76], 1u);              /* immediate true value */
      EXPECT_EQ(f[68], 1u);
      EXPECT_EQ(f[93], 1u);              /* immediate false value */
      EXPECT_EQ(f[85], 0u);
      EXPECT_EQ(f[110], (uint64_t)AGX_FCOND_LT);
      RecordProperty(half ? "fsel16" : "fsel32", hex(code, ref_len));
   }

   /* Apple's own slot-1 uniform select (r22 > u126 ? r32 : r22, see
    * test-agx3-fsel-uniform.cpp) decodes to the same uniform fields. */
   const uint8_t apple[] = {0xb2, 0x17, 0x67, 0x7f, 0x82,
                            0x20, 0x02, 0x02, 0x00, 0x16};
   std::string name;
   EXPECT_EQ(walk(apple, sizeof(apple), &name), (signed)sizeof(apple));
   EXPECT_EQ(name, "fsel");
   uint64_t f[AGX3_MAX_FIELDS];
   ASSERT_TRUE(decode(apple, sizeof(apple), AGX3_MN_fsel, f));
   EXPECT_EQ(f[59], 3u);
   EXPECT_EQ(f[53], 126u);
   EXPECT_EQ(f[35], 22u);
}

/* The widened integer comparison packs to the native 32-bit select with a
 * slot-1 uniform (f51, f57 = 3) and immediate alternatives. */
TEST_F(AGX3UniformComparands, WideIntegerComparePacksAndDecodes)
{
   agx_builder *b = builder(AGX_ISA_ENCODING_AGX3B, MESA_SHADER_COMPUTE);
   key.dev.chip = AGX_CHIP_G17P;
   agx_icmpsel_to(b, agx_register(2, AGX_SIZE_32),
                  agx_register(4, AGX_SIZE_32), agx_uniform(44, AGX_SIZE_32),
                  agx_immediate(1), agx_immediate(0), AGX_ICOND_ULT);
   const std::vector<uint8_t> code = pack(b);
   ASSERT_GT(code.size(), 4u);

   struct agx3_isel_desc d = {};
   d.dest = 2;
   d.cmp0 = 4;
   d.cmp1 = 44;
   d.cmp1_uniform = true;
   d.cond = AGX3_ISEL_ICOND_ULT;
   d.explicit_hints = true;
   d.true_kind = AGX3_ISEL_IMMEDIATE;
   d.sel_true = 1;
   d.false_kind = AGX3_ISEL_IMMEDIATE;
   d.sel_false = 0;
   uint8_t ref[AGX3_ISEL_MAX_LENGTH], ref_len = 0;
   ASSERT_TRUE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, ref, &ref_len));
   ASSERT_LE(ref_len, code.size());
   EXPECT_EQ(memcmp(code.data(), ref, ref_len), 0);

   std::string name;
   EXPECT_EQ(walk(code.data(), code.size(), &name), ref_len);
   EXPECT_EQ(name, "isel");

   uint64_t f[AGX3_MAX_FIELDS];
   ASSERT_TRUE(decode(code.data(), ref_len, AGX3_MN_isel, f))
      << hex(code, ref_len);
   EXPECT_EQ(f[3], 2u);    /* destination half */
   EXPECT_EQ(f[11], 2u);   /* 32-bit result */
   EXPECT_EQ(f[35], 4u);   /* register comparand */
   EXPECT_EQ(f[42], 2u);
   EXPECT_EQ(f[57], 3u);   /* uniform comparand kind */
   EXPECT_EQ(f[51], 44u);  /* ... its half index */
   EXPECT_EQ(f[72], 1u);   /* immediate true value */
   EXPECT_EQ(f[64], 1u);
   EXPECT_EQ(f[89], 1u);   /* immediate false value */
   EXPECT_EQ(f[81], 0u);
   EXPECT_EQ(f[106], (uint64_t)AGX3_ISEL_ICOND_ULT);
   RecordProperty("isel32", hex(code, ref_len));

   /* The 16-bit result form has no uniform comparand. */
   d.dest_16 = true;
   EXPECT_FALSE(agx3_isel_encode(AGX_ISA_ENCODING_AGX3B, &d, ref, &ref_len));
}
