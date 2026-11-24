/*
 * Copyright 2021 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

#define CASE(instr, expected, size)                                            \
   INSTRUCTION_CASE(                                                           \
      {                                                                        \
         UNUSED agx_index out = agx_temp(b->shader, AGX_SIZE_##size);          \
         instr;                                                                \
         agx_unit_test(b, out);                                                \
      },                                                                       \
      {                                                                        \
         UNUSED agx_index out = agx_temp(b->shader, AGX_SIZE_##size);          \
         expected;                                                             \
         agx_unit_test(b, out);                                                \
      },                                                                       \
      agx_opt_compact_constants)

#define NEGCASE(instr, size)    CASE(instr, instr, size)
#define CASE32(instr, expected) CASE(instr, expected, 32)
#define NEGCASE32(instr)        NEGCASE(instr, 32)

class CompactConstants : public testing::Test {
 protected:
   CompactConstants()
   {
      mem_ctx = ralloc_context(NULL);

      wx = agx_register(0, AGX_SIZE_32);
   }

   ~CompactConstants()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
   agx_index wx;
};

TEST_F(CompactConstants, FP32)
{
   CASE32(agx_fadd_to(b, out, wx, agx_mov_imm(b, 32, fui(32768.0))),
          agx_fadd_to(b, out, wx, agx_mov_imm(b, 16, 0x7800)));
}

TEST_F(CompactConstants, InexactFP32)
{
   NEGCASE32(agx_fadd_to(b, out, wx, agx_mov_imm(b, 32, fui(32769.0))));
}

/* An AGX3 f32 add takes at most one half register (f32 + f16), so a constant
 * added to a half there keeps its 32 bits, in either operand order. Beside a
 * word register it still compacts: that is the mixed f32 + f16 form. Returns
 * the width the constant ends up with. */
static enum agx_size
compacted_constant_size(void *mem_ctx, enum agx_isa_encoding enc,
                        bool half_operand, bool constant_first)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   agx_shader_key *key = rzalloc(mem_ctx, agx_shader_key);
   key->dev.encoding = enc;
   b->shader->key = key;

   agx_index x = agx_preload(
      b, agx_register(0, half_operand ? AGX_SIZE_16 : AGX_SIZE_32));
   agx_index c = agx_mov_imm(b, 32, fui(5.15625));
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *add = constant_first ? agx_fadd_to(b, out, c, x)
                                   : agx_fadd_to(b, out, x, c);
   agx_unit_test(b, out);

   agx_opt_compact_constants(b->shader);

   const agx_index read = add->src[constant_first ? 0 : 1];
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV_IMM)
         EXPECT_EQ(I->dest[0].size, read.size);
   }
   return read.size;
}

TEST_F(CompactConstants, Agx3HalfAddKeepsWordConstant)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (bool constant_first : {false, true}) {
         EXPECT_EQ(compacted_constant_size(mem_ctx, enc, true, constant_first),
                   AGX_SIZE_32)
            << enc << " " << constant_first;
      }
   }
}

TEST_F(CompactConstants, Agx3WordAddCompactsConstant)
{
   for (enum agx_isa_encoding enc :
        {AGX_ISA_ENCODING_AGX3B, AGX_ISA_ENCODING_AGX3C}) {
      for (bool constant_first : {false, true}) {
         EXPECT_EQ(compacted_constant_size(mem_ctx, enc, false, constant_first),
                   AGX_SIZE_16)
            << enc << " " << constant_first;
      }
   }
}
