/*
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

static void
pass(agx_context *ctx)
{
   agx_repair_ssa(ctx);
   agx_reindex_ssa(ctx);
}

#define CASE(instr)                                                            \
   INSTRUCTION_CASE(                                                           \
      {                                                                        \
         bool repaired = false;                                                \
         instr                                                                 \
      },                                                                       \
      {                                                                        \
         bool repaired = true;                                                 \
         instr                                                                 \
      },                                                                       \
      pass)

class RepairSSA : public testing::Test {
 protected:
   RepairSSA()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~RepairSSA()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
};

static agx_index
agx_phi_2(agx_builder *b, agx_index x, agx_index y)
{
   agx_index idx = agx_temp(b->shader, x.size);
   agx_instr *phi = agx_phi_to(b, idx, 2);
   phi->src[0] = x;
   phi->src[1] = y;
   return idx;
}

TEST_F(RepairSSA, Local)
{
   CASE({
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_index y = agx_mov_imm(b, AGX_SIZE_32, 0xefac);

      if (repaired) {
         agx_unit_test(b, agx_fadd(b, y, x));
      } else {
         agx_fadd_to(b, x, y, x);
         agx_unit_test(b, x);
      }
   });
}

/*      A
 *     / \
 *    B   C
 *     \ /
 *      D
 */
TEST_F(RepairSSA, IfElse)
{
   CASE({
      agx_block *A = agx_start_block(b->shader);
      agx_block *B = agx_test_block(b->shader);
      agx_block *C = agx_test_block(b->shader);
      agx_block *D = agx_test_block(b->shader);

      agx_block_add_successor(A, B);
      agx_block_add_successor(A, C);

      agx_block_add_successor(B, D);
      agx_block_add_successor(C, D);

      b->cursor = agx_after_block(B);
      agx_index x = agx_mov_imm(b, 32, 0xcafe);
      agx_index y = agx_mov_imm(b, 32, 0xbade);

      b->cursor = agx_after_block(C);
      agx_index x2 = repaired ? agx_temp(b->shader, AGX_SIZE_32) : x;
      agx_mov_imm_to(b, x2, 0xefac);
      agx_index y2 = agx_mov_imm(b, 32, 0xbade);

      b->cursor = agx_after_block(D);
      if (repaired)
         x = agx_phi_2(b, x, x2);

      agx_index y3 = agx_phi_2(b, y, y2);
      agx_unit_test(b, agx_fadd(b, x, y3));
   });
}

TEST_F(RepairSSA, UseModifiersDoNotLeakIntoPhi)
{
   for (unsigned modifiers = 1; modifiers <= 3; ++modifiers) {
      CASE({
         agx_block *A = agx_start_block(b->shader);
         agx_block *B = agx_test_block(b->shader);
         agx_block *C = agx_test_block(b->shader);
         agx_block *D = agx_test_block(b->shader);
         agx_block_add_successor(A, B);
         agx_block_add_successor(A, C);
         agx_block_add_successor(B, D);
         agx_block_add_successor(C, D);

         b->cursor = agx_after_block(B);
         agx_index x = agx_mov_imm(b, 32, 1);
         b->cursor = agx_after_block(C);
         agx_index x2 = repaired ? agx_temp(b->shader, AGX_SIZE_32) : x;
         agx_mov_imm_to(b, x2, 2);

         b->cursor = agx_after_block(D);
         if (repaired)
            x = agx_phi_2(b, x, x2);

         agx_index modified = x;
         modified.abs = modifiers & 1;
         modified.neg = modifiers & 2;
         agx_unit_test(b, modifiers == 2
                            ? agx_iadd(b, agx_immediate(10), modified, 0)
                            : agx_fadd(b, modified, agx_negzero()));
         agx_unit_test(b, x);
      });
   }
}

/* A value defined before a join and read after it gets a phi of itself from
 * the repair. The repair removes that trivial phi on every chip. */
TEST_F(RepairSSA, RemovesTrivialJoinPhi)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G17P, AGX_CHIP_G17S}) {
      agx_shader_key key = {};
      key.dev.encoding = chip == AGX_CHIP_G17S ? AGX_ISA_ENCODING_AGX3C
                         : chip == AGX_CHIP_G17P ? AGX_ISA_ENCODING_AGX3B
                                                 : AGX_ISA_ENCODING_AGX2;
      key.dev.chip = chip;

      CASE({
         b->shader->key = &key;
         agx_block *A = agx_start_block(b->shader);
         agx_block *B = agx_test_block(b->shader);
         agx_block *C = agx_test_block(b->shader);
         agx_block *D = agx_test_block(b->shader);
         agx_block_add_successor(A, B);
         agx_block_add_successor(A, C);
         agx_block_add_successor(B, D);
         agx_block_add_successor(C, D);

         b->cursor = agx_after_block(A);
         agx_index x = agx_mov_imm(b, 32, 0xcafe);

         b->cursor = agx_after_block(D);
         agx_unit_test(b, x);
      });
   }
}

/* Exercise def_map growth (>16 definitions in a block) across a diamond join. */
TEST_F(RepairSSA, ManyDefsGrowMap)
{
   CASE({
      agx_block *A = agx_start_block(b->shader);
      agx_block *B = agx_test_block(b->shader);
      agx_block *C = agx_test_block(b->shader);
      agx_block *D = agx_test_block(b->shader);

      agx_block_add_successor(A, B);
      agx_block_add_successor(A, C);
      agx_block_add_successor(B, D);
      agx_block_add_successor(C, D);

      constexpr unsigned N = 32;
      agx_index x[N];
      agx_index x2[N];

      b->cursor = agx_after_block(B);
      for (unsigned i = 0; i < N; ++i)
         x[i] = agx_mov_imm(b, 32, 0x100 + i);

      b->cursor = agx_after_block(C);
      for (unsigned i = 0; i < N; ++i) {
         x2[i] = repaired ? agx_temp(b->shader, AGX_SIZE_32) : x[i];
         agx_mov_imm_to(b, x2[i], 0x200 + i);
      }

      b->cursor = agx_after_block(D);
      if (repaired) {
         for (int i = N - 1; i >= 0; --i)
            x[i] = agx_phi_2(b, x[i], x2[i]);
      }
      for (unsigned i = 0; i < N; ++i)
         agx_unit_test(b, x[i]);
   });
}
