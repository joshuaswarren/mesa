/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <vector>
#include "agx_test.h"

#include <gtest/gtest.h>

/*
 * A constant program's ILP schedule may raise a block's register pressure up
 * to a cap. Values live across the block hold their registers through all of
 * it, so they count toward that cap too.
 *
 * The middle block of a three-block G17P constant program holds independent
 * ALU chains (immediate, add, observe). The ILP order issues every immediate
 * first, holding all of them at once; the source order holds one at a time.
 * The first block defines values the last block reads.
 */
class IlpCprogCap : public testing::Test {
 protected:
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};

   ~IlpCprogCap()
   {
      ralloc_free(mem);
   }

   /* Whether the ILP schedule reordered the middle block. */
   bool reorders(unsigned live_across, unsigned chains)
   {
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->is_preamble = true;

      agx_block *last = list_first_entry(&b->shader->blocks, agx_block, link);
      auto next_block = [&]() {
         agx_block *next = agx_test_block(b->shader);
         last->successors[0] = next;
         util_dynarray_append(&next->predecessors, last);
         last = next;
         b->cursor = agx_after_block(last);
      };

      std::vector<agx_index> across;
      for (unsigned i = 0; i < live_across; ++i)
         across.push_back(agx_mov_imm(b, 32, i));

      next_block();
      agx_block *middle = last;
      for (unsigned i = 0; i < chains; ++i) {
         agx_index v = agx_mov_imm(b, 32, 1000 + i);
         agx_unit_test(b, agx_iadd(b, v, v, 0));
      }

      next_block();
      for (agx_index v : across)
         agx_unit_test(b, v);

      std::vector<agx_instr *> original;
      agx_foreach_instr_in_block(middle, I)
         original.push_back(I);

      agx_ilp_schedule(b->shader);

      unsigned i = 0;
      bool changed = false;
      agx_foreach_instr_in_block(middle, I) {
         EXPECT_LT(i, original.size());
         changed |= i >= original.size() || I != original[i];
         ++i;
      }
      EXPECT_EQ(i, original.size());
      return changed;
   }
};

/* 16 halves live across plus 80 for the immediates: inside the cap. */
TEST_F(IlpCprogCap, SchedulesBlockThatFits)
{
   EXPECT_TRUE(reorders(8, 40));
}

/* The same block with 64 halves live across would need 144: it keeps its
 * order, although what the block itself adds still fits. */
TEST_F(IlpCprogCap, CountsValuesLiveAcrossTheBlock)
{
   EXPECT_FALSE(reorders(32, 40));
}
