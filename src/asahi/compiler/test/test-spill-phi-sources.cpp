/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"

/*
 * Phis whose incoming value is spilled on one edge.
 *
 * entry -> {then, els} -> merge -> tail. A phi in merge takes `incoming` from
 * then and x from els. els holds more values than the register file at its
 * end, and x, read only by the phi, is the value furthest from its next use
 * there, so x leaves the register file in els.
 */

namespace {

class SpillPhiSources : public testing::Test {
 protected:
   SpillPhiSources()
   {
      mem = ralloc_context(NULL);
      b = agx_test_builder(mem);
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.has_scratch = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->any_cf = true;

      entry = agx_start_block(b->shader);
      then = agx_test_block(b->shader);
      els = agx_test_block(b->shader);
      merge = agx_test_block(b->shader);
      tail = agx_test_block(b->shader);
      agx_block_add_successor(entry, then);
      agx_block_add_successor(entry, els);
      agx_block_add_successor(then, merge);
      agx_block_add_successor(els, merge);
      agx_block_add_successor(merge, tail);
   }

   ~SpillPhiSources()
   {
      ralloc_free(mem);
   }

   agx_index u(unsigned i)
   {
      return agx_uniform(2 * i, AGX_SIZE_32);
   }

   /* A value that cannot be computed again */
   agx_index leaf(unsigned i)
   {
      return agx_rcp(b, u(i));
   }

   /* Builds the program with the phi read at the start of merge, or only in
    * tail after values that keep merge busy. Returns the phi.
    */
   agx_instr *build(agx_index incoming, bool read_in_merge)
   {
      b->cursor = agx_after_block(entry);
      agx_index x = leaf(1);

      b->cursor = agx_after_block(els);
      agx_index t[8];
      for (unsigned i = 0; i < ARRAY_SIZE(t); ++i)
         t[i] = leaf(10 + i);
      for (unsigned i = 0; i < ARRAY_SIZE(t); ++i)
         agx_unit_test(b, t[i]);

      b->cursor = agx_after_block(merge);
      agx_index p = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *phi = agx_phi_to(b, p, 2);
      phi->src[0] = incoming;
      phi->src[1] = x;

      if (read_in_merge)
         agx_unit_test(b, p);

      agx_index m[4];
      for (unsigned i = 0; i < ARRAY_SIZE(m); ++i)
         m[i] = leaf(30 + i);
      for (unsigned i = 0; i < ARRAY_SIZE(m); ++i)
         agx_unit_test(b, m[i]);

      b->cursor = agx_after_block(tail);
      agx_unit_test(b, p);
      return phi;
   }

   /* 16 halves: 24 less the 8 kept for memory swaps. els holds x and eight
    * 32-bit values, two halves over. */
   void spill()
   {
      agx_compute_liveness(b->shader);
      agx_spill(b->shader, 24, false);
   }

   agx_instr *phi_in(agx_block *block)
   {
      agx_foreach_phi_in_block(block, I)
         return I;
      return NULL;
   }

   unsigned memory_writes(agx_block *block)
   {
      unsigned n = 0;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_MOV && I->dest[0].memory)
            n++;
      }
      return n;
   }

   unsigned fills(agx_block *block)
   {
      unsigned n = 0;
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
            n++;
      }
      return n;
   }

   void *mem;
   agx_builder *b;
   agx_shader_key key = {};
   agx_block *entry, *then, *els, *merge, *tail;
};

} // namespace

/* A spilled phi keeps its uniform and immediate sources. Parallel copy
 * lowering writes them to the phi's slot on the edge. No memory variable is
 * made for them at the start of the program, where it would stay live until
 * the phi, and the edge would copy it from memory to memory.
 */
TEST_F(SpillPhiSources, SpilledPhiKeepsUniformSource)
{
   agx_instr *phi = build(u(20), false);
   spill();

   phi = phi_in(merge);
   ASSERT_NE(phi, nullptr);
   EXPECT_TRUE(phi->dest[0].memory);
   EXPECT_EQ(phi->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(phi->src[0].value, 40u);
   EXPECT_TRUE(phi->src[1].memory);
   EXPECT_EQ(memory_writes(entry), 0u);
   EXPECT_TRUE(b->shader->has_spill_pcopy_reserved);
}

TEST_F(SpillPhiSources, SpilledPhiKeepsImmediateSource)
{
   agx_instr *phi = build(agx_immediate(9), false);
   spill();

   phi = phi_in(merge);
   ASSERT_NE(phi, nullptr);
   EXPECT_TRUE(phi->dest[0].memory);
   EXPECT_EQ(phi->src[0].type, AGX_INDEX_IMMEDIATE);
   EXPECT_EQ(phi->src[0].value, 9u);
   EXPECT_EQ(memory_writes(entry), 0u);
}

/* A uniform the program writes is read where the phi's value is defined, at
 * the start of the program, so it still goes to memory there.
 */
TEST_F(SpillPhiSources, WrittenUniformIsCopiedAtTheStart)
{
   b->cursor = agx_before_block(entry);
   agx_uniform_store(b, agx_mov_imm(b, 32, 5), agx_immediate(60), 1);

   agx_instr *phi = build(u(20), false);
   spill();

   phi = phi_in(merge);
   ASSERT_NE(phi, nullptr);
   EXPECT_TRUE(phi->dest[0].memory);
   EXPECT_TRUE(phi->src[0].memory);
   EXPECT_EQ(memory_writes(entry), 1u);
}

/* A phi read in its own block stays in registers although x was spilled: x is
 * filled on its edge in place of a fill before the read, and nothing is
 * written to memory for the uniform's edge.
 */
TEST_F(SpillPhiSources, PhiReadInItsBlockStaysInRegisters)
{
   agx_instr *phi = build(u(20), true);
   spill();

   phi = phi_in(merge);
   ASSERT_NE(phi, nullptr);
   EXPECT_FALSE(phi->dest[0].memory);
   EXPECT_EQ(phi->src[0].type, AGX_INDEX_UNIFORM);
   EXPECT_FALSE(phi->src[1].memory);
   EXPECT_EQ(fills(els), 1u);
   EXPECT_EQ(fills(merge), 0u);
   EXPECT_EQ(memory_writes(then), 0u);
   EXPECT_EQ(memory_writes(merge), 0u);
}

/* Without the read in merge the phi is spilled, as before, and read from its
 * slot in tail.
 */
TEST_F(SpillPhiSources, PhiReadOnlyLaterIsSpilled)
{
   agx_instr *phi = build(u(20), false);
   spill();

   phi = phi_in(merge);
   ASSERT_NE(phi, nullptr);
   EXPECT_TRUE(phi->dest[0].memory);
   EXPECT_EQ(fills(tail), 1u);
}
