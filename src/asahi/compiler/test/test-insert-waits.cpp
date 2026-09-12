/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

/*
 * Capacity and hazard tests for agx_insert_waits, including the divergent
 * if/else scoreboard carry. The IR scoreboard field is a single bit
 * (agx_instr::scoreboard), so every test also asserts assignments stay in
 * {0, 1}.
 */

#include "agx_builder.h"
#include "agx_compile.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include "util/macros.h"
#include <gtest/gtest.h>

namespace {

class InsertWaits : public testing::Test {
 protected:
   InsertWaits() { mem_ctx = ralloc_context(NULL); }
   ~InsertWaits() { ralloc_free(mem_ctx); }

   void *mem_ctx;

   enum agx_format i32 = AGX_FORMAT_I32;
   unsigned mask1 = BITFIELD_MASK(1);

   agx_index reg(unsigned n) { return agx_register(n * 2, AGX_SIZE_32); }

   agx_instr *load(agx_builder *b, agx_index dst)
   {
      return agx_device_load_to(b, dst, agx_register(0, AGX_SIZE_64),
                                agx_immediate(0), i32, mask1, 0, false);
   }

   /* A fresh block appended to ctx, plus a builder cursor at its start */
   agx_builder *new_block(agx_context *ctx)
   {
      agx_block *blk = agx_test_block(ctx);
      agx_builder *b = rzalloc(mem_ctx, agx_builder);
      b->shader = ctx;
      b->cursor = agx_before_block(blk);
      return b;
   }

   unsigned count_waits(agx_context *ctx)
   {
      unsigned n = 0;
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            if (I->op == AGX_OPCODE_WAIT)
               n++;
         }
      }
      return n;
   }

   /* The scoreboard field is one bit: nothing may ever be assigned 2 or 3 */
   void assert_slots_valid(agx_context *ctx)
   {
      agx_foreach_block(ctx, block) {
         agx_foreach_instr_in_block(block, I) {
            ASSERT_TRUE(I->scoreboard <= 1)
               << "slot " << (unsigned)I->scoreboard
               << " does not fit the 1-bit IR field";
         }
      }
   }

   /* Waits in a block strictly before the given instruction */
   unsigned waits_before(agx_block *block, agx_instr *I)
   {
      unsigned n = 0;
      agx_foreach_instr_in_block(block, ins) {
         if (ins == I)
            break;
         if (ins->op == AGX_OPCODE_WAIT)
            n++;
      }
      return n;
   }

   /* Block 0 of the shader (the if block under test) */
   agx_block *first_block(agx_context *ctx)
   {
      return list_first_entry(&ctx->blocks, agx_block, link);
   }

   /* Standard divergent if/else shape: if block, then arm, else arm (ending
    * in pop_exec), merge - the shape emit_if lowers to.
    */
   void build_if_else(agx_builder *b0)
   {
      agx_context *ctx = b0->shader;
      agx_block *then_blk = agx_test_block(ctx);
      agx_builder *b1 = new_block(ctx);
      agx_block *else_blk = agx_test_block(ctx);
      agx_builder *b2 = new_block(ctx);
      agx_block *merge = agx_test_block(ctx);
      agx_builder *b3 = new_block(ctx);

      (void)b1;
      (void)b3;

      /* else arm restores the exec mask */
      agx_pop_exec(b2, 1);

      agx_if_icmp(b0, agx_zero(), agx_zero(), 1, AGX_ICOND_UEQ, true,
                  else_blk);
      agx_block_add_successor(first_block(ctx), then_blk);
      agx_block_add_successor(first_block(ctx), else_blk);
      agx_block_add_successor(then_blk, merge);
      agx_block_add_successor(else_blk, merge);
   }
};

/* A load followed by a read of its destination waits before the read. */
TEST_F(InsertWaits, RawHazardWaitsAtUse)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   agx_index dst = reg(1), out = reg(2);

   agx_instr *ld = load(b, dst);
   agx_iadd_to(b, out, dst, dst, 0);
   (void)ld;

   agx_insert_waits(b->shader);

   EXPECT_EQ(count_waits(b->shader), 1u);
   assert_slots_valid(b->shader);
}

/* Seventeen loads with no intervening reads overflow a slot: the pre-issue
 * capacity check must wait before the ninth message on that slot rather
 * than let it issue.
 */
TEST_F(InsertWaits, SlotCapacityWaitsBeforeIssue)
{
   agx_builder *b = agx_test_builder(mem_ctx);

   for (unsigned i = 0; i < 17; ++i)
      load(b, reg(i + 1));

   agx_insert_waits(b->shader);

   EXPECT_EQ(count_waits(b->shader), 1u);
   assert_slots_valid(b->shader);
}

/* Divergent if/else: the then arm loads a register that the else arm
 * overwrites (register allocation may legally reuse it after this pass).
 * The carried pending state must force a wait in the else arm before the
 * overwriting write, and the then arm itself must not be drained.
 */
TEST_F(InsertWaits, DivergentWawWaitsInElseArm)
{
   agx_builder *b0 = agx_test_builder(mem_ctx);
   agx_context *ctx = b0->shader;

   agx_block *then_blk = agx_test_block(ctx);
   agx_builder *b1 = new_block(ctx);
   agx_block *else_blk = agx_test_block(ctx);
   agx_builder *b2 = new_block(ctx);
   agx_block *merge = agx_test_block(ctx);

   agx_index dst = reg(1);

   agx_instr *ld = load(b1, dst);

   /* Else arm overwrites the load's register, then pops the exec mask */
   agx_instr *overwrite = agx_mov_to(b2, dst, agx_immediate(0));
   agx_pop_exec(b2, 1);

   agx_if_icmp(b0, agx_zero(), agx_zero(), 1, AGX_ICOND_UEQ, true, else_blk);
   agx_block_add_successor(first_block(ctx), then_blk);
   agx_block_add_successor(first_block(ctx), else_blk);
   agx_block_add_successor(then_blk, merge);
   agx_block_add_successor(else_blk, merge);

   agx_insert_waits(ctx);

   EXPECT_EQ(count_waits(ctx), 1u);
   assert_slots_valid(ctx);

   /* No wait in the then arm: the load carries, it does not drain */
   EXPECT_EQ(waits_before(then_blk, ld), 0u);

   /* Exactly one wait in the else arm, before the overwriting mov */
   EXPECT_EQ(waits_before(else_blk, overwrite), 1u);
}

/* The then arm's pending state is not forgotten at the merge: a block after
 * the region reading the loaded register waits, even though the load issued
 * in a different block. No block-exit drains fire inside the region.
 */
TEST_F(InsertWaits, MergeReaderSeesCarriedPending)
{
   agx_builder *b0 = agx_test_builder(mem_ctx);
   agx_context *ctx = b0->shader;

   agx_block *then_blk = agx_test_block(ctx);
   agx_builder *b1 = new_block(ctx);
   agx_block *else_blk = agx_test_block(ctx);
   agx_builder *b2 = new_block(ctx);
   agx_block *merge = agx_test_block(ctx);
   agx_builder *b3 = new_block(ctx);

   agx_index dst = reg(1), out = reg(3);

   load(b1, dst);
   agx_pop_exec(b2, 1);

   /* Merge reads the loaded register */
   agx_iadd_to(b3, out, dst, dst, 0);

   agx_if_icmp(b0, agx_zero(), agx_zero(), 1, AGX_ICOND_UEQ, true, else_blk);
   agx_block_add_successor(first_block(ctx), then_blk);
   agx_block_add_successor(first_block(ctx), else_blk);
   agx_block_add_successor(then_blk, merge);
   agx_block_add_successor(else_blk, merge);

   agx_insert_waits(ctx);

   /* One wait total: at the merge-block read. The exit block never drains,
    * and in-region block exits carry.
    */
   EXPECT_EQ(count_waits(ctx), 1u);
   assert_slots_valid(ctx);
}

} // namespace
