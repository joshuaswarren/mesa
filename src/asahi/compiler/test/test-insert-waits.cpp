/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

static unsigned
count_waits_before(agx_block *block, agx_instr *stop)
{
   unsigned waits = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I == stop)
         break;
      if (I->op == AGX_OPCODE_WAIT)
         waits++;
   }
   return waits;
}

class InsertWaits : public testing::Test {
 protected:
   InsertWaits() { mem_ctx = ralloc_context(NULL); }
   ~InsertWaits() { ralloc_free(mem_ctx); }
   void *mem_ctx;
};

TEST_F(InsertWaits, ReorderableDeviceLoadStaysInFlightAcrossThreadgroupBarrier)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   agx_block *blk = agx_start_block(b->shader);

   agx_index addr = agx_register(0, AGX_SIZE_64);
   agx_index off = agx_register(4, AGX_SIZE_32);
   agx_index dst = agx_register(6, AGX_SIZE_32);
   agx_index sum = agx_register(8, AGX_SIZE_32);

   agx_instr *ld =
      agx_device_load_to(b, dst, addr, off, AGX_FORMAT_I32, 0x1, 0, 0);
   ld->memory_can_reorder = true;
   agx_instr *barrier = agx_threadgroup_barrier(b);
   agx_instr *use = agx_iadd_to(b, sum, dst, off, 0);

   agx_insert_waits(b->shader);

   EXPECT_EQ(count_waits_before(blk, barrier), 0u);
   EXPECT_EQ(count_waits_before(blk, use), 1u);
}

TEST_F(InsertWaits, NonReorderableDeviceLoadDrainsAtThreadgroupBarrier)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   agx_block *blk = agx_start_block(b->shader);

   agx_index addr = agx_register(0, AGX_SIZE_64);
   agx_index off = agx_register(4, AGX_SIZE_32);
   agx_index dst = agx_register(6, AGX_SIZE_32);
   agx_index sum = agx_register(8, AGX_SIZE_32);

   agx_instr *ld =
      agx_device_load_to(b, dst, addr, off, AGX_FORMAT_I32, 0x1, 0, 0);
   ld->memory_can_reorder = false;
   agx_instr *barrier = agx_threadgroup_barrier(b);
   agx_instr *use = agx_iadd_to(b, sum, dst, off, 0);

   agx_insert_waits(b->shader);

   EXPECT_EQ(count_waits_before(blk, barrier), 1u);
   EXPECT_EQ(count_waits_before(blk, use), 1u);
}

TEST_F(InsertWaits, ReorderableDeviceLoadDrainsAtMemoryBarrier)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   agx_block *blk = agx_start_block(b->shader);

   agx_index addr = agx_register(0, AGX_SIZE_64);
   agx_index off = agx_register(4, AGX_SIZE_32);
   agx_index dst = agx_register(6, AGX_SIZE_32);
   agx_index sum = agx_register(8, AGX_SIZE_32);

   agx_instr *ld =
      agx_device_load_to(b, dst, addr, off, AGX_FORMAT_I32, 0x1, 0, 0);
   ld->memory_can_reorder = true;
   agx_instr *barrier = agx_memory_barrier(b);
   agx_instr *use = agx_iadd_to(b, sum, dst, off, 0);

   agx_insert_waits(b->shader);

   EXPECT_EQ(count_waits_before(blk, barrier), 1u);
   EXPECT_EQ(count_waits_before(blk, use), 1u);
}
