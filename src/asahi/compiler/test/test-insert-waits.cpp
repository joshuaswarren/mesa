/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <chrono>
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

static agx_instr *
emit_reg_load(agx_builder *b, unsigned reg_16)
{
   return agx_device_load_to(b, agx_register(reg_16, AGX_SIZE_32),
                             agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
                             AGX_FORMAT_I32, 1, 0, false);
}

static unsigned
count_waits_before_first_alu(agx_block *blk)
{
   unsigned waits = 0;
   agx_foreach_instr_in_block(blk, I) {
      if (I->op == AGX_OPCODE_WAIT)
         waits++;
      else if (I->op != AGX_OPCODE_DEVICE_LOAD)
         break;
   }
   return waits;
}

TEST_F(InsertWaits, SmartWaitsAvoidsMidBurstOverflowOnG14X)
{
   auto build_case = [&](enum agx_chip chip, mesa_shader_stage stage) {
      agx_builder *b = agx_test_builder(mem_ctx);
      b->shader->stage = stage;
      struct agx_shader_key *key = rzalloc(mem_ctx, struct agx_shader_key);
      key->dev.chip = chip;
      b->shader->key = key;

      emit_reg_load(b, 0);
      for (unsigned i = 0; i < 8; ++i)
         emit_reg_load(b, 2 + 2 * i);
      emit_reg_load(b, 18);
      emit_reg_load(b, 20);

      agx_iadd_to(b, agx_register(30, AGX_SIZE_32),
                  agx_register(0, AGX_SIZE_32), agx_register(18, AGX_SIZE_32), 0);
      agx_iadd_to(b, agx_register(32, AGX_SIZE_32),
                  agx_register(30, AGX_SIZE_32), agx_register(20, AGX_SIZE_32), 0);

      for (unsigned i = 0; i < 70; ++i) {
         agx_iadd_to(b, agx_register(34, AGX_SIZE_32),
                     agx_register(34, AGX_SIZE_32), agx_immediate(1), 0);
      }

      for (unsigned i = 0; i < 8; ++i) {
         agx_iadd_to(b, agx_register(34, AGX_SIZE_32),
                     agx_register(34, AGX_SIZE_32),
                     agx_register(2 + 2 * i, AGX_SIZE_32), 0);
      }

      agx_insert_waits(b->shader);
      return b->shader;
   };

   agx_context *g13g = build_case(AGX_CHIP_G13G, MESA_SHADER_COMPUTE);
   agx_context *g13x = build_case(AGX_CHIP_G13X, MESA_SHADER_COMPUTE);
   agx_context *g14g = build_case(AGX_CHIP_G14G, MESA_SHADER_COMPUTE);
   agx_context *g14x = build_case(AGX_CHIP_G14X, MESA_SHADER_COMPUTE);
   agx_context *frag = build_case(AGX_CHIP_G14X, MESA_SHADER_FRAGMENT);

   EXPECT_GE(count_waits_before_first_alu(agx_start_block(g13g)), 2u);
   EXPECT_GE(count_waits_before_first_alu(agx_start_block(g13x)), 2u);
   EXPECT_GE(count_waits_before_first_alu(agx_start_block(g14g)), 2u);
   EXPECT_EQ(count_waits_before_first_alu(agx_start_block(g14x)), 1u);
   EXPECT_TRUE(agx_shader_equal(g13g, frag));
}

TEST_F(InsertWaits, SmartWaitsLinearOnLongIndependentLoadBlock)
{
   agx_builder *b = agx_test_builder(mem_ctx);
   b->shader->stage = MESA_SHADER_COMPUTE;
   struct agx_shader_key *key = rzalloc(mem_ctx, struct agx_shader_key);
   key->dev.chip = AGX_CHIP_G14X;
   b->shader->key = key;

   const unsigned num_loads = 512;
   for (unsigned i = 0; i < num_loads; ++i) {
      unsigned reg = (i % 64) * 2;
      emit_reg_load(b, reg);
      for (unsigned k = 0; k < 3; ++k) {
         agx_iadd_to(b, agx_register(140, AGX_SIZE_32),
                     agx_register(140, AGX_SIZE_32), agx_immediate(1), 0);
      }
   }
   for (unsigned r = 0; r < 64; ++r) {
      agx_iadd_to(b, agx_register(142, AGX_SIZE_32),
                  agx_register(142, AGX_SIZE_32),
                  agx_register(r * 2, AGX_SIZE_32), 0);
   }

   auto t0 = std::chrono::steady_clock::now();
   agx_insert_waits(b->shader);
   auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - t0)
                        .count();

   EXPECT_LT(elapsed_ms, 200);

   unsigned pending[2] = {0, 0};
   unsigned total_waits = 0;
   agx_foreach_instr_in_block(agx_start_block(b->shader), I) {
      if (I->op == AGX_OPCODE_WAIT) {
         pending[I->scoreboard] = 0;
         total_waits++;
      } else if (I->op == AGX_OPCODE_DEVICE_LOAD) {
         pending[I->scoreboard]++;
         EXPECT_LE(pending[I->scoreboard], 8u);
      }
   }
   EXPECT_GT(total_waits, 0u);
}
