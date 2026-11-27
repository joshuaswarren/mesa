/*
 * SPDX-License-Identifier: MIT
 */

#include "agx_builder.h"
#include "agx_compiler.h"
#include "agx_test.h"

#include <gtest/gtest.h>

static void
pass(agx_context *ctx)
{
   agx_hoist_loads(ctx);
   agx_reindex_ssa(ctx);
}

#define CASE(instr, expected) INSTRUCTION_CASE(instr, expected, pass)
#define NEGCASE(instr)        INSTRUCTION_CASE(instr, instr, pass)

class HoistLoads : public testing::Test {
 protected:
   HoistLoads()
   {
      mem_ctx = ralloc_context(NULL);
   }

   ~HoistLoads()
   {
      ralloc_free(mem_ctx);
   }

   void *mem_ctx;
};

static agx_index
load(agx_builder *b, agx_index off, bool coherent)
{
   agx_index v = agx_temp(b->shader, AGX_SIZE_32);
   agx_device_load_to(b, v, agx_uniform(0, AGX_SIZE_64), off, AGX_FORMAT_I32, 1,
                      0, coherent);
   return v;
}

TEST_F(HoistLoads, AboveAluAndThreadgroupBarrier)
{
   CASE(
      {
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index y = agx_fadd(b, x, x);
         agx_threadgroup_barrier(b);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      },
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index v = load(b, off, false);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index y = agx_fadd(b, x, x);
         agx_threadgroup_barrier(b);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      });
}

TEST_F(HoistLoads, HoistsPureAddrProducerChain)
{
   CASE(
      {
         agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index y = agx_fadd(b, x, x);
         agx_threadgroup_barrier(b);
         agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      },
      {
         agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
         agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index y = agx_fadd(b, x, x);
         agx_threadgroup_barrier(b);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      });
}

TEST_F(HoistLoads, HoistsTwoStepAddrChainPreservingOrder)
{
   CASE(
      {
         agx_index idx = agx_mov_imm(b, AGX_SIZE_32, 8);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index y = agx_fadd(b, x, x);
         agx_index scaled = agx_iadd(b, idx, idx, 0);
         agx_index off = agx_iadd(b, scaled, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      },
      {
         agx_index idx = agx_mov_imm(b, AGX_SIZE_32, 8);
         agx_index scaled = agx_iadd(b, idx, idx, 0);
         agx_index off = agx_iadd(b, scaled, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_index y = agx_fadd(b, x, x);
         agx_unit_test(b, v);
         agx_unit_test(b, y);
      });
}

TEST_F(HoistLoads, AddrProducerNeverMovesDownward)
{
   CASE(
      {
         agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
         agx_memory_barrier(b);
         agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
         agx_index mid = agx_iadd(b, off, agx_immediate(8), 0);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
         agx_unit_test(b, mid);
      },
      {
         agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
         agx_memory_barrier(b);
         agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_index mid = agx_iadd(b, off, agx_immediate(8), 0);
         agx_unit_test(b, v);
         agx_unit_test(b, mid);
      });
}

TEST_F(HoistLoads, NotAboveDeviceStore)
{
   NEGCASE({
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_device_store(b, x, agx_uniform(0, AGX_SIZE_64), off, AGX_FORMAT_I32,
                       1, 0, false);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, NotAboveMemoryBarrier)
{
   NEGCASE({
      agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
      agx_memory_barrier(b);
      agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, NotAboveDoorbell)
{
   NEGCASE({
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_doorbell(b, 1);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, NotAboveUniformStoreConflict)
{
   NEGCASE({
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_uniform_store(b, x, agx_immediate(20), 1);
      agx_index off = agx_iadd(b, agx_uniform(20, AGX_SIZE_32), agx_immediate(4), 0);
      agx_index v = load(b, off, false);
      agx_unit_test(b, v);
   });
}

TEST_F(HoistLoads, AboveUniformStoreDisjoint)
{
   CASE(
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_uniform_store(b, x, agx_immediate(32), 1);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
      },
      {
         agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
         agx_index v = load(b, off, false);
         agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
         agx_uniform_store(b, x, agx_immediate(32), 1);
         agx_unit_test(b, v);
      });
}

TEST_F(HoistLoads, CoherentLoadStays)
{
   NEGCASE({
      agx_index x = agx_mov_imm(b, AGX_SIZE_32, 0xcafe);
      agx_index off = agx_mov_imm(b, AGX_SIZE_32, 4);
      agx_index y = agx_fadd(b, x, x);
      agx_index v = load(b, off, true);
      agx_unit_test(b, v);
      agx_unit_test(b, y);
   });
}

TEST_F(HoistLoads, ComputeSyncCrossesThreadgroupBarrierStopsAtMemoryBarrier)
{
   /* In compute shaders, threadgroup_barrier synchronizes threadgroup/shared
    * execution only (not global device memory), so an independent non-coherent
    * device_load may hoist across it. A preceding memory_barrier orders global
    * memory and must stop the load from crossing past the memory boundary.
    */
   CASE(
      {
         agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
         agx_memory_barrier(b);
         agx_threadgroup_barrier(b);
         agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_unit_test(b, v);
      },
      {
         agx_index base = agx_mov_imm(b, AGX_SIZE_32, 0x100);
         agx_memory_barrier(b);
         agx_index off = agx_iadd(b, base, agx_immediate(4), 0);
         agx_index v = load(b, off, false);
         agx_threadgroup_barrier(b);
         agx_unit_test(b, v);
      });
}
