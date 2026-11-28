/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include <array>
extern "C" {
#include "agx3_slot_core.h"
}

static agx3_slot_instr
load(unsigned reg, bool join = true, bool prefer = false)
{
   agx3_slot_instr i{};
   i.is_async = true;
   i.nr_dsts = 1;
   i.dst[0] = reg;
   i.join_class = join ? 1 : 0;
   i.join_limit = join ? 8 : 0;
   i.prefer_join = prefer;
   return i;
}

TEST(DeviceLoadBatches, IndependentReadsShareQueues)
{
   std::array<agx3_slot_instr, 32> i;
   std::array<agx3_slot_result, 32> r;
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n);
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
   for (unsigned n = 0; n < i.size(); ++n) {
      EXPECT_EQ(r[n].wait_mask, 0);
      EXPECT_EQ(r[n].slot, n % 8 + 1);
   }
}

TEST(DeviceLoadBatches, RetiredAliasesDoNotConsumeNewGeneration)
{
   std::array<agx3_slot_instr, 37> i{};
   std::array<agx3_slot_result, 37> r;
   for (unsigned n = 0; n < 32; ++n) i[n] = load(n);
   i[32].nr_srcs = i[33].nr_srcs = i[35].nr_srcs = i[36].nr_srcs = 1;
   i[32].src[0] = 0; i[33].src[0] = 8;
   i[34] = load(40); i[34].preferred_slot = 1;
   i[35].src[0] = 16; i[36].src[0] = 40;
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
   EXPECT_EQ(r[32].wait_mask, 1);
   EXPECT_EQ(r[33].wait_mask, 0);
   EXPECT_EQ(r[35].wait_mask, 0);
   EXPECT_EQ(r[36].wait_mask, 1);
}

TEST(DeviceLoadBatches, BoundAndMixedEngineRetirement)
{
   std::array<agx3_slot_instr, 9> i;
   std::array<agx3_slot_result, 9> r;
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n);
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 1, r.data()));
   for (unsigned n = 0; n < 8; ++n) EXPECT_EQ(r[n].wait_mask, 0);
   EXPECT_EQ(r[8].wait_mask, 1);
   i[2] = load(15, false); i[2].preferred_slot = 1;
   ASSERT_TRUE(agx3_slot_assign(i.data(), 3, 1, r.data()));
   EXPECT_EQ(r[2].wait_mask, 1);
   i[2] = {}; i[2].nr_dsts = 1; i[2].dst[0] = 0;
   ASSERT_TRUE(agx3_slot_assign(i.data(), 3, 1, r.data()));
   EXPECT_EQ(r[2].wait_mask, 1);
}

TEST(DeviceLoadBatches, GroupedPolicyAndLegacyPolicy)
{
   std::array<agx3_slot_instr, 32> i;
   std::array<agx3_slot_result, 32> r;
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n, true, true);
   ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
   for (unsigned n = 0; n < i.size(); ++n) {
      EXPECT_EQ(r[n].wait_mask, 0);
      EXPECT_EQ(r[n].slot, n / 8 + 1);
   }
   for (unsigned n = 0; n < i.size(); ++n) i[n] = load(n, false);
   ASSERT_TRUE(agx3_slot_assign(i.data(), 9, 8, r.data()));
   EXPECT_NE(r[8].wait_mask, 0);
}

TEST(DeviceLoadBatches, FreeSlotPreferredOverLiveJoin)
{
   /* When !prefer_join, pick an already-waited/free slot before joining a live
    * slot that appears earlier in round-robin order. Once every slot up to cap
    * is live again, still join a compatible live slot without stalling. */
   {
      std::array<agx3_slot_instr, 13> i{};
      std::array<agx3_slot_result, 13> r{};
      for (unsigned n = 0; n < 8; ++n)
         i[n] = load(n);
      i[8].nr_srcs = 1;
      i[8].src[0] = 1; /* retire slot 2 while slot 1 stays live */
      i[9] = load(20);
      i[10] = load(21);
      i[11].nr_srcs = 1;
      i[11].src[0] = 0;
      i[12].nr_srcs = 1;
      i[12].src[0] = 20;
      ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 8, r.data()));
      EXPECT_EQ(r[8].wait_mask, 1u << 1);
      EXPECT_EQ(r[9].slot, 2);
      EXPECT_EQ(r[9].wait_mask, 0);
      EXPECT_EQ(r[10].slot, 3);
      EXPECT_EQ(r[10].wait_mask, 0);
      EXPECT_EQ(r[11].wait_mask, 1u << 0);
      EXPECT_EQ(r[12].wait_mask, 1u << 1);
   }

   /* An earlier live joinable slot (placed via preferred_slot) does not steal
    * an instruction from untouched free slots later in round-robin order. */
   {
      std::array<agx3_slot_instr, 3> i{};
      std::array<agx3_slot_result, 3> r{};
      i[0] = load(0);
      i[0].preferred_slot = 2;
      i[1] = load(1);
      i[2] = load(2);
      ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 6, r.data()));
      EXPECT_EQ(r[0].slot, 2);
      EXPECT_EQ(r[1].slot, 1);
      EXPECT_EQ(r[2].slot, 3);
      EXPECT_EQ(r[2].wait_mask, 0);
   }

   /* max_slot caps both the free-slot pass and the live-join pass. */
   {
      std::array<agx3_slot_instr, 6> i{};
      std::array<agx3_slot_result, 6> r{};
      for (unsigned n = 0; n < 3; ++n) {
         i[n] = load(n);
         i[n].max_slot = 3;
      }
      i[3].nr_srcs = 1;
      i[3].src[0] = 1; /* retire slot 2 while slots 1 and 3 stay live */
      i[4] = load(10);
      i[4].max_slot = 3;
      i[5] = load(11);
      i[5].max_slot = 3;
      ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 6, r.data()));
      EXPECT_EQ(r[3].wait_mask, 1u << 1);
      EXPECT_EQ(r[4].slot, 2);
      EXPECT_EQ(r[4].wait_mask, 0);
      EXPECT_EQ(r[5].slot, 3);
      EXPECT_EQ(r[5].wait_mask, 0);
   }

   /* prefer_join keeps joining a live compatible slot (r[1]) even while other
    * slots are free, then takes a free slot (r[3]) once slot 1 is waited. */
   {
      std::array<agx3_slot_instr, 4> i{};
      std::array<agx3_slot_result, 4> r{};
      i[0] = load(0, true, true);
      i[1] = load(1, true, true);
      i[2].nr_srcs = 1;
      i[2].src[0] = 1; /* wait slot 1, leaving slot 2 free */
      i[3] = load(2, true, true);
      ASSERT_TRUE(agx3_slot_assign(i.data(), i.size(), 6, r.data()));
      EXPECT_EQ(r[0].slot, 1);
      EXPECT_EQ(r[1].slot, 1);
      EXPECT_EQ(r[1].wait_mask, 0);
      EXPECT_EQ(r[2].wait_mask, 1u << 0);
      EXPECT_EQ(r[3].slot, 2);
      EXPECT_EQ(r[3].wait_mask, 0);
   }
}

TEST(DeviceLoadBatches, InvalidContractsFailClosed)
{
   agx3_slot_instr i = load(0);
   agx3_slot_result r;
   i.join_limit = 0; EXPECT_FALSE(agx3_slot_assign(&i, 1, 8, &r));
   i.join_limit = 9; EXPECT_FALSE(agx3_slot_assign(&i, 1, 8, &r));
   i.join_limit = 8; i.is_async = false;
   EXPECT_FALSE(agx3_slot_assign(&i, 1, 8, &r));
}
