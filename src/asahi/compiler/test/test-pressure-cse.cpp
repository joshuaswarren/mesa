/* SPDX-License-Identifier: MIT */
#include <vector>
#include <gtest/gtest.h>
#include "agx_test.h"

static unsigned
count_uniform_copies(agx_context *ctx, unsigned uniform)
{
   unsigned count = 0;
   agx_foreach_instr_global(ctx, I) {
      count += I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM &&
               I->src[0].value == uniform;
   }
   return count;
}

TEST(PressureCSE, SharesOverlappingCopies)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index a = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_index c = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_instr *use_a = agx_unit_test(b, a);
   agx_instr *use_c = agx_unit_test(b, c);
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 1u);
   EXPECT_TRUE(agx_is_equiv(use_a->src[0], use_c->src[0]));
   ralloc_free(mem);
}

TEST(PressureCSE, PreservesCopiesAcrossFullPressureGap)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_index x = agx_mov(b, agx_uniform(10, AGX_SIZE_32));
   agx_index y = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
   agx_unit_test(b, x);
   agx_unit_test(b, y);
   agx_index next = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_index last = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_instr *use_next = agx_unit_test(b, next);
   agx_instr *use_last = agx_unit_test(b, last);
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   EXPECT_TRUE(agx_is_equiv(use_next->src[0], use_last->src[0]));
   ralloc_free(mem);
}

TEST(PressureCSE, DoesNotExtendLifetimesIntoPressureHeadroom)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(10, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_index x = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
   agx_index y = agx_mov(b, agx_uniform(14, AGX_SIZE_32));
   agx_index z = agx_mov(b, agx_uniform(16, AGX_SIZE_32));
   agx_unit_test(b, x);
   agx_unit_test(b, y);
   agx_unit_test(b, z);
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   ralloc_free(mem);
}

TEST(PressureCSE, PreservesIndependentBlockDefinitions)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_block *block = agx_test_block(b->shader);
   b->cursor = agx_after_block(block);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_opt_cse_pressure(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   ralloc_free(mem);
}

static unsigned
peak_pressure(agx_context *ctx)
{
   std::vector<unsigned> first(ctx->alloc, ~0u), last(ctx->alloc, 0),
      width(ctx->alloc, 0);
   unsigned position = 0;
   agx_foreach_instr_global(ctx, I) {
      ++position;
      agx_foreach_ssa_dest(I, d) {
         first[I->dest[d].value] = position;
         width[I->dest[d].value] = agx_index_size_16(I->dest[d]);
      }
      agx_foreach_ssa_src(I, s)
         last[I->src[s].value] = position;
   }
   unsigned peak = 0;
   for (unsigned p = 1; p <= position; ++p) {
      unsigned live = 0;
      for (unsigned v = 0; v < ctx->alloc; ++v)
         if (first[v] <= p && p <= last[v])
            live += width[v];
      peak = MAX2(peak, live);
   }
   return peak;
}

static std::vector<unsigned>
observe_uniforms(agx_context *ctx)
{
   std::vector<unsigned> values(ctx->alloc, ~0u), observed;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV) {
         EXPECT_EQ(I->src[0].type, AGX_INDEX_UNIFORM);
         values[I->dest[0].value] = I->src[0].value;
      } else if (I->op == AGX_OPCODE_UNIT_TEST) {
         observed.push_back(values[I->src[0].value]);
      }
   }
   return observed;
}

TEST(PressureCSE, RandomizedValuesAndPressureInvariant)
{
   for (unsigned seed = 1; seed <= 128; ++seed) {
      SCOPED_TRACE(seed);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      std::vector<agx_index> values;
      unsigned state = seed;
      auto random = [&state]() {
         state = state * 1664525u + 1013904223u;
         return state;
      };
      for (unsigned i = 0; i < 64; ++i) {
         auto size = random() & 1 ? AGX_SIZE_16 : AGX_SIZE_32;
         values.push_back(agx_mov(b, agx_uniform(2 * (random() % 8), size)));
         if (random() & 3)
            agx_unit_test(b, values[random() % values.size()]);
      }
      for (unsigned i = 0; i < 32; ++i)
         agx_unit_test(b, values[random() % values.size()]);
      auto expected = observe_uniforms(b->shader);
      unsigned pressure = peak_pressure(b->shader);
      agx_opt_cse_pressure(b->shader);
      agx_dce(b->shader, true);
      EXPECT_EQ(observe_uniforms(b->shader), expected);
      EXPECT_LE(peak_pressure(b->shader), pressure);
      ralloc_free(mem);
   }
}

TEST(PressureCSE, ReplacementRangesStayWithinTheirBlock)
{
   for (unsigned gap : {0u, 31u, 4095u}) {
      for (bool empty_middle : {false, true}) {
         SCOPED_TRACE(gap);
         SCOPED_TRACE(empty_middle);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_block *first =
            list_first_entry(&b->shader->blocks, agx_block, link);
         agx_index a = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
         agx_index c = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
         agx_instr *local_c = agx_unit_test(b, c);
         b->shader->alloc += gap;
         agx_index d = agx_mov(b, agx_uniform(10, AGX_SIZE_32));
         agx_index e = agx_mov(b, agx_uniform(10, AGX_SIZE_32));
         agx_instr *local_e = agx_unit_test(b, e);

         agx_block *next = agx_test_block(b->shader);
         first->successors[0] = next;
         util_dynarray_append(&next->predecessors, first);
         if (empty_middle) {
            agx_block *last = agx_test_block(b->shader);
            next->successors[0] = last;
            util_dynarray_append(&last->predecessors, next);
            next = last;
         }
         b->cursor = agx_after_block(next);
         agx_instr *external_c = agx_unit_test(b, c);
         agx_instr *external_e = agx_unit_test(b, e);

         agx_opt_cse(b->shader);
         EXPECT_TRUE(agx_is_equiv(local_c->src[0], a));
         EXPECT_TRUE(agx_is_equiv(local_e->src[0], d));
         EXPECT_TRUE(agx_is_equiv(external_c->src[0], c));
         EXPECT_TRUE(agx_is_equiv(external_e->src[0], e));
         ralloc_free(mem);
      }
   }
}

TEST(PressureCSE, ReplacementKeyZeroAndRepeatedEmptyBlocks)
{
   for (unsigned gap : {0u, 31u, 4095u}) {
      for (bool pressure : {false, true}) {
         SCOPED_TRACE(gap);
         SCOPED_TRACE(pressure);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_block *last =
            list_first_entry(&b->shader->blocks, agx_block, link);
         agx_index c = agx_temp(b->shader, AGX_SIZE_32);
         agx_index a = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
         agx_mov_to(b, c, agx_uniform(8, AGX_SIZE_32));
         agx_unit_test(b, a);
         agx_instr *local_c = agx_unit_test(b, c);
         b->shader->alloc += gap;
         agx_index d = agx_mov(b, agx_uniform(10, AGX_SIZE_32));
         agx_index e = agx_mov(b, agx_uniform(10, AGX_SIZE_32));
         agx_unit_test(b, d);
         agx_instr *local_e = agx_unit_test(b, e);

         for (unsigned n = 0; n < 3; ++n) {
            agx_block *next = agx_test_block(b->shader);
            last->successors[0] = next;
            util_dynarray_append(&next->predecessors, last);
            last = next;
         }
         b->cursor = agx_after_block(last);
         agx_instr *external_c = pressure ? NULL : agx_unit_test(b, c);
         agx_instr *external_e = pressure ? NULL : agx_unit_test(b, e);
         agx_index f = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
         agx_index g = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
         agx_unit_test(b, f);
         agx_instr *local_g = agx_unit_test(b, g);

         if (pressure)
            agx_opt_cse_pressure(b->shader);
         else
            agx_opt_cse(b->shader);
         EXPECT_TRUE(agx_is_equiv(local_c->src[0], a));
         EXPECT_TRUE(agx_is_equiv(local_e->src[0], d));
         EXPECT_TRUE(agx_is_equiv(local_g->src[0], f));
         if (!pressure) {
            EXPECT_TRUE(agx_is_equiv(external_c->src[0], c));
            EXPECT_TRUE(agx_is_equiv(external_e->src[0], e));
         }
         ralloc_free(mem);
      }
   }
}

TEST(PressureCSE, BoundedSharesCopiesBelowThePeak)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index x = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
   agx_index y = agx_mov(b, agx_uniform(14, AGX_SIZE_32));
   agx_index z = agx_mov(b, agx_uniform(16, AGX_SIZE_32));
   agx_unit_test(b, x);
   agx_unit_test(b, y);
   agx_unit_test(b, z);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(10, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   unsigned pressure = peak_pressure(b->shader);
   agx_opt_cse_bounded(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 1u);
   EXPECT_LE(peak_pressure(b->shader), pressure);
   ralloc_free(mem);
}

TEST(PressureCSE, BoundedKeepsCopiesAcrossThePeak)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_index x = agx_mov(b, agx_uniform(12, AGX_SIZE_32));
   agx_index y = agx_mov(b, agx_uniform(14, AGX_SIZE_32));
   agx_unit_test(b, x);
   agx_unit_test(b, y);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_opt_cse_bounded(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 2u);
   ralloc_free(mem);
}

TEST(PressureCSE, BoundedRandomizedValuesAndPressureInvariant)
{
   for (unsigned seed = 1; seed <= 128; ++seed) {
      SCOPED_TRACE(seed);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      std::vector<agx_index> values;
      unsigned state = seed;
      auto random = [&state]() {
         state = state * 1664525u + 1013904223u;
         return state;
      };
      for (unsigned i = 0; i < 64; ++i) {
         auto size = random() & 1 ? AGX_SIZE_16 : AGX_SIZE_32;
         values.push_back(agx_mov(b, agx_uniform(2 * (random() % 8), size)));
         if (random() & 3)
            agx_unit_test(b, values[random() % values.size()]);
      }
      for (unsigned i = 0; i < 32; ++i)
         agx_unit_test(b, values[random() % values.size()]);
      auto expected = observe_uniforms(b->shader);
      unsigned pressure = peak_pressure(b->shader);
      agx_opt_cse_bounded(b->shader);
      agx_dce(b->shader, true);
      EXPECT_EQ(observe_uniforms(b->shader), expected);
      EXPECT_LE(peak_pressure(b->shader), pressure);
      ralloc_free(mem);
   }
}

TEST(PressureCSE, BoundedMultiBlockPressureReuse)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_block *b0 = list_first_entry(&b->shader->blocks, agx_block, link);

   /* Block 0 establishes a high global pressure peak (4 x 32-bit = 8 halves). */
   agx_index p0 = agx_mov(b, agx_uniform(0, AGX_SIZE_32));
   agx_index p1 = agx_mov(b, agx_uniform(2, AGX_SIZE_32));
   agx_index p2 = agx_mov(b, agx_uniform(4, AGX_SIZE_32));
   agx_index p3 = agx_mov(b, agx_uniform(6, AGX_SIZE_32));
   agx_unit_test(b, p0);
   agx_unit_test(b, p1);
   agx_unit_test(b, p2);
   agx_unit_test(b, p3);

   /* Block 1 has low pressure, so extending uniform 8 across uniform 10 fits
    * strictly under block 0's peak. */
   agx_block *b1 = agx_test_block(b->shader);
   b0->successors[0] = b1;
   util_dynarray_append(&b1->predecessors, b0);
   b->cursor = agx_after_block(b1);
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(10, AGX_SIZE_32)));
   agx_unit_test(b, agx_mov(b, agx_uniform(8, AGX_SIZE_32)));

   /* Block 2 reaches the global peak between two copies of uniform 12, so it
    * must not merge them across the peak. */
   agx_block *b2 = agx_test_block(b->shader);
   b1->successors[0] = b2;
   util_dynarray_append(&b2->predecessors, b1);
   b->cursor = agx_after_block(b2);
   agx_unit_test(b, agx_mov(b, agx_uniform(12, AGX_SIZE_32)));
   agx_index q0 = agx_mov(b, agx_uniform(14, AGX_SIZE_32));
   agx_index q1 = agx_mov(b, agx_uniform(16, AGX_SIZE_32));
   agx_index q2 = agx_mov(b, agx_uniform(18, AGX_SIZE_32));
   agx_index q3 = agx_mov(b, agx_uniform(20, AGX_SIZE_32));
   agx_unit_test(b, q0);
   agx_unit_test(b, q1);
   agx_unit_test(b, q2);
   agx_unit_test(b, q3);
   agx_unit_test(b, agx_mov(b, agx_uniform(12, AGX_SIZE_32)));

   agx_opt_cse_bounded(b->shader);
   agx_dce(b->shader, true);
   EXPECT_EQ(count_uniform_copies(b->shader, 8), 1u);
   EXPECT_EQ(count_uniform_copies(b->shader, 12), 2u);
   ralloc_free(mem);
}

/* Note: the incremental-vs-reference delta cross-check inside
 * agx_pressure_schedule() is compiled under #ifndef NDEBUG, so this test
 * verifies exact delta equivalence in assert builds and non-regressing peak
 * pressure in both assert and release builds. */
TEST(PressureSchedule, IncrementalDeltaMatchesWithDuplicateModifierSources)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   struct agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->key = &key;

   /* Build an interleaved high-pressure sequence that also exercises same-SSA
    * operands with different float modifiers (fadd x, -x), which are not
    * agx_is_equiv duplicates. */
   agx_index a = agx_mov(b, agx_uniform(0, AGX_SIZE_32));
   agx_index c = agx_mov(b, agx_uniform(2, AGX_SIZE_32));
   agx_index d = agx_mov(b, agx_uniform(4, AGX_SIZE_32));
   agx_index e = agx_mov(b, agx_uniform(6, AGX_SIZE_32));
   agx_index s0 = agx_fadd(b, a, agx_neg(a));
   agx_index s1 = agx_fadd(b, c, c);
   agx_index s2 = agx_fmul(b, d, agx_abs(d));
   agx_index s3 = agx_fadd(b, s0, e);
   agx_index s4 = agx_fadd(b, s1, s2);
   agx_index out = agx_fadd(b, s3, s4);
   agx_unit_test(b, out);

   unsigned before = peak_pressure(b->shader);
   agx_pressure_schedule(b->shader);
   EXPECT_LE(peak_pressure(b->shader), before);
   ralloc_free(mem);
}

TEST(RegisterAllocate, FindRegsSimpleWordBoundaryMasks)
{
   auto ref_find = [](const BITSET_WORD *used, unsigned count, unsigned align,
                      unsigned minimum, unsigned limit, unsigned *out) -> bool {
      unsigned start = ALIGN_POT(minimum, align);
      for (unsigned reg = start; reg + count <= limit; reg += align) {
         if (!BITSET_TEST_COUNT(used, reg, count)) {
            *out = reg;
            return true;
         }
      }
      return false;
   };

   const BITSET_WORD patterns[][3] = {
      {0x00000000u, 0x00000000u, 0x00000000u},
      {0xffffffffu, 0x00000000u, 0xffffffffu},
      {0x7fffffffu, 0xffffffffu, 0x7fffffffu}, /* only bit 31 free in word 0/2 */
      {0x0000ffffu, 0xffff0000u, 0x00010000u}, /* align-16 half-word boundaries */
      {0x00010001u, 0x80008000u, 0x00000000u}, /* single bit blocks each 16-slot */
      {0x55555555u, 0xaaaaaaaau, 0xf0f0f0f0u},
      {0x80000001u, 0x00008000u, 0x00010000u},
   };
   const unsigned aligns[] = {1, 2, 4, 8, 16};
   const unsigned minimums[] = {0, 1, 7, 15, 16, 17, 30, 31, 32, 33, 48, 63};
   const unsigned limits[] = {16, 31, 32, 33, 47, 48, 63, 64, 80, 96};

   for (const auto &pat : patterns) {
      BITSET_DECLARE(used, 96);
      used[0] = pat[0];
      used[1] = pat[1];
      used[2] = pat[2];

      for (unsigned align : aligns) {
         for (unsigned min : minimums) {
            for (unsigned lim : limits) {
               unsigned got = 0xdeadbeefu, ref = 0xdeadbeefu;
               bool ok = agx_ra_find_regs_simple(used, align, align, min, lim,
                                                 &got);
               bool ref_ok = ref_find(used, align, align, min, lim, &ref);
               ASSERT_EQ(ok, ref_ok)
                  << "align=" << align << " min=" << min << " lim=" << lim;
               if (ok) {
                  ASSERT_EQ(got, ref)
                     << "align=" << align << " min=" << min << " lim=" << lim;
               }
            }
         }
      }

      /* Also check the count != align fallback path across word boundaries. */
      for (unsigned min : {0u, 15u, 30u, 31u, 32u}) {
         for (unsigned lim : {32u, 35u, 64u, 96u}) {
            unsigned got = 0xdeadbeefu, ref = 0xdeadbeefu;
            bool ok = agx_ra_find_regs_simple(used, 3, 2, min, lim, &got);
            bool ref_ok = ref_find(used, 3, 2, min, lim, &ref);
            ASSERT_EQ(ok, ref_ok) << "count=3 align=2 min=" << min << " lim=" << lim;
            if (ok)
               ASSERT_EQ(got, ref);
         }
      }
   }
}

TEST(PressureCSE, NormfltDistinguishesFormatAndChannels)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index src = agx_preload(b, agx_register(0, AGX_SIZE_16));

   agx_index u8_pair_a = agx_temp(b->shader, AGX_SIZE_16);
   u8_pair_a.channels_m1 = 1;
   agx_normflt_agx3_to(b, u8_pair_a, src, AGX_FORMAT_U8NORM);

   agx_index u8_pair_b = agx_temp(b->shader, AGX_SIZE_16);
   u8_pair_b.channels_m1 = 1;
   agx_normflt_agx3_to(b, u8_pair_b, src, AGX_FORMAT_U8NORM);

   agx_index s8_pair = agx_temp(b->shader, AGX_SIZE_16);
   s8_pair.channels_m1 = 1;
   agx_normflt_agx3_to(b, s8_pair, src, AGX_FORMAT_S8NORM);

   agx_index u8_scalar = agx_temp(b->shader, AGX_SIZE_16);
   u8_scalar.channels_m1 = 0;
   agx_normflt_agx3_to(b, u8_scalar, src, AGX_FORMAT_U8NORM);

   agx_instr *use_a = agx_unit_test(b, u8_pair_a);
   agx_instr *use_b = agx_unit_test(b, u8_pair_b);
   agx_instr *use_s8 = agx_unit_test(b, s8_pair);
   agx_instr *use_scalar = agx_unit_test(b, u8_scalar);

   agx_opt_cse(b->shader);
   agx_dce(b->shader, true);

   /* Identical normflt ops merge; differing format or channels_m1 stay distinct. */
   EXPECT_TRUE(agx_is_equiv(use_a->src[0], use_b->src[0]));
   EXPECT_FALSE(agx_is_equiv(use_a->src[0], use_s8->src[0]));
   EXPECT_FALSE(agx_is_equiv(use_a->src[0], use_scalar->src[0]));

   unsigned normflt_count = 0;
   agx_foreach_instr_global(b->shader, I)
      normflt_count += I->op == AGX_OPCODE_NORMFLT_AGX3;
   EXPECT_EQ(normflt_count, 3u);
   ralloc_free(mem);
}
