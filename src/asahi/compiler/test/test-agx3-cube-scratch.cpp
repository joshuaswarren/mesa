/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_builder.h"
#include "agx_test.h"

/* A cube sample gets its own CEM scratch: an undefined four-word vector it
 * reads, defined right before it, so RA places the scratch rather than the
 * packer stacking it above the allocation. */
static agx_instr *
cube_sample(agx_builder *b, enum agx_dim dim, enum agx_lod_mode lod)
{
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   return agx_texture_sample_to(b, agx_temp(b->shader, AGX_SIZE_16), coords,
                                agx_immediate(0), agx_immediate(0),
                                agx_immediate(0), agx_immediate(0),
                                agx_null(), dim, lod, 0xf, false, false, false,
                                AGX_GATHER_NONE, false);
}

TEST(AGX3CubeScratch, CubeSamplesReadAnUndefinedScratchVector)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   cube_sample(b, AGX_DIM_CUBE_ARRAY, AGX_LOD_MODE_AUTO_LOD);
   cube_sample(b, AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD);
   /* The bias with minimum keeps the shared scratch and the capped file. */
   cube_sample(b, AGX_DIM_CUBE_ARRAY, AGX_LOD_MODE_AUTO_LOD_BIAS_MIN);
   cube_sample(b, AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD);

   ASSERT_TRUE(agx3_reserve_cube_scratch(b->shader));

   agx_instr *collect = NULL;
   unsigned samples = 0, own = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_COLLECT) {
         collect = I;
         continue;
      }
      ASSERT_EQ(I->op, AGX_OPCODE_TEXTURE_SAMPLE);
      ++samples;
      const bool expect_own = (I->dim == AGX_DIM_CUBE ||
                               I->dim == AGX_DIM_CUBE_ARRAY) &&
                              I->lod_mode != AGX_LOD_MODE_AUTO_LOD_BIAS_MIN;
      if (!expect_own) {
         EXPECT_EQ(I->nr_srcs, 6u);
         continue;
      }
      ++own;
      ASSERT_EQ(I->nr_srcs, 7u);
      ASSERT_NE(collect, nullptr);
      EXPECT_EQ(collect->dest[0].value, I->src[6].value);
      EXPECT_EQ(agx_index_size_16(I->src[6]), 8u);
      ASSERT_EQ(collect->nr_srcs, 4u);
      agx_foreach_src(collect, s)
         EXPECT_EQ(collect->src[s].type, AGX_INDEX_UNDEF);
      collect = NULL;
   }
   EXPECT_EQ(samples, 4u);
   EXPECT_EQ(own, 2u);

   /* RA may be retried after a failed allocation: no second scratch. */
   EXPECT_FALSE(agx3_reserve_cube_scratch(b->shader));
   ralloc_free(mem);
}

/* A cube sample that reads its scratch is valid IR: the scratch source is
 * four words wide. agx_validate checks only in builds with assertions. */
TEST(AGX3CubeScratch, ScratchSourceValidates)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   for (unsigned array = 0; array < 2; ++array) {
      const unsigned comps = 3 + array;
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, comps);
      agx_instr *collect = agx_collect_to(b, coords, comps);
      agx_foreach_src(collect, c)
         collect->src[c] = agx_undef(AGX_SIZE_32);
      agx_texture_sample_to(b, agx_vec_temp(b->shader, AGX_SIZE_16, 4), coords,
                            agx_immediate(0), agx_immediate(0),
                            agx_immediate(0), agx_immediate(0), agx_null(),
                            array ? AGX_DIM_CUBE_ARRAY : AGX_DIM_CUBE,
                            AGX_LOD_MODE_AUTO_LOD, 0xf, false, false, false,
                            AGX_GATHER_NONE, false);
   }
   agx_validate(b->shader, "cube sample test shader");
   ASSERT_TRUE(agx3_reserve_cube_scratch(b->shader));
   agx_validate(b->shader, "cube-array scratch");
   ralloc_free(mem);
}

/* On AGX3C, cube sampling reserves halves 8..15 for the low CEM scratch bank.
 * Linked shaders may preload ABI inputs in halves 8..15 and keep them live
 * across a cube sample; RA must evict those preloads out of 8..15 at entry so
 * CEM cannot clobber them. */
TEST(AGX3CubeScratch, Agx3cEvictsPreloadsOutOfLowCemBank)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_shader_info out = {};
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->out = &out;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;

   agx_index p4 = agx_preload(b, agx_register(8, AGX_SIZE_32));
   agx_index p6 = agx_preload(b, agx_register(12, AGX_SIZE_32));

   agx_index c_vals[3];
   for (unsigned c = 0; c < 3; ++c)
      c_vals[c] = agx_mov_imm(b, 32, 0x3f800000);
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
   agx_instr *collect = agx_collect_to(b, coords, 3);
   for (unsigned c = 0; c < 3; ++c)
      collect->src[c] = c_vals[c];

   agx_index sampled = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_texture_sample_to(b, sampled, coords, agx_immediate(0),
                         agx_immediate(0), agx_uniform(0, AGX_SIZE_64),
                         agx_immediate(0), agx_null(), AGX_DIM_CUBE,
                         AGX_LOD_MODE_LOD_MIN, 0xf, false, false, false,
                         AGX_GATHER_NONE, false);
   agx_instr *split = agx_split(b, 4, sampled);
   for (unsigned c = 0; c < 4; ++c)
      split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);

   agx_index sum = agx_iadd(b, p4, p6, 0);
   agx_instr *final_add = agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                      sum, split->dest[0], 0);
   agx_export(b, final_add->dest[0], 16);
   agx_stop(b);

   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   ASSERT_TRUE(b->shader->agx3_cem_low_scratch);
   agx_validate(b->shader, "agx3c cem preload eviction");

   /* After the entry eviction moves, the live operands consumed after the cube
    * sample must not sit in halves 8..15. */
   bool saw_sum = false;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_IADD && I->src[0].type == AGX_INDEX_REGISTER &&
          I->src[1].type == AGX_INDEX_REGISTER) {
         saw_sum = true;
         EXPECT_GE(I->src[0].value, 16u);
         EXPECT_GE(I->src[1].value, 16u);
      }
   }
   EXPECT_TRUE(saw_sum);
   ralloc_free(mem);
}
