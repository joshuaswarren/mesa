/* SPDX-License-Identifier: MIT */
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <string>
#include <gtest/gtest.h>
#include "agx3_pack.h"
#include "agx3_tex_core.h"
#include "agx_test.h"

TEST(M5PackedLOD, NativeSharedAndLiveLoadWords)
{
   uint8_t out[AGX3_TEX_MAX_LENGTH], len;
   const uint8_t shared[] = {
      0x27,0x02,0x00,0x3a,0x2d,0x00,0xaa,0x02,0x00,0x00,0x40,0x30,0x10,0x00,
   };
   ASSERT_TRUE(agx3_tex_pack_float_lod_m5(6,0,false,true,0,out,&len));
   ASSERT_EQ(len, sizeof(shared));
   EXPECT_EQ(memcmp(out, shared, len), 0);

   /* The device-half value is read again after sampling. Native preserves
    * it and waits its slot8 load directly, without a carrier or fence. */
   const uint8_t live_load[] = {
      0x2f,0x00,0x00,0x9a,0x29,0x00,0xaa,0x02,0x00,0x00,0x40,0x30,0x10,0x00,
   };
   ASSERT_TRUE(agx3_tex_pack_float_lod_m5(16,0,false,false,128,out,&len));
   ASSERT_EQ(len, sizeof(live_load));
   EXPECT_EQ(memcmp(out, live_load, len), 0);
   for (unsigned waits=0; waits<256; ++waits)
      EXPECT_TRUE(agx3_tex_pack_float_lod_m5(16,0,false,false,waits,out,&len));
   EXPECT_FALSE(agx3_tex_pack_float_lod_m5(128,0,false,false,0,out,&len));
   EXPECT_FALSE(agx3_tex_pack_float_lod_m5(16,1,false,false,0,out,&len));
}

TEST(M5PackedLOD, PureProducersShareThroughCSE)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index half = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, half, agx_register(6, AGX_SIZE_16));
   agx_index word[2];
   agx_instr *samples[2];
   for (unsigned i=0; i<2; ++i) {
      word[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_pack_sampler_lod_agx3_to(b, word[i], half);
      agx_index coords = agx_register(8, AGX_SIZE_32);
      coords.channels_m1 = 1;
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      samples[i] = agx_texture_sample_to(
         b, result, coords, word[i], agx_zero(), agx_immediate(i),
         agx_zero(), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      samples[i]->memory_is_packed = true;
      samples[i]->mask = 15;
      agx_unit_test(b, result);
   }
   agx_opt_cse(b->shader);
   agx_dce(b->shader, true);
   unsigned packs = 0;
   agx_foreach_instr_global(b->shader, I)
      packs += I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3;
   EXPECT_EQ(packs, 1u);
   EXPECT_TRUE(agx_is_equiv(samples[0]->src[1], samples[1]->src[1]));
   EXPECT_EQ(samples[0]->src[1].size, AGX_SIZE_32);
   EXPECT_EQ(samples[1]->src[1].size, AGX_SIZE_32);
   ralloc_free(mem);
}

TEST(M5PackedLOD, ReusedWordPublishesWithoutDiscardingFirstUse)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.fs.ignore_tib_dependencies = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_register(6, AGX_SIZE_16);
   input.discard = true;
   agx_pack_sampler_lod_agx3_to(b, agx_register(0, AGX_SIZE_32), input);
   for (unsigned i=0; i<2; ++i) {
      agx_index word = agx_register(0, AGX_SIZE_32);
      word.discard = i == 1;
      agx_index coords = agx_register(8 + i*4, AGX_SIZE_32);
      coords.channels_m1 = 1;
      coords.discard = true;
      agx_index result = agx_register(16 + i*8, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_instr *I = agx_texture_sample_to(
         b, result, coords, word, agx_zero(), agx_immediate(i),
         agx_zero(), agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0,
         false, false, false, AGX_GATHER_NONE, false);
      I->memory_is_packed = true;
      I->mask = 15;
   }
   agx_stop(b);
   util_dynarray bin;
   util_dynarray_init(&bin, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &bin));
   /* Two independent eight-half groups retain both samples in flight. */
   EXPECT_EQ(b->shader->agx3_staging_halfs, 16u);
   const uint8_t first[] = {0x3b,0x80,0x21,0x04};
   const uint8_t *start = (const uint8_t *)bin.data;
   EXPECT_NE(std::search(start, start+bin.size, std::begin(first),
                        std::end(first)), start+bin.size);
   ralloc_free(mem);
}

TEST(M5PackedLOD, VertexImageReadPublishesCoordinates)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_VERTEX;
   agx_instr *I = agx_alloc_instr(b, AGX_OPCODE_IMAGE_LOAD, 1, 6);
   I->dest[0] = agx_register(16, AGX_SIZE_32);
   I->dest[0].channels_m1 = 3;
   I->src[0] = agx_register(4, AGX_SIZE_32);
   I->src[0].channels_m1 = 1;
   I->src[1] = agx_immediate(0);
   I->src[2] = agx_uniform(4, AGX_SIZE_64);
   I->src[3] = agx_register(24, AGX_SIZE_64);
   I->src[4] = agx_immediate(0);
   I->src[5] = agx_null();
   I->dim = AGX_DIM_2D;
   I->lod_mode = AGX_LOD_MODE_AUTO_LOD;
   I->mask = 15;
   agx_builder_insert(&b->cursor, I);
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_TRUE(b->shader->agx3_staging_known);
   EXPECT_EQ(b->shader->agx3_staging_halfs, 4u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(M5PackedLOD, Bank8MinimumOwnsCompleteStagingRange)
{
   for (bool shadow : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.fs.ignore_tib_dependencies = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index coords = agx_register(8, AGX_SIZE_32);
      coords.channels_m1 = 2;
      agx_index result = agx_register(24, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_index bias_min = agx_register(0, AGX_SIZE_32);
      agx_index offset = agx_register(16, AGX_SIZE_32);
      offset.channels_m1 = shadow ? 1 : 0;
      agx_instr *I = agx_texture_sample_to(
         b, result, coords, bias_min, agx_zero(), agx_zero(),
         agx_zero(), offset, shadow ? AGX_DIM_2D_ARRAY : AGX_DIM_3D,
         AGX_LOD_MODE_AUTO_LOD_BIAS_MIN, 0, true, shadow, false,
         AGX_GATHER_NONE, false);
      I->mask = shadow ? 1 : 15;
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
      EXPECT_TRUE(b->shader->agx3_staging_known);
      EXPECT_EQ(b->shader->agx3_staging_halfs, 10u);
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }

}

TEST(M5PackedLOD, LastUseSplitMovesHighHalfIntoSamplerBank)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   key.has_scratch = true;
   agx_shader_info info = {};
   b->shader->out = &info;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   b->shader->nir->info.workgroup_size[0] = 32;
   b->shader->nir->info.workgroup_size[1] = 1;
   b->shader->nir->info.workgroup_size[2] = 1;

   agx_index pair = agx_temp(b->shader, AGX_SIZE_32);
   agx_preload_to(b, pair, agx_register(188, AGX_SIZE_32));
   agx_instr *split = agx_split(b, 2, pair);
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_16);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_16);
   agx_index packed = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *pack = agx_pack_sampler_lod_agx3_to(b, packed, split->dest[0]);
   agx_unit_test(b, split->dest[1]);
   agx_unit_test(b, packed);

   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_LT(pack->src[0].value, 128u);
   ralloc_free(mem);
}

TEST(M5PackedLOD, LoopInvariantSingleUseScalarLodPacks)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   body->loop_header = true;
   agx_block_add_successor(entry, body);
   agx_block_add_successor(body, body);
   agx_block_add_successor(body, exit);

   b->cursor = agx_after_block(entry);
   agx_index lod = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, lod, agx_register(6, AGX_SIZE_16));
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, agx_register(8, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *sample = agx_texture_sample_to(
      b, result, coords, lod, agx_zero(), agx_immediate(0), agx_zero(),
      agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   sample->mask = 15;
   agx_unit_test(b, result);

   /* Single static use in a uniform loop is still live across the back-edge. */
   agx_lower_uniform_sources(b->shader);

   unsigned packs = 0;
   agx_foreach_instr_global(b->shader, I)
      packs += I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3;
   EXPECT_EQ(packs, 1u);
   EXPECT_TRUE(sample->memory_is_packed);
   EXPECT_EQ(sample->src[1].size, AGX_SIZE_32);
   ralloc_free(mem);

   /* A straight-line killed single-use scalar LOD stays on the direct path. */
   mem = ralloc_context(NULL);
   b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   lod = agx_temp(b->shader, AGX_SIZE_16);
   agx_preload_to(b, lod, agx_register(6, AGX_SIZE_16));
   coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, agx_register(8, AGX_SIZE_32));
   result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   sample = agx_texture_sample_to(
      b, result, coords, lod, agx_zero(), agx_immediate(0), agx_zero(),
      agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   sample->mask = 15;
   agx_unit_test(b, result);

   agx_lower_uniform_sources(b->shader);

   packs = 0;
   agx_foreach_instr_global(b->shader, I)
      packs += I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3;
   EXPECT_EQ(packs, 0u);
   EXPECT_FALSE(sample->memory_is_packed);
   EXPECT_EQ(sample->src[1].size, AGX_SIZE_16);
   ralloc_free(mem);
}

TEST(M5PackedLOD, PostCseMergedScalarLodPacks)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;

   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, agx_register(8, AGX_SIZE_32));

   agx_instr *samples[2];
   for (unsigned i = 0; i < 2; ++i) {
      agx_index lod = agx_mov(b, agx_uniform(14, AGX_SIZE_16));
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      samples[i] = agx_texture_sample_to(
         b, result, coords, lod, agx_zero(), agx_immediate(i), agx_zero(),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
         AGX_GATHER_NONE, false);
      samples[i]->mask = 15;
      agx_unit_test(b, result);
   }

   /* CSE merges the two single-use copies into one live SSA; the follow-up
    * lowering must pack both samples and share one producer in the block. */
   agx_opt_cse(b->shader);
   agx3_lower_m5_live_scalar_lods(b->shader);

   unsigned packs = 0;
   agx_foreach_instr_global(b->shader, I)
      packs += I->op == AGX_OPCODE_PACK_SAMPLER_LOD_AGX3;
   EXPECT_EQ(packs, 1u);
   EXPECT_TRUE(samples[0]->memory_is_packed);
   EXPECT_TRUE(samples[1]->memory_is_packed);
   EXPECT_TRUE(agx_is_equiv(samples[0]->src[1], samples[1]->src[1]));
   EXPECT_EQ(samples[0]->src[1].size, AGX_SIZE_32);
   ralloc_free(mem);
}
