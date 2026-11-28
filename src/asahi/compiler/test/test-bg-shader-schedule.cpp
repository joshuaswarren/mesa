/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_test.h"

#include <gtest/gtest.h>

/* On AGX3B, agx3_tile_store_bank() assumes per-sample ST_TILE stores to the
 * same colour target / pixel_offset are serialized and reuse one register
 * vector in r0..r7 (value < 16 in agx3_emit_tile()).  Hoisting independent
 * sample loads ahead of repeated ST_TILEs to the same offset keeps multiple
 * payloads live simultaneously and pushes later store sources into r8+. */

namespace {

struct tile_schedule_shader {
   void *mem;
   agx_shader_key key;
   agx_builder *b;
   agx_block *body;
};

static void
shader_init(tile_schedule_shader *s, enum agx_chip chip, bool background)
{
   s->mem = ralloc_context(NULL);
   s->key = {};
   s->key.dev.chip = chip;
   s->key.dev.encoding = agx_chip_isa_encoding(chip);
   s->key.fs.background = background;
   s->key.fs.nr_samples = 4;

   s->b = agx_test_builder(s->mem);
   s->b->shader->key = &s->key;
   s->b->shader->stage = MESA_SHADER_FRAGMENT;
   s->b->shader->nir = rzalloc(s->mem, nir_shader);
   s->b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   s->body = agx_start_block(s->b->shader);
}

/* Emit n per-sample texture loads targeting the same pixel_offset (0).
 * With channels == 2 (4 halves per payload), hoisting all 4 loads adds 12
 * halves of pressure (within AGX_ILP_FS_SAMPLE_SLACK = 16), keeping 16 halves
 * of tile-store payloads live alongside a live preloaded value (2 halves in
 * r0), which forces at least one ST_TILE source out of AGX3B's 16-half r0..r7
 * single-target tile-store bank (value >= 16). */
static void
emit_multi_sample_tile_stores(tile_schedule_shader *s, unsigned n,
                              unsigned channels, bool through_alu)
{
   agx_builder *b = s->b;
   b->cursor = agx_after_block(s->body);

   agx_index coords = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index live = agx_preload(b, agx_register(2, AGX_SIZE_32));
   for (unsigned i = 0; i < n; ++i) {
      agx_index sample_coords = coords;
      if (through_alu)
         sample_coords = agx_fadd(b, coords, agx_immediate(i));

      agx_index texel = agx_vec_temp(b->shader, AGX_SIZE_32, channels);
      agx_texture_load_to(b, texel, sample_coords, agx_immediate(i),
                          agx_immediate(0), agx_immediate(0), agx_zero(),
                          agx_null(), AGX_DIM_2D_MS, AGX_LOD_MODE_AUTO_LOD,
                          BITFIELD_MASK(channels), false, false);

      agx_st_tile(b, texel, agx_immediate(1u << i), agx_null(),
                  AGX_FORMAT_I32, BITFIELD_MASK(channels), 0, false, 0);
   }
   /* Keep `live` in r1 across the hoisted texture loads so RA cannot reuse its
    * slot before the first ST_TILE fires. */
   agx_st_tile(b, live, agx_immediate(0xff), agx_null(),
               AGX_FORMAT_I32, 0x1, 8, false, 0);
   agx_stop(b);
}

static unsigned
loads_before_first_tile_store(agx_block *block)
{
   unsigned loads = 0;
   agx_foreach_instr_in_block(block, I) {
      if (I->op == AGX_OPCODE_ST_TILE)
         break;
      loads += I->op == AGX_OPCODE_TEXTURE_LOAD;
   }
   return loads;
}

static void
expect_tile_stores_in_low_bank(agx_context *ctx, unsigned expected_stores)
{
   unsigned stores = 0;
   agx_foreach_instr_global(ctx, I) {
      if (I->op != AGX_OPCODE_ST_TILE)
         continue;

      ++stores;
      EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
      EXPECT_LT(I->src[0].value, 16u);
   }
   EXPECT_EQ(stores, expected_stores);
}

} /* namespace */

TEST(BgShaderSchedule, BackgroundReloadKeepsLowTileStoreBankOnG17P)
{
   tile_schedule_shader s;
   shader_init(&s, AGX_CHIP_G17P, true);
   emit_multi_sample_tile_stores(&s, 4, 2, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(loads_before_first_tile_store(s.body), 1u);

   agx_ra(s.b->shader);
   expect_tile_stores_in_low_bank(s.b->shader, 5u);
   ralloc_free(s.mem);
}

TEST(BgShaderSchedule, BackgroundReloadKeepsLowTileStoreBankOnG16G)
{
   tile_schedule_shader s;
   shader_init(&s, AGX_CHIP_G16G, true);
   emit_multi_sample_tile_stores(&s, 4, 2, false);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(loads_before_first_tile_store(s.body), 1u);

   agx_ra(s.b->shader);
   expect_tile_stores_in_low_bank(s.b->shader, 5u);
   ralloc_free(s.mem);
}

TEST(BgShaderSchedule, NonBackgroundMultiSampleFragmentKeepsLowTileStoreBankOnG17P)
{
   tile_schedule_shader s;
   shader_init(&s, AGX_CHIP_G17P, false);
   emit_multi_sample_tile_stores(&s, 4, 2, true);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(loads_before_first_tile_store(s.body), 1u);

   agx_ra(s.b->shader);
   expect_tile_stores_in_low_bank(s.b->shader, 5u);
   ralloc_free(s.mem);
}

TEST(BgShaderSchedule, NonBackgroundMultiSampleFragmentKeepsLowTileStoreBankOnG16G)
{
   tile_schedule_shader s;
   shader_init(&s, AGX_CHIP_G16G, false);
   emit_multi_sample_tile_stores(&s, 4, 2, true);
   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(loads_before_first_tile_store(s.body), 1u);

   agx_ra(s.b->shader);
   expect_tile_stores_in_low_bank(s.b->shader, 5u);
   ralloc_free(s.mem);
}

TEST(BgShaderSchedule, NormalSingleStoreFragmentStillHoistsSamples)
{
   tile_schedule_shader s;
   shader_init(&s, AGX_CHIP_G17P, false);
   agx_builder *b = s.b;
   b->cursor = agx_after_block(s.body);

   agx_index coords = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index sum = agx_null();
   for (unsigned i = 0; i < 4; ++i) {
      agx_index texel = agx_temp(b->shader, AGX_SIZE_32);
      agx_texture_load_to(b, texel, coords, agx_immediate(i),
                          agx_immediate(0), agx_immediate(0), agx_zero(),
                          agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD,
                          0x1, false, false);
      sum = i == 0 ? texel : agx_fadd(b, sum, texel);
   }
   agx_st_tile(b, sum, agx_immediate(0xff), agx_null(),
               AGX_FORMAT_I32, 0x1, 0, false, 0);
   agx_stop(b);

   agx_ilp_schedule(s.b->shader);
   EXPECT_EQ(loads_before_first_tile_store(s.body), 4u);

   agx_ra(s.b->shader);
   expect_tile_stores_in_low_bank(s.b->shader, 1u);
   ralloc_free(s.mem);
}

/* Regression test for issue #1253: on AGX3C, layered 8x MSAA RGBA32 reload
 * background shaders emit 8 four-channel (8-half) ST_TILE stores to pixel
 * offset 0 (16-half bank). ILP must not hoist four 8-half payloads (32 halves)
 * ahead of wait_pix + st_tile and overflow agx3_tile_store_bank(). */
static void
test_layered_8x_rgba32_reload(enum agx_chip chip)
{
   tile_schedule_shader s;
   shader_init(&s, chip, true);
   s.key.fs.nr_samples = 8;

   agx_builder *b = s.b;
   b->cursor = agx_after_block(s.body);

   agx_index coords = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index layer = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_wait_pix(b, 0x0001);
   for (unsigned i = 0; i < 8; ++i) {
      agx_index sample_idx = agx_mov_imm(b, 32, i);
      agx_index sample_coord = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *cc = agx_collect_to(b, sample_coord, 3);
      cc->src[0] = coords;
      cc->src[1] = layer;
      cc->src[2] = sample_idx;

      agx_index texel = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_load_to(b, texel, sample_coord, agx_immediate(0),
                          agx_immediate(0), agx_immediate(0), agx_zero(),
                          agx_null(), AGX_DIM_2D_MS_ARRAY,
                          AGX_LOD_MODE_AUTO_LOD, 0xf, false, false);

      agx_st_tile(b, texel, agx_immediate(1u << i), agx_null(),
                  AGX_FORMAT_I32, 0xf, 0, false, 0);
   }
   agx_stop(b);

   agx_ilp_schedule(s.b->shader);
   EXPECT_LE(loads_before_first_tile_store(s.body), 2u);

   agx_ra(s.b->shader);
   EXPECT_FALSE(s.b->shader->ra_failed);
   expect_tile_stores_in_low_bank(s.b->shader, 8u);
   ralloc_free(s.mem);
}

TEST(BgShaderSchedule, Layered8xRgba32ReloadKeepsTileStoreBankOnG17S)
{
   test_layered_8x_rgba32_reload(AGX_CHIP_G17S);
}

TEST(BgShaderSchedule, Layered8xRgba32ReloadKeepsTileStoreBankOnG17G)
{
   test_layered_8x_rgba32_reload(AGX_CHIP_G17G);
}

/* A per-sample reload of an R32 target on AGX3 stores channel 0 split from
 * each eight-half 2D_MS texel fetch. RA places the whole fetched vector
 * in the tile-store bank (a split channel takes its vector's register), so
 * hoisting all four fetches ahead of the first ST_TILE needs 32 halves of the
 * 16-half bank even though the stored channels only total eight. ILP must
 * count the vectors, not just the stored channels. */
static void
test_split_ms_reload(enum agx_chip chip)
{
   tile_schedule_shader s;
   shader_init(&s, chip, true);

   agx_builder *b = s.b;
   b->cursor = agx_after_block(s.body);

   agx_index coords = agx_preload(b, agx_register(0, AGX_SIZE_32));
   for (unsigned i = 0; i < 4; ++i) {
      agx_index texel = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_load_to(b, texel, coords, agx_immediate(i),
                          agx_immediate(0), agx_immediate(0), agx_zero(),
                          agx_null(), AGX_DIM_2D_MS, AGX_LOD_MODE_AUTO_LOD,
                          0xf, false, false);

      agx_instr *split = agx_split(b, 4, texel);
      for (unsigned c = 0; c < 4; ++c)
         split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);

      agx_st_tile(b, split->dest[0], agx_immediate(1u << i), agx_null(),
                  AGX_FORMAT_I32, 0x1, 0, false, 0);
   }
   agx_stop(b);

   agx_ilp_schedule(s.b->shader);
   EXPECT_LE(loads_before_first_tile_store(s.body), 2u);

   agx_ra(s.b->shader);
   EXPECT_FALSE(s.b->shader->ra_failed);
   expect_tile_stores_in_low_bank(s.b->shader, 4u);
   ralloc_free(s.mem);
}

TEST(BgShaderSchedule, SplitMsR32ReloadKeepsTileStoreBankOnG17P)
{
   test_split_ms_reload(AGX_CHIP_G17P);
}

TEST(BgShaderSchedule, SplitMsR32ReloadKeepsTileStoreBankOnG16G)
{
   test_split_ms_reload(AGX_CHIP_G16G);
}
