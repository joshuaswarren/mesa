#include "agx3_select_core.h"
#include "agx3_alu_core.h"
#include "agx3_falu_mod_core.h"
#include "agx3_iadd_core.h"
#include "agx3_pack.h"
#include "agx3_stage_core.h"
#include "agx3_tex_core.h"
#include "agx_test.h"
#include "agx3_tile_core.h"
#include "agx3_sr_core.h"
#include "agx3_subgroup_core.h"
#include "agx3_matrix_core.h"
#include "agx3_vary_core.h"
extern "C" {
#include "agx3_compose/agx3_decode.h"
}
extern "C" {
#include "agx_nir.h"
#include "asahi/layout/layout.h"
}
#include "agx_compile.h"
#include "agx_debug.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir_m5_mesh.h"
#include "agx3_mesh_core.h"
#include "agx_mesh_abi.h"
#include <gtest/gtest.h>
#include <vector>
#include <csignal>
#include <cstdlib>

TEST(NativeForms, CollectSplitSameWidthAndChains)
{
   for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_index input[3];
      for (unsigned i = 0; i < 3; ++i)
         input[i] = agx_preload(b, agx_register(4 * i, size));
      agx_index previous = input[0];
      for (unsigned i = 0; i < 8; ++i) {
         agx_index vector = agx_vec_temp(b->shader, size, 2);
         agx_instr *collect = agx_collect_to(b, vector, 2);
         collect->src[0] = previous;
         collect->src[1] = input[1 + (i & 1)];
         agx_instr *split = agx_split(b, 2, vector);
         split->dest[0] = agx_temp(b->shader, size);
         split->dest[1] = agx_null();
         previous = split->dest[0];
      }
      agx_instr *use = agx_unit_test(b, previous);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      EXPECT_TRUE(agx_is_equiv(use->src[0], input[0]));
      EXPECT_EQ(use->src[0].size, size);
      agx_dce(b->shader, true);
      agx_foreach_instr_global(b->shader, I) {
         EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
         EXPECT_NE(I->op, AGX_OPCODE_SPLIT);
      }
      EXPECT_FALSE(agx_opt_collect_split(b->shader));
      ralloc_free(mem);
   }
}

TEST(NativeForms, CollectSplitPaddingRetainsScalarAffinity)
{
   for (bool dead_normal : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index occupied = agx_preload(b, agx_register(1, AGX_SIZE_16));
      agx_index input = agx_preload(b, agx_register(16, AGX_SIZE_32));
      agx_instr *producers[3];
      for (unsigned i = 0; i < 3; ++i)
         producers[i] = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                    input, input);
      agx_index padding = dead_normal ? agx_mov_imm(b, 32, 7)
                                     : agx_undef(AGX_SIZE_32);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *collect = agx_collect_to(b, vector, 4);
      for (unsigned i = 0; i < 3; ++i)
         collect->src[i] = producers[i]->dest[0];
      collect->src[3] = padding;
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 3; ++i) {
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_unit_test(b, split->dest[i]);
      }
      split->dest[3] = agx_null();
      agx_unit_test(b, occupied);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      agx_dce(b->shader, true);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(producers[0]->dest[0].value % 8, 0u);
      EXPECT_EQ(producers[1]->dest[0].value, producers[0]->dest[0].value + 2);
      EXPECT_EQ(producers[2]->dest[0].value, producers[0]->dest[0].value + 4);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CollectSplitEarlyInteriorExportAffinity)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      bool all_exported = mode == 1;
      unsigned exported_component = mode == 2 ? 3 : (mode == 4 ? 0 : 1);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index high =
         mode == 3 ? agx_preload(b, agx_register(31, AGX_SIZE_16)) : agx_null();
      agx_instr *producers[4];
      unsigned order[4] = {0, 1, 2, 3};
      if (mode == 2) {
         order[0] = 3;
         order[3] = 0;
      }
      for (unsigned component : order)
         producers[component] =
            agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), input, input);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *collect = agx_collect_to(b, vector, 4);
      for (unsigned i = 0; i < 4; ++i)
         collect->src[i] = producers[i]->dest[0];
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 4; ++i) {
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         if (i == exported_component)
            agx_export(b, split->dest[i], 14);
         else if (all_exported)
            agx_export(b, split->dest[i], 12 + 2 * i);
         else
            agx_unit_test(b, split->dest[i]);
      }
      if (!agx_is_null(high))
         agx_unit_test(b, high);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      agx_dce(b->shader, true);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(producers[exported_component]->dest[0].value == 14u,
                all_exported || mode == 2 || mode == 3 || mode == 4);
      bool exported = false;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_EXPORT && I->imm == 14) {
            exported = true;
            EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
            EXPECT_EQ(I->src[0].value, 14u);
         }
      }
      EXPECT_TRUE(exported);
      ralloc_free(mem);
   }
}

TEST(NativeForms, EarlyPairExportLeavesLaterWideGroupAvailable)
{
   for (unsigned mode = 0; mode < 6; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_instr *pair[2];
      for (unsigned i = 0; i < 2; ++i) {
         unsigned component = mode == 2 ? i : 1 - i;
         pair[component] =
            agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), input, input);
      }
      agx_index pair_vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, pair_vector, 2);
      for (unsigned i = 0; i < 2; ++i)
         collect->src[i] = pair[i]->dest[0];
      agx_instr *pair_split = agx_split(b, 2, pair_vector);
      for (unsigned i = 0; i < 2; ++i)
         pair_split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *wide[4];
      for (unsigned i = 0; i < 4; ++i)
         wide[i] =
            agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), input, input);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *wide_collect = agx_collect_to(b, vector, 4);
      for (unsigned i = 0; i < 4; ++i)
         wide_collect->src[i] = wide[i]->dest[0];
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 4; ++i) {
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_unit_test(b, split->dest[i]);
      }
      if (mode != 1) {
         if (mode >= 3) {
            agx_publish_texture_scalar_agx3(b, pair_split->dest[0], input,
                                            agx_null(), 1);
            if (mode >= 4) {
               agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
               resource.type = AGX_INDEX_RESOURCE;
               agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
               agx_texture_sample_to(
                  b, result, resource, agx_zero(), agx_zero(),
                  agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
                  agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false,
                  false, false, AGX_GATHER_NONE, false);
               if (mode == 5) {
                  agx_index later = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
                  agx_texture_sample_to(
                     b, later, resource, agx_zero(), agx_zero(),
                     agx_uniform(32, AGX_SIZE_64), agx_uniform(36, AGX_SIZE_64),
                     agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false,
                     false, false, AGX_GATHER_NONE, false);
                  agx_instr *later_split = agx_split(b, 4, later);
                  later_split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
                  for (unsigned i = 1; i < 4; ++i)
                     later_split->dest[i] = agx_null();
                  agx_unit_test(b, later_split->dest[0]);
               }
               agx_instr *result_split = agx_split(b, 4, result);
               result_split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
               for (unsigned i = 1; i < 4; ++i)
                  result_split->dest[i] = agx_null();
               agx_unit_test(b, agx_fmul(b, result_split->dest[0], input));
            }
         } else
            agx_unit_test(b, agx_fmul(b, pair_split->dest[0], input));
      }
      if (mode == 1)
         agx_export(b, pair_split->dest[0], 12);
      agx_export(b, pair_split->dest[1], 14);
      ASSERT_TRUE(agx_opt_collect_split(b->shader));
      agx_dce(b->shader, true);
      agx_validate(b->shader, "early pair export allocation");
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(pair[1]->dest[0].value == 14u, mode != 0 && mode != 4);
      if (mode == 0 || mode == 4) {
         EXPECT_LT(pair[1]->dest[0].value, 8u);
         for (unsigned i = 0; i < 4; ++i)
            EXPECT_EQ(wide[i]->dest[0].value, 8 + 2 * i);
      }
      bool exported = false;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_EXPORT && I->imm == 14) {
            exported = true;
            EXPECT_EQ(I->src[0].type, AGX_INDEX_REGISTER);
            EXPECT_EQ(I->src[0].value, 14u);
         }
      }
      EXPECT_TRUE(exported);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CollectSplitKeepsVectorConsumersAndModifiers)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, vector, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   agx_instr *split = agx_split(b, 2, vector);
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *vector_use = agx_unit_test(b, vector);
   agx_index result = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *add = agx_fadd_to(
      b, result, agx_neg(agx_abs(split->dest[0])), split->dest[1]);
   agx_unit_test(b, result);
   ASSERT_TRUE(agx_opt_collect_split(b->shader));
   EXPECT_TRUE(agx_is_equiv(add->src[0], agx_neg(agx_abs(x))));
   EXPECT_TRUE(add->src[0].abs);
   EXPECT_TRUE(add->src[0].neg);
   EXPECT_TRUE(agx_is_equiv(add->src[1], y));
   EXPECT_TRUE(agx_is_equiv(vector_use->src[0], vector));
   agx_dce(b->shader, true);
   bool kept_vector = false;
   agx_foreach_instr_global(b->shader, I) {
      kept_vector |= I == collect;
      EXPECT_NE(I->op, AGX_OPCODE_SPLIT);
   }
   EXPECT_TRUE(kept_vector);
   ralloc_free(mem);
}

TEST(NativeForms, CollectSplitDoesNotChangeWidthsOrOperandClasses)
{
   for (agx_size size : {AGX_SIZE_32, AGX_SIZE_64}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_index x = agx_preload(b, agx_register(0, size));
      agx_index vector = agx_vec_temp(b->shader, size, 2);
      agx_instr *collect = agx_collect_to(b, vector, 2);
      collect->src[0] = collect->src[1] = x;
      agx_instr *split = agx_split(b, 4, vector);
      for (unsigned i = 0; i < 4; ++i) {
         split->dest[i] = agx_temp(b->shader, agx_size(size - 1));
         agx_unit_test(b, split->dest[i]);
      }
      EXPECT_FALSE(agx_opt_collect_split(b->shader));
      ralloc_free(mem);
   }
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, vector, 2);
   collect->src[0] = x;
   collect->src[1] = agx_uniform(20, AGX_SIZE_32);
   agx_instr *split = agx_split(b, 2, vector);
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *first = agx_unit_test(b, split->dest[0]);
   agx_instr *second = agx_unit_test(b, split->dest[1]);
   agx_index original_second = second->src[0];
   ASSERT_TRUE(agx_opt_collect_split(b->shader));
   EXPECT_TRUE(agx_is_equiv(first->src[0], x));
   EXPECT_TRUE(agx_is_equiv(second->src[0], original_second));
   agx_dce(b->shader, true);
   bool kept_split = false;
   agx_foreach_instr_global(b->shader, I)
      kept_split |= I == split;
   EXPECT_TRUE(kept_split);
   ralloc_free(mem);
}

TEST(NativeForms, BankedScalarCollectAffinity)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.secondary = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index occupied = agx_preload(b, agx_register(1, AGX_SIZE_16));
   agx_index inputs[4];
   for (unsigned i = 0; i < 4; ++i)
      inputs[i] = agx_preload(b, agx_register(8 + 2 * i, AGX_SIZE_32));
   agx_instr *packs[2];
   for (unsigned i = 0; i < 2; ++i) {
      agx_index dest = agx_temp(b->shader, AGX_SIZE_16);
      packs[i] = agx_fltnorm_agx3_to(
         b, dest, inputs[2 * i], inputs[2 * i + 1], AGX_FORMAT_U8NORM);
   }
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_16, 2);
   agx_instr *collect = agx_collect_to(b, vector, 2);
   collect->src[0] = packs[0]->dest[0];
   collect->src[1] = packs[1]->dest[0];
   agx_unit_test(b, vector);
   for (unsigned i = 0; i < 2; ++i) {
      agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_ST_TILE, 0, 3);
      store->src[0] = packs[i]->dest[0];
      store->src[1] = agx_immediate(1);
      store->src[2] = agx_null();
      store->format = AGX_FORMAT_I16;
      store->mask = 1;
      store->pixel_offset = i;
      agx_builder_insert(&b->cursor, store);
   }
   agx_unit_test(b, occupied);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(packs[0]->dest[0].value & 1, 0u);
   EXPECT_EQ(packs[1]->dest[0].value, packs[0]->dest[0].value + 1);
   EXPECT_LT(packs[1]->dest[0].value, agx3_tile_store_bank(b->shader));
   ralloc_free(mem);
}

TEST(NativeForms, PhiWebAtRegisterBoundaryAvoidsEdgeCopies)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, join);
   agx_index input = agx_preload(b, agx_register(30, AGX_SIZE_32));
   b->cursor = agx_after_block(join);
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *phi = agx_phi_to(b, value, 1);
   phi->src[0] = input;
   agx_instr *use = agx_unit_test(b, value);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 30u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, SpillPhiUsesActualOtherEdgeWithoutCopyingLiveSlot)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->has_spill_pcopy_reserved = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index input = agx_preload(b, agx_register(20, AGX_SIZE_32));
   agx_index second = agx_preload(b, agx_register(22, AGX_SIZE_32));
   agx_index padding[4];
   for (agx_index &slot : padding) {
      slot = agx_temp(b->shader, AGX_SIZE_32);
      slot.memory = true;
      agx_mov_to(b, slot, input);
   }
   agx_index held = agx_temp(b->shader, AGX_SIZE_32);
   held.memory = true;
   agx_mov_to(b, held, input);
   agx_index other = agx_temp(b->shader, AGX_SIZE_32);
   other.memory = true;
   for (agx_block *block : {left, right}) {
      b->cursor = agx_after_block(block);
      if (block == right)
         agx_mov_to(b, other, second);
      for (agx_index slot : padding)
         agx_unit_test(b, agx_mov(b, slot));
   }
   b->cursor = agx_after_block(join);
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   value.memory = true;
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = held;
   phi->src[1] = other;
   agx_instr *reload = agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32), value);
   agx_unit_test(b, reload->dest[0]);
   agx_instr *held_reload =
      agx_mov_to(b, agx_temp(b->shader, AGX_SIZE_32), held);
   agx_unit_test(b, held_reload->dest[0]);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_TRUE(reload->src[0].memory);
   EXPECT_EQ(reload->src[0].value, 10u);
   EXPECT_EQ(held_reload->src[0].value, 8u);
   unsigned stores = 0;
   agx_foreach_instr_global(b->shader, I)
      stores += I->op == AGX_OPCODE_MOV && I->dest[0].memory;
   EXPECT_EQ(stores, 7u);
   EXPECT_EQ(b->shader->scratch_size_B, 24u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiAffinityKeepsCrossedEdgeInputsOutOfSwapCycles)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index x = agx_preload(b, agx_register(10, AGX_SIZE_16));
   agx_index y = agx_preload(b, agx_register(12, AGX_SIZE_16));
   b->cursor = agx_after_block(join);
   agx_index first = agx_temp(b->shader, AGX_SIZE_16);
   agx_index second = agx_temp(b->shader, AGX_SIZE_16);
   agx_instr *a = agx_phi_to(b, first, 2);
   a->src[0] = x;
   a->src[1] = y;
   agx_instr *c = agx_phi_to(b, second, 2);
   c->src[0] = y;
   c->src[1] = x;
   agx_instr *use_first = agx_unit_test(b, first);
   agx_instr *use_second = agx_unit_test(b, second);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use_first->src[0].value, 10u);
   EXPECT_EQ(use_second->src[0].value, 1u);
   unsigned copies = 0, swaps = 0;
   agx_foreach_instr_global(b->shader, I) {
      copies += I->op == AGX_OPCODE_MOV;
      swaps += I->op == AGX_OPCODE_SWAP;
   }
   EXPECT_EQ(copies, 3u);
   EXPECT_EQ(swaps, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiEdgeAffinityRespectsTileBankAndLiveValues)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      SCOPED_TRACE(mode);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.is_helper = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->any_cf = true;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *left = agx_test_block(b->shader);
      agx_block *right = agx_test_block(b->shader);
      agx_block *join = agx_test_block(b->shader);
      agx_block_add_successor(entry, left);
      agx_block_add_successor(entry, right);
      agx_block_add_successor(left, join);
      agx_block_add_successor(right, join);
      agx_index held =
         mode == 2 ? agx_preload(b, agx_register(10, AGX_SIZE_16)) : agx_null();
      agx_index incoming[2];
      unsigned side = 0;
      for (agx_block *block : {left, right}) {
         b->cursor = agx_after_block(block);
         unsigned reg = side == 0 ? 20 : (mode == 0 ? 10 : 22);
         incoming[side++] =
            mode == 2 ? held : agx_preload(b, agx_register(reg, AGX_SIZE_16));
      }
      b->cursor = agx_after_block(join);
      agx_index value = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *phi = agx_phi_to(b, value, 2);
      phi->src[0] = incoming[0];
      phi->src[1] = incoming[1];
      agx_instr *use = agx_unit_test(b, value);
      agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_ST_TILE, 0, 3);
      store->src[0] = value;
      store->src[1] = agx_immediate(1);
      store->src[2] = agx_null();
      store->format = AGX_FORMAT_I16;
      store->mask = 1;
      agx_builder_insert(&b->cursor, store);
      agx_instr *held_use = mode == 2 ? agx_unit_test(b, held) : nullptr;
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_LT(use->src[0].value, agx3_tile_store_bank(b->shader));
      EXPECT_EQ(use->src[0].value, mode == 0 ? 10u : 1u);
      if (held_use) {
         EXPECT_EQ(held_use->src[0].value, 10u);
      }
      unsigned copies = 0;
      agx_foreach_instr_global(b->shader, I)
         copies += I->op == AGX_OPCODE_MOV;
      EXPECT_EQ(copies, mode == 0 ? 1u : 2u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, PhiUsesOtherEdgeWhenPreferredRegisterIsLive)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index held = agx_preload(b, agx_register(10, AGX_SIZE_16));
   b->cursor = agx_after_block(right);
   agx_index other = agx_preload(b, agx_register(12, AGX_SIZE_16));
   b->cursor = agx_after_block(join);
   agx_index value = agx_temp(b->shader, AGX_SIZE_16);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = held;
   phi->src[1] = other;
   agx_instr *use = agx_unit_test(b, value);
   agx_instr *held_use = agx_unit_test(b, held);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 12u);
   EXPECT_EQ(held_use->src[0].value, 10u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 1u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiWithAgreeingInputsIsAssignedBeforeEarlierPhis)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index incoming[2];
   for (unsigned side = 0; side < 2; ++side) {
      b->cursor = agx_after_block(side ? right : left);
      incoming[side] = agx_preload(b, agx_register(2, AGX_SIZE_32));
   }
   b->cursor = agx_after_block(join);
   /* The constant phi comes first and has no register to follow. */
   agx_index constant = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *first = agx_phi_to(b, constant, 2);
   first->src[0] = agx_immediate(1);
   first->src[1] = agx_immediate(2);
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = incoming[0];
   phi->src[1] = incoming[1];
   agx_instr *use = agx_unit_test(b, value);
   agx_instr *constant_use = agx_unit_test(b, constant);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 2u);
   EXPECT_NE(constant_use->src[0].value, 2u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, PhiFallbackLeavesLaterPhiInputsFree)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.is_helper = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);
   header->loop_header = true;
   agx_index input = agx_preload(b, agx_register(2, AGX_SIZE_32));
   b->cursor = agx_after_block(header);
   /* The counter phi comes first and knows none of its inputs yet. */
   agx_index counter = agx_temp(b->shader, AGX_SIZE_32);
   agx_index next_counter = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *count_phi = agx_phi_to(b, counter, 2);
   count_phi->src[0] = agx_immediate(0);
   count_phi->src[1] = next_counter;
   agx_index value = agx_temp(b->shader, AGX_SIZE_32);
   agx_index next_value = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *phi = agx_phi_to(b, value, 2);
   phi->src[0] = input;
   phi->src[1] = next_value;
   b->cursor = agx_after_block(body);
   agx_iadd_to(b, next_value, value, agx_immediate(1), 0);
   agx_iadd_to(b, next_counter, counter, agx_immediate(1), 0);
   b->cursor = agx_after_block(exit);
   agx_instr *use = agx_unit_test(b, value);
   agx_unit_test(b, counter);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_EQ(use->src[0].value, 2u);
   unsigned copies = 0;
   agx_foreach_instr_global(b->shader, I)
      copies += I->op == AGX_OPCODE_MOV;
   EXPECT_EQ(copies, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, TransientTextureVectorBecomesScalarBeforeAnyEviction)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      SCOPED_TRACE(mode);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index held[3];
      held[0] = agx_preload(b, agx_register(0, AGX_SIZE_16));
      held[1] = agx_preload(b, agx_register(8, AGX_SIZE_16));
      held[2] = agx_preload(b, agx_register(18, AGX_SIZE_16));
      agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_index coords_reg = agx_register(4, AGX_SIZE_32);
      coords_reg.channels_m1 = 1;
      agx_preload_to(b, coords, coords_reg);
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, result, coords, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD,
         mode == 3 ? 1 : 15, false, false, false, AGX_GATHER_NONE, false);
      agx_index extra = agx_null();
      if (mode == 2)
         extra = agx_fadd(b, agx_uniform(0, AGX_SIZE_32),
                          agx_uniform(2, AGX_SIZE_32));
      agx_instr *split = agx_split(b, 4, result);
      for (unsigned i = 0; i < 4; ++i)
         split->dest[i] = mode == 3 && i
                            ? agx_null() : agx_temp(b->shader, AGX_SIZE_32);
      if (mode == 3) {
         extra = agx_fadd(b, agx_uniform(0, AGX_SIZE_32),
                          agx_uniform(2, AGX_SIZE_32));
         agx_export(b, extra, 12);
      }
      agx_index second = mode == 3 ? agx_uniform(4, AGX_SIZE_32)
                                   : split->dest[3];
      agx_unit_test(b, agx_fadd(b, split->dest[0], second));
      if (mode != 3) {
         agx_unit_test(b, split->dest[1]);
         agx_unit_test(b, split->dest[2]);
      }
      if (mode == 1)
         agx_unit_test(b, result);
      if (mode == 2)
         agx_unit_test(b, extra);
      for (agx_index value : held)
         agx_unit_test(b, value);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(sample->dest[0].value, mode == 0 ? 10u : 24u);
      EXPECT_EQ(agx_channels(sample->dest[0]), 4u);
      if (mode == 0) {
         EXPECT_EQ(b->shader->max_reg, 18u);
         for (unsigned i = 0; i < 4; ++i)
            EXPECT_EQ(split->dest[i].value, 10u + 2 * i);
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TransientTextureResultPreservesEarlierAsyncWriteSpan)
{
   for (bool complete : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index low = agx_preload(b, agx_register(0, AGX_SIZE_16));
      agx_index high = agx_preload(b, agx_register(18, AGX_SIZE_16));
      agx_index x = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index y = agx_preload(b, agx_register(6, AGX_SIZE_32));
      agx_publish_texture_scalar_agx3(b, x, y, agx_null(), 1);
      agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index previous = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *first = agx_texture_sample_to(
         b, previous, resource, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 8,
         false, false, false, AGX_GATHER_NONE, false);
      agx_unit_test(b, low);
      agx_instr *parts = agx_split(b, 4, previous);
      for (unsigned i = 0; i < 3; ++i)
         parts->dest[i] = agx_null();
      parts->dest[3] = agx_temp(b->shader, AGX_SIZE_32);
      agx_index retained = parts->dest[3];
      if (complete)
         retained = agx_fadd(b, retained, agx_uniform(0, AGX_SIZE_32));
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, result, resource, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64),
         agx_null(), AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15,
         false, false, false, AGX_GATHER_NONE, false);
      agx_instr *split = agx_split(b, 4, result);
      for (unsigned i = 0; i < 4; ++i)
         split->dest[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_unit_test(b, agx_fadd(b, split->dest[0], split->dest[3]));
      agx_unit_test(b, split->dest[1]);
      agx_unit_test(b, split->dest[2]);
      agx_unit_test(b, retained);
      if (complete)
         agx_unit_test(b, parts->dest[3]);
      agx_unit_test(b, low);
      agx_unit_test(b, high);
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      EXPECT_EQ(first->dest[0].value, 8u);
      EXPECT_EQ(sample->dest[0].value, complete ? 4u : 24u);
      EXPECT_EQ(agx_channels(sample->dest[0]), 4u);
      ralloc_free(mem);
   }
}
#include <algorithm>

TEST(NativeForms, M5Matrix8MetalCompilerWitnesses)
{
   const uint8_t zero[] = {
      0xa7, 0x02, 0x05, 0x2a, 0x20, 0x20, 0xaf,
      0x12, 0x28, 0x06, 0x44, 0x00, 0x02, 0x01,
   };
   const uint8_t accumulator[] = {
      0x2f, 0x02, 0x05, 0x6b, 0x20, 0x20, 0xaf,
      0x02, 0x28, 0x08, 0x14, 0x00, 0x00, 0x01,
   };
   agx3_matrix_mma8_desc d = {};
   d.dest = 24;
   d.a = 2;
   d.b = 20;
   d.c_zero = true;
   d.discard[0] = d.discard[1] = true;
   uint8_t bytes[14], length;
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &d,
                                     bytes, &length));
   ASSERT_EQ(length, sizeof(zero));
   EXPECT_EQ(memcmp(bytes, zero, sizeof(zero)), 0);

   d.dest = d.c = 0;
   d.a = 10;
   d.b = 6;
   d.c_zero = false;
   d.discard[2] = true;
   d.wait_mask = 0x80;
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &d,
                                     bytes, &length));
   ASSERT_EQ(length, sizeof(accumulator));
   EXPECT_EQ(memcmp(bytes, accumulator, sizeof(accumulator)), 0);
}

static unsigned
m5_absolute_comparison_instrs(bool repeat_absolute)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "M5 absolute comparisons");
   nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
   nir_def *x = nir_channel(&b, input, 0);
   nir_def *abs_x = nir_fabs(&b, x);
   nir_def *first = nir_bcsel(&b, nir_flt(&b, abs_x, nir_channel(&b, input, 1)),
                              nir_channel(&b, input, 2),
                              nir_channel(&b, input, 3));
   nir_def *second =
      nir_bcsel(&b,
                nir_flt(&b, repeat_absolute ? abs_x : x,
                        nir_channel(&b, input, 2)),
                nir_channel(&b, input, 1), nir_channel(&b, input, 3));
   agx_test_store_local_pixel_agx(
      &b, nir_vec4(&b, first, second, first, second),
      nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), 0, 0xf,
      (enum pipe_format)AIL_ISA_FORMAT_I32);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.reserved_preamble = 16;
   key.fs.nr_samples = 1;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   unsigned instrs = part.info.stats.instrs;
   free(part.binary);
   ralloc_free(b.shader);
   return instrs;
}

/* M5 compares |x| through a plain copy. Comparing the same |x| twice in a
 * block costs no more than comparing |x| and x. */
TEST(NativeForms, M5RepeatedAbsoluteComparandSharesItsCopy)
{
   unsigned once = m5_absolute_comparison_instrs(false);
   EXPECT_GT(once, 0u);
   EXPECT_EQ(m5_absolute_comparison_instrs(true), once);
}

TEST(NativeForms, M5SignedByteToHalfReachesPacker)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_instr *convert = agx_alloc_instr(b, AGX_OPCODE_CONVERT, 1, 2);
   convert->dest[0] = agx_register(10, AGX_SIZE_16);
   convert->src[0] = agx_immediate(AGX_CONVERT_S8_TO_F);
   convert->src[1] = agx_register(10, AGX_SIZE_16);
   convert->round = AGX_ROUND_RTE;
   agx_builder_insert(&b->cursor, convert);
   agx_stop(b);
   util_dynarray bytes;
   util_dynarray_init(&bytes, mem);
   EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
   EXPECT_GT(bytes.size, 4u);
   ralloc_free(mem);
}

TEST(NativeForms, BoundedUnaryRegistersUnderPressure)
{
   for (agx_opcode op : {AGX_OPCODE_TRUNC, AGX_OPCODE_DFDX, AGX_OPCODE_DFDY}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index live[64];
      for (unsigned i = 0; i < 64; ++i)
         live[i] = agx_preload(b, agx_register(2 * i, AGX_SIZE_32));
      agx_index input = agx_mov_imm(b, 32, 0x40500000);
      agx_index output = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *unary = agx_alloc_instr(b, op, 1, 1);
      unary->src[0] = input;
      unary->dest[0] = output;
      agx_builder_insert(&b->cursor, unary);
      agx_index sum = output;
      for (unsigned i = 0; i < 64; ++i)
         sum = agx_fadd(b, sum, live[i]);
      agx_export(b, sum, 8);
      agx_stop(b);
      agx_ra(b->shader);
      EXPECT_LT(unary->src[0].value, 128u) << op;
      EXPECT_LT(unary->dest[0].value, 128u) << op;
      unsigned high = 0;
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_dest(I, d)
            high += I->dest[d].type == AGX_INDEX_REGISTER && I->dest[d].value >= 128;
      }
      EXPECT_GT(high, 0u) << "Unrelated live values must retain the upper bank";
      ralloc_free(mem);
   }
}

TEST(NativeForms, NegatedSqrtStepReachesPacker)
{
   for (bool absolute : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index src = agx_register(112, AGX_SIZE_32);
      src.neg = true;
      src.abs = absolute;
      src.cache = true;
      agx_instr *step = agx_alloc_instr(b, AGX_OPCODE_SRSQRT, 1, 1);
      step->src[0] = src;
      step->dest[0] = agx_register(122, AGX_SIZE_32);
      agx_builder_insert(&b->cursor, step);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 4u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, G17PHalfSqrtStepAcceptsLastUse)
{
   for (unsigned hint : {0u, 1u, 2u}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 7;
      agx_index src = agx_register(3, AGX_SIZE_16);
      src.cache = hint == 1;
      src.discard = hint == 2;
      agx_srsqrt_to(b, agx_register(4, AGX_SIZE_32), src);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      uint8_t expected[AGX3_ALU_MAX_LENGTH], length;
      ASSERT_TRUE(agx3_alu_sqrt_step_half(
         AGX_ISA_ENCODING_AGX3B, 4, 3, AGX3_ALU_HINT_CACHE, true, 0,
         expected, &length));
      ASSERT_GE(binary.size, length);
      EXPECT_EQ(memcmp(binary.data, expected, length), 0);
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, G17PHalfComparisonSelectsIntegerBits)
{
   for (unsigned selected : {1u, 2u, 0xffu, 0x3c00u, 0x8000u, 0xffffu}) {
      for (unsigned arm : {2u, 3u}) {
         SCOPED_TRACE(selected);
         SCOPED_TRACE(arm);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         b->shader->key = &key;
         agx_index x = agx_preload(b, agx_register(4, AGX_SIZE_16));
         agx_index y = agx_preload(b, agx_register(5, AGX_SIZE_16));
         agx_index alternatives[] = {agx_immediate(selected), x};
         agx_instr *sel = agx_fcmpsel_to(
            b, agx_temp(b->shader, AGX_SIZE_16), x, y,
            alternatives[arm == 3], alternatives[arm == 2], AGX_FCOND_LT);
         agx_materialize_agx3_alu_constants(b->shader);
         const agx_instr *constant = NULL;
         agx_foreach_instr_global(b->shader, I) {
            if (I->nr_dests && agx_is_equiv(I->dest[0], sel->src[arm]))
               constant = I;
         }
         ASSERT_NE(constant, nullptr);
         EXPECT_EQ(constant->op, AGX_OPCODE_MOV_IMM);
         EXPECT_EQ(constant->imm, selected);
         EXPECT_EQ(constant->dest[0].size, AGX_SIZE_16);
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, HalfModifiedMultiplyAndSelect)
{
   for (bool select : {false, true}) {
      SCOPED_TRACE(select);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_FRAGMENT, &agx_nir_options, "half modified operands");
      nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
      nir_def *x = nir_f2f16(&b, nir_channel(&b, input, 0));
      nir_def *result;
      if (select) {
         nir_def *cmp = agx_test_load_preamble(&b, 2, 16, 12);
         x = nir_channel(&b, cmp, 0);
         nir_def *y = nir_channel(&b, cmp, 1);
         result = nir_bcsel(&b, nir_flt(&b, x, nir_fneg(&b, y)),
                           nir_channel(&b, input, 2),
                           nir_channel(&b, input, 3));
      } else {
         result = nir_f2f32(
            &b, nir_fmul(&b, nir_fabs(&b, x), nir_imm_floatN_t(&b, 0.5, 16)));
      }
      agx_test_store_local_pixel_agx(
         &b, nir_vec4(&b, nir_channel(&b, input, 0), result, result, result),
         nir_imm_intN_t(&b, 0xff, 16), nir_undef(&b, 2, 16), 0, 0xf,
         PIPE_FORMAT_R8_UNORM);
      b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      key.fs.nr_samples = 1;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
   }
}

TEST(NativeForms, ImageStoreUsesDataTypeAcrossChipsAndFormats)
{
   const struct {
      nir_alu_type type;
      pipe_format format;
   } cases[] = {
      {nir_type_uint32, PIPE_FORMAT_NONE},
      {nir_type_uint32, PIPE_FORMAT_R32_UINT},
      {nir_type_uint32, PIPE_FORMAT_R32G32B32A32_UINT},
      {nir_type_int32, PIPE_FORMAT_NONE},
      {nir_type_int32, PIPE_FORMAT_R32_SINT},
      {nir_type_int32, PIPE_FORMAT_R32G32B32A32_SINT},
      {nir_type_float32, PIPE_FORMAT_NONE},
      {nir_type_float32, PIPE_FORMAT_R32_FLOAT},
      {nir_type_float32, PIPE_FORMAT_R32G32B32A32_FLOAT},
      {nir_type_invalid, PIPE_FORMAT_R32_UINT},
   };
   agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));
   /* The store is the program's last instruction before its end. Code around
    * it may differ by chip (G17P runs its own uniform-copy CSE, which changes
    * register allocation), so compare only the store. Integer and float data
    * each select one encoding whatever the format; a typeless store takes its
    * type from the format. */
   std::vector<uint8_t> reference[2];
   for (const auto &c : cases) {
      const bool is_float =
         nir_alu_type_get_base_type(c.type) == nir_type_float;
      for (agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G16G}) {
         SCOPED_TRACE(::testing::Message() << chip << " " << c.type << " "
                                          << c.format);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "image store data type");
         nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
         nir_intrinsic_instr *store = nir_image_store(
            &b, nir_imm_intN_t(&b, 1, 16),
            nir_imm_ivec4(&b, 0, 0, 0, 0), nir_imm_intN_t(&b, 0, 16),
            input, nir_imm_intN_t(&b, 0, 16));
         nir_intrinsic_set_image_dim(store, GLSL_SAMPLER_DIM_2D);
         nir_intrinsic_set_format(store, c.format);
         nir_intrinsic_set_src_type(store, c.type);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = chip;
         key.no_preamble = true;
         key.reserved_preamble = 16;
         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         ASSERT_GT(part.info.binary_size, 0u);
         const uint8_t *code = static_cast<const uint8_t *>(part.binary);
         const unsigned size = part.info.binary_size;
         ASSERT_GE(size, AGX3_IMAGE_WRITE_LENGTH + end.length);
         EXPECT_EQ(memcmp(code + size - end.length, end.bytes, end.length), 0);
         const uint8_t *write =
            code + size - end.length - AGX3_IMAGE_WRITE_LENGTH;
         std::vector<uint8_t> encoded(write, write + AGX3_IMAGE_WRITE_LENGTH);
         if (reference[is_float].empty())
            reference[is_float] = encoded;
         else
            EXPECT_EQ(encoded, reference[is_float]);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
   EXPECT_NE(reference[0], reference[1]);
}

/* A G17P state-0 storage image that the shader also reads names its
 * texture/PBE descriptor pair through state 0 (hk's read/write ABI): record 0
 * is the texture, which read as a PBE describes a far larger surface, so the
 * store must take record 1. A write-only image's state 0 is the PBE itself and
 * keeps record 0. Only the immediate descriptor index may differ. */
TEST(NativeForms, G17PState0PairedStoreTakesTheSecondRecord)
{
   agx3_encoded_end end;
   ASSERT_TRUE(agx3_pack_end(AGX_ISA_ENCODING_AGX3B, &end));

   std::vector<uint8_t> store[2];
   for (bool pair : {false, true}) {
      SCOPED_TRACE(pair ? "read/write pair" : "write-only");
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "state-0 store record");
      nir_def *input = agx_test_load_preamble(&b, 4, 32, 4);
      nir_intrinsic_instr *intr = nir_image_store(
         &b, nir_imm_intN_t(&b, 0, 16), nir_imm_ivec4(&b, 0, 0, 0, 0),
         nir_imm_intN_t(&b, 0, 16), input, nir_imm_intN_t(&b, 0, 16));
      nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
      nir_intrinsic_set_format(intr, PIPE_FORMAT_R32_UINT);
      nir_intrinsic_set_src_type(intr, nir_type_uint32);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      key.g17p_storage_image_pair = pair;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      const uint8_t *code = static_cast<const uint8_t *>(part.binary);
      const unsigned size = part.info.binary_size;
      ASSERT_GE(size, AGX3_IMAGE_WRITE_LENGTH + end.length);
      const uint8_t *write = code + size - end.length - AGX3_IMAGE_WRITE_LENGTH;
      store[pair].assign(write, write + AGX3_IMAGE_WRITE_LENGTH);
      free(part.binary);
      ralloc_free(b.shader);
   }

   /* The encoder's own difference between records 0 and 1 at one base. */
   agx3_image_write_desc d = {};
   d.image_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   d.dim = AGX3_IMAGE_WRITE_2D;
   uint8_t record[2][AGX3_IMAGE_WRITE_LENGTH], length[2];
   for (unsigned i = 0; i < 2; ++i) {
      d.image_value = i;
      ASSERT_TRUE(agx3_tex_encode_image_write_2d(AGX_ISA_ENCODING_AGX3B, &d, 0,
                                                 1, record[i], &length[i]));
   }
   ASSERT_EQ(length[0], AGX3_IMAGE_WRITE_LENGTH);
   for (unsigned i = 0; i < AGX3_IMAGE_WRITE_LENGTH; ++i) {
      SCOPED_TRACE(i);
      EXPECT_EQ(store[0][i] ^ store[1][i], record[0][i] ^ record[1][i]);
   }
   EXPECT_NE(store[0], store[1]);
}

TEST(NativeForms, VertexImageLoadRegisterLod)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_VERTEX, &agx_nir_options, "vertex image load register LOD");
   nir_def *id = nir_load_vertex_id(&b);
   nir_def *coord = nir_vec4(&b, id, id, id, nir_imm_int(&b, 0));
   nir_def *value = nir_image_load(
      &b, 4, 32, nir_imm_intN_t(&b, 3, 16), coord, nir_imm_int(&b, 0),
      agx_test_load_preamble(&b, 1, 16, 0));
   nir_intrinsic_instr *load = nir_def_as_intrinsic(value);
   nir_intrinsic_set_image_dim(load, GLSL_SAMPLER_DIM_3D);
   nir_intrinsic_set_format(load, PIPE_FORMAT_R32G32B32A32_UINT);
   nir_intrinsic_set_dest_type(load, nir_type_uint32);
   for (unsigned c = 0; c < 4; ++c) {
      nir_store_uvs_agx(&b, nir_imm_float(&b, c == 3 ? 1.0f : 0.0f),
                       nir_imm_intN_t(&b, c, 16));
      nir_store_uvs_agx(&b, nir_channel(&b, value, c),
                       nir_imm_intN_t(&b, 4 + c, 16));
   }
   b.shader->info.outputs_written = VARYING_BIT_POS |
                                   BITFIELD64_BIT(VARYING_SLOT_VAR0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.reserved_preamble = 64;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   EXPECT_GT(part.info.main_size, 0u);
   free(part.binary);
}

TEST(NativeForms, TexturePublicationWaitPreservesUnrelatedResult)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;
   agx_index coords = agx_register(10, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_device_load_to(b, agx_register(0, AGX_SIZE_32),
                     agx_register(28, AGX_SIZE_64), agx_immediate(0),
                     AGX_FORMAT_I32, 1, 0, false);
   agx_device_load_to(b, coords, agx_register(28, AGX_SIZE_64),
                     agx_immediate(4), AGX_FORMAT_I32, 3, 0, false);
   agx_publish_texture_agx3(b, coords, agx_null(), 1);
   agx_index resource = coords;
   resource.type = AGX_INDEX_RESOURCE;
   resource.value = 1;
   agx_index result = agx_register(20, AGX_SIZE_32);
   result.channels_m1 = 3;
   agx_texture_sample_to(b, result, resource, agx_immediate(0),
                        agx_immediate(0), agx_register(24, AGX_SIZE_64),
                        agx_register(26, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
                        AGX_LOD_MODE_LOD_MIN, 15, false, false, false,
                        AGX_GATHER_NONE, false);
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, 10,
                                         2, carrier, &len));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   bool found = false;
   for (unsigned i = 0; i + len <= binary.size; ++i)
      found |= memcmp(bytes + i, carrier, len) == 0;
   EXPECT_TRUE(found);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

/* A vertex output whose staging COPY reuses the operand staging of a sample
 * still in flight retires that sample first. The r0 carrier used for that
 * rewrites r0, so it must not run while a load is still writing r0 (the
 * texturelodoffset_pcoffset *shadow_vertex position load): the retirement
 * must also wait for the stored value's own producer. */
TEST(NativeForms, M5VertexOutputStagingWaitCoversStoredLoad)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_VERTEX;
   b->shader->max_reg = 63;

   /* Earlier loads occupy the first slots, so the sample's slot is not the
    * one the first output's staging COPY takes. */
   for (unsigned i = 0; i < 3; ++i) {
      agx_index value = agx_register(80 + 8 * i, AGX_SIZE_32);
      value.channels_m1 = 3;
      agx_device_load_to(b, value, agx_register(8, AGX_SIZE_64),
                         agx_immediate(i), AGX_FORMAT_I32, 15, 0, false);
   }

   /* The CTS vertex shader's sample, as register allocation left it: the
    * reference and the packed level/offset word in r14..r15. */
   agx_index coords = agx_register(20, AGX_SIZE_32);
   coords.channels_m1 = 1;
   coords.discard = true;
   agx_index level = agx_register(3, AGX_SIZE_16);
   level.discard = true;
   agx_index texture = agx_register(12, AGX_SIZE_64);
   texture.discard = true;
   agx_index sampler = agx_register(24, AGX_SIZE_64);
   sampler.discard = true;
   agx_index compare = agx_register(28, AGX_SIZE_32);
   compare.channels_m1 = 1;
   compare.discard = true;
   agx_index result = agx_register(32, AGX_SIZE_32);
   result.channels_m1 = 3;
   agx_texture_sample_to(b, result, coords, level, agx_zero(), texture,
                         sampler, compare, AGX_DIM_2D,
                         AGX_LOD_MODE_LOD_MIN, 1, true, true, false,
                         AGX_GATHER_NONE, false);
   /* The position load into r0..r3 while the sample is in flight. */
   agx_index position = agx_register(0, AGX_SIZE_32);
   position.channels_m1 = 3;
   agx_device_load_to(b, position, agx_register(8, AGX_SIZE_64),
                      agx_register(6, AGX_SIZE_32), AGX_FORMAT_I32, 15, 0,
                      false);
   for (unsigned i = 0; i < 4; ++i)
      agx_st_vary(b, agx_immediate(i), agx_register(2 * i, AGX_SIZE_32));
   agx_st_vary(b, agx_immediate(4), agx_register(32, AGX_SIZE_32));
   for (unsigned i = 0; i < 3; ++i)
      agx_st_vary(b, agx_immediate(5 + i), agx_register(80 + 8 * i, AGX_SIZE_32));
   agx_stop(b);

   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

   /* The one identity carrier, found by its wait mask, waits for both the
    * sample (staging) and the load (r0). */
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned carriers = 0;
   for (unsigned mask = 1; mask < 0x80; ++mask) {
      uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3C, 0,
                                              mask, carrier, &len));
      for (unsigned i = 0; i + len <= binary.size; ++i) {
         if (memcmp(bytes + i, carrier, len) != 0)
            continue;
         ++carriers;
         EXPECT_EQ(util_bitcount(mask), 2u) << "mask " << mask;
      }
   }
   EXPECT_EQ(carriers, 1u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, MinmaxReusesUniformCopiedForComparison)
{
   for (agx_fcond cond : {AGX_FCOND_LTN, AGX_FCOND_GTN}) {
      for (unsigned immediate : {0u, 176u}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_instr *I = agx_alloc_instr(b, AGX_OPCODE_FCMPSEL, 1, 4);
         I->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
         I->src[0] = I->src[2] = agx_uniform(8, AGX_SIZE_32);
         I->src[1] = I->src[3] = agx_immediate(immediate);
         I->fcond = cond;
         agx_builder_insert(&b->cursor, I);
         agx_lower_uniform_sources(b->shader);
         EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
         EXPECT_TRUE(agx_is_equiv(I->src[0], I->src[2]));
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, AtomicVertexOutputsReachTerminalBundle)
{
   for (bool position : {false, true}) {
      SCOPED_TRACE(position);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_VERTEX, &agx_nir_options, "atomic vertex outputs");
      nir_def *address = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *value = nir_global_atomic_agx(
         &b, 32, address, nir_imm_int(&b, 0), nir_imm_int(&b, 1));
      nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(value), nir_atomic_op_iadd);
      nir_store_uvs_agx(&b, value, nir_imm_intN_t(&b, 4, 16));
      nir_def *second = nir_global_atomic_agx(
         &b, 32, address, nir_imm_int(&b, 4), nir_imm_int(&b, -1));
      nir_intrinsic_set_atomic_op(nir_def_as_intrinsic(second), nir_atomic_op_iadd);
      nir_store_uvs_agx(&b, nir_iadd(&b, value, second),
                       nir_imm_intN_t(&b, 5, 16));
      if (position) {
         for (unsigned c = 0; c < 4; ++c)
            nir_store_uvs_agx(&b, nir_imm_float(&b, c == 3 ? 1.0 : 0.0),
                             nir_imm_intN_t(&b, c, 16));
      }
      b.shader->info.outputs_written = BITFIELD64_BIT(VARYING_SLOT_VAR0) |
                                     (position ? VARYING_BIT_POS : 0);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.reserved_preamble = 4;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.main_size, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, RectangleImplicitLodOutsideFragment)
{
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_COMPUTE,
                                   MESA_SHADER_FRAGMENT}) {
      for (bool sampler_bias : {false, true}) {
         SCOPED_TRACE(stage);
         SCOPED_TRACE(sampler_bias);
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "rectangle implicit lod");
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 1);
         tex->op = nir_texop_tex;
         tex->sampler_dim = GLSL_SAMPLER_DIM_RECT;
         tex->dest_type = nir_type_float32;
         tex->coord_components = 2;
         tex->src[0] = nir_tex_src_for_ssa(
            nir_tex_src_coord, nir_imm_vec2(&b, 3.0, 5.0));
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         agx_nir_lower_g17p_texture_early(b.shader, sampler_bias, true);
         EXPECT_EQ(tex->sampler_dim, GLSL_SAMPLER_DIM_RECT);
         EXPECT_GE(nir_tex_instr_src_index(tex, nir_tex_src_coord), 0);
         if (stage == MESA_SHADER_FRAGMENT) {
            EXPECT_EQ(tex->op, sampler_bias ? nir_texop_txb : nir_texop_tex);
         } else {
            EXPECT_EQ(tex->op, nir_texop_txl);
            EXPECT_LT(nir_tex_instr_src_index(tex, nir_tex_src_bias), 0);
            int lod = nir_tex_instr_src_index(tex, nir_tex_src_lod);
            ASSERT_GE(lod, 0);
            EXPECT_EQ(nir_src_as_float(tex->src[lod].src), 0.0);
         }
         nir_validate_shader(b.shader, "rectangle level zero");
         ralloc_free(b.shader);
      }
   }
}

TEST(NativeForms, ImageWriteHighUniformHeap)
{
   for (unsigned uniform : {252u, 256u, 296u, 508u}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G16G;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      agx_index data = agx_register(16, AGX_SIZE_32);
      data.channels_m1 = 3;
      agx_index coord = agx_register(24, AGX_SIZE_32);
      coord.channels_m1 = 1;
      agx_image_write(b, data, coord, agx_zero(),
                      agx_uniform(uniform, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_2D, true);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &binary)) << uniform;
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ImageFenceWaitsRotatingWriteSlots)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(16, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coord = agx_register(24, AGX_SIZE_32);
   coord.channels_m1 = 1;
   for (unsigned i = 0; i < 3; ++i) {
      agx_image_write(b, data, coord, agx_zero(), agx_uniform(8, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_2D, true);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

   agx3_encoded_scoped_barrier fence;
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3B,
                                       AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT,
                                       &fence));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned fences = 0;
   for (unsigned i = 0; i + fence.length <= binary.size; ++i) {
      if (memcmp(bytes + i, fence.bytes, fence.length))
         continue;
      /* The first write uses the fence's embedded slot1 wait. Later writes
       * rotate to slots2/3 and must finish before the cache fence, not after. */
      if (fences > 0) {
         uint8_t carrier[AGX3_ALU_MAX_LENGTH], len = 0;
         ASSERT_TRUE(agx3_alu_band_identity_wait(
            AGX_ISA_ENCODING_AGX3B, 0, 1u << fences, carrier, &len));
         ASSERT_GE(i, len);
         EXPECT_EQ(memcmp(bytes + i - len, carrier, len), 0) << fences;
      }
      ++fences;
      i += fence.length - 1;
   }
   EXPECT_EQ(fences, 3u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, ScalarCubeCoordinateAdmissionAndRetainedConsumers)
{
   for (unsigned invalid = 0; invalid < 19; ++invalid) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index sources[3];
      for (unsigned c = 0; c < 3; ++c)
         sources[c] = agx_preload(b, agx_register(4 + 2 * c, AGX_SIZE_32));
      if (invalid == 1)
         sources[0].abs = true;
      if (invalid == 2)
         sources[1].neg = true;
      if (invalid == 3)
         sources[2].size = AGX_SIZE_16;
      if (invalid == 4)
         sources[1] = agx_uniform(8, AGX_SIZE_32);
      if (invalid == 5)
         sources[2] = agx_register(128, AGX_SIZE_32);
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, vector, 3);
      for (unsigned c = 0; c < 3; ++c)
         collect->src[c] = sources[c];
      agx_index out = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *original = agx_texture_sample_to(
         b, out, vector, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
         agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_CUBE,
         AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE,
         false);
      if (invalid == 6)
         original->dim = AGX_DIM_CUBE_ARRAY;
      if (invalid == 7)
         original->lod_mode = AGX_LOD_MODE_LOD_MIN;
      if (invalid == 8)
         original->shadow = true;
      if (invalid == 9)
         original->offset = true;
      if (invalid == 10)
         original->query_lod = true;
      if (invalid == 11)
         original->sparse = true;
      if (invalid == 12)
         original->gather = AGX_GATHER_R;
      if (invalid == 13)
         original->src[3] = agx_register(24, AGX_SIZE_64);
      if (invalid == 14)
         original->src[1] = agx_immediate(1);
      if (invalid == 15)
         b->shader->stage = MESA_SHADER_COMPUTE;
      if (invalid == 16)
         b->shader->is_preamble = true;
      if (invalid == 17)
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      if (invalid == 18)
         agx_unit_test(b, vector);
      agx_instr *use = agx_unit_test(b, out);
      bool progress = agx3_lower_scalar_cube_coordinates(b->shader);
      EXPECT_EQ(progress, invalid == 0) << invalid;
      if (!progress) {
         EXPECT_EQ(original->nr_srcs, 6u);
         EXPECT_FALSE(original->agx3_cube_scalar_coords);
      } else {
         agx_instr *sample = nullptr;
         agx_foreach_instr_global(b->shader, I)
            if (I->op == AGX_OPCODE_TEXTURE_SAMPLE)
               sample = I;
         ASSERT_NE(sample, nullptr);
         EXPECT_EQ(sample->nr_srcs, 8u);
         EXPECT_TRUE(sample->agx3_cube_scalar_coords);
         EXPECT_TRUE(agx_is_equiv(sample->dest[0], out));
         for (unsigned c = 0; c < 3; ++c)
            EXPECT_TRUE(agx_is_equiv(sample->src[c ? c + 5 : 0], sources[c]));
         EXPECT_TRUE(agx_is_equiv(use->src[0], out));
         agx_dce(b->shader, true);
         unsigned collects = 0;
         agx_foreach_instr_global(b->shader, I)
            collects += I->op == AGX_OPCODE_COLLECT;
         EXPECT_EQ(collects, 0u);
         EXPECT_FALSE(agx3_lower_scalar_cube_coordinates(b->shader));
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, ScalarCubePacksNoncontiguousCemOperands)
{
   for (unsigned flags = 0; flags < 8; ++flags) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 31;
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, vector, 3);
      const unsigned source_regs[] = {4, 6, 20};
      for (unsigned c = 0; c < 3; ++c) {
         collect->src[c] = agx_register(source_regs[c], AGX_SIZE_32);
         collect->src[c].discard = flags & (1u << c);
      }
      agx_index result = agx_register(24, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, vector, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                            false, false, AGX_GATHER_NONE, false);
      agx_stop(b);
      ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
      agx_remove_instruction(collect);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary)) << flags;
      const uint8_t *bytes = (const uint8_t *)binary.data;
      for (unsigned axis = 0; axis < 3; ++axis) {
         uint8_t cem[AGX3_TEX_MAX_LENGTH], length = 0;
         const bool discard[3] = {
            axis == 2 && bool(flags & 1),
            axis == 2 && bool(flags & 2),
            axis == 2 && bool(flags & 4),
         };
         ASSERT_TRUE(
            agx3_tex_cube_cem_calc_hints_to(AGX_ISA_ENCODING_AGX3B, 4, 6, 20,
                                            axis == 0   ? 32
                                            : axis == 1 ? 36
                                                        : 38,
                                            axis, discard, cem, &length));
         unsigned matches = 0;
         for (unsigned i = 0; i + length <= binary.size; ++i)
            matches += memcmp(bytes + i, cem, length) == 0;
         EXPECT_EQ(matches, 1u) << flags << ":" << axis;
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ScalarCubeWaitsForAsynchronousYZ)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 31;
   agx_index input = agx_register(16, AGX_SIZE_32);
   input.channels_m1 = 1;
   agx_index pending = agx_register(4, AGX_SIZE_32);
   pending.channels_m1 = 3;
   agx_texture_sample_to(
      b, pending, input, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE, false);
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
   agx_instr *collect = agx_collect_to(b, vector, 3);
   const unsigned source_regs[] = {20, 4, 6};
   for (unsigned c = 0; c < 3; ++c)
      collect->src[c] = agx_register(source_regs[c], AGX_SIZE_32);
   agx_index result = agx_register(24, AGX_SIZE_32);
   result.channels_m1 = 3;
   agx_texture_sample_to(
      b, result, vector, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_CUBE,
      AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE, false);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
   agx_remove_instruction(collect);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   uint8_t cem[AGX3_TEX_MAX_LENGTH], cem_len = 0;
   ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 20, 4, 6, 32,
                                         0, false, cem, &cem_len));
   unsigned cem_offset = binary.size;
   for (unsigned i = 0; i + cem_len <= binary.size; ++i)
      if (memcmp(bytes + i, cem, cem_len) == 0)
         cem_offset = i;
   ASSERT_LT(cem_offset, binary.size);
   bool waits = false;
   for (unsigned reg : {0u, 24u}) {
      for (unsigned slot = 0; slot < 6; ++slot) {
         uint8_t carrier[AGX3_ALU_MAX_LENGTH], length = 0;
         ASSERT_TRUE(agx3_alu_band_identity_wait(AGX_ISA_ENCODING_AGX3B, reg,
                                                 1u << slot, carrier, &length));
         for (unsigned i = 0; i + length <= cem_offset; ++i)
            waits |= memcmp(bytes + i, carrier, length) == 0;
      }
   }
   EXPECT_TRUE(waits);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, NativeCubeIndependentLastUses)
{
   struct fixture {
      unsigned x, y, z, dest, axis, discard;
      uint8_t bytes[12];
   };
   const fixture vectors[] = {
      {4,
       6,
       14,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x04, 0x0d, 0x3a, 0x54, 0x21, 0x92,
        0x00}},
      {4,
       6,
       14,
       8,
       1,
       0,
       {0x17, 0x01, 0x54, 0x04, 0x03, 0x04, 0x0d, 0x3a, 0x54, 0x21, 0x92,
        0x04}},
      {4,
       6,
       14,
       4,
       2,
       7,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x04, 0x0c, 0x38, 0x50, 0x2f, 0x92,
        0x08}},
      {8,
       4,
       6,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x08, 0x09, 0x1a, 0x54, 0x21, 0x92,
        0x00}},
      {8,
       4,
       6,
       10,
       1,
       0,
       {0x17, 0x01, 0x54, 0x05, 0x03, 0x08, 0x09, 0x1a, 0x54, 0x21, 0x92,
        0x04}},
      {8,
       4,
       6,
       4,
       2,
       6,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x08, 0x08, 0x18, 0x50, 0x2d, 0x92,
        0x08}},
      {4,
       8,
       6,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x04, 0x11, 0x1a, 0x54, 0x21, 0x92,
        0x00}},
      {4,
       8,
       6,
       10,
       1,
       0,
       {0x17, 0x01, 0x54, 0x05, 0x03, 0x04, 0x11, 0x1a, 0x54, 0x21, 0x92,
        0x04}},
      {4,
       8,
       6,
       4,
       2,
       5,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x04, 0x10, 0x18, 0x50, 0x2b, 0x92,
        0x08}},
      {4,
       6,
       8,
       0,
       0,
       0,
       {0x17, 0x01, 0x54, 0x00, 0x03, 0x04, 0x0d, 0x22, 0x54, 0x21, 0x92,
        0x00}},
      {4,
       6,
       8,
       10,
       1,
       0,
       {0x17, 0x01, 0x54, 0x05, 0x03, 0x04, 0x0d, 0x22, 0x54, 0x21, 0x92,
        0x04}},
      {4,
       6,
       8,
       4,
       2,
       3,
       {0x17, 0x01, 0x54, 0x02, 0x03, 0x04, 0x0c, 0x20, 0x50, 0x27, 0x92,
        0x08}},
   };
   for (const fixture &v : vectors) {
      const bool discard[3] = {bool(v.discard & 1), bool(v.discard & 2),
                               bool(v.discard & 4)};
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], length = 0;
      ASSERT_TRUE(agx3_tex_cube_cem_calc_hints_to(AGX_ISA_ENCODING_AGX3B, v.x,
                                                  v.y, v.z, v.dest, v.axis,
                                                  discard, bytes, &length));
      ASSERT_EQ(length, 12u);
      EXPECT_EQ(memcmp(bytes, v.bytes, length), 0);
   }
}

TEST(NativeForms, ScalarCubeAllocationPreservesSharedInputs)
{
   for (unsigned mode = 0; mode < 6; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      agx_index occupied = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index occupied2 = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index sources[3];
      for (unsigned c = 0; c < 3; ++c)
         sources[c] = agx_preload(b, agx_register(4 + 2 * c, AGX_SIZE_32));
      if (mode == 2)
         sources[2] = sources[1];
      agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
      agx_instr *collect = agx_collect_to(b, vector, 3);
      for (unsigned c = 0; c < 3; ++c)
         collect->src[c] = sources[c];
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_sample_to(b, result, vector, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                            false, false, AGX_GATHER_NONE, false);
      if (mode == 3) {
         agx_index second = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
         agx_texture_sample_to(b, second, vector, agx_zero(), agx_zero(),
                               agx_uniform(32, AGX_SIZE_64),
                               agx_uniform(36, AGX_SIZE_64), agx_null(),
                               AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                               false, false, AGX_GATHER_NONE, false);
         agx_instr *split = agx_split(b, 4, second);
         for (unsigned c = 0; c < 4; ++c) {
            split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
            agx_unit_test(b, split->dest[c]);
         }
      }
      agx_instr *keep = mode == 1 ? agx_unit_test(b, sources[1]) : nullptr;
      agx_instr *split = agx_split(b, 4, result);
      for (unsigned c = 0; c < 4; ++c) {
         split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
         if (mode == 4)
            agx_export(b, split->dest[c], 8 + 2 * c);
         else
            agx_unit_test(b, split->dest[c]);
      }
      if (mode != 4) {
         agx_unit_test(b, occupied);
         agx_unit_test(b, occupied2);
      }
      ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
      agx_dce(b->shader, true);
      agx_validate(b->shader, "scalar cube allocation test");
      if (mode == 5) {
         agx_reindex_ssa(b->shader);
         b->shader->scratch_size_B = 16;
      }
      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);
      unsigned samples = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op != AGX_OPCODE_TEXTURE_SAMPLE)
            continue;
         ++samples;
         EXPECT_TRUE(I->agx3_cube_scalar_coords);
         EXPECT_EQ(I->dest[0].value % 8, 0u);
         for (unsigned src : {0u, 6u, 7u}) {
            EXPECT_EQ(I->src[src].type, AGX_INDEX_REGISTER);
            EXPECT_LT(I->src[src].value, 128u);
         }
         if (keep) {
            EXPECT_EQ(I->src[6].value, keep->src[0].value);
            EXPECT_TRUE(I->dest[0].value + 8 <= keep->src[0].value ||
                        I->dest[0].value >= keep->src[0].value + 2);
         }
         if (mode == 2) {
            EXPECT_EQ(I->src[6].value, I->src[7].value);
         }
         if (mode == 0 || mode == 4) {
            EXPECT_EQ(I->dest[0].value, 8u);
         }
      }
      EXPECT_EQ(samples, mode == 3 ? 2u : 1u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ScalarCubeSpillRebuildsCurrentAffinity)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(16, AGX_SIZE_32));
   std::vector<agx_index> live;
   for (unsigned i = 0; i < 100; ++i)
      live.push_back(agx_fmul(b, input, agx_uniform(2 * i, AGX_SIZE_32)));
   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 3);
   agx_instr *collect = agx_collect_to(b, vector, 3);
   for (unsigned c = 0; c < 3; ++c)
      collect->src[c] = live[c];
   agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_texture_sample_to(
      b, result, vector, agx_zero(), agx_zero(), agx_uniform(240, AGX_SIZE_64),
      agx_uniform(244, AGX_SIZE_64), agx_null(), AGX_DIM_CUBE,
      AGX_LOD_MODE_AUTO_LOD, 15, false, false, false, AGX_GATHER_NONE, false);
   for (agx_index value : live)
      agx_unit_test(b, value);
   agx_instr *split = agx_split(b, 4, result);
   for (unsigned c = 0; c < 4; ++c) {
      split->dest[c] = agx_temp(b->shader, AGX_SIZE_32);
      agx_unit_test(b, split->dest[c]);
   }
   ASSERT_TRUE(agx3_lower_scalar_cube_coordinates(b->shader));
   agx_dce(b->shader, true);
   agx_validate(b->shader, "scalar cube before spilling");
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   EXPECT_GT(b->shader->scratch_size_B, 0u);
   bool sample = false;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op != AGX_OPCODE_TEXTURE_SAMPLE)
         continue;
      sample = true;
      EXPECT_TRUE(I->agx3_cube_scalar_coords);
      for (unsigned src : {0u, 6u, 7u}) {
         EXPECT_EQ(I->src[src].type, AGX_INDEX_REGISTER);
         EXPECT_FALSE(I->src[src].memory);
         EXPECT_LT(I->src[src].value, 128u);
      }
   }
   EXPECT_TRUE(sample);
   ralloc_free(mem);
}

TEST(NativeForms, CubeScratchUsesDeadRegionsWithoutAsyncDestinations)
{
   for (unsigned mode = 0; mode < 5; ++mode) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 103;
      if (mode == 3)
         b->shader->any_cf = true;
      if (mode == 1) {
         agx_index input = agx_register(8, AGX_SIZE_32);
         input.channels_m1 = 1;
         agx_index pending = agx_register(36, AGX_SIZE_32);
         pending.channels_m1 = 3;
         agx_texture_sample_to(b, pending, input, agx_zero(), agx_zero(),
                               agx_uniform(24, AGX_SIZE_64),
                               agx_uniform(28, AGX_SIZE_64), agx_null(),
                               AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false,
                               false, false, AGX_GATHER_NONE, false);
      }
      agx_index coords = agx_register(4, AGX_SIZE_32);
      coords.channels_m1 = 2;
      agx_index result = agx_register(24, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, coords, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_CUBE, AGX_LOD_MODE_AUTO_LOD, 15, false,
                            false, false, AGX_GATHER_NONE, false);
      if (mode == 4)
         agx_fmul_to(b, agx_register(36, AGX_SIZE_32),
                     agx_register(12, AGX_SIZE_32),
                     agx_uniform(8, AGX_SIZE_32));
      for (unsigned reg = 16; reg < 104; reg += 2) {
         bool hole =
            reg == 36 || (reg >= 64 && reg < 68) || (reg >= 70 && reg < 74);
         if (mode == 2 && (reg == 64 || reg == 70))
            hole = false;
         if (!hole)
            agx_fmul_to(b, agx_register(0, AGX_SIZE_32),
                        agx_register(reg, AGX_SIZE_32),
                        agx_uniform(8, AGX_SIZE_32));
      }
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary)) << mode;
      bool reused = mode == 0 || mode == 1 || mode == 4;
      EXPECT_EQ(b->shader->max_reg, reused ? 103u : 111u);
      const unsigned expected[] = {
         reused ? 64u : 104u,
         reused ? ((mode == 1 || mode == 4) ? 70u : 36u) : 108u,
         reused ? ((mode == 1 || mode == 4) ? 72u : 70u) : 110u,
      };
      const uint8_t *bytes = (const uint8_t *)binary.data;
      for (unsigned axis = 0; axis < 3; ++axis) {
         uint8_t cem[AGX3_TEX_MAX_LENGTH], length = 0;
         ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 4, 6, 8,
                                               expected[axis], axis, false, cem,
                                               &length));
         unsigned matches = 0;
         for (unsigned i = 0; i + length <= binary.size; ++i)
            matches += memcmp(bytes + i, cem, length) == 0;
         EXPECT_EQ(matches, 1u) << mode << ":" << axis;
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

TEST(NativeForms, ImmediateCubeGathersReserveCemScratch)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;

   /* This is the hardware reproducer's allocated shape. The red gather
    * remains live in r0 while the remaining gathers compute their cube
    * coordinates. The old immediate-descriptor path used r0 as CEM scratch
    * and overwrote red with the next face calculation. */
   agx_index coords = agx_register(16, AGX_SIZE_32);
   coords.channels_m1 = 3;
   const unsigned destinations[] = {0, 8, 24, 16};
   for (unsigned component = 0; component < 4; ++component) {
      agx_index dst = agx_register(destinations[component], AGX_SIZE_32);
      dst.channels_m1 = 3;
      agx_texture_sample_to(b, dst, coords, agx_immediate(0),
                           agx_immediate(0), agx_immediate(0),
                           agx_immediate(1), agx_null(), AGX_DIM_CUBE_ARRAY,
                           AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
                           (enum agx_gather)(AGX_GATHER_R + 2 * component), false);
   }
   agx_stop(b);

   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GE(b->shader->max_reg, 39u);

   /* Check the actual emitted CEM destination, not just the reservation.
    * All four axis-zero operations must write above the live result bank. */
   uint8_t cem[AGX3_TEX_MAX_LENGTH], len = 0;
   ASSERT_TRUE(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 16, 18, 20,
                                         32, 0, false, cem, &len));
   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned count = 0;
   for (unsigned i = 0; i + len <= binary.size; ++i)
      count += memcmp(bytes + i, cem, len) == 0;
   EXPECT_EQ(count, 4u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}


TEST(NativeForms, LateCoverageStaysAfterImageSideEffects)
{
   for (bool depth : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index data = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_device_load_to(b, data, agx_uniform(8, AGX_SIZE_64), agx_immediate(0),
                         AGX_FORMAT_I32, 15, 0, false);
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_device_load_to(b, coord, agx_uniform(8, AGX_SIZE_64), agx_immediate(4),
                         AGX_FORMAT_I32, 3, 0, false);
      agx_instr *store = agx_image_write(
         b, data, coord, agx_zero(), agx_uniform(12, AGX_SIZE_64),
         agx_mov_imm(b, 32, 1), AGX_DIM_2D, true);
      agx_index mask = agx_mov_imm(b, 16, 1);
      agx_index coverage[8];
      for (unsigned i = 0; i < 8; ++i)
         coverage[i] = agx_get_coverage_1x_agx3(b, 16);
      agx_index word = coverage[0];
      for (unsigned i = 1; i < 8; ++i)
         word = agx_iadd(b, word, coverage[i], 0);
      word = agx_iadd(b, mask, word, 4);
      agx_index wide = agx_temp(b->shader, AGX_SIZE_32);
      agx_mov_to(b, wide, word);
      agx_instr *tests = depth ? agx_zs_emit_agx3(b, wide, 1)
                              : agx_sample_mask_agx3(b, mask, wide, 2);
      agx_export(b, data, 8);
      agx_pressure_schedule(b->shader);
      bool saw_store = false;
      agx_foreach_instr_global(b->shader, I) {
         saw_store |= I == store;
         if (I == tests) {
            EXPECT_TRUE(saw_store) << "Native late tests preceded imageStore";
         }
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, VertexOutputsAllowTrailingMemoryFence)
{
   for (unsigned count : {5u, 6u, 11u}) {
      SCOPED_TRACE(count);
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_VERTEX;
      for (unsigned i = 0; i < count; ++i)
         agx_st_vary(b, agx_immediate(i), agx_register(2 * i, AGX_SIZE_32));
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 0u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, CubeArrayShadowZeroLodUsesNativeDescriptors)
{
   for (mesa_shader_stage stage : {MESA_SHADER_VERTEX, MESA_SHADER_TESS_CTRL,
                                MESA_SHADER_TESS_EVAL, MESA_SHADER_GEOMETRY,
                                MESA_SHADER_FRAGMENT, MESA_SHADER_COMPUTE}) {
      for (float lod : {0.0f, 1.0f}) {
         SCOPED_TRACE(::testing::Message() << "stage=" << stage << " lod=" << lod);
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "cube array shadow native descriptors");
         nir_def *texture = nir_bindless_image_agx(&b, nir_imm_int(&b, 0));
         nir_def *sampler = nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, 5, 16));
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
         tex->op = nir_texop_txl;
         tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
         tex->is_array = true;
         tex->is_shadow = true;
         tex->is_new_style_shadow = true;
         tex->coord_components = 4;
         tex->dest_type = nir_type_float32;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
                                         nir_imm_vec4(&b, 1, 0, 0, 0));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                         nir_imm_float(&b, 0.5));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, lod));
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
         tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
         nir_def_init(&tex->instr, &tex->def, 1, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
         EXPECT_EQ(agx_nir_lower_g17p_indexed_samplers(b.shader, true), lod == 0.0f);
         ralloc_free(b.shader);
      }
   }
}

/* A constant descriptor may be read ahead of its sample. A dynamic index is
 * only known to be in range where the sample runs, and there are no soft
 * faults: speculating `if (i >= 0) sample(tex[i])` would read element -1. */
TEST(NativeForms, HeapIndexReadsSpeculateOnlyForConstantDescriptors)
{
   for (bool dynamic : {false, true}) {
      SCOPED_TRACE(dynamic);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "heap index access");
      nir_def *texture = nir_bindless_image_agx(
         &b, dynamic ? nir_load_local_invocation_index(&b) : nir_imm_int(&b, 0));
      nir_def *sampler = nir_bindless_sampler_agx(
         &b, nir_imm_int(&b, 32), nir_imm_intN_t(&b, 5, 16));
      nir_tex_instr *tex = nir_tex_instr_create(b.shader, 5);
      tex->op = nir_texop_txl;
      tex->sampler_dim = GLSL_SAMPLER_DIM_CUBE;
      tex->is_array = true;
      tex->is_shadow = true;
      tex->is_new_style_shadow = true;
      tex->coord_components = 4;
      tex->dest_type = nir_type_float32;
      tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
                                      nir_imm_vec4(&b, 1, 0, 0, 0));
      tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_comparator,
                                      nir_imm_float(&b, 0.5));
      tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0));
      tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
      tex->src[4] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
      nir_def_init(&tex->instr, &tex->def, 1, 32);
      nir_builder_instr_insert(&b, &tex->instr);
      nir_store_ssbo(&b, &tex->def, nir_imm_int(&b, 0), nir_imm_int(&b, 0));
      ASSERT_TRUE(agx_nir_lower_g17p_indexed_samplers(b.shader, true));

      unsigned reads = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic !=
                   nir_intrinsic_load_global_constant)
               continue;
            reads++;
            EXPECT_EQ(nir_intrinsic_access(nir_instr_as_intrinsic(instr)) &
                         ACCESS_CAN_SPECULATE,
                      dynamic ? 0u : (unsigned)ACCESS_CAN_SPECULATE);
         }
      }
      EXPECT_EQ(reads, 1u);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, EndOfTilePreservesSamplesAndResolvesSameTile)
{
   /* Both outputs read offset zero. Their descriptors are distinct, and
    * only the second descriptor names a single-sample image. Keep all four
    * dimension combinations as controls; no host check establishes pixels. */
   for (unsigned resolves = 0; resolves < 4; ++resolves) {
      SCOPED_TRACE(resolves);
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "EOT store and resolve");
      nir_def *xy = nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
      for (unsigned pbe = 0; pbe < 2; ++pbe) {
         nir_intrinsic_instr *intr = nir_image_store_block_agx(
            &b, nir_imm_intN_t(&b, pbe, 16), nir_imm_int(&b, 0), xy);
         nir_intrinsic_set_format(intr, (enum pipe_format)AIL_ISA_FORMAT_U8NORM);
         nir_intrinsic_set_image_dim(intr, GLSL_SAMPLER_DIM_2D);
         nir_intrinsic_set_image_array(intr, false);
         nir_intrinsic_set_explicit_coord(intr, true);
      }
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.no_preamble = true;
      key.g17p_rt0_eot_multisample = true;
      key.g17p_mrt_eot_count = 2;
      key.g17p_mrt_eot_resolve_mask = resolves;
      key.g17p_mrt_eot_selectors[0] = 0x75;
      key.g17p_mrt_eot_selectors[1] = 0x75;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      bool ok = agx_compile_shader_nir(b.shader, &key, &part);
      EXPECT_TRUE(ok);
      if (ok) {
         EXPECT_EQ(part.info.binary_size, 120u);
         EXPECT_EQ(part.info.main_size, 120u);
         EXPECT_TRUE(part.info.agx3_staging_known);
         EXPECT_EQ(part.info.agx3_staging_halfs, 16u);
         if (part.info.binary_size == 120) {
            const uint8_t *code = static_cast<const uint8_t *>(part.binary);
            /* Native single-target MS and SS dimension fields, independently
             * selected for each store of the existing dual-target program. */
            for (unsigned pbe = 0; pbe < 2; ++pbe) {
               bool ms = !(resolves & (1u << pbe));
               EXPECT_EQ(code[pbe ? 63 : 43],
                         (pbe ? 0x0c : 0x04) | (ms ? 0x02 : 0x00));
               EXPECT_EQ(code[92 + 12 * pbe + 8], ms ? 0x28 : 0xa8);
               EXPECT_EQ(code[92 + 12 * pbe + 9], ms ? 0x72 : 0x75);
            }
         }
      }
      free(part.binary);
      ralloc_free(b.shader);
   }
}

static agx_shader_info
compile_partial_varying(unsigned components, unsigned bits,
                        glsl_interp_mode mode)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "partial varying cache");
   nir_def *bary = nir_load_barycentric_pixel(&b, 32);
   nir_intrinsic_set_interp_mode(nir_def_as_intrinsic(bary), mode);
   nir_def *input = nir_load_interpolated_input(&b, components, bits, bary,
                                                nir_imm_int(&b, 0));
   nir_intrinsic_instr *load = nir_def_as_intrinsic(input);
   nir_intrinsic_set_component(load, components == 1 ? 3 : 0);
   nir_intrinsic_set_dest_type(
      load, bits == 16 ? nir_type_float16 : nir_type_float32);
   nir_intrinsic_set_io_semantics(
      load, nir_io_semantics{.location = VARYING_SLOT_VAR0});
   nir_def *value = nir_channel(&b, input, components - 1);
   if (bits == 16)
      value = nir_f2f32(&b, value);
   agx_test_store_local_pixel_agx(&b, nir_vec4(&b, value, value, value, value),
                                  nir_imm_intN_t(&b, 0xff, 16),
                                  nir_undef(&b, 2, 16), 0, 0xf,
                                  PIPE_FORMAT_R32_UINT);
   b.shader->info.inputs_read = BITFIELD64_BIT(VARYING_SLOT_VAR0);
   b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.no_preamble = true;
   key.fs.nr_samples = 1;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   agx_shader_info info = part.info;
   free(part.binary);
   ralloc_free(b.shader);
   return info;
}

TEST(NativeForms, PartialVaryingOmitsUnusedChannels)
{
   for (unsigned bits : {16u, 32u}) {
      for (glsl_interp_mode mode :
           {INTERP_MODE_SMOOTH, INTERP_MODE_NOPERSPECTIVE}) {
         SCOPED_TRACE(bits);
         SCOPED_TRACE(mode);
         agx_shader_info scalar = compile_partial_varying(1, bits, mode);
         agx_shader_info vector = compile_partial_varying(4, bits, mode);
         ASSERT_GT(scalar.main_size, 0u);
         EXPECT_LE(vector.main_size, scalar.main_size + 8);
         EXPECT_LE(vector.nr_gprs, 2 * scalar.nr_gprs);
      }
   }
}

TEST(NativeForms, TextureFmaMatchesMetalStaging)
{
   struct vector {
      uint16_t dest;
      uint8_t slot;
      uint16_t src[3];
      bool live[3], discard[3];
      uint16_t wait;
      uint8_t bytes[16];
      uint8_t length;
   };
   const vector vectors[] = {
      {0,
       1,
       {46, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x09, 0x87, 0x26, 0x2f, 0x01, 0x84, 0x42, 0x02},
       8},
      {4,
       2,
       {48, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x29, 0x87, 0x26, 0x31, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {8,
       2,
       {50, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x49, 0x87, 0x26, 0x33, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {12,
       2,
       {52, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x69, 0x87, 0x26, 0x35, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {2,
       1,
       {46, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x19, 0x83, 0x26, 0x2f, 0x01, 0x80, 0x42, 0x02},
       8},
      {6,
       2,
       {48, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x39, 0x83, 0x26, 0x31, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {10,
       2,
       {50, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x59, 0x83, 0x26, 0x33, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {14,
       2,
       {52, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x79, 0x83, 0x26, 0x35, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {16,
       2,
       {54, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x89, 0x87, 0x26, 0x37, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {20,
       2,
       {56, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xa9, 0x87, 0x26, 0x39, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
       10},
      {28,
       4,
       {58, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xe9, 0x87, 0x26, 0x3b, 0x02, 0x84, 0x42, 0x02, 0x00, 0x0c},
       10},
      {36,
       5,
       {62, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x29, 0x87, 0x66, 0x3f, 0x02, 0x84, 0x42, 0x02, 0x00, 0x10},
       10},
      {40,
       5,
       {64, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x49, 0x87, 0x66, 0x41, 0x02, 0x84, 0x42, 0x02, 0x00, 0x10},
       10},
      {42,
       5,
       {64, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x59, 0x83, 0x66, 0x41, 0x02, 0x80, 0x42, 0x02, 0x00, 0x10},
       10},
      {38,
       5,
       {62, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x39, 0x83, 0x66, 0x3f, 0x02, 0x80, 0x42, 0x02, 0x00, 0x10},
       10},
      {44,
       6,
       {60, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x69, 0x87, 0x66, 0x3d, 0x02, 0x84, 0x42, 0x02, 0x00, 0x14},
       10},
      {46,
       6,
       {60, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x79, 0x83, 0x66, 0x3d, 0x02, 0x80, 0x42, 0x02, 0x00, 0x14},
       10},
      {30,
       4,
       {58, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xf9, 0x83, 0x26, 0x3b, 0x02, 0x80, 0x42, 0x02, 0x00, 0x0c},
       10},
      {22,
       2,
       {56, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xb9, 0x83, 0x26, 0x39, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {18,
       2,
       {54, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x99, 0x83, 0x26, 0x37, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
       10},
      {32,
       6,
       {66, 6, 4},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0x09, 0x87, 0x66, 0x43, 0x02, 0x84, 0x42, 0x02, 0x00, 0x14},
       10},
      {56,
       4,
       {68, 6, 4},
       {0, 0, 0},
       {0, 1, 1},
       0,
       {0xc9, 0x07, 0x6e, 0x45, 0x82, 0x04, 0x42, 0x02, 0x00, 0x0c},
       10},
      {58,
       4,
       {68, 2, 0},
       {0, 1, 1},
       {0, 0, 0},
       0,
       {0xd9, 0x83, 0x66, 0x45, 0x02, 0x80, 0x42, 0x02, 0x00, 0x0c},
       10},
      {34,
       6,
       {66, 2, 0},
       {0, 0, 0},
       {0, 1, 1},
       0,
       {0x19, 0x03, 0x6e, 0x43, 0x82, 0x00, 0x42, 0x02, 0x00, 0x14},
       10},
   };
   for (const auto &v : vectors) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_stage_vertex_alu(AGX_ISA_ENCODING_AGX3B, true, v.dest,
                                        v.slot, v.src, true, v.live, v.discard,
                                        v.wait, bytes, &length));
      ASSERT_EQ(length, v.length);
      EXPECT_EQ(memcmp(bytes, v.bytes, length), 0);
   }
}

TEST(NativeForms, TextureFmaStagingAdmission)
{
   for (unsigned invalid = 0; invalid < 15; ++invalid) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index x = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index y = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index z = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index multiplier = agx_uniform(20, AGX_SIZE_32);
      if (invalid == 12)
         multiplier = agx_preload(b, agx_register(6, AGX_SIZE_32));
      if (invalid == 13)
         multiplier = agx_immediate_f(0.5f);
      if (invalid == 14)
         multiplier = agx_immediate(256);
      if (invalid == 1)
         multiplier.size = AGX_SIZE_16;
      if (invalid == 2)
         y = agx_uniform(22, AGX_SIZE_32);
      if (invalid == 3)
         z = agx_uniform(22, AGX_SIZE_32);
      if (invalid == 4)
         x.neg = true;
      if (invalid == 10)
         multiplier.value = 256;
      if (invalid == 11)
         multiplier.value = 21;
      agx_index u = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, u, x, multiplier, z);
      agx_instr *first =
         agx_last_instr(list_first_entry(&b->shader->blocks, agx_block, link));
      if (invalid == 5)
         first->saturate = true;
      agx_index v = agx_temp(b->shader, AGX_SIZE_32);
      agx_ffma_to(b, v, multiplier, y, z);
      agx_instr *second =
         agx_last_instr(list_first_entry(&b->shader->blocks, agx_block, link));
      if (invalid == 6)
         agx_unit_test(b, u);
      if (invalid == 7) {
         agx_instr *store = agx_alloc_instr(b, AGX_OPCODE_UNIFORM_STORE, 0, 2);
         store->src[0] = z;
         store->src[1] = agx_immediate(20);
         store->mask = 1;
         agx_builder_insert(&b->cursor, store);
      }
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = u;
      collect->src[1] = v;
      agx_index out = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, out, coord, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
         agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
         AGX_LOD_MODE_AUTO_LOD, 0xf, false, false, false, AGX_GATHER_NONE,
         false);
      if (invalid == 8)
         sample->dim = AGX_DIM_3D;
      if (invalid == 9)
         sample->lod_mode = AGX_LOD_MODE_LOD_MIN;
      agx_unit_test(b, out);
      agx3_lower_scalar_texture_publications(b->shader);
      agx_dce(b->shader, true);
      agx_instr *publication = nullptr;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3)
            publication = I;
      }
      const bool progress = agx3_fuse_fma_texture_publications(b->shader);
      EXPECT_EQ(progress, (invalid == 0 || invalid == 9 || invalid == 12 ||
                           invalid == 13))
         << invalid;
      EXPECT_EQ(first->agx3_tex_coord_stage, (invalid == 0 || invalid == 9 ||
                                              invalid == 12 || invalid == 13))
         << invalid;
      EXPECT_EQ(second->agx3_tex_coord_stage, (invalid == 0 || invalid == 9 ||
                                               invalid == 12 || invalid == 13))
         << invalid;
      const bool supported =
         invalid == 0 || invalid == 9 || invalid == 12 || invalid == 13;
      EXPECT_FALSE(sample->agx3_tex_coords_staged);
      if (publication) {
         EXPECT_EQ(publication->agx3_tex_coords_staged, supported);
      }
      if (supported) {
         ASSERT_NE(publication, nullptr);
         EXPECT_TRUE(first->agx3_tex_coord_start);
         EXPECT_EQ(first->link.next, &second->link);
         EXPECT_EQ(second->link.next, &publication->link);
         EXPECT_EQ(sample->src[0].type, AGX_INDEX_RESOURCE);
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TextureFmaImmediateMatchesMetal)
{
   const uint16_t sources[2][3] = {{32, 6, 4}, {32, 2, 0}};
   const uint8_t expected[2][10] = {
      {0xc9, 0x87, 0x26, 0xa1, 0x02, 0x84, 0x42, 0x02, 0x00, 0x04},
      {0xd9, 0x83, 0x26, 0xa1, 0x02, 0x80, 0x42, 0x02, 0x00, 0x04},
   };
   const bool live[3] = {false, true, true};
   const bool discard[3] = {};
   for (unsigned i = 0; i < 2; ++i) {
      uint8_t bytes[16], length = 0;
      ASSERT_TRUE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 24 + 2 * i, 2,
                                 sources[i], AGX3_STAGE_FMA_IMMEDIATE, live,
                                 discard, 0, bytes, &length));
      ASSERT_EQ(length, sizeof(expected[i]));
      EXPECT_EQ(memcmp(bytes, expected[i], length), 0);
   }
   uint8_t bytes[16], length = 0;
   const uint16_t invalid_sources[3] = {256, 2, 0};
   EXPECT_FALSE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 0, 1, invalid_sources,
                               AGX3_STAGE_FMA_IMMEDIATE, live, discard, 0,
                               bytes, &length));
   const bool invalid_live[3] = {true, true, true};
   EXPECT_FALSE(agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 0, 1, sources[0],
                               AGX3_STAGE_FMA_IMMEDIATE, invalid_live, discard,
                               0, bytes, &length));
}

TEST(NativeForms, TextureFmaStagingPreservesDemand)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index operands[6];
   for (unsigned i = 0; i < 6; ++i)
      operands[i] = agx_iadd(b, input, agx_immediate(i + 1), 0);
   agx_index x = agx_ffma(b, operands[0], operands[1], operands[2]);
   agx_index live[10];
   for (unsigned i = 0; i < 10; ++i)
      live[i] = agx_iadd(b, input, agx_immediate(i + 10), 0);
   agx_index y = agx_ffma(b, operands[3], operands[4], operands[5]);
   agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, coord, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   agx_index out = agx_temp(b->shader, AGX_SIZE_32);
   agx_texture_sample_to(
      b, out, coord, agx_zero(), agx_zero(), agx_uniform(24, AGX_SIZE_64),
      agx_uniform(28, AGX_SIZE_64), agx_null(), AGX_DIM_2D,
      AGX_LOD_MODE_AUTO_LOD, 1, false, false, false, AGX_GATHER_NONE, false);
   for (agx_index value : live)
      agx_unit_test(b, value);
   agx_unit_test(b, out);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   std::vector<const agx_instr *> order;
   agx_foreach_instr_global(b->shader, I)
      order.push_back(I);
   agx_compute_liveness(b->shader);
   unsigned before = agx_calc_register_demand(b->shader, false);
   EXPECT_FALSE(agx3_fuse_fma_texture_publications(b->shader));
   agx_compute_liveness(b->shader);
   EXPECT_EQ(agx_calc_register_demand(b->shader, false), before);
   unsigned index = 0;
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_FALSE(I->agx3_tex_coords_staged);
      if (I->op == AGX_OPCODE_FFMA) {
         EXPECT_EQ(I->dest[0].type, AGX_INDEX_NORMAL);
      }
      if (agx_is_texture_publication(I)) {
         EXPECT_EQ(I->src[0].type, AGX_INDEX_NORMAL);
         EXPECT_EQ(I->src[1].type, AGX_INDEX_NORMAL);
      }
      ASSERT_LT(index, order.size());
      EXPECT_EQ(I, order[index++]);
   }
   EXPECT_EQ(index, order.size());
   ralloc_free(mem);
}

TEST(NativeForms, MultipleTextureFmaPublications)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index output[3];
   for (unsigned i = 0; i < 3; ++i) {
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_index x = agx_ffma(b, input, agx_uniform(2 * i, AGX_SIZE_32), input);
      agx_index y = agx_ffma(b, agx_uniform(6 + 2 * i, AGX_SIZE_32), input, input);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = x;
      collect->src[1] = y;
      output[i] = agx_temp(b->shader, AGX_SIZE_32);
      agx_index sampled = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_texture_sample_to(
         b, sampled, coord, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
      agx_instr *split = agx_split(b, 4, sampled);
      split->dest[0] = output[i];
      for (unsigned c = 1; c < 4; ++c)
         split->dest[c] = agx_null();
   }
   for (unsigned i = 0; i < 3; ++i)
      agx_export(b, output[i], 8 + 2 * i);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   ASSERT_TRUE(agx3_fuse_fma_texture_publications(b->shader));
   agx_dce(b->shader, true);
   unsigned stages = 0, publications = 0;
   agx_foreach_instr_global(b->shader, I) {
      stages += I->agx3_tex_coord_stage;
      if (I->agx3_tex_coord_stage) {
         EXPECT_TRUE(agx_is_null(I->dest[0]));
      }
      if (agx_is_texture_publication(I)) {
         publications++;
         EXPECT_TRUE(I->agx3_tex_coords_staged);
         EXPECT_TRUE(agx_is_null(I->src[0]));
         EXPECT_TRUE(agx_is_null(I->src[1]));
      }
   }
   EXPECT_EQ(stages, 6u);
   EXPECT_EQ(publications, 3u);
   EXPECT_FALSE(agx3_fuse_fma_texture_publications(b->shader));
   agx_ra(b->shader);
   EXPECT_FALSE(b->shader->ra_failed);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPrecedesEarlierSamples)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index x = agx_ffma(b, input, agx_uniform(0, AGX_SIZE_32), input);
   agx_index y = agx_ffma(b, input, agx_uniform(2, AGX_SIZE_32), input);
   agx_instr *samples[2];
   for (unsigned i = 0; i < 2; ++i) {
      agx_index coordinates = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coordinates, 2);
      collect->src[0] = i ? x : input;
      collect->src[1] = i ? y : input;
      agx_index result = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      samples[i] = agx_texture_sample_to(
         b, result, coordinates, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
      agx_unit_test(b, result);
   }
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   ASSERT_TRUE(agx3_fuse_fma_texture_publications(b->shader));
   bool published = false;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_PUBLISH_TEXTURE_SCALAR_AGX3 &&
          I->imm == agx_texture_resource(samples[1])) {
         EXPECT_TRUE(I->agx3_tex_coords_staged);
         published = true;
      }
      if (I == samples[0]) {
         EXPECT_TRUE(published);
      }
   }
   EXPECT_TRUE(published);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPressure)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
   agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
   agx_index coordinates[65][2];
   for (unsigned i = 0; i < 65; ++i) {
      coordinates[i][0] = agx_ffma(b, input, agx_uniform(2 * i, AGX_SIZE_32), input);
      coordinates[i][1] = agx_ffma(b, input, agx_uniform(2 * i + 2, AGX_SIZE_32), input);
   }
   agx_instr *first_sample = nullptr;
   agx_index sum = agx_null();
   for (unsigned i = 0; i < 65; ++i) {
      agx_index coord = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
      agx_instr *collect = agx_collect_to(b, coord, 2);
      collect->src[0] = coordinates[i][0];
      collect->src[1] = coordinates[i][1];
      agx_index sampled = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
      agx_instr *sample = agx_texture_sample_to(
         b, sampled, coord, agx_zero(), agx_zero(),
         agx_uniform(200, AGX_SIZE_64), agx_uniform(204, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
      if (!i)
         first_sample = sample;
      agx_index x = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *split = agx_split(b, 4, sampled);
      split->dest[0] = x;
      for (unsigned c = 1; c < 4; ++c)
         split->dest[c] = agx_null();
      sum = i ? agx_fadd(b, sum, x) : x;
   }
   agx_export(b, sum, 8);
   agx_stop(b);
   ASSERT_TRUE(agx3_lower_scalar_texture_publications(b->shader));
   agx_dce(b->shader, true);
   ASSERT_TRUE(agx3_fuse_fma_texture_publications(b->shader));
   unsigned early = 0, fused = 0;
   bool before_sample = true;
   agx_foreach_instr_global(b->shader, I) {
      if (I == first_sample)
         before_sample = false;
      if (agx_is_texture_publication(I) && I->agx3_tex_coords_staged) {
         fused++;
         early += before_sample;
      }
   }
   EXPECT_EQ(fused, 65u);
   EXPECT_EQ(early, 64u);
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   agx_lower_pseudo(b->shader);
   agx_opt_register_cache(b->shader);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_GT(binary.size, 0u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaStagingRespectsCoverageOrdering)
{
   for (unsigned position = 0; position < 3; ++position) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      agx_index input = agx_preload(b, agx_register(0, AGX_SIZE_32));
      agx_index outputs[2];
      for (unsigned i = 0; i < 3; ++i) {
         if (i == position) {
            agx_instr *coverage =
               agx_alloc_instr(b, AGX_OPCODE_SAMPLE_MASK_AGX3, 0, 2);
            coverage->src[0] = input;
            coverage->src[1] = agx_zero();
            agx_builder_insert(&b->cursor, coverage);
         }
         if (i < 2)
            outputs[i] = agx_ffma(b, input, agx_uniform(2 * i, AGX_SIZE_32), input);
      }
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), 1);
      std::vector<const agx_instr *> order;
      agx_foreach_instr_global(b->shader, I)
         order.push_back(I);
      EXPECT_EQ(agx3_fuse_fma_texture_publications(b->shader), position == 0);
      EXPECT_EQ(publication->agx3_tex_coords_staged, position == 0);
      if (position != 0) {
         unsigned index = 0;
         agx_foreach_instr_global(b->shader, I) {
            ASSERT_LT(index, order.size());
            EXPECT_EQ(I, order[index++]);
         }
         EXPECT_EQ(index, order.size());
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, TextureFmaPublicationsPreserveReusedCoordinates)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->max_reg = 63;
   for (unsigned pair = 0; pair < 2; ++pair) {
      agx_index outputs[2];
      for (unsigned c = 0; c < 2; ++c) {
         outputs[c] = agx_register(16 + 4 * pair + 2 * c, AGX_SIZE_32);
         agx_instr *fma = agx_ffma_to(
            b, outputs[c], agx_uniform(12 + 4 * pair + 2 * c, AGX_SIZE_32),
            agx_register(2, AGX_SIZE_32), agx_register(4, AGX_SIZE_32));
         fma->agx3_tex_coord_stage = true;
         fma->agx3_tex_coord_start = c == 0;
         fma->agx3_tex_coord_component = c;
      }
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), pair + 1);
      publication->agx3_tex_coords_staged = true;
   }
   const unsigned values[3] = {1, 2, 1};
   for (unsigned i = 0; i < 3; ++i) {
      agx_index resource = agx_get_vec_index(values[i], AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index result = agx_register(40 + 8 * i, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(
         b, result, resource, agx_zero(), agx_zero(),
         agx_uniform(24, AGX_SIZE_64), agx_uniform(28, AGX_SIZE_64), agx_null(),
         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false, false,
         AGX_GATHER_NONE, false);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
   EXPECT_EQ(b->shader->agx3_staging_halfs, 8u);
   const uint8_t *bytes = (const uint8_t *)binary.data;
   const bool hints[3] = {};
   for (unsigned pair = 0; pair < 2; ++pair) {
      for (unsigned c = 0; c < 2; ++c) {
         const uint16_t src[3] = {uint16_t(12 + 4 * pair + 2 * c), 2, 4};
         unsigned found = 0;
         for (unsigned slot = 1; slot <= 6; ++slot) {
            for (unsigned wait = 0; wait < 64; ++wait) {
               uint8_t expected[16], length = 0;
               ASSERT_TRUE(agx3_stage_fma(
                  AGX_ISA_ENCODING_AGX3B, 4 * pair + 2 * c, slot, src,
                  AGX3_STAGE_FMA_UNIFORM, hints, hints, wait, expected, &length));
               for (unsigned offset = 0; offset + length <= binary.size; ++offset)
                  found += memcmp(bytes + offset, expected, length) == 0;
            }
         }
         EXPECT_EQ(found, 1u) << pair << ":" << c;
      }
   }
   for (unsigned i = 0; i < 3; ++i) {
      agx3_tex_sample_desc desc = {};
      desc.dest = 40 + 8 * i;
      desc.coord = 4 * (values[i] - 1);
      desc.width = AGX3_TEX_W32;
      desc.mask = 15;
      desc.lod = AGX3_TEX_LOD_AUTO;
      desc.texture_kind = desc.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
      desc.texture_value = 24;
      desc.sampler_value = 28;
      desc.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
      desc.coord_kill = i != 0;
      desc.final = desc.sample_group_end = i == 2;
      desc.sample_group_nonterminal = i != 2;
      unsigned found = 0;
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait = 0; wait <= 6; ++wait) {
            desc.slot = slot;
            desc.wait_mask = wait ? 1u << (wait - 1) : 0;
            uint8_t expected[AGX3_TEX_MAX_LENGTH], length = 0;
            ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &desc,
                                              expected, &length));
            for (unsigned offset = 0; offset + length <= binary.size; ++offset)
               found += memcmp(bytes + offset, expected, length) == 0;
         }
      }
      EXPECT_EQ(found, 1u) << i;
   }
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

TEST(NativeForms, TextureFmaPublicationPacksWithoutCopy)
{
   for (bool loaded : {false, true}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->max_reg = 31;
      if (loaded)
         agx_device_load_to(b, agx_register(2, AGX_SIZE_32),
                            agx_register(28, AGX_SIZE_64), agx_zero(),
                            AGX_FORMAT_I32, 1, 0, false);
      const agx_index outputs[2] = {
         agx_register(16, AGX_SIZE_32),
         agx_register(18, AGX_SIZE_32),
      };
      agx_instr *x = agx_ffma_to(b, outputs[0], agx_uniform(20, AGX_SIZE_32),
                                 agx_register(2, AGX_SIZE_32),
                                 agx_register(4, AGX_SIZE_32));
      agx_instr *y = agx_ffma_to(b, outputs[1], agx_register(6, AGX_SIZE_32),
                                 agx_uniform(20, AGX_SIZE_32),
                                 agx_register(8, AGX_SIZE_32));
      x->agx3_tex_coord_stage = x->agx3_tex_coord_start = true;
      y->agx3_tex_coord_stage = true;
      y->agx3_tex_coord_component = 1;
      agx_instr *publication = agx_publish_texture_scalar_agx3(
         b, outputs[0], outputs[1], agx_null(), 1);
      publication->agx3_tex_coords_staged = true;
      agx_index resource = agx_get_vec_index(1, AGX_SIZE_32, 2);
      resource.type = AGX_INDEX_RESOURCE;
      agx_index result = agx_register(20, AGX_SIZE_32);
      result.channels_m1 = 3;
      agx_texture_sample_to(b, result, resource, agx_zero(), agx_zero(),
                            agx_uniform(24, AGX_SIZE_64),
                            agx_uniform(28, AGX_SIZE_64), agx_null(),
                            AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 15, false, false,
                            false, AGX_GATHER_NONE, false);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      const uint8_t *bytes = (const uint8_t *)binary.data;
      const uint16_t src[2][3] = {{20, 2, 4}, {20, 6, 8}};
      const bool hints[3] = {};
      unsigned found[2] = {};
      unsigned publication_slot[2] = {};
      for (unsigned slot = 1; slot <= 6; ++slot) {
         for (unsigned wait = 0; wait < 64; ++wait) {
            for (unsigned c = 0; c < 2; ++c) {
               uint8_t expected[16], length = 0;
               if (!agx3_stage_fma(AGX_ISA_ENCODING_AGX3B, 2 * c, slot, src[c],
                                   AGX3_STAGE_FMA_UNIFORM, hints, hints, wait,
                                   expected, &length))
                  continue;
               for (unsigned offset = 0; offset + length <= binary.size;
                    ++offset) {
                  if (memcmp(bytes + offset, expected, length))
                     continue;
                  found[c]++;
                  if (c == 0) {
                     EXPECT_EQ(wait != 0, loaded);
                  }
                  publication_slot[c] = slot;
               }
            }
         }
      }
      EXPECT_EQ(found[0], 1u);
      EXPECT_EQ(found[1], 1u);
      EXPECT_EQ(publication_slot[0], publication_slot[1]);
      for (unsigned c = 0; c < 2; ++c) {
         for (unsigned slot = 1; slot <= 6; ++slot) {
            uint8_t copy[AGX3_STAGE_MAX_LENGTH], length = 0;
            ASSERT_TRUE(agx3_stage_copy_source(AGX_ISA_ENCODING_AGX3B, 2 * c,
                                               outputs[c].value, false, slot, 0,
                                               copy, &length));
            for (unsigned offset = 0; offset + length <= binary.size; ++offset)
               EXPECT_NE(memcmp(bytes + offset, copy, length), 0);
         }
      }
      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

#ifndef NDEBUG

static agx_index
validation_span_value(agx_builder *b, bool memory, unsigned reg,
                      agx_size size, unsigned channels)
{
   agx_index value = agx_vec_temp(b->shader, size, channels);
   value.has_reg = true;
   value.reg = reg;
   value.memory = memory;
   agx_index uniform = agx_uniform(0, size);
   uniform.channels_m1 = channels - 1;
   agx_mov_to(b, value, uniform);
   return value;
}

static void
validation_span_diamond(bool memory, unsigned reg, agx_size size,
                        unsigned channels, bool clobber)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);
   agx_index value = validation_span_value(b, memory, reg, size, channels);
   if (clobber) {
      b->cursor = agx_after_block(right);
      validation_span_value(b, memory, reg + agx_index_size_16(value) - 1,
                            AGX_SIZE_16, 1);
   }
   b->cursor = agx_after_block(join);
   agx_unit_test(b, value);
   agx_validate_ra(b->shader);
   ralloc_free(mem);
}

TEST(NativeForms, RegisterValidationTracksMergeSpansAndClobbers)
{
   for (bool memory : {false, true}) {
      for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
         for (unsigned channels : {1u, 3u, 4u}) {
            unsigned width = channels * agx_size_align_16(size);
            for (unsigned reg : {8u, 128u, AGX_NUM_MODELED_REGS - width}) {
               for (bool clobber : {false, true}) {
                  EXPECT_EXIT({
                     std::signal(SIGABRT, +[](int) { std::_Exit(77); });
                     validation_span_diamond(memory, reg, size, channels, clobber);
                     std::_Exit(0);
                  }, ::testing::ExitedWithCode(clobber ? 77 : 0),
                     clobber ? "failed to validate RA" : "");
               }
            }
         }
      }
   }
}

TEST(NativeForms, RegisterValidationTracksPhysicalMovesAndSwaps)
{
   for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
      for (unsigned channels : {1u, 3u, 4u}) {
         EXPECT_EXIT({
            std::signal(SIGABRT, +[](int) { std::_Exit(77); });
            void *mem = ralloc_context(NULL);
            agx_builder *b = agx_test_builder(mem);
            agx_block *entry = agx_start_block(b->shader);
            agx_block *next = agx_test_block(b->shader);
            agx_block_add_successor(entry, next);
            agx_index a = validation_span_value(b, false, 16, size, channels);
            agx_index z = validation_span_value(b, false, 64, size, channels);
            unsigned spill_reg = AGX_NUM_MODELED_REGS - agx_index_size_16(a);
            b->cursor = agx_after_block(next);
            agx_index spill = agx_memory_register(spill_reg, size);
            spill.channels_m1 = channels - 1;
            spill.memory = true;
            agx_index source = agx_register(16, size);
            source.channels_m1 = channels - 1;
            agx_mov_to(b, spill, source);
            agx_index other = agx_register(64, size);
            other.channels_m1 = channels - 1;
            agx_swap(b, spill, other);
            a.reg = 64;
            z.memory = true;
            z.reg = spill_reg;
            agx_unit_test(b, a);
            agx_unit_test(b, z);
            agx_validate_ra(b->shader);
            ralloc_free(mem);
            std::_Exit(0);
         }, ::testing::ExitedWithCode(0), "");
      }
   }
}

TEST(NativeForms, RegisterValidationTracksPhiCopies)
{
   for (bool physical : {false, true}) {
      for (agx_size size : {AGX_SIZE_16, AGX_SIZE_32, AGX_SIZE_64}) {
         for (unsigned channels : {1u, 3u, 4u}) {
            EXPECT_EXIT({
               std::signal(SIGABRT, +[](int) { std::_Exit(77); });
               void *mem = ralloc_context(NULL);
               agx_builder *b = agx_test_builder(mem);
               agx_block *entry = agx_start_block(b->shader);
               agx_block *left = agx_test_block(b->shader);
               agx_block *right = agx_test_block(b->shader);
               agx_block *join = agx_test_block(b->shader);
               agx_block_add_successor(entry, left);
               agx_block_add_successor(entry, right);
               agx_block_add_successor(left, join);
               agx_block_add_successor(right, join);
               agx_index value = validation_span_value(b, false, 128, size, channels);
               b->cursor = agx_after_block(join);
               agx_index dest = value;
               dest.reg = 192;
               if (physical) {
                  dest = agx_register(192, size);
                  dest.channels_m1 = channels - 1;
               }
               agx_instr *phi = agx_phi_to(b, dest, 2);
               for (unsigned q = 0; q < 2; ++q) {
                  phi->src[q] = physical ? agx_register(128, size) : value;
                  phi->src[q].channels_m1 = channels - 1;
               }
               value.reg = 192;
               agx_unit_test(b, value);
               agx_validate_ra(b->shader);
               ralloc_free(mem);
               std::_Exit(0);
            }, ::testing::ExitedWithCode(0), "");
         }
      }
   }
}

TEST(NativeForms, RegisterValidationRetainsLoopHeaderRules)
{
   for (bool invalid_livein : {false, true}) {
      EXPECT_EXIT({
         std::signal(SIGABRT, +[](int) { std::_Exit(77); });
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_block *entry = agx_start_block(b->shader);
         agx_block *header = agx_test_block(b->shader);
         agx_block *body = agx_test_block(b->shader);
         agx_block *exit = agx_test_block(b->shader);
         agx_block_add_successor(entry, header);
         agx_block_add_successor(header, body);
         agx_block_add_successor(header, exit);
         agx_block_add_successor(body, header);
         header->loop_header = true;
         agx_index incoming = validation_span_value(b, false, 128, AGX_SIZE_32, 1);
         b->cursor = agx_after_block(body);
         agx_index back = validation_span_value(b, false, 144, AGX_SIZE_32, 1);
         b->cursor = agx_after_block(header);
         agx_index value = agx_temp(b->shader, AGX_SIZE_32);
         value.has_reg = true;
         value.reg = 160;
         agx_instr *phi = agx_phi_to(b, value, 2);
         phi->src[0] = incoming;
         phi->src[1] = back;
         agx_unit_test(b, invalid_livein ? incoming : value);
         agx_validate_ra(b->shader);
         ralloc_free(mem);
         std::_Exit(0);
      }, ::testing::ExitedWithCode(invalid_livein ? 77 : 0),
         invalid_livein ? "failed to validate RA" : "");
   }
}

TEST(NativeForms, RegisterValidationTracksNarrowPhysicalCopySources)
{
   for (unsigned mode = 0; mode < 3; ++mode) {
      for (unsigned channels : {3u, 4u}) {
         EXPECT_EXIT({
            std::signal(SIGABRT, +[](int) { std::_Exit(77); });
            void *mem = ralloc_context(NULL);
            agx_builder *b = agx_test_builder(mem);
            agx_block *entry = agx_start_block(b->shader);
            agx_block *next = agx_test_block(b->shader);
            agx_block_add_successor(entry, next);
            bool source_memory = mode == 1;
            agx_index value = validation_span_value(
               b, source_memory, source_memory ? AGX_NUM_MODELED_REGS - 2 * channels
                                              : 128,
               AGX_SIZE_32, channels);
            agx_index source = agx_as_register(value);
            source.channels_m1 = 0;
            agx_index dest = agx_register(16, AGX_SIZE_32);
            dest.channels_m1 = channels - 1;
            dest.memory = mode == 0;
            b->cursor = agx_after_block(next);
            if (mode == 2) {
               agx_instr *phi = agx_phi_to(b, dest, 1);
               phi->src[0] = source;
            } else {
               agx_mov_to(b, dest, source);
            }
            value.reg = dest.value;
            value.memory = dest.memory;
            agx_unit_test(b, value);
            agx_validate_ra(b->shader);
            ralloc_free(mem);
            std::_Exit(0);
         }, ::testing::ExitedWithCode(0), "");
      }
   }
}

TEST(NativeForms, RegisterValidationRejectsOutOfModelCopySpans)
{
   for (bool phi_copy : {false, true}) {
      EXPECT_EXIT({
         std::signal(SIGABRT, +[](int) { std::_Exit(77); });
         std::signal(SIGSEGV, +[](int) { std::_Exit(78); });
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_block *entry = agx_start_block(b->shader);
         agx_block *next = agx_test_block(b->shader);
         agx_block_add_successor(entry, next);
         b->cursor = agx_after_block(next);
         agx_index source = agx_register(0, AGX_SIZE_16);
         source.value = AGX_NUM_MODELED_REGS - 1;
         source.memory = !phi_copy;
         agx_index dest = agx_register(16, AGX_SIZE_32);
         if (phi_copy) {
            agx_instr *phi = agx_phi_to(b, dest, 1);
            phi->src[0] = source;
         } else {
            agx_mov_to(b, dest, source);
         }
         agx_validate_ra(b->shader);
         ralloc_free(mem);
         std::_Exit(0);
      }, ::testing::ExitedWithCode(77),
         "width <= AGX_NUM_MODELED_REGS - reg");
   }
}

#endif

TEST(NativeForms, SchedulerSsaBookkeepingStaysWithinEachBlock)
{
   for (bool ilp : {false, true}) {
      for (unsigned gap : {0u, 31u, 4095u}) {
         for (unsigned empty_blocks : {0u, 2u}) {
            SCOPED_TRACE(ilp);
            SCOPED_TRACE(gap);
            SCOPED_TRACE(empty_blocks);
            void *mem = ralloc_context(NULL);
            agx_shader_key key = {};
            key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
            key.dev.chip = AGX_CHIP_G17P;
            agx_builder *b = agx_test_builder(mem);
            b->shader->key = &key;
            b->shader->stage = MESA_SHADER_COMPUTE;
            b->shader->is_preamble = true;
            agx_block *last =
               list_first_entry(&b->shader->blocks, agx_block, link);
            agx_index a = agx_mov_imm(b, 32, 3);
            b->shader->alloc += gap;
            agx_index c = agx_mov_imm(b, 32, 5);
            agx_instr *observe_a = agx_unit_test(b, a);
            agx_instr *observe_c = agx_unit_test(b, c);
            auto next_block = [&]() {
               agx_block *next = agx_test_block(b->shader);
               last->successors[0] = next;
               util_dynarray_append(&next->predecessors, last);
               last = next;
               b->cursor = agx_after_block(last);
            };
            for (unsigned n = 0; n < empty_blocks; ++n)
               next_block();
            next_block();
            agx_index small = agx_mov_imm(b, 32, 7);
            next_block();
            agx_index sum = agx_iadd(b, a, c, 0);
            sum = agx_iadd(b, sum, agx_mov_imm(b, 32, 11), 0);
            sum = agx_iadd(b, sum, small, 0);
            agx_instr *observe_sum = agx_unit_test(b, sum);
            agx_index x = agx_mov_imm(b, 32, 13);
            agx_index y = agx_mov_imm(b, 32, 17);
            agx_instr *observe_other = agx_unit_test(b, agx_iadd(b, x, y, 0));

            std::vector<std::vector<agx_instr *>> original;
            agx_foreach_block(b->shader, block) {
               original.emplace_back();
               agx_foreach_instr_in_block(block, I)
                  original.back().push_back(I);
            }
            if (ilp)
               agx_ilp_schedule(b->shader);
            else
               agx_pressure_schedule(b->shader);

            std::vector<uint64_t> values(b->shader->alloc, UINT64_MAX);
            unsigned block_index = 0;
            unsigned observations = 0;
            agx_foreach_block(b->shader, block) {
               unsigned count = 0;
               agx_foreach_instr_in_block(block, I) {
                  ++count;
                  bool belonged = false;
                  for (agx_instr *before : original[block_index])
                     belonged |= before == I;
                  ASSERT_TRUE(belonged);
                  agx_foreach_ssa_src(I, src) {
                     ASSERT_LT(I->src[src].value, values.size());
                     ASSERT_NE(values[I->src[src].value], UINT64_MAX);
                  }
                  if (I->op == AGX_OPCODE_MOV_IMM) {
                     values[I->dest[0].value] = I->imm;
                  } else if (I->op == AGX_OPCODE_IADD) {
                     values[I->dest[0].value] =
                        uint32_t(values[I->src[0].value] +
                                 values[I->src[1].value]);
                  } else {
                     ASSERT_EQ(I->op, AGX_OPCODE_UNIT_TEST);
                     const unsigned expected = I == observe_a ? 3 :
                        I == observe_c ? 5 : I == observe_sum ? 26 : 30;
                     ASSERT_TRUE(I == observe_a || I == observe_c ||
                                 I == observe_sum || I == observe_other);
                     EXPECT_EQ(values[I->src[0].value], expected);
                     ++observations;
                  }
               }
               EXPECT_EQ(count, original[block_index].size());
               ++block_index;
            }
            EXPECT_EQ(block_index, original.size());
            EXPECT_EQ(observations, 4u);
            ralloc_free(mem);
         }
      }
   }
}

/* AGX2 preamble lowering replaces an unpromoted bindless sampler token with
 * its heap index. The main shader must still request the sampler heap. */
TEST(LegacySamplers, UnpromotedIndexNeedsHeap)
{
   for (bool dynamic : {false, true}) {
      for (bool border : {false, true}) {
         SCOPED_TRACE(::testing::Message() << "dynamic=" << dynamic
                                         << " border=" << border);
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "AGX2 sampler heap");
         b.shader->info.workgroup_size[0] = 1;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;
         nir_def *index = dynamic
            ? nir_u2u16(&b, nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0))
            : nir_imm_intN_t(&b, 28, 16);
         nir_def *sampler = border ? index : nir_bindless_sampler_agx(
            &b, nir_imm_int(&b, 24), index);
         nir_def *texture = nir_bindless_image_agx(&b, nir_imm_int(&b, 0));
         nir_tex_instr *tex = nir_tex_instr_create(b.shader, 4);
         tex->op = nir_texop_txl;
         tex->sampler_dim = GLSL_SAMPLER_DIM_2D;
         tex->coord_components = 2;
         tex->dest_type = nir_type_float32;
         tex->src[0] = nir_tex_src_for_ssa(nir_tex_src_backend1,
            nir_vec2(&b, nir_imm_float(&b, 0.25), nir_imm_float(&b, 0.5)));
         tex->src[1] = nir_tex_src_for_ssa(nir_tex_src_lod, nir_imm_float(&b, 0));
         tex->src[2] = nir_tex_src_for_ssa(nir_tex_src_texture_handle, texture);
         tex->src[3] = nir_tex_src_for_ssa(nir_tex_src_sampler_handle, sampler);
         nir_def_init(&tex->instr, &tex->def, 4, 32);
         nir_builder_instr_insert(&b, &tex->instr);
         nir_store_global(&b, &tex->def, nir_imm_int64(&b, 0x1000), .align_mul = 16);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX2;
         key.dev.chip = AGX_CHIP_G13G;
         key.promote_textures = false;
         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         bool ok = agx_compile_shader_nir(b.shader, &key, &part);
         EXPECT_TRUE(ok);
         if (ok) {
            EXPECT_TRUE(part.info.uses_sampler_heap);
         }
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(LegacyFragments, SideEffectsRetainUpstreamLateDepth)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "legacy side effects");
   b.shader->info.writes_memory = true;
   nir_store_global(&b, nir_imm_int(&b, 1), nir_imm_int64(&b, 0x1000),
                    .align_mul = 4);
   EXPECT_TRUE(agx_nir_lower_frag_sidefx(b.shader, false));
   EXPECT_NE(b.shader->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH), 0u);
   unsigned never = 0, masks = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic) continue;
         auto op = nir_instr_as_intrinsic(instr)->intrinsic;
         never += op == nir_intrinsic_load_depth_never_agx;
         masks += op == nir_intrinsic_sample_mask_agx;
      }
   }
   EXPECT_EQ(never, 1u);
   EXPECT_EQ(masks, 0u);
   ralloc_free(b.shader);
}

/* AGX3C bitwise operations with an 8-bit immediate operand, against native
 * G17G witnesses (the native G17G compiler; port/cwit k_or1, k_and255,
 * k_xor1, k_or1h). Each follows a load whose scoreboard wait is slot 7. */
TEST(NativeForms, M5BitwiseImmediateWitnesses)
{
   struct {
      enum agx3_alu_op op;
      bool narrow;
      uint16_t dest, src;
      uint8_t imm;
      std::vector<uint8_t> expect;
   } rows[] = {
      {AGX3_ALU_BOR, false, 0, 0, 1,
       {0x2b, 0x00, 0x07, 0x80, 0x32, 0x38, 0xa3, 0x02, 0x44, 0x80}},
      {AGX3_ALU_BAND, false, 0, 0, 255,
       {0x2b, 0x00, 0x07, 0x80, 0x30, 0x30, 0xa3, 0x02, 0xc4, 0xbf}},
      {AGX3_ALU_BXOR, false, 0, 0, 1,
       {0x2b, 0x00, 0x06, 0x80, 0x32, 0x38, 0xa3, 0x02, 0x44, 0x80}},
      {AGX3_ALU_BOR, true, 2, 2, 1,
       {0x3b, 0x02, 0x07, 0x00, 0x32, 0x28, 0xa3, 0x02, 0x44, 0x80}},
   };
   for (const auto &row : rows) {
      uint8_t out[AGX3_ALU_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_alu_bit_imm(AGX_ISA_ENCODING_AGX3C, row.op, row.narrow,
                                   row.dest, row.src, row.imm,
                                   AGX3_ALU_HINT_DISCARD, 0x80, out, &len))
         << row.op;
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), row.expect) << row.op;
   }
}


/* AGX3C runtime texture offsets: PACK_SMPIR into bank 6, against native G17G
 * witnesses (port/texwit3 d_auto/d_gath: X = h2, Y = h4; d_cmp: h4/h5). */
TEST(NativeForms, M5RuntimeTextureOffsetWitnesses)
{
   uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
   ASSERT_TRUE(agx3_tex_stage_offset_2d(AGX_ISA_ENCODING_AGX3C, 2, 4, 1, 0,
                                        false, true, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x37, 0x02, 0x00, 0x0a, 0x25, 0x00, 0xaa,
                                   0x0a, 0x00, 0x01, 0x60, 0x30, 0x00, 0x00}));
   ASSERT_TRUE(agx3_tex_stage_gather_offset(AGX_ISA_ENCODING_AGX3C, 4, false,
                                            0, 1, false, true, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x37, 0x00, 0x00, 0x2a, 0x25, 0x00, 0xaa,
                                   0x0a, 0x40, 0x01, 0x60, 0x30, 0x00, 0x00}));
}

/* Native G17G (texwit3) SMPs that read a runtime offset staged in bank 6:
 * d_auto (implicit LOD), d_lvlr (register LOD in the same word, f139 = 4) and
 * d_arr (array layer in the same word, f77 = 2). Each waits on the staging
 * slot. The word itself has G17P's layout: layer in bits 0..11, X/Y nibbles
 * at 12/28, LOD x64 in 16..27 (the PACK_SMPIR f74/f75 pairs of d_lvlr/d_arr). */
TEST(NativeForms, M5StagedOffsetSampleWitnesses)
{
   const struct agx3_tex_sample_desc d_auto = {
      .width = AGX3_TEX_W32, .mask = 15, .texture = 0, .sampler = 0,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true,
      .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .offset = true, .offset_staged = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   struct agx3_tex_sample_desc d_lvlr = d_auto;
   d_lvlr.lod = AGX3_TEX_LOD_REGISTER;
   struct agx3_tex_sample_desc d_arr = d_auto;
   d_arr.array = true;
   d_arr.offset_staged_array = true;
   const struct {
      const struct agx3_tex_sample_desc *d;
      std::vector<uint8_t> native;
   } cases[] = {
      {&d_auto, {0x0f, 0x04, 0x16, 0x00, 0x41, 0x80, 0x63, 0x29, 0x01, 0x10, 0x07,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00}},
      {&d_lvlr, {0x0f, 0x05, 0x16, 0x00, 0x45, 0x80, 0x63, 0x29, 0x01, 0x10, 0x07,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00}},
      {&d_arr,  {0x0f, 0x05, 0x16, 0x00, 0x41, 0x80, 0x62, 0x29, 0x01, 0x18, 0x07,
                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00}},
   };
   for (const auto &c : cases) {
      uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, c.d, out, &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native);
   }

   /* texwit6 g_arr / g_arr_y: an array gather reads the same layer word
    * (f77 = 2) with the gather selector f144 (and f145 for component y). */
   struct agx3_tex_sample_desc g_arr = d_arr;
   g_arr.gather = AGX3_TEX_GATHER_R;
   struct agx3_tex_sample_desc g_arr_y = d_arr;
   g_arr_y.gather = AGX3_TEX_GATHER_G;
   /* Comparisons (f143) with the reference staged beside the same word. */
   struct agx3_tex_sample_desc g_cmp_arr = g_arr;
   g_cmp_arr.shadow = true;
   struct agx3_tex_sample_desc c_arr = d_arr;
   c_arr.shadow = true;
   c_arr.mask = 1;
   struct agx3_tex_sample_desc c_lvlr = d_lvlr;
   c_lvlr.shadow = true;
   c_lvlr.mask = 1;
   const struct {
      const struct agx3_tex_sample_desc *d;
      std::vector<uint8_t> native;
   } gathers[] = {
      {&g_arr, {0x0f, 0x05, 0x16, 0x00, 0x41, 0x80, 0x02, 0x21, 0x01, 0x18, 0x16,
                0x00, 0x00, 0x00}},
      {&g_arr_y, {0x0f, 0x05, 0x16, 0x00, 0x41, 0x80, 0x22, 0x21, 0x01, 0x18, 0x16,
                  0x00, 0x00, 0x00}},
      {&g_cmp_arr, {0x0f, 0x06, 0x16, 0x00, 0x41, 0x80, 0x02, 0x21, 0x01, 0x18, 0x16,
                    0x00, 0x00, 0x00}},
      {&c_arr, {0x0f, 0x06, 0x16, 0x00, 0x41, 0x80, 0x02, 0x21, 0x01, 0x18, 0x06,
                0x00, 0x00, 0x00}},
      {&c_lvlr, {0x0f, 0x06, 0x16, 0x00, 0x45, 0x80, 0x03, 0x21, 0x01, 0x10, 0x06,
                 0x00, 0x00, 0x00}},
   };
   for (const auto &c : gathers) {
      uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, c.d, out, &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native);
   }
}

/* Native G17G (port/cwit3) ALU forms behind the spirv_assembly.type 8/16-bit
 * refusals: `ushort(1) << uint(n)` is one 32-bit-count sll with the
 * immediate one. */
TEST(NativeForms, M5ShiftOneWitness)
{
   uint8_t out[AGX3_ALU_MAX_LENGTH] = {0}, len = 0;
   ASSERT_TRUE(agx3_alu_sll_one_m5(0, false, 0, AGX3_ALU_HINT_DISCARD, false,
                                   0, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x27, 0x01, 0x20, 0x1a, 0x26, 0x00, 0xa0,
                                   0x02, 0x00, 0x80, 0x20, 0x34, 0x00, 0x00}));
   ASSERT_TRUE(agx3_alu_sll_one_m5(3, true, 4, AGX3_ALU_HINT_NONE, false, 0,
                                   out, &len));
   EXPECT_EQ(len, 14);
}

/* Native G17G (port/cwit3) addsat/subsat, popcount(ushort) and ~x. */
TEST(NativeForms, M5SpirvIntegerWitnesses)
{
   const auto eq = [](const uint8_t *out, uint8_t len,
                      std::vector<uint8_t> native) {
      return std::vector<uint8_t>(out, out + len) == native;
   };
   uint8_t out[AGX3_ALU_MAX_LENGTH] = {0}, len = 0;

   /* 32-bit: iadd32/isub32 register point with saturation; each native
    * instance waits on slot 8 (f53), as its loads publish there. */
   for (unsigned v = 0; v < 3; ++v) {
      const struct agx3_iadd_desc d = {
         .src_a_is_register = true, .src_a_reg = 0,
         .src_a_sign_extend = v == 1, .src_a_kill = true, .src_kill = true,
         .dest = 0, .src = 4, .sign_extend = v == 1,
         .wait_mask = 0x80, .subtract = v == 2, .saturate = true,
      };
      uint8_t b[AGX3_IADD_MAX_LENGTH] = {0}, n = 0;
      ASSERT_TRUE(agx3_iadd_encode(AGX_ISA_ENCODING_AGX3C, &d, b, &n)) << v;
      const std::vector<uint8_t> native[] = {
         {0x2f, 0x00, 0x04, 0x1a, 0x21, 0x00, 0xa3, 0x02, 0x38, 0x81, 0x20, 0x00},
         {0x2f, 0x00, 0x04, 0x1a, 0x21, 0x00, 0xa3, 0x02, 0x38, 0x81, 0xe0, 0x00},
         {0x2f, 0x00, 0x04, 0x1a, 0x25, 0x00, 0xa2, 0x02, 0x18, 0x81, 0x20, 0x00},
      };
      EXPECT_EQ(std::vector<uint8_t>(b, b + n), native[v]) << v;
   }

   /* 16-bit: half result and operands. */
   const uint16_t src[2] = {4, 2};
   const enum agx3_alu_hint discard[2] = {AGX3_ALU_HINT_DISCARD,
                                          AGX3_ALU_HINT_DISCARD};
   ASSERT_TRUE(agx3_alu_addsub_sat16_m5(2, src, discard, false, false, false,
                                        0x80, out, &len));
   EXPECT_TRUE(eq(out, len, {0x3f, 0x00, 0x04, 0x38, 0x21, 0x00, 0xa3, 0x02,
                             0xb0, 0x00, 0x20, 0x00}));
   ASSERT_TRUE(agx3_alu_addsub_sat16_m5(2, src, discard, true, false, false,
                                        0x80, out, &len));
   EXPECT_TRUE(eq(out, len, {0x3f, 0x00, 0x04, 0x38, 0x21, 0x00, 0xa3, 0x02,
                             0xb0, 0x00, 0xe0, 0x00}));
   ASSERT_TRUE(agx3_alu_addsub_sat16_m5(2, src, discard, true, true, false,
                                        0x80, out, &len));
   EXPECT_TRUE(eq(out, len, {0x3f, 0x00, 0x04, 0x38, 0x25, 0x08, 0xa2, 0x02,
                             0x90, 0x00, 0x60, 0x00}));

   /* popcount(ushort), waiting on slot 8. */
   ASSERT_TRUE(agx3_alu_popcount(AGX_ISA_ENCODING_AGX3C, 2, 2, true,
                                 AGX3_ALU_HINT_DISCARD, false, 0x80, out,
                                 &len));
   EXPECT_TRUE(eq(out, len, {0x3f, 0x02, 0x00, 0x00, 0x32, 0x20, 0xaa, 0x02,
                             0x04, 0x00}));

   /* ~ushort and ~uint, waiting on slot 8. */
   ASSERT_TRUE(agx3_alu_bit_not_m5(2, 2, false, AGX3_ALU_HINT_DISCARD, false,
                                   0x80, out, &len));
   EXPECT_TRUE(eq(out, len, {0x3b, 0x02, 0x26, 0x18, 0x21, 0x08, 0xa0, 0x02}));
   ASSERT_TRUE(agx3_alu_bit_not_m5(0, 0, true, AGX3_ALU_HINT_DISCARD, false,
                                   0x80, out, &len));
   EXPECT_TRUE(eq(out, len, {0x2b, 0x00, 0x26, 0x1a, 0x21, 0x08, 0xa8, 0x02}));
}

/* texwit6 minimum-LOD samples with a runtime offset in bank 6: the bias
 * (f139 = 3) or gradient (f139 = 5) and the minimum (f140 = 2) come from the
 * staged word / banks, as on G17P. */
TEST(NativeForms, M5MinimumLodOffsetSampleWitnesses)
{
   struct agx3_tex_sample_desc base = {
      .width = AGX3_TEX_W32, .mask = 15, .texture = 0, .sampler = 0,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .offset = true, .offset_staged = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   struct agx3_tex_sample_desc m_bias = base;
   m_bias.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   m_bias.min_lod = true;
   struct agx3_tex_sample_desc m_grad = base;
   m_grad.lod = AGX3_TEX_LOD_GRADIENT;
   m_grad.min_lod = true;
   const struct {
      const char *name;
      const struct agx3_tex_sample_desc *d;
      std::vector<uint8_t> native;
   } cases[] = {
      {"m_bias", &m_bias, {0x0f, 0x06, 0x16, 0x00, 0x45, 0x80, 0x60, 0x29, 0x01, 0x18, 0x07,
                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00}},
      {"m_grad", &m_grad, {0x0f, 0x06, 0x12, 0x00, 0x41, 0x80, 0x60, 0x29, 0x01, 0x18, 0x07,
                           0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00}},
   };
   /* texwit7: the same with a comparison (f143, mask 1, 14-byte form). */
   struct agx3_tex_sample_desc c_bias_min = m_bias;
   c_bias_min.shadow = true;
   c_bias_min.mask = 1;
   struct agx3_tex_sample_desc c_arr_grad_min = m_grad;
   c_arr_grad_min.shadow = true;
   c_arr_grad_min.mask = 1;
   c_arr_grad_min.array = true;
   c_arr_grad_min.offset_staged_array = true;
   const struct {
      const char *name;
      const struct agx3_tex_sample_desc *d;
      std::vector<uint8_t> native;
   } cmp[] = {
      {"c_bias_min", &c_bias_min, {0x0f, 0x07, 0x16, 0x00, 0x45, 0x80, 0x00, 0x21, 0x01, 0x18, 0x06,
                                   0x00, 0x00, 0x00}},
      {"c_arr_grad_min", &c_arr_grad_min, {0x0f, 0x04, 0x12, 0x00, 0x45, 0x80, 0x03, 0x21, 0x01, 0x18, 0x06,
                                           0x00, 0x00, 0x00}},
   };
   for (const auto &c : cmp) {
      uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
      const bool ok = agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, c.d, out, &len);
      EXPECT_TRUE(ok) << c.name;
      if (ok) {
         EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native) << c.name;
      }
   }
   for (const auto &c : cases) {
      uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
      const bool ok = agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, c.d, out, &len);
      EXPECT_TRUE(ok) << c.name;
      if (ok) {
         EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native) << c.name;
      }
   }
}

TEST(NativeForms, M5MinimumBank8Witness)
{
   uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
   ASSERT_TRUE(agx3_tex_stage_minimum_bank8_m5(6, 1, 0, false, true, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x27, 0x02, 0x00, 0x2a, 0x25, 0x00, 0xaa,
                                   0x12, 0x00, 0x00, 0x4a, 0x30, 0x18, 0x00}));

   /* The SMPs that read that minimum: texwit8 ca_br_min_d (2D-array
    * comparison, register bias, runtime offset) and m3_br_min_d (3D). */
   struct agx3_tex_sample_desc ca = {
      .width = AGX3_TEX_W32, .mask = 1, .lod = AGX3_TEX_LOD_BIAS_REGISTER,
      .min_lod = true, .slot = 1, .wait_mask = 1, .coord_kill = true,
      .final = true, .array = true,
      .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE, .shadow = true,
      .offset = true, .offset_staged = true, .offset_staged_array = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
   };
   ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &ca, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x0f, 0x04, 0x16, 0x00, 0x41, 0x80, 0x00,
                                   0x21, 0x01, 0x00, 0x00, 0x00, 0x00, 0x80}));
   struct agx3_tex_sample_desc cc = ca;
   cc.array = false;
   cc.cube = true;
   cc.offset = false;
   cc.offset_staged = false;
   cc.offset_staged_array = false;
   ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &cc, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x0f, 0x06, 0x16, 0x00, 0x41, 0x80, 0x03,
                                   0x21, 0x01, 0x00, 0x02, 0x00, 0x00, 0x00}));

   struct agx3_tex_sample_desc m3 = ca;
   m3.shadow = false;
   m3.mask = 15;
   m3.array = false;
   m3.offset_staged_array = false;
   m3.three_d = true;
   ASSERT_TRUE(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &m3, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x0f, 0x04, 0x16, 0x00, 0x41, 0x80, 0x62,
                                   0x29, 0x01, 0x18, 0x07, 0x00, 0x00, 0x00,
                                   0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00,
                                   0x00}));
}


/* Native G17G at-sample interpolation (lane-mesh/cwit-interp ias_k1,
 * ias_k1_np, sp@4): ITRCOORDS f38=2 on the sample index, then f32interp
 * token leaves (f39=2, f32 = token, f41 live / f43 final) with the projective
 * RHW leaf (f54) first. */
TEST(NativeForms, M5SampleInterpolationWitnesses)
{
   const auto bytes = [](const uint8_t *out, uint8_t len) {
      return std::vector<uint8_t>(out, out + len);
   };
   uint8_t out[AGX3_VARY_MAX_LENGTH] = {0}, len = 0;

   const struct {
      uint16_t dest, src;
      std::vector<uint8_t> native;
   } coords[] = {
      {6, 0, {0x37, 0x00, 0x24, 0x1a, 0x2a, 0x00, 0xab, 0x0a, 0x00, 0x02}},
      {10, 4, {0x37, 0x00, 0x24, 0x3a, 0x2a, 0x00, 0xab, 0x12, 0x00, 0x02}},
      {14, 1, {0x37, 0x01, 0x24, 0x1a, 0x2a, 0x00, 0xab, 0x1a, 0x00, 0x02}},
   };
   for (const auto &c : coords) {
      ASSERT_TRUE(agx3_vary_sample_coords(AGX_ISA_ENCODING_AGX3C, c.dest,
                                          c.src, true, 0, out, &len));
      EXPECT_EQ(bytes(out, len), c.native) << c.dest;
   }

   ASSERT_TRUE(agx3_vary_projective_rhw_sample(AGX_ISA_ENCODING_AGX3C, 0, 0,
                                               6, false, out, &len));
   EXPECT_EQ(bytes(out, len),
             (std::vector<uint8_t>{0x27, 0x00, 0x04, 0x0a, 0x2a, 0x20, 0xaa,
                                   0x02, 0x06, 0x88}));
   ASSERT_TRUE(agx3_vary_projective_rhw_sample(AGX_ISA_ENCODING_AGX3C, 0, 0,
                                               10, false, out, &len));
   EXPECT_EQ(bytes(out, len),
             (std::vector<uint8_t>{0x27, 0x00, 0x04, 0x0a, 0x2a, 0x20, 0xaa,
                                   0x02, 0x0a, 0x88}));

   const struct {
      uint16_t dest, coef;
      bool perspective;
      uint16_t coords;
      bool kill;
      std::vector<uint8_t> native;
   } leaves[] = {
      {0, 1, true, 6, false,
       {0x27, 0x01, 0x04, 0x0a, 0x2a, 0x20, 0xaa, 0x02, 0x06, 0x80}},
      {8, 2, true, 6, false,
       {0x27, 0x02, 0x04, 0x0a, 0x2a, 0x20, 0xaa, 0x12, 0x06, 0x80}},
      {2, 3, true, 6, false,
       {0x37, 0x03, 0x04, 0x0a, 0x2a, 0x20, 0xaa, 0x02, 0x06, 0x80}},
      {6, 4, true, 6, true,
       {0x37, 0x00, 0x04, 0x1a, 0x2a, 0x20, 0xaa, 0x0a, 0x06, 0x04}},
      {0, 0, false, 4, false,
       {0x27, 0x00, 0x04, 0x0a, 0x2a, 0x20, 0xaa, 0x02, 0x04, 0x80}},
      {4, 3, false, 4, true,
       {0x27, 0x03, 0x04, 0x0a, 0x2a, 0x20, 0xaa, 0x0a, 0x04, 0x04}},
   };
   for (const auto &l : leaves) {
      ASSERT_TRUE(agx3_vary_interp_sample(AGX_ISA_ENCODING_AGX3C, l.dest,
                                          l.coef, l.perspective, l.coords,
                                          l.kill, out, &len));
      EXPECT_EQ(bytes(out, len), l.native) << l.dest << " " << l.coef;
   }
}

/* Native G17G quad ballots of a boolean immediate (lane-mesh/cwit-quad
 * q_term/q_vote, cwit-quad2 q_any_mem): quad_any(false) and the quad
 * active-lane ballot(1 != 0). */
TEST(NativeForms, M5QuadImmediateBallotWitnesses)
{
   const struct {
      uint16_t dest, value;
      std::vector<uint8_t> native;
   } rows[] = {
      {14, 0, {0x37, 0x00, 0x20, 0x18, 0x2d, 0x00, 0xa6, 0x1a, 0x30, 0x00, 0x08, 0x40}},
      {4, 1, {0x27, 0x01, 0x20, 0x18, 0x2d, 0x00, 0xa6, 0x0a, 0x30, 0x00, 0x08, 0x40}},
      {2, 1, {0x37, 0x01, 0x20, 0x18, 0x2d, 0x00, 0xa6, 0x02, 0x30, 0x00, 0x08, 0x40}},
   };
   for (const auto &r : rows) {
      uint8_t out[AGX3_SUBGROUP_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_subgroup_encode_quad_ballot_imm(r.dest, true, r.value, 0,
                                                       out, &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), r.native) << r.dest;
   }
}

/* Native G17G centroid inputs (lane-mesh/cwit-interp iac, cwit-interp2
 * cp/cnp at 1 and 4 samples): the fragment opens with the marked SR39 read,
 * whose slot 1 ITRCOORDS f38=1 waits on; the token then feeds the same
 * f32interp leaves as at-sample interpolation. */
TEST(NativeForms, M5CentroidInterpolationWitnesses)
{
   struct agx3_sr_desc sr = {};
   sr.width = AGX3_SR_W16;
   sr.sr = 39;
   sr.slot = 1;
   sr.first = true;
   uint8_t srb[AGX3_SR_MAX_LENGTH] = {0}, srlen = 0;
   ASSERT_TRUE(agx3_sr_encode(AGX_ISA_ENCODING_AGX3C, &sr, srb, &srlen));
   EXPECT_EQ(std::vector<uint8_t>(srb, srb + srlen),
             (std::vector<uint8_t>{0x24, 0x82, 0x01, 0x00, 0x20, 0x00, 0xb0,
                                   0x02}));

   const struct {
      uint16_t dest;
      std::vector<uint8_t> native;
   } coords[] = {
      {6, {0x37, 0x04, 0x24, 0x1a, 0x2a, 0x00, 0xab, 0x0a, 0x00, 0x00}},
      {4, {0x27, 0x04, 0x24, 0x1a, 0x2a, 0x00, 0xab, 0x0a, 0x00, 0x00}},
   };
   for (const auto &c : coords) {
      uint8_t out[AGX3_VARY_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_vary_centroid_coords(AGX_ISA_ENCODING_AGX3C, c.dest, 0,
                                            true, 0x1, out, &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native) << c.dest;
   }
}

/* AGX3C texture barriers wait in their own first word on the result slots of
 * the image writes in flight. Native G17G bytes from port/fencewit:
 * k_fence (slot2), k_fence2 (slots 2+3), k_simdbar_tex (slot1), k_tgbar_tex,
 * k_atf_tg. */
TEST(NativeForms, M5TextureFenceWitnesses)
{
   const struct {
      agx3_barrier_scope scope;
      uint16_t wait;
      std::vector<uint8_t> native;
   } rows[] = {
      {AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT, 0x2,
       {0x87, 0x26, 0x00, 0x06, 0x00, 0x08, 0x87, 0x1e, 0x00, 0x06, 0x00, 0x00}},
      {AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT, 0x6,
       {0x87, 0x26, 0x00, 0x06, 0x00, 0x18, 0x87, 0x1e, 0x00, 0x06, 0x00, 0x00}},
      {AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT, 0x1,
       {0x87, 0x26, 0x00, 0x06, 0x00, 0x04, 0x87, 0x1e, 0x00, 0x06, 0x00, 0x00}},
      {AGX3_BARRIER_SCOPE_TEXTURE, 0x2,
       {0x87, 0x67, 0x00, 0x06, 0x00, 0x08, 0x87, 0x5f, 0x00, 0x06, 0x00, 0x00}},
      {AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP, 0x2,
       {0x87, 0x27, 0x00, 0x06, 0x00, 0x08, 0x87, 0x1f, 0x00, 0x06, 0x00, 0x00}},
      {AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT, 0,
       {0x87, 0x26, 0x00, 0x06, 0x00, 0x00, 0x87, 0x1e, 0x00, 0x06, 0x00, 0x00}},
   };
   for (const auto &r : rows) {
      agx3_encoded_scoped_barrier enc = {};
      ASSERT_TRUE(agx3_pack_scoped_barrier_wait(AGX_ISA_ENCODING_AGX3C,
                                                r.scope, r.wait, &enc))
         << r.scope << " " << r.wait;
      EXPECT_EQ(std::vector<uint8_t>(enc.bytes, enc.bytes + enc.length),
                r.native) << r.scope << " " << r.wait;
   }

   agx3_encoded_scoped_barrier enc = {};
   EXPECT_FALSE(agx3_pack_scoped_barrier(
      AGX_ISA_ENCODING_AGX3B, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_THREADGROUP,
      &enc));
   EXPECT_FALSE(agx3_pack_scoped_barrier(
      AGX_ISA_ENCODING_AGX3C, AGX3_BARRIER_SCOPE_TEXTURE_FENCE, &enc));
   EXPECT_FALSE(agx3_pack_scoped_barrier_wait(
      AGX_ISA_ENCODING_AGX3B, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT, 0x2,
      &enc));
   EXPECT_FALSE(agx3_pack_scoped_barrier_wait(
      AGX_ISA_ENCODING_AGX3C, AGX3_BARRIER_SCOPE_DEVICE, 0x2, &enc));
}

/* Each AGX3C image write's fence waits on that write's own result slot in
 * the barrier word, with no separate retirement, as k_fence_twice does. */
TEST(NativeForms, M5ImageFenceWaitsWriteSlotInBarrier)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_index data = agx_register(16, AGX_SIZE_32);
   data.channels_m1 = 3;
   agx_index coord = agx_register(24, AGX_SIZE_32);
   coord.channels_m1 = 1;
   for (unsigned i = 0; i < 3; ++i) {
      agx_image_write(b, data, coord, agx_zero(), agx_uniform(8, AGX_SIZE_64),
                      agx_register(28, AGX_SIZE_32), AGX_DIM_2D, true);
      agx_scoped_barrier_agx3(b, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT);
   }
   agx_stop(b);
   util_dynarray binary;
   util_dynarray_init(&binary, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));

   const uint8_t *bytes = (const uint8_t *)binary.data;
   unsigned fences = 0;
   for (unsigned i = 0; i + 12 <= binary.size; ++i) {
      if (bytes[i] != 0x87 || bytes[i + 1] != 0x26 || bytes[i + 6] != 0x87 ||
          bytes[i + 7] != 0x1e)
         continue;
      /* One wait bit, on a slot an imgwci before it signals. */
      const uint16_t wait = (bytes[i + 5] >> 2) | (((bytes[i + 3] >> 5) & 3) << 6);
      EXPECT_EQ(util_bitcount(wait), 1u) << fences;
      agx3_encoded_scoped_barrier enc = {};
      ASSERT_TRUE(agx3_pack_scoped_barrier_wait(
         AGX_ISA_ENCODING_AGX3C, AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT, wait,
         &enc));
      EXPECT_EQ(memcmp(bytes + i, enc.bytes, 12), 0) << fences;
      ++fences;
      i += 11;
   }
   EXPECT_EQ(fences, 3u);
   util_dynarray_fini(&binary);
   ralloc_free(mem);
}

/* G17G 2D-MS texel fetches staged through PACK_SMPIR: the vertex form
 * (port/vmswit v_ms_f) is the fragment form (f_ms_f) with sampler index 0. */
TEST(NativeForms, M5VertexMultisampleFetchWitness)
{
   const struct {
      uint16_t sampler;
      std::vector<uint8_t> native;
   } rows[] = {
      {0, {0x0f, 0x04, 0x1a, 0x00, 0x45, 0x80, 0x61, 0x29, 0x01, 0x08, 0x02,
           0x00, 0x00, 0x00}},
      {1, {0x0f, 0x04, 0x1a, 0x00, 0x45, 0x81, 0x61, 0x29, 0x01, 0x08, 0x02,
           0x00, 0x00, 0x00}},
   };
   for (const auto &r : rows) {
      agx3_tex_fetch_desc d = {};
      d.width = AGX3_TEX_W32;
      d.mask = 0xf;
      d.slot = 1;
      d.wait_mask = 1;
      d.coord_kill = true;
      d.final = true;
      d.multisample = true;
      d.staged_sample = true;
      d.sampler_value = r.sampler;
      uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, out, &len))
         << r.sampler;
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), r.native) << r.sampler;
   }
}

/* AGX3C TEX destinations are eight-bit half indices: G17G places a read at
 * half 174 under register pressure (port/rawit k_read_hi). */
TEST(NativeForms, M5HighTextureDestinationWitness)
{
   agx3_tex_fetch_desc d = {};
   d.width = AGX3_TEX_W32;
   d.dest = 174;
   d.mask = 0xf;
   d.slot = 2;
   d.wait_mask = 1;
   d.coord_kill = true;
   d.final = true;
   uint8_t out[AGX3_TEX_MAX_LENGTH] = {0}, len = 0;
   ASSERT_TRUE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, out, &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x7f, 0x06, 0x5a, 0x00, 0xc8, 0x80, 0x60,
                                   0x39}));
   d.dest = 128;
   EXPECT_FALSE(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
}

/* G17G saturate(half): the f16add -0.0h form with f14, plain and with the
 * negate/abs modifiers (port/hsatwit k_hsat, k_hsat_neg, k_hsat_abs,
 * k_hsat_live). */
TEST(NativeForms, M5HalfSaturateWitness)
{
   const struct {
      uint16_t dest, src;
      bool abs, neg;
      enum agx3_alu_hint hint;
      std::vector<uint8_t> native;
   } cases[] = {
      {2, 2, false, false, AGX3_ALU_HINT_DISCARD,
       {0x38, 0x02, 0x04, 0x02, 0x22, 0x20, 0xa0, 0x02, 0x84, 0x16}},
      {2, 2, false, true, AGX3_ALU_HINT_DISCARD,
       {0x38, 0x02, 0x04, 0x02, 0x22, 0x20, 0xa0, 0x02, 0x84, 0x96}},
      {2, 2, true, false, AGX3_ALU_HINT_DISCARD,
       {0x38, 0x02, 0x04, 0x02, 0x22, 0x20, 0xa0, 0x02, 0x84, 0x36}},
      {5, 4, false, false, AGX3_ALU_HINT_NONE,
       {0x28, 0x00, 0x04, 0x0b, 0x22, 0x20, 0xa0, 0x0a, 0x80, 0x16}},
   };
   for (const auto &c : cases) {
      uint8_t out[AGX3_ALU_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_alu_half_add_negzero_m5(c.dest, c.src, c.abs, c.neg,
                                               true, c.hint, false, 0x80, out,
                                               &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native);
   }
}

/* G17G float(h) * minifloat into a float (port/hmulwit). */
TEST(NativeForms, M5HalfTimesMinifloatWitness)
{
   const struct {
      uint16_t dest, src, imm;
      bool abs, sat;
      enum agx3_alu_hint hint;
      uint16_t wait;
      std::vector<uint8_t> native;
   } cases[] = {
      {0, 0, 32, false, false, AGX3_ALU_HINT_DISCARD, 0x80,
       {0x29, 0x00, 0x05, 0x10, 0x22, 0x80, 0xac, 0x02, 0x02, 0x80, 0x20,
        0x00, 0x02, 0x10}},
      {0, 0, 32, false, true, AGX3_ALU_HINT_DISCARD, 0x80,
       {0x29, 0x00, 0x05, 0x10, 0x22, 0x80, 0xac, 0x02, 0x02, 0x80, 0x20,
        0x02, 0x02, 0x10}},
      {0, 0, 32, true, false, AGX3_ALU_HINT_DISCARD, 0x80,
       {0x29, 0x00, 0x05, 0x10, 0x22, 0x80, 0xac, 0x02, 0x02, 0x80, 0x20,
        0x08, 0x02, 0x10}},
      {0, 4, 32, false, false, AGX3_ALU_HINT_NONE, 0x80,
       {0x29, 0x00, 0x05, 0x10, 0x22, 0x80, 0xac, 0x02, 0x02, 0x81, 0x00,
        0x00, 0x02, 0x10}},
      {0, 6, 32, false, false, AGX3_ALU_HINT_CACHE, 0x80,
       {0x29, 0x00, 0x05, 0x10, 0x22, 0x80, 0xac, 0x02, 0x82, 0x81, 0x10,
        0x00, 0x02, 0x10}},
      {2, 6, 64, false, false, AGX3_ALU_HINT_CACHE, 0,
       {0x19, 0xc1, 0x05, 0x86, 0x80, 0x00}},
      {6, 6, 80, false, false, AGX3_ALU_HINT_DISCARD, 0,
       {0x39, 0xd1, 0x15, 0x06, 0x80, 0x00}},
   };
   for (const auto &c : cases) {
      uint8_t out[AGX3_ALU_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_alu_fmul_half_minifloat_m5(c.dest, c.src, c.imm, false,
                                                  c.abs, c.sat, c.hint,
                                                  c.wait, out, &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native);
   }
}

/* G17G half(fma(a, k, b)): the immediate-multiplier FMA with f10=1
 * (port/hfmawit). */
TEST(NativeForms, M5NarrowingFmaImmediateWitness)
{
   const struct {
      uint16_t dest, src1, add;
      uint8_t k;
      bool src1_16, sat, src_cache, src_discard, add_cache, add_discard;
      std::vector<uint8_t> native;
   } cases[] = {
      {2, 2, 4, 32, false, false, false, true, false, true,
       {0x39, 0x02, 0x07, 0x08, 0x20, 0x30, 0xa3, 0x02, 0x30, 0x88, 0x98,
        0x00, 0x04, 0x08, 0x00, 0x80}},
      {2, 2, 4, 160, false, false, false, true, false, true,
       {0x39, 0x02, 0x07, 0x08, 0x20, 0x30, 0xa3, 0x02, 0x30, 0xa8, 0x98,
        0x00, 0x04, 0x08, 0x00, 0x80}},
      {2, 2, 4, 32, false, true, false, true, false, true,
       {0x39, 0x02, 0x07, 0x08, 0x20, 0x30, 0xa3, 0x02, 0x30, 0x88, 0x98,
        0x02, 0x04, 0x08, 0x00, 0x80}},
      {10, 4, 6, 32, false, false, true, false, true, false,
       {0x39, 0x80, 0x07, 0x18, 0x20, 0x30, 0xa3, 0x12, 0x10, 0x88, 0x98,
        0x00, 0x06, 0x04, 0x00, 0x80}},
      {2, 4, 2, 32, true, false, false, true, false, true,
       {0x39, 0x00, 0x07, 0x18, 0x20, 0x30, 0xa7, 0x02, 0x20, 0x88, 0x98,
        0x00, 0x02, 0x08, 0x00, 0x80}},
   };
   for (const auto &c : cases) {
      const struct agx3_falu_mods m = {.src1_16 = c.src1_16,
                                       .saturate = c.sat};
      const struct agx3_falu_runtime rt = {
         .cache = {false, c.src_cache, c.add_cache},
         .discard = {false, c.src_discard, c.add_discard},
         .dest_cache = false,
         .wait_mask = 0x80,
      };
      uint8_t out[AGX3_FALU_MOD_MAX_LENGTH] = {0}, len = 0;
      ASSERT_TRUE(agx3_falu_fma_mul_imm_narrow_m5(c.dest, c.k, c.src1, c.add,
                                                  &m, &rt, out, &len));
      EXPECT_EQ(std::vector<uint8_t>(out, out + len), c.native);
   }
}

/* G17G B5G6R5Unorm framebuffer fetch: three 16-bit lanes (port/fbwit). */
TEST(NativeForms, M5Rgb565TileLoadWitness)
{
   uint8_t out[AGX3_TILE_MAX_LENGTH] = {0}, len = 0;
   ASSERT_TRUE(agx3_tile_load(AGX_ISA_ENCODING_AGX3C, 0, AGX3_TILE_W16, 8, 7,
                              0, AGX3_TILE_SAMPLE_1X, 0, true, true, out,
                              &len));
   EXPECT_EQ(std::vector<uint8_t>(out, out + len),
             (std::vector<uint8_t>{0x27, 0x01, 0x05, 0x00, 0x01, 0x87, 0xbc,
                                   0x02, 0x00, 0x00, 0x40, 0xd8, 0x20, 0x00}));
}
TEST(NativeForms, M5MeshUVBNativeOperands)
{
   uint8_t bytes[8], length = 0;
   const uint8_t allocation[] = {0x35,0x01,0x0f,0x00,0x20,0x00,0xa2,0x0a};
   const uint8_t primitive[] = {0x25,0x00,0x0b,0x18,0x20,0x00,0xa2,0x0a};
   const uint8_t position[] = {0x35,0x02,0x0b,0x0a,0x20,0x00,0xa2,0x0a};
   ASSERT_TRUE(agx3_mesh_allocate_uvb(AGX_ISA_ENCODING_AGX3C,1,10,true,bytes,&length));
   EXPECT_EQ(length,8u); EXPECT_EQ(memcmp(bytes,allocation,8),0);
   ASSERT_TRUE(agx3_mesh_uvst(AGX_ISA_ENCODING_AGX3C,12,true,true,bytes,&length));
   EXPECT_EQ(memcmp(bytes,primitive,8),0);
   ASSERT_TRUE(agx3_mesh_uvst(AGX_ISA_ENCODING_AGX3C,22,false,true,bytes,&length));
   EXPECT_EQ(memcmp(bytes,position,8),0);
   EXPECT_TRUE(agx3_mesh_allocate_uvb(AGX_ISA_ENCODING_AGX3B,1,10,true,bytes,&length));
   EXPECT_FALSE(agx3_mesh_uvst(AGX_ISA_ENCODING_AGX3C,1,false,true,bytes,&length));
   EXPECT_FALSE(agx3_mesh_uvst(AGX_ISA_ENCODING_AGX3C,254,false,true,bytes,&length));
}

TEST(NativeForms, M5MeshUVBReachesNativePacker)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                          &agx_nir_options, "M5 native UVB descriptor");
   b.shader->info.workgroup_size[0] = 3;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *index = nir_load_local_invocation_index(&b);
   /* Dynamic values deliberately exercise SSA -> scheduling -> register
    * allocation. The backend must see both words of the hidden descriptor. */
   nir_allocate_mesh_uvb_agx(&b,nir_u2u16(&b,index),
                              nir_u2u16(&b,nir_iadd_imm(&b,index,3)));
   nir_def *descriptor = nir_vec2(&b,index,nir_iadd_imm(&b,index,32));
   nir_store_mesh_uvb_agx(&b,descriptor,.flags=0);
   nir_store_mesh_uvb_agx(&b,descriptor,.flags=1);
   nir_validate_shader(b.shader,"M5 native UVB");
   agx_preprocess_nir(b.shader);
   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S;
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.no_preamble = true;
   agx_shader_part part = {};
   bool compiled = agx_compile_shader_nir(b.shader,&key,&part);
   EXPECT_TRUE(compiled) << agx_last_compile_refusal();
   if (compiled) { EXPECT_GT(part.info.binary_size,24u); free(part.binary); }
   ralloc_free(b.shader);
}


TEST(NativeForms, M5ObjectChildcountNativeLeaf)
{
   uint8_t bytes[8], size;
   const uint8_t native[] = {0x35,0x04,0x33,0x00,0x24,0x00,0xa8,0x02};
   ASSERT_TRUE(agx3_mesh_object_childcount(AGX_ISA_ENCODING_AGX3C,bytes,&size));
   ASSERT_EQ(size,sizeof(native));
   EXPECT_EQ(memcmp(bytes,native,size),0);
   EXPECT_TRUE(agx3_mesh_object_childcount(AGX_ISA_ENCODING_AGX3B,bytes,&size));
}

TEST(NativeForms, M5ObjectChildcountSSACompiles)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_TASK,
                     &agx_nir_options,"M5 native object childcount SSA");
   b.shader->info.workgroup_size[0]=3;
   b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.io_lowered=true;
   nir_def *pointer=nir_pack_64_2x32(&b,nir_load_preamble(&b,2,32,.base=0));
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_def *value=nir_load_global_constant(&b,1,32,
      nir_iadd(&b,pointer,nir_u2u64(&b,nir_imul_imm(&b,lane,4))),.align_mul=4);
   nir_store_object_childcount_agx(&b,value);
   nir_validate_shader(b.shader,"object childcount");
   agx_preprocess_nir_m5(b.shader);
   agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;
   key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
   agx_shader_part part={};
   bool compiled=agx_compile_shader_nir(b.shader,&key,&part);
   EXPECT_TRUE(compiled)<<agx_last_compile_refusal();
   if(compiled){EXPECT_EQ(part.info.scratch_size,0u);free(part.binary);}
   ralloc_free(b.shader);
}

/* The record task's child grid takes 16 bytes past the body's threadgroup
 * memory; a body using all 32 KiB aliases word 0 after a rendezvous. */
TEST(NativeForms, M5TaskGridFitsFullThreadgroupMemory)
{
   for (unsigned shared : {64u, 32768u}) {
      for (bool payload : {false, true}) {
         nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_TASK,&agx_nir_options,"task grid");
         b.shader->info.workgroup_size[0]=32;
         b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
         b.shader->info.shared_size=shared;b.shader->info.task_payload_size=payload ? 4 : 0;
         if (payload)
            nir_store_task_payload(&b,nir_imm_int(&b,7),nir_imm_int(&b,0),.align_mul=4);
         nir_launch_mesh_workgroups(&b,nir_imm_ivec3(&b,2,1,1));
         ASSERT_TRUE(agx_nir_lower_m5_task(b.shader,payload ? 16384 : 0,payload));
         nir_validate_shader(b.shader,"task grid");
         EXPECT_LE(b.shader->info.shared_size,32768u);
         if (payload) {
            EXPECT_EQ(b.shader->info.shared_size,shared==64 ? 128u : 32768u);
         }
         ralloc_free(b.shader);
      }
   }
}

/* The constant byte offset past the payload address of a lowered payload
 * access, or UINT64_MAX if the address is not payload address + constant. */
static uint64_t
m5_payload_access_offset(nir_def *addr)
{
   nir_scalar a = nir_scalar_resolved(addr, 0);
   if (!nir_scalar_is_alu(a) || nir_scalar_alu_op(a) != nir_op_iadd)
      return UINT64_MAX;
   for (unsigned k = 0; k < 2; ++k) {
      nir_scalar c = nir_scalar_chase_alu_src(a, k);
      if (nir_scalar_is_const(c))
         return nir_scalar_as_uint(c);
   }
   return UINT64_MAX;
}

/* Without the grid record (a mesh that never reads gl_NumWorkGroups) the
 * task stores no child grid at the payload address and both stages address
 * the payload from +0; with it, the grid takes +0 and the payload starts at
 * +16. A payload word at byte 1020 is the last of a full 1 KiB payload. */
TEST(NativeForms, M5TaskPayloadAtZeroWithoutGridRecord)
{
   for (bool grid_record : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TASK,
         &agx_nir_options, "task payload layout");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.task_payload_size = 1024;
      nir_store_task_payload(&b, nir_imm_int(&b, 7), nir_imm_int(&b, 1020),
                             .align_mul = 4);
      nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 256, 1, 1));
      ASSERT_TRUE(agx_nir_lower_m5_task(b.shader, 16384, grid_record));
      nir_validate_shader(b.shader, "task payload layout");
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      unsigned grid_stores = 0, payload_stores = 0;
      uint64_t payload_offset = UINT64_MAX;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            EXPECT_NE(i->intrinsic, nir_intrinsic_store_task_payload);
            if (i->intrinsic != nir_intrinsic_store_global)
               continue;
            if (i->src[0].ssa->num_components == 3) {
               grid_stores++;
            } else {
               payload_stores++;
               payload_offset = m5_payload_access_offset(i->src[1].ssa);
            }
         }
      }
      EXPECT_EQ(grid_stores, grid_record ? 1u : 0u);
      EXPECT_EQ(payload_stores, 1u);
      EXPECT_EQ(payload_offset, grid_record ? 1036u : 1020u);
      ralloc_free(b.shader);
   }

   /* The mesh side of the same two layouts. */
   for (unsigned base : {0u, 16u}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
         &agx_nir_options, "mesh payload layout");
      b.shader->info.task_payload_size = 1024;
      nir_def *value = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 1020),
                                             .align_mul = 4);
      nir_store_global(&b, value, nir_imm_int64(&b, 0x2000000),
                       .write_mask = 1, .align_mul = 4);
      ASSERT_TRUE(agx_nir_lower_m5_task_payload(b.shader, 16384, base));
      nir_validate_shader(b.shader, "mesh payload layout");
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      uint64_t load_offset = UINT64_MAX;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            EXPECT_NE(i->intrinsic, nir_intrinsic_load_task_payload);
            if (i->intrinsic == nir_intrinsic_load_global)
               load_offset = m5_payload_access_offset(i->src[0].ssa);
         }
      }
      EXPECT_EQ(load_offset, 1020u + base);
      ralloc_free(b.shader);
   }
}

/* A payload-free task on a kernel that backs the payload keeps the
 * rendezvous form without the grid record: only a kernel without payload
 * storage takes the direct form. */
TEST(NativeForms, M5PayloadFreeTaskWithoutGridRecordKeepsRendezvous)
{
   for (unsigned max_payload_B : {0u, 16384u}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_TASK,
         &agx_nir_options, "payload-free task");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      nir_launch_mesh_workgroups(&b, nir_imm_ivec3(&b, 2, 1, 1));
      ASSERT_TRUE(agx_nir_lower_m5_task(b.shader, max_payload_B, false));
      nir_validate_shader(b.shader, "payload-free task");
      bool exit_barrier = false, global_stores = false;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            exit_barrier |=
               i->intrinsic == nir_intrinsic_object_payload_exit_barrier_agx;
            global_stores |= i->intrinsic == nir_intrinsic_store_global;
         }
      }
      EXPECT_EQ(exit_barrier, max_payload_B != 0);
      EXPECT_FALSE(global_stores);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5MeshUserSharedBarrierCompile)
{
   for (unsigned shared : {0u,4u,64u,128u}) {
      nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,
                            &agx_nir_options,"native mesh shared output readback");
      b.shader->info.workgroup_size[0]=3;
      b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
      b.shader->info.mesh.max_vertices_out=3;
      b.shader->info.mesh.max_primitives_out=1;
      b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;
      b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES;
      b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;
      b.shader->info.io_lowered=true;b.shader->info.shared_size=shared;
      nir_def *lane=nir_load_local_invocation_index(&b);
      nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,3),nir_imm_int(&b,1),
                                         nir_imm_int(&b,0),.stream_id=0);
      nir_def *pos=nir_vec4(&b,nir_u2f32(&b,lane),nir_imm_float(&b,1),
                            nir_imm_float(&b,.25),nir_imm_float(&b,1));
      nir_store_per_vertex_output(&b,pos,lane,nir_imm_int(&b,0),
         .write_mask=15,.src_type=nir_type_float32,
         .io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
      nir_push_if(&b,nir_ieq_imm(&b,lane,0));
      nir_store_per_primitive_output(&b,nir_imm_ivec3(&b,0,1,2),
         nir_imm_int(&b,0),nir_imm_int(&b,0),.write_mask=7,.src_type=nir_type_uint32,
         .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
      if(shared)nir_store_shared(&b,nir_imm_int(&b,0x531a),nir_imm_int(&b,0),
                                .write_mask=1,.align_mul=4);
      nir_pop_if(&b,NULL);
      nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
         .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
      nir_def *token=shared?nir_load_shared(&b,1,32,nir_imm_int(&b,0),.align_mul=4):nir_imm_int(&b,0);
      nir_store_global(&b,token,nir_iadd(&b,nir_imm_int64(&b,0x2000000),
            nir_u2u64(&b,nir_imul_imm(&b,lane,4))),.write_mask=1,.align_mul=4);
      ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
      EXPECT_EQ(b.shader->info.shared_size,((shared+63)&~63u)+128);
      unsigned output_reads=0;bool user_address=false;
      nir_foreach_block(block,b.impl)nir_foreach_instr(in,block) {
         if(in->type!=nir_instr_type_intrinsic)continue;
         nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
         EXPECT_NE(i->intrinsic,nir_intrinsic_load_per_vertex_output);
         EXPECT_NE(i->intrinsic,nir_intrinsic_load_per_primitive_output);
         if(i->intrinsic==nir_intrinsic_load_shared) {
            output_reads++;
            if(nir_src_is_const(i->src[0])&&nir_src_as_uint(i->src[0])==0&&i->def.bit_size==32)
               user_address=true;
         }
      }
      EXPECT_GE(output_reads,2u);EXPECT_EQ(user_address,shared!=0);
      nir_validate_shader(b.shader,"mesh output load arena");
      agx_preprocess_nir(b.shader);agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;
      key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
      agx_shader_part part={};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader,&key,&part))<<agx_last_compile_refusal();
      free(part.binary);ralloc_free(b.shader);
   }
}

/* A padded mesh workgroup keeps its API body under the exact invocation
 * mask; a top-level execution barrier splits that body and is passed by every
 * physical lane. */
TEST(NativeForms, M5MeshPaddedTopLevelBarrierStaysUnmasked)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"padded top-level barrier");
   b.shader->info.workgroup_size[0]=b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=3;b.shader->info.mesh.max_primitives_out=1;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;b.shader->info.shared_size=4;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;
   nir_def *index=nir_load_local_invocation_index(&b);
   nir_store_shared(&b,index,nir_imm_int(&b,0),.align_mul=4);
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
      .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
   nir_set_vertex_and_primitive_count(&b,nir_iadd_imm(&b,index,3),nir_imm_int(&b,1),nir_imm_int(&b,0));
   ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
   EXPECT_GT(b.shader->info.workgroup_size[0],1u);
   nir_validate_shader(b.shader,"padded top-level barrier");
   unsigned user_barriers=0;
   nir_foreach_block(block,nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(in,block) {
         if(in->type!=nir_instr_type_intrinsic)continue;
         nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
         if(i->intrinsic==nir_intrinsic_barrier &&
            nir_intrinsic_execution_scope(i)!=SCOPE_NONE) {
            EXPECT_EQ(block->cf_node.parent->type,nir_cf_node_function);
            user_barriers++;
         }
      }
   }
   EXPECT_GE(user_barriers,1u);
   ralloc_free(b.shader);
}

TEST(NativeForms, M5MeshPaddedNestedBarrierRefusesWithoutMutation)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"padded nested barrier refusal");
   b.shader->info.workgroup_size[0]=b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=3;b.shader->info.mesh.max_primitives_out=1;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;b.shader->info.shared_size=4;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;
   nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,3),nir_imm_int(&b,1),nir_imm_int(&b,0));
   nir_push_if(&b,nir_ieq_imm(&b,nir_load_local_invocation_index(&b),0));
   nir_barrier(&b,.execution_scope=SCOPE_WORKGROUP,.memory_scope=SCOPE_WORKGROUP,
      .memory_semantics=NIR_MEMORY_ACQ_REL,.memory_modes=nir_var_mem_shared);
   nir_pop_if(&b,NULL);
   EXPECT_FALSE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
   EXPECT_EQ(b.shader->info.shared_size,4u);EXPECT_EQ(b.shader->info.workgroup_size[0],1u);
   ralloc_free(b.shader);
}

TEST(NativeForms, M5MeshPrimitiveDeclarationsRespectWrittenMask)
{
   for (bool written : {false, true}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "mesh primitive declaration");
      b.shader->info.io_lowered = true;
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = 256;
      b.shader->info.mesh.max_primitives_out = 256;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      b.shader->info.outputs_written = VARYING_BIT_POS |
         VARYING_BIT_PRIMITIVE_INDICES | (written ? VARYING_BIT_VAR(0) : 0);
      b.shader->info.per_primitive_outputs =
         VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_VAR(0);
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                       nir_imm_int(&b, 1), nir_imm_int(&b, 0));
      for (unsigned vertex = 0; vertex < 3; ++vertex) {
         nir_store_per_vertex_output(&b, nir_imm_vec4(&b, vertex, 0, 0, 1),
            nir_imm_int(&b, vertex), nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      }
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
      if (written) {
         nir_store_per_primitive_output(&b, nir_imm_vec4(&b, 0, 1, 0, 1),
            nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 15,
            .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_VAR0, .num_slots = 1});
      }
      /* A written per-primitive word would need 768 private vertices; the
       * provoking vertex carries it instead (mesh_layout_provoking): 784
       * index and count bytes, 8 planes of 256 slots, then the slot map. */
      ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
      EXPECT_EQ(b.shader->info.shared_size, written ? 10048u : 4928u);
      EXPECT_EQ(b.shader->info.mesh.max_vertices_out, 256u);
      nir_validate_shader(b.shader, "mesh primitive declaration");
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5MeshVertexColorUVSTReachesRA)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"native mesh color UVST");
   b.shader->info.workgroup_size[0]=3;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=3;b.shader->info.mesh.max_primitives_out=1;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|VARYING_BIT_VAR(0);
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;b.shader->info.io_lowered=true;
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,3),nir_imm_int(&b,1),nir_imm_int(&b,0));
   nir_store_per_vertex_output(&b,nir_imm_vec4(&b,0,1,0,1),lane,nir_imm_int(&b,0),
      .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_VAR0,.num_slots=1});
   nir_store_per_vertex_output(&b,nir_imm_vec4(&b,0,0,.25,1),lane,nir_imm_int(&b,0),
      .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
   nir_push_if(&b,nir_ieq_imm(&b,lane,0));
   nir_store_per_primitive_output(&b,nir_imm_ivec3(&b,0,1,2),nir_imm_int(&b,0),
      nir_imm_int(&b,0),.write_mask=7,.src_type=nir_type_uint32,
      .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
   nir_pop_if(&b,NULL);
   ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));EXPECT_EQ(b.shader->info.shared_size,192u);
   unsigned uvst=0;nir_foreach_block(block,b.impl)nir_foreach_instr(in,block)
      if(in->type==nir_instr_type_intrinsic&&nir_instr_as_intrinsic(in)->intrinsic==nir_intrinsic_store_mesh_uvb_agx)uvst++;
   EXPECT_EQ(uvst,9u);nir_validate_shader(b.shader,"native color planes");agx_preprocess_nir(b.shader);
   agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
   agx_shader_part part={};ASSERT_TRUE(agx_compile_shader_nir(b.shader,&key,&part))<<agx_last_compile_refusal();
   free(part.binary);ralloc_free(b.shader);
}

TEST(NativeForms, M5MeshNativeLargeOutputBatchesCompile)
{
   for (unsigned vertices : {3u,4u,5u,8u,9u,16u,32u,33u,64u,256u}) {
      const unsigned primitives=vertices==64?126:vertices==256?256:1;
      nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,
         &agx_nir_options,"native mesh larger UVB rounds");
      b.shader->info.workgroup_size[0]=32;
      b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
      b.shader->info.mesh.max_vertices_out=vertices;
      b.shader->info.mesh.max_primitives_out=primitives;
      b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;
      b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|VARYING_BIT_VAR(0);
      b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;
      b.shader->info.io_lowered=true;
      nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,vertices),nir_imm_int(&b,primitives),nir_imm_int(&b,0));
      /* The UVS layout comes from the per-vertex stores themselves. */
      nir_store_per_vertex_output(&b,nir_imm_vec4(&b,0,1,0,1),nir_load_local_invocation_index(&b),
         nir_imm_int(&b,0),.write_mask=15,.src_type=nir_type_float32,
         .io_semantics={.location=VARYING_SLOT_VAR0,.num_slots=1});
      ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
      unsigned varying=0,primitive=0,round2=0;
      const unsigned position=((3*primitives+15)&~15u)+16;
      const unsigned stride=((vertices+3)&~3u)*4;
      nir_foreach_block(block,b.impl)nir_foreach_instr(in,block) {
         if(in->type!=nir_instr_type_intrinsic)continue;
         nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
         if(i->intrinsic!=nir_intrinsic_store_mesh_uvb_agx)continue;
         ASSERT_TRUE(nir_src_is_const(i->src[0]));
         unsigned target=nir_src_comp_as_uint(i->src[0],0), address=nir_src_comp_as_uint(i->src[0],1);
         if(nir_intrinsic_flags(i)) {primitive++;EXPECT_EQ(address,(target>>16)*4);}
         else {varying++;EXPECT_EQ(address,position+(target&65535)*stride+4*(target>>16));
            if((target>>16)==32)round2++;}
      }
      EXPECT_EQ(varying,8*((vertices+31)/32));
      EXPECT_EQ(primitive,((3*primitives+3)/4+31)/32);
      EXPECT_EQ(round2,vertices>32?8u:0u);
      EXPECT_EQ(b.shader->info.shared_size,(position+8*stride+63)&~63u);
      nir_validate_shader(b.shader,"native larger mesh batch descriptors");
      agx_preprocess_nir(b.shader);agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;
      key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
      agx_shader_part part={};ASSERT_TRUE(agx_compile_shader_nir(b.shader,&key,&part))<<agx_last_compile_refusal();
      free(part.binary);ralloc_free(b.shader);
   }
}

/* Mesh shader with three per-vertex user slots of different interpolation
 * classes: VAR0 vec4 smooth, VAR1 ivec2 flat (stored as 16-bit), VAR2 float
 * noperspective, plus position. */
static nir_shader *
m5_mesh_multi_varying_shader(void)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"native mesh multi varyings");
   b.shader->info.workgroup_size[0]=4;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=4;b.shader->info.mesh.max_primitives_out=2;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|
      VARYING_BIT_VAR(0)|VARYING_BIT_VAR(1)|VARYING_BIT_VAR(2);
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;b.shader->info.io_lowered=true;
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,4),nir_imm_int(&b,2),nir_imm_int(&b,0));
   nir_store_per_vertex_output(&b,nir_imm_vec4(&b,0,0,.25,1),lane,nir_imm_int(&b,0),
      .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
   nir_store_per_vertex_output(&b,nir_vec4(&b,nir_u2f32(&b,lane),nir_imm_float(&b,1),
      nir_imm_float(&b,2),nir_imm_float(&b,3)),lane,nir_imm_int(&b,0),
      .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_VAR0,.num_slots=1});
   nir_store_per_vertex_output(&b,nir_u2u16(&b,nir_vec2(&b,lane,nir_imm_int(&b,7))),lane,nir_imm_int(&b,0),
      .write_mask=3,.src_type=nir_type_int16,.io_semantics={.location=VARYING_SLOT_VAR1,.num_slots=1});
   nir_store_per_vertex_output(&b,nir_imm_float(&b,5),lane,nir_imm_int(&b,0),
      .write_mask=1,.component=1,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_VAR2,.num_slots=1});
   nir_push_if(&b,nir_ieq_imm(&b,lane,0));
   nir_store_per_primitive_output(&b,nir_imm_ivec3(&b,0,1,2),nir_imm_int(&b,0),
      nir_imm_int(&b,0),.write_mask=7,.src_type=nir_type_uint32,
      .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
   nir_pop_if(&b,NULL);
   return b.shader;
}

/* Point and line meshes with user and per-primitive words: the point size
 * word follows the user words (agx_nir_lower_uvs group order) and every
 * primitive gets one private vertex per corner for its flat words. */
TEST(NativeForms, M5PointLineMeshPerPrimitiveWords)
{
   for (auto primitive : {MESA_PRIM_POINTS, MESA_PRIM_LINES}) {
      const unsigned corners = primitive == MESA_PRIM_POINTS ? 1 : 2;
      nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"point/line per-primitive");
      b.shader->info.workgroup_size[0]=2;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
      b.shader->info.mesh.max_vertices_out=2;b.shader->info.mesh.max_primitives_out=2;
      b.shader->info.mesh.primitive_type=primitive;b.shader->info.io_lowered=true;
      b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|VARYING_BIT_VAR(0)|
         (primitive==MESA_PRIM_POINTS ? VARYING_BIT_PSIZ : 0);
      b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES|VARYING_BIT_VAR(0);
      nir_def *lane=nir_load_local_invocation_index(&b);
      nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,2),nir_imm_int(&b,2),nir_imm_int(&b,0));
      nir_store_per_vertex_output(&b,nir_imm_vec4(&b,0,0,.25,1),lane,nir_imm_int(&b,0),
         .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
      if (primitive==MESA_PRIM_POINTS)
         nir_store_per_vertex_output(&b,nir_imm_float(&b,1),lane,nir_imm_int(&b,0),
            .write_mask=1,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_PSIZ,.num_slots=1});
      nir_store_per_primitive_output(&b,nir_imm_vec4(&b,0,0,1,1),lane,nir_imm_int(&b,0),
         .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_VAR0,.num_slots=1});
      nir_store_per_primitive_output(&b,primitive==MESA_PRIM_POINTS ? lane : nir_vec2(&b,lane,lane),
         lane,nir_imm_int(&b,0),.write_mask=BITFIELD_MASK(corners),.src_type=nir_type_uint32,
         .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_m5_mesh_gather_varyings(b.shader,&v));
      EXPECT_EQ(v.first[VARYING_SLOT_VAR0],4u);EXPECT_FALSE(v.dynamic);
      if (primitive==MESA_PRIM_POINTS) {
         EXPECT_EQ(v.first[VARYING_SLOT_PSIZ],8u);EXPECT_EQ(v.size,9u);
      } else {
         EXPECT_EQ(v.size,8u);
      }
      ASSERT_TRUE(agx_nir_lower_m5_mesh(b.shader,&v));
      EXPECT_EQ(b.shader->info.mesh.max_vertices_out,2u*corners);
      nir_validate_shader(b.shader,"point/line per-primitive");
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5MeshMultiVaryingPlanesFollowLinkedUVS)
{
   nir_shader *s=m5_mesh_multi_varying_shader();
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));
   EXPECT_EQ(v.components[VARYING_SLOT_VAR0],4u);EXPECT_EQ(v.components[VARYING_SLOT_VAR1],2u);
   EXPECT_EQ(v.components[VARYING_SLOT_VAR2],2u);EXPECT_EQ(v.size,12u);EXPECT_TRUE(v.dynamic);
   /* Linked order: smooth VAR0, linear VAR2 last, flat VAR1 between. */
   v.first[VARYING_SLOT_VAR0]=4;v.first[VARYING_SLOT_VAR1]=8;v.first[VARYING_SLOT_VAR2]=10;v.dynamic=false;
   ASSERT_TRUE(agx_nir_lower_m5_mesh(s,&v));
   const unsigned stride=16,position=16+16;
   EXPECT_EQ(s->info.shared_size,(position+12*stride+63)&~63u);
   unsigned planes=0;bool seen[12]={};
   nir_foreach_block(block,nir_shader_get_entrypoint(s))nir_foreach_instr(in,block) {
      if(in->type!=nir_instr_type_intrinsic)continue;
      nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
      EXPECT_NE(i->intrinsic,nir_intrinsic_load_uvs_index_agx);
      if(i->intrinsic!=nir_intrinsic_store_mesh_uvb_agx||nir_intrinsic_flags(i))continue;
      ASSERT_TRUE(nir_src_is_const(i->src[0]));
      unsigned target=nir_src_comp_as_uint(i->src[0],0),address=nir_src_comp_as_uint(i->src[0],1);
      ASSERT_LT(target&65535,12u);seen[target&65535]=true;planes++;
      EXPECT_EQ(address,position+(target&65535)*stride);
   }
   EXPECT_EQ(planes,12u);for(unsigned k=0;k<12;k++)EXPECT_TRUE(seen[k])<<k;
   /* Stores land in the linked planes: VAR2.y is plane 11. */
   bool var2y=false;unsigned narrow=0;
   nir_foreach_block(block,nir_shader_get_entrypoint(s))nir_foreach_instr(in,block) {
      if(in->type!=nir_instr_type_intrinsic)continue;
      nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
      if(i->intrinsic==nir_intrinsic_store_shared) {
         /* Only the vertex/primitive counts are 16-bit; INT16 VAR1 is widened. */
         narrow+=i->src[0].ssa->bit_size==16;
         nir_scalar a=nir_scalar_resolved(i->src[1].ssa,0);
         if(nir_scalar_is_alu(a)&&nir_scalar_alu_op(a)==nir_op_iadd) {
            for(unsigned k=0;k<2;k++){nir_scalar c=nir_scalar_chase_alu_src(a,k);
               if(nir_scalar_is_const(c)&&nir_scalar_as_uint(c)==position+11*stride)var2y=true;}
         }
      }
   }
   EXPECT_TRUE(var2y);EXPECT_EQ(narrow,2u);
   nir_validate_shader(s,"native multi varying planes");agx_preprocess_nir(s);
   agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
   agx_shader_part part={};ASSERT_TRUE(agx_compile_shader_nir(s,&key,&part))<<agx_last_compile_refusal();
   free(part.binary);ralloc_free(s);
}

TEST(NativeForms, M5MeshDynamicVaryingSelectorsReadRootIndex)
{
   nir_shader *s=m5_mesh_multi_varying_shader();
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));ASSERT_TRUE(v.dynamic);
   ASSERT_TRUE(agx_nir_lower_m5_mesh(s,&v));
   unsigned index_loads=0,dynamic=0,fixed=0;
   nir_foreach_block(block,nir_shader_get_entrypoint(s))nir_foreach_instr(in,block) {
      if(in->type!=nir_instr_type_intrinsic)continue;
      nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
      if(i->intrinsic==nir_intrinsic_load_uvs_index_agx)index_loads++;
      if(i->intrinsic!=nir_intrinsic_store_mesh_uvb_agx||nir_intrinsic_flags(i))continue;
      if(nir_src_is_const(i->src[0])){fixed++;EXPECT_LT(nir_src_comp_as_uint(i->src[0],0)&65535,4u);}
      else dynamic++;
   }
   EXPECT_EQ(index_loads,3u);EXPECT_EQ(fixed,4u);EXPECT_EQ(dynamic,8u);
   nir_validate_shader(s,"native dynamic varying selectors");ralloc_free(s);
}

TEST(NativeForms, M5MeshClipDistancesPublishVaryingAndClippingPlanes)
{
   for (unsigned count : {2u, 8u}) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "native mesh clipping planes");
      b.shader->info.io_lowered = true;
      b.shader->info.workgroup_size[0] = 3;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = 3;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      b.shader->info.clip_distance_array_size = count;
      b.shader->info.outputs_written = VARYING_BIT_POS |
         VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CLIP_DIST0 |
         (count > 4 ? VARYING_BIT_CLIP_DIST1 : 0);
      b.shader->info.per_primitive_outputs = VARYING_BIT_PRIMITIVE_INDICES;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                       nir_imm_int(&b, 1), nir_imm_int(&b, 0));
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      for (unsigned c = 0; c < count; ++c) {
         nir_store_per_vertex_output(&b, nir_imm_float(&b, c + 0.25f),
            nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 1,
            .component = c % 4, .src_type = nir_type_float32,
            .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0 + c / 4,
                             .num_slots = 1});
      }
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                          .num_slots = 1});
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_m5_mesh_gather_varyings(b.shader, &v));
      EXPECT_EQ(v.clip_distance_count, count);
      EXPECT_EQ(v.clip_distance_first, 4u + count);
      EXPECT_EQ(v.size, 4u + 2 * count);
      EXPECT_EQ(v.dynamic, count > 4);
      v.dynamic = false;
      ASSERT_TRUE(agx_nir_lower_m5_mesh(b.shader, &v));
      NIR_PASS(_, b.shader, nir_opt_constant_folding);
      unsigned copies[8] = {0};
      nir_foreach_block(block, b.impl) nir_foreach_instr(in, block) {
         if (in->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
         if (i->intrinsic != nir_intrinsic_store_shared ||
             !nir_src_is_const(i->src[1]))
            continue;
         unsigned address = nir_src_as_uint(i->src[1]);
         if (address < 96)
            continue;
         ASSERT_LT(address, 32u + 16 * v.size);
         unsigned component = ((address - 96) / 16) % count;
         ASSERT_TRUE(nir_src_is_const(i->src[0]));
         EXPECT_FLOAT_EQ(nir_src_as_float(i->src[0]), component + 0.25f);
         copies[component]++;
      }
      for (unsigned c = 0; c < count; ++c)
         EXPECT_EQ(copies[c], 2u);
      nir_validate_shader(b.shader, "native mesh clip-distance publication");
      agx_preprocess_nir_m5(b.shader);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17S;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.no_preamble = true;
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, Gen1MeshClipCullPointSizeFollowHardwareUVSOrder)
{
   /* Two clip and one cull distance plus gl_CullPrimitiveEXT. User words
    * (the CLIP_DIST slot) come first, then point size, then the clip group,
    * as agx_nir_lower_uvs lays out the hardware vertex. */
   for (mesa_prim prim : {MESA_PRIM_TRIANGLES, MESA_PRIM_POINTS}) {
      const unsigned corners = prim == MESA_PRIM_POINTS ? 1 : 3;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "gen1 mesh clip/cull");
      b.shader->info.io_lowered = true;
      b.shader->info.workgroup_size[0] = 3;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = 3;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.mesh.primitive_type = prim;
      b.shader->info.clip_distance_array_size = 2;
      b.shader->info.cull_distance_array_size = 1;
      b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_CLIP_DIST0 |
         VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CULL_PRIMITIVE;
      b.shader->info.per_primitive_outputs =
         VARYING_BIT_PRIMITIVE_INDICES | VARYING_BIT_CULL_PRIMITIVE;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 3),
                                       nir_imm_int(&b, 1), nir_imm_int(&b, 0));
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 15,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_vertex_output(&b, nir_imm_vec3(&b, 1, 1, -1),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0), .write_mask = 7,
         .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 1});
      nir_store_per_primitive_output(&b, nir_imm_ivec3(&b, 0, 1, 2),
         nir_imm_int(&b, 0), nir_imm_int(&b, 0),
         .write_mask = BITFIELD_MASK(corners), .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                          .num_slots = 1});
      nir_store_per_primitive_output(&b, nir_imm_true(&b), nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_bool1,
         .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE,
                          .num_slots = 1});
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_gen1_mesh_gather_varyings(b.shader, &v));
      const unsigned psiz = prim == MESA_PRIM_POINTS;
      EXPECT_TRUE(v.cull_primitive);
      EXPECT_EQ(v.components[VARYING_SLOT_CLIP_DIST0], 3u);
      EXPECT_EQ(v.first[VARYING_SLOT_CLIP_DIST0], 4u);
      EXPECT_EQ(v.components[VARYING_SLOT_PSIZ], psiz);
      if (psiz) {
         EXPECT_EQ(v.first[VARYING_SLOT_PSIZ], 7u);
      }
      EXPECT_EQ(v.clip_first, 7u + psiz);
      EXPECT_EQ(v.clip_count, 2u);
      EXPECT_EQ(v.size, 9u + psiz);
      ASSERT_TRUE(agx_nir_lower_gen1_mesh(b.shader, &v));
      EXPECT_EQ(b.shader->info.cull_distance_array_size, 0u);
      nir_validate_shader(b.shader, "gen1 mesh clip/cull");
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, Gen1MeshLayerExportsFromThreadgroupMemory)
{
   /* A per-primitive layer (hk's multiview lowering writes one) reaches the
    * exported layer/viewport vertex word from threadgroup memory: the export
    * must not read it back from the primitive's record in device memory. The
    * record keeps its copy for a fragment shader reading gl_Layer.
    *
    * near: the shader's own threadgroup memory leaves room for the export
    * arena (1,920 + 1,792 bytes from 28,992) but not for the 128 bytes of
    * layer words; it still compiles, keeps the arena, and reads the record. */
   for (unsigned variant = 0; variant < 3; ++variant) {
      const bool cull = variant == 1, near = variant == 2;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_MESH, &agx_nir_options, "gen1 mesh layer");
      b.shader->info.io_lowered = true;
      b.shader->info.workgroup_size[0] = 40;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.max_vertices_out = 64;
      b.shader->info.mesh.max_primitives_out = 32;
      b.shader->info.mesh.primitive_type = MESA_PRIM_LINES;
      b.shader->info.shared_size = near ? 28992 : 0;
      b.shader->info.outputs_written = VARYING_BIT_POS | VARYING_BIT_LAYER |
         VARYING_BIT_PRIMITIVE_INDICES | (cull ? VARYING_BIT_CULL_PRIMITIVE : 0);
      b.shader->info.per_primitive_outputs = VARYING_BIT_LAYER |
         VARYING_BIT_PRIMITIVE_INDICES | (cull ? VARYING_BIT_CULL_PRIMITIVE : 0);
      nir_def *lane = nir_load_local_invocation_index(&b);
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 64),
                                       nir_imm_int(&b, 32), nir_imm_int(&b, 0));
      nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), lane,
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
      nir_store_per_primitive_output(&b, nir_imm_ivec2(&b, 0, 1), lane,
         nir_imm_int(&b, 0), .write_mask = 3, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES,
                          .num_slots = 1});
      if (cull) {
         nir_store_per_primitive_output(&b, nir_imm_false(&b), lane,
            nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_bool1,
            .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE,
                             .num_slots = 1});
      }
      nir_store_per_primitive_output(&b, nir_imm_int(&b, 1), lane,
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_LAYER, .num_slots = 1});
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_gen1_mesh_gather_varyings(b.shader, &v));
      EXPECT_TRUE(v.lv);
      EXPECT_EQ(v.primitive_stride_B, 4u);
      const unsigned shared_before = b.shader->info.shared_size;
      ASSERT_TRUE(agx_nir_lower_gen1_mesh(b.shader, &v));
      nir_validate_shader(b.shader, "gen1 mesh layer");
      unsigned global_loads = 0, global_stores = 0;
      nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
         nir_foreach_instr(in, block) {
            if (in->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
            global_loads += i->intrinsic == nir_intrinsic_load_global;
            global_stores += i->intrinsic == nir_intrinsic_store_global;
         }
      }
      /* Culling still moves the records themselves (one load per record
       * word); near the limit the export reads the layer from the record;
       * nothing else reads device memory. */
      EXPECT_EQ(global_loads, (cull || near) ? 1u : 0u);
      EXPECT_GE(global_stores, 1u);
      EXPECT_LE(b.shader->info.shared_size, 32768u);
      if (near) {
         /* The export arena is kept: 2 private vertices per line. */
         EXPECT_EQ(b.shader->info.mesh.max_vertices_out, 64u);
         EXPECT_EQ(b.shader->info.shared_size, 28992u + 1920u + 1792u);
      } else {
         EXPECT_GE(b.shader->info.shared_size, shared_before + 4u * 32u);
      }
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5MeshVaryingRefusalsLeaveShaderUntouched)
{
   /* A placement that does not cover every written component is refused. */
   nir_shader *s=m5_mesh_multi_varying_shader();
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));
   v.first[VARYING_SLOT_VAR2]=v.first[VARYING_SLOT_VAR1];
   EXPECT_FALSE(agx_nir_lower_m5_mesh(s,&v));
   EXPECT_EQ(s->info.shared_size,0u);EXPECT_EQ(s->info.workgroup_size[0],4u);
   ralloc_free(s);
}

TEST(NativeForms, M5MeshPaddedGlobalInvocationIdUsesApiSize)
{
   /* API workgroup 8 threads padded to 24 physical lanes (max_vertices 24):
    * the hardware thread position in grid counts physical lanes, so the
    * lowering must rebuild gl_GlobalInvocationID from the API size. */
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"padded global id");
   b.shader->info.workgroup_size[0]=8;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=24;b.shader->info.mesh.max_primitives_out=8;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;b.shader->info.io_lowered=true;
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,24),nir_imm_int(&b,8),nir_imm_int(&b,0));
   nir_def *gid=nir_channel(&b,nir_load_global_invocation_id(&b,32),0);
   nir_store_per_vertex_output(&b,nir_vec4(&b,nir_u2f32(&b,gid),nir_imm_float(&b,0),
      nir_imm_float(&b,0),nir_imm_float(&b,1)),lane,nir_imm_int(&b,0),
      .write_mask=15,.src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
   nir_store_per_primitive_output(&b,nir_imm_ivec3(&b,0,1,2),lane,nir_imm_int(&b,0),
      .write_mask=7,.src_type=nir_type_uint32,
      .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
   ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
   EXPECT_EQ(b.shader->info.workgroup_size[0],24u);
   unsigned global=0,workgroup=0;
   nir_foreach_block(block,b.impl)nir_foreach_instr(in,block) {
      if(in->type!=nir_instr_type_intrinsic)continue;
      nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
      global+=i->intrinsic==nir_intrinsic_load_global_invocation_id;
      workgroup+=i->intrinsic==nir_intrinsic_load_workgroup_id;
   }
   EXPECT_EQ(global,0u);EXPECT_GE(workgroup,1u);
   nir_validate_shader(b.shader,"padded global id");
   agx_preprocess_nir(b.shader);agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;
   key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
   agx_shader_part part={};ASSERT_TRUE(agx_compile_shader_nir(b.shader,&key,&part))<<agx_last_compile_refusal();
   free(part.binary);ralloc_free(b.shader);
}

TEST(NativeForms, M5SaturatingAddOfHalfIndexCompiles)
{
   /* nir_opt_idiv_const turns an unsigned divide by 7 into
    * umul_high(uadd_sat(x, 1), magic); x here is the 16-bit local index
    * special register, zero-extended. The AGX3C saturating add only has a
    * witnessed 32-bit register form, so the extension must stay a move. */
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,&agx_nir_options,"udiv7 of local index");
   b.shader->info.workgroup_size[0]=64;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_def *q=nir_udiv_imm(&b,lane,7);
   nir_store_global(&b,q,nir_iadd(&b,nir_imm_int64(&b,0x2000000),
      nir_u2u64(&b,nir_imul_imm(&b,lane,4))),.write_mask=1,.align_mul=4);
   agx_preprocess_nir(b.shader);
   agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
   agx_shader_part part={};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader,&key,&part))<<agx_last_compile_refusal();
   free(part.binary);ralloc_free(b.shader);
}

TEST(NativeForms, M5NativeTriangleMeshProbeCompiles)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,
                     &agx_nir_options,"M5 native triangle mesh probe");
   b.shader->info.workgroup_size[0]=3;b.shader->info.workgroup_size[1]=1;
   b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=3;b.shader->info.mesh.max_primitives_out=1;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES;
   b.shader->info.io_lowered=true;
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,3),nir_imm_int(&b,1),
                                       nir_imm_int(&b,0),.stream_id=0);
   nir_def *x=nir_bcsel(&b,nir_ieq_imm(&b,lane,2),nir_imm_float(&b,0),
                nir_bcsel(&b,nir_ieq_imm(&b,lane,0),nir_imm_float(&b,-1),
                                                     nir_imm_float(&b,1)));
   nir_def *y=nir_bcsel(&b,nir_ieq_imm(&b,lane,2),nir_imm_float(&b,1),
                                                     nir_imm_float(&b,-1));
   nir_def *pos=nir_vec4(&b,x,y,nir_imm_float(&b,0.25),nir_imm_float(&b,1));
   nir_store_per_vertex_output(&b,pos,lane,nir_imm_int(&b,0),
     .write_mask=15,.src_type=nir_type_float32,
     .io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
   nir_push_if(&b,nir_ieq_imm(&b,lane,0));
   nir_store_per_primitive_output(&b,nir_imm_ivec3(&b,0,1,2),
     nir_imm_int(&b,0),nir_imm_int(&b,0),.write_mask=7,.src_type=nir_type_uint32,
     .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
   nir_pop_if(&b,NULL);
   ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
   EXPECT_EQ(b.shader->info.shared_size,128u);
   nir_validate_shader(b.shader,"native triangle mesh probe");
   agx_preprocess_nir(b.shader);
   agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;
   key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
   agx_shader_part part={};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader,&key,&part))<<agx_last_compile_refusal();
   free(part.binary);
   ralloc_free(b.shader);
}

/* cts-coverage/fp-controls/native/nanmin.gpubin, G17G with -fno-fast-math:
 * IEEE minNum/maxNum retain condition 3/7 and the compare/select ties. */
TEST(NativeForms, M5MinNumMaxNumNaNWitnesses)
{
   for (bool maximum : {false, true}) {
      agx3_fsel32_gpu1_desc d = {};
      d.dest = 2;
      d.cmp0 = 2; d.cmp1 = 3;
      d.cmp0_16 = d.cmp1_16 = true;
      d.cmp0_hint = d.cmp1_hint = AGX3_ALU_HINT_DISCARD;
      d.sel[0] = 2; d.sel[1] = 3;
      d.sel_hint[0] = d.sel_hint[1] = AGX3_ALU_HINT_DISCARD;
      d.sel_integer = true;
      d.condition = maximum ? 7 : 3;
      d.wait_mask = 0x80;
      uint8_t out[AGX3_ALU_MAX_LENGTH] = {}, length = 0;
      ASSERT_TRUE(agx3_alu_fcmp_select16_gpu1(&d, out, &length));
      const std::vector<uint8_t> native = maximum
         ? std::vector<uint8_t>{0x32,0x02,0x05,0x0a,0x23,0x00,0xa0,0x02,
                               0xcc,0x80,0xa0,0x64,0x26,0x00,0x80,0x00}
         : std::vector<uint8_t>{0x32,0x02,0x45,0x0a,0x23,0x00,0xa0,0x02,
                               0xcc,0x80,0xa0,0x60,0x26,0x00,0x80,0x00};
      EXPECT_EQ(std::vector<uint8_t>(out, out + length), native);
      d.sel[1] = 5;
      EXPECT_FALSE(agx3_alu_fcmp_select16_gpu1(&d, out, &length));
   }
   agx3_fsel32_gpu1_desc d = {};
   d.dest = 4; d.dest_cache = true;
   d.cmp0 = 4; d.cmp1 = 6;
   d.cmp0_hint = d.cmp1_hint = AGX3_ALU_HINT_DISCARD;
   d.sel[0] = 4; d.sel[1] = 6;
   d.sel_hint[0] = d.sel_hint[1] = AGX3_ALU_HINT_DISCARD;
   d.sel_integer = true; d.condition = 7; d.wait_mask = 0x80;
   uint8_t out[AGX3_ALU_MAX_LENGTH] = {}, length = 0;
   ASSERT_TRUE(agx3_alu_fcmp_select32_gpu1(&d, out, &length));
   EXPECT_EQ(std::vector<uint8_t>(out, out + length),
      (std::vector<uint8_t>{0x22,0x00,0x07,0x2b,0x2b,0x00,0xa0,
                           0x0a,0x96,0x01,0xa1,0x64,0x20,0x40}));
   d.dest = 2; d.dest_cache = false; d.cmp1 = d.sel[1] = 2;
   d.condition = 3; d.wait_mask = 1;
   ASSERT_TRUE(agx3_alu_fcmp_select32_gpu1(&d, out, &length));
   EXPECT_EQ(std::vector<uint8_t>(out, out + length),
             (std::vector<uint8_t>{0x12,0x05,0x1e,0x03,0x01,0x20}));
   d.sel[0] = 8;
   EXPECT_FALSE(agx3_alu_fcmp_select32_gpu1(&d, out, &length));
}

TEST(NativeForms, M5MappedBufferPointerOffsets)
{
   nir_builder builder = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "M5 mapped buffer pointers");
   nir_builder *b = &builder;
   std::vector<nir_intrinsic_instr *> stores;
   auto store_test_val = [&](nir_def *value) {
      nir_intrinsic_instr *store = nir_intrinsic_instr_create(
         b->shader, nir_intrinsic_store_global);
      store->num_components = 1;
      store->src[0] = nir_src_for_ssa(value);
      store->src[1] = nir_src_for_ssa(nir_imm_int64(b, 0x1000 + 4 * stores.size()));
      nir_intrinsic_set_align(store, 4, 0);
      nir_intrinsic_set_write_mask(store, 1);
      nir_builder_instr_insert(b, &store->instr);
      stores.push_back(store);
   };
   auto test_val = [&](unsigned i) { return stores[i]->src[0].ssa; };
   for (nir_address_format format : {
        nir_address_format_64bit_global_32bit_offset_map,
        nir_address_format_64bit_bounded_global_map}) {
      nir_def *parts[] = {nir_imm_int(b, 0x1000), nir_imm_int(b, 2),
         nir_imm_int(b, 4096), nir_imm_int(b, 12), nir_imm_int(b, 0x1234)};
      nir_def *ptr = nir_vec(b, parts, 5);
      ptr = nir_build_addr_iadd(b, ptr, format, nir_var_mem_ssbo,
                               nir_imm_int(b, 28));
      ptr = nir_bcsel(b, nir_imm_false(b), nir_build_imm(b, 5, 32, nir_address_format_null_value(format)), ptr);
      unsigned start = stores.size();
      for (unsigned c = 0; c < 5; ++c)
         store_test_val(nir_channel(b, ptr, c));
      nir_def *other = nir_vector_insert_imm(b, ptr, nir_imm_int(b, 0x4321), 4);
      store_test_val(nir_b2i32(b, nir_build_addr_ieq(b, ptr, other, format)));
      nir_def *changed = nir_build_addr_iadd(b, ptr, format, nir_var_mem_ssbo,
                                           nir_imm_int(b, 4));
      store_test_val(nir_b2i32(b, nir_build_addr_ieq(b, ptr, changed, format)));
      NIR_PASS(_, b->shader, nir_opt_constant_folding);
      const uint32_t expected[] = {0x1000, 2, 4096, 40, 0x1234, 1, 0};
      for (unsigned c = 0; c < 7; ++c)
         EXPECT_EQ(nir_src_as_uint(nir_src_for_ssa(test_val(start + c))), expected[c]);
   }
   nir_validate_shader(b->shader, "mapped buffer pointers");
   ralloc_free(b->shader);
}

TEST(NativeForms, M5GemmNativeCapturedInstructions)
{
   struct Case { agx3_matrix_gemm16_desc desc; uint8_t bytes[10]; };
   const Case cases[] = {
      {{16, 80, 72, AGX3_MATRIX_INPUT_BF16, true,false,false, 1}, {0xaf,0x00,0x25,0x82,0x2a,0x02,0xa0,0x02,0x00,0x12}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_BF16, false,false,true, 0}, {0xa7,0x04,0x25,0x80,0x22,0x02,0xa0,0x02,0x00,0x0e}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_BF16, true,false,false, 1}, {0xaf,0x00,0x25,0x80,0x2a,0x02,0xa0,0x02,0x00,0x0e}},
      {{16, 48, 0, AGX3_MATRIX_INPUT_BF16, false,false,true, 0}, {0xa7,0x04,0x25,0x03,0x22,0x02,0xa0,0x02,0x00,0x00}},
      {{16, 72, 64, AGX3_MATRIX_INPUT_BF16, true,false,false, 1}, {0xaf,0x00,0x25,0x90,0x2a,0x02,0xa0,0x02,0x00,0x10}},
      {{16, 56, 48, AGX3_MATRIX_INPUT_BF16, false,false,true, 0}, {0xa7,0x04,0x25,0x13,0x22,0x02,0xa0,0x02,0x00,0x0c}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_BF16, true,false,false, 1}, {0xaf,0x00,0x25,0x80,0x2a,0x02,0xa0,0x02,0x00,0x0e}},
      {{16, 48, 0, AGX3_MATRIX_INPUT_BF16, false,false,true, 0}, {0xa7,0x04,0x25,0x03,0x22,0x02,0xa0,0x02,0x00,0x00}},
      {{16, 80, 72, AGX3_MATRIX_INPUT_F16, true,false,false, 1}, {0xaf,0x00,0x25,0x82,0x2a,0x02,0xa4,0x42,0x00,0x12}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_F16, false,false,true, 0}, {0xa7,0x04,0x25,0x80,0x22,0x02,0xa4,0x42,0x00,0x0e}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_F16, true,false,false, 1}, {0xaf,0x00,0x25,0x80,0x2a,0x02,0xa4,0x42,0x00,0x0e}},
      {{16, 48, 0, AGX3_MATRIX_INPUT_F16, false,false,true, 0}, {0xa7,0x04,0x25,0x03,0x22,0x02,0xa4,0x42,0x00,0x00}},
      {{16, 72, 64, AGX3_MATRIX_INPUT_F16, true,false,false, 1}, {0xaf,0x00,0x25,0x90,0x2a,0x02,0xa4,0x42,0x00,0x10}},
      {{16, 56, 48, AGX3_MATRIX_INPUT_F16, false,false,true, 0}, {0xa7,0x04,0x25,0x13,0x22,0x02,0xa4,0x42,0x00,0x0c}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_F16, true,false,false, 1}, {0xaf,0x00,0x25,0x80,0x2a,0x02,0xa4,0x42,0x00,0x0e}},
      {{16, 48, 0, AGX3_MATRIX_INPUT_F16, false,false,true, 0}, {0xa7,0x04,0x25,0x03,0x22,0x02,0xa4,0x42,0x00,0x00}},
      {{16, 80, 72, AGX3_MATRIX_INPUT_F16, true,true,false, 1}, {0xaf,0x00,0x25,0x82,0x2a,0x02,0xa4,0x42,0x08,0x12}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_F16, false,true,true, 0}, {0xa7,0x04,0x25,0x80,0x22,0x02,0xa4,0x42,0x08,0x0e}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_F16, true,true,false, 1}, {0xaf,0x00,0x25,0x80,0x2a,0x02,0xa4,0x42,0x08,0x0e}},
      {{16, 48, 0, AGX3_MATRIX_INPUT_F16, false,true,true, 0}, {0xa7,0x04,0x25,0x03,0x22,0x02,0xa4,0x42,0x08,0x00}},
      {{16, 72, 64, AGX3_MATRIX_INPUT_F16, true,true,false, 1}, {0xaf,0x00,0x25,0x90,0x2a,0x02,0xa4,0x42,0x08,0x10}},
      {{16, 56, 48, AGX3_MATRIX_INPUT_F16, true,true,true, 0}, {0xa7,0x04,0x25,0x13,0x2a,0x02,0xa4,0x42,0x08,0x0c}},
      {{16, 64, 56, AGX3_MATRIX_INPUT_F16, true,true,false, 1}, {0xaf,0x00,0x25,0x80,0x2a,0x02,0xa4,0x42,0x08,0x0e}},
      {{16, 48, 0, AGX3_MATRIX_INPUT_F16, false,true,true, 0}, {0xa7,0x04,0x25,0x03,0x22,0x02,0xa4,0x42,0x08,0x00}},
   };
   for (const auto &c : cases) {
      uint8_t out[10] = {}, length = 0;
      ASSERT_TRUE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3C,
                                           &c.desc, out, &length));
      ASSERT_EQ(length, 10u);
      EXPECT_EQ(std::vector<uint8_t>(out, out + length),
                std::vector<uint8_t>(c.bytes, c.bytes + 10));
      EXPECT_FALSE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3B,
                                            &c.desc, out, &length));
   }
   auto invalid = cases[0].desc;
   uint8_t out[10] = {}, length = 0;
   invalid.dest = 17;
   EXPECT_FALSE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3C,
                                         &invalid, out, &length));
   invalid = cases[0].desc;
   invalid.a = 81;
   EXPECT_FALSE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3C,
                                         &invalid, out, &length));
   invalid = cases[0].desc;
   invalid.input_type = static_cast<agx3_matrix_input_type>(2);
   EXPECT_FALSE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3C,
                                         &invalid, out, &length));
}

TEST(NativeForms, M5MeshCullDistancesKeepVertexAndClipDomains)
{
   nir_shader *s = m5_mesh_multi_varying_shader();
   s->info.clip_distance_array_size = 1;
   s->info.cull_distance_array_size = 5;
   s->info.outputs_written |= VARYING_BIT_CLIP_DIST0 | VARYING_BIT_CLIP_DIST1;
   nir_builder b = nir_builder_at(nir_after_cf_list(&nir_shader_get_entrypoint(s)->body));
   const float distances[] = {-7.0f, -1.0f, 0.0f, 2.0f, -2.0f, 1.0f};
   for (unsigned c = 0; c < ARRAY_SIZE(distances); ++c) {
      nir_store_per_vertex_output(&b, nir_imm_float(&b, distances[c]),
         nir_imm_int(&b, 2), nir_imm_int(&b, c / 4),
         .write_mask = 1, .component = c % 4, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_CLIP_DIST0, .num_slots = 2});
   }
   ASSERT_TRUE(agx_nir_lower_cull_distance_vs(s));
   NIR_PASS(_, s, nir_opt_constant_folding);
   unsigned masks = 0;
   bool found[5] = {};
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) nir_foreach_instr(in, block) {
      if (in->type != nir_instr_type_intrinsic) continue;
      nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
      if (i->intrinsic != nir_intrinsic_store_per_vertex_output)
         continue;
      const unsigned slot = nir_intrinsic_io_semantics(i).location;
      if (slot < VARYING_SLOT_CULL_PRIMITIVE ||
          slot > VARYING_SLOT_CULL_PRIMITIVE + 1)
         continue;
      unsigned c = 4 * (slot - VARYING_SLOT_CULL_PRIMITIVE +
                        nir_src_as_uint(i->src[2])) + nir_intrinsic_component(i);
      ASSERT_LT(c, 5u);
      EXPECT_FALSE(found[c]); found[c] = true;
      EXPECT_EQ(nir_src_as_uint(i->src[1]), 2u);
      ASSERT_TRUE(nir_src_is_const(i->src[0]));
      EXPECT_FLOAT_EQ(nir_src_as_float(i->src[0]), distances[c + 1] >= 0.0f ? 1.0f : 0.0f);
      masks++;
   }
   EXPECT_EQ(masks, 5u);
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s, &v));
   EXPECT_EQ(v.components[VARYING_SLOT_CULL_PRIMITIVE], 4u);
   EXPECT_EQ(v.components[VARYING_SLOT_CULL_PRIMITIVE + 1], 1u);
   EXPECT_EQ(v.clip_distance_count, 1u);
   v.dynamic = false;
   ASSERT_TRUE(agx_nir_lower_m5_mesh(s, &v));
   nir_validate_shader(s, "mesh cull and clip distance domains");
   agx_preprocess_nir_m5(s);
   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G17S; key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.no_preamble = true;
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(s, &key, &part)) << agx_last_compile_refusal();
   free(part.binary); ralloc_free(s);
}

TEST(NativeForms, M5IntegerRegisterBallotPredicatesReachPacker)
{
   for (agx_icond condition : {AGX_ICOND_UEQ, AGX_ICOND_SEQ, AGX_ICOND_ULT,
                              AGX_ICOND_UGT, AGX_ICOND_SLT, AGX_ICOND_SGT}) {
      for (bool invert : {false, true}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.dev.chip = AGX_CHIP_G17S;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = MESA_SHADER_COMPUTE;
         agx_instr *ballot = agx_alloc_instr(b, AGX_OPCODE_ICMP_BALLOT, 1, 2);
         ballot->dest[0] = agx_register(22, AGX_SIZE_32);
         ballot->src[0] = agx_register(20, AGX_SIZE_32);
         ballot->src[1] = agx_register(16, AGX_SIZE_32);
         ballot->icond = condition;
         ballot->invert_cond = invert;
         agx_builder_insert(&b->cursor, ballot);
         agx_stop(b);
         util_dynarray bytes;
         util_dynarray_init(&bytes, mem);
         EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes)) << condition << " invert=" << invert;
         EXPECT_GT(bytes.size, 4u);
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, M5GemmIndependentIntrinsicWitnesses)
{
   struct Case { unsigned a, b; bool zero, transpose, preserve; uint8_t bytes[10]; };
   const Case cases[] = {
      {8, 0, true, false, true, {0x2f,0x00,0x05,0x10,0x22,0x00,0xa4,0x42,0x02,0x00}},
      {8, 0, true, true, true, {0x2f,0x00,0x05,0x10,0x22,0x00,0xa4,0x42,0x0e,0x00}},
      {24,16,false,false,false,{0x2f,0x00,0x25,0x12,0x22,0x02,0xa4,0x42,0x00,0x04}},
   };
   for (const auto &c : cases) {
      agx3_matrix_gemm16_desc desc = {};
      desc.a = c.a; desc.b = c.b; desc.input_type = AGX3_MATRIX_INPUT_F16;
      desc.wait_mask = 128; desc.zero_accumulator = c.zero;
      desc.transpose_a = c.transpose; desc.transpose_b = c.transpose;
      desc.preserve_a = c.preserve; desc.preserve_b = c.preserve;
      uint8_t bytes[10], length;
      ASSERT_TRUE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3C, &desc, bytes, &length));
      ASSERT_EQ(length, 10u);
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + length), std::vector<uint8_t>(c.bytes, c.bytes + 10));
   }
}

TEST(NativeForms, M5MmaNativeOperandTypeWitnesses)
{
   struct Case { agx3_matrix_mma8_desc desc; uint8_t bytes[14]; };
   const Case cases[] = {
      {{14,78,52,14,false,true,{false,false,false},{true,true,true},0,AGX3_MATRIX_INPUT_F32},
       {0x37,0x02,0x05,0x7a,0x28,0x20,0xaf,0x1a,0x31,0x08,0x18,0x00,0x1a,0x19}},
      {{102,102,94,98,false,true,{false,true,true},{true,false,false},128,AGX3_MATRIX_INPUT_BF16},
       {0x3f,0x02,0x85,0xbb,0x28,0x20,0xaf,0x2a,0x61,0x04,0x19,0x00,0x86,0x04}},
   };
   for (const auto &c : cases) {
      uint8_t bytes[14], length;
      ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &c.desc, bytes, &length));
      ASSERT_EQ(length, 14u);
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + length), std::vector<uint8_t>(c.bytes, c.bytes + 14));
   }
}

TEST(NativeForms, M5GemmMixedOperandExecutionWitnesses)
{
   agx3_matrix_gemm16_desc d = {};
   d.a = 24; d.b = 16; d.wait_mask = 128;
   for (unsigned swap = 0; swap < 2; ++swap) {
      d.input_type = swap ? AGX3_MATRIX_INPUT_F16 : AGX3_MATRIX_INPUT_BF16;
      d.b_input_type = swap ? AGX3_MATRIX_INPUT_BF16 : AGX3_MATRIX_INPUT_F16;
      const uint8_t expected[2][10] = {
         {0x2f,0x00,0x25,0x12,0x22,0x02,0xa0,0x42,0x00,0x04},
         {0x2f,0x00,0x25,0x12,0x22,0x02,0xa4,0x02,0x00,0x04},
      };
      uint8_t bytes[10], length;
      ASSERT_TRUE(agx3_matrix_encode_gemm16(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
      ASSERT_EQ(length, 10u);
      EXPECT_EQ(std::vector<uint8_t>(bytes, bytes + length), std::vector<uint8_t>(expected[swap], expected[swap] + 10));
   }
}

TEST(NativeForms, M5VectorAccumulatorCopiesEveryComponent)
{
   for (unsigned channels : {2u, 4u, 8u}) {
      for (unsigned dest : {8u, 34u, 64u}) {
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         agx_index src = agx_register(32, AGX_SIZE_32);
         agx_index dst = agx_register(dest, AGX_SIZE_32);
         src.channels_m1 = dst.channels_m1 = channels - 1;
         agx_mov_to(b, dst, src);
         agx_stop(b);
         agx_lower_pseudo(b->shader);
         uint32_t registers[128];
         for (unsigned i = 0; i < ARRAY_SIZE(registers); ++i)
            registers[i] = 0x10000000 + i;
         unsigned copies = 0;
         agx_foreach_instr_global(b->shader, I) {
            if (I->op == AGX_OPCODE_STOP)
               continue;
            ASSERT_EQ(I->op, AGX_OPCODE_BITOP);
            ASSERT_EQ(agx_channels(I->dest[0]), 1u);
            ASSERT_EQ(agx_channels(I->src[0]), 1u);
            registers[I->dest[0].value] = registers[I->src[0].value];
            ++copies;
         }
         EXPECT_EQ(copies, channels);
         for (unsigned c = 0; c < channels; ++c)
            EXPECT_EQ(registers[dest + 2 * c], 0x10000000u + 32 + 2 * c);
         util_dynarray bytes;
         util_dynarray_init(&bytes, mem);
         EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, M5QuadHelpersKeepFragmentEntryUnmarked)
{
   for (unsigned helpers : {0u, 1u, 2u}) {
      SCOPED_TRACE(helpers);
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;
      b->shader->nir->info.fs.needs_coarse_quad_helper_invocations = helpers == 1;
      b->shader->nir->info.fs.needs_full_quad_helper_invocations = helpers == 2;
      agx_interp_agx3_to(b, agx_register(0, AGX_SIZE_32), agx_immediate(1),
                         agx_immediate(0), AGX_INTERPOLATION_CENTER, false);
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      uint8_t expected[AGX3_VARY_MAX_LENGTH], len = 0;
      ASSERT_TRUE(agx3_vary_interp_at_start(AGX_ISA_ENCODING_AGX3C,
                     0, 1, false, AGX3_VARY_CENTER, helpers == 0, expected, &len));
      ASSERT_GE(binary.size, len);
      EXPECT_EQ(memcmp(binary.data, expected, len), 0);
      ralloc_free(mem);
   }
}


/* A G17P eMRT epilog shares one high-register payload across samples too
 * (vulkan-msaa-mrt rgba16f,r32f 2 4 load). */
TEST(NativeForms, MultisampleTileDataRemainsLiveUntilLastStore)
{
   for (bool g17p : {false, true})
   for (unsigned targets : {1u, 4u}) {
      SCOPED_TRACE(targets);
      SCOPED_TRACE(g17p);
      const enum agx_isa_encoding enc =
         g17p ? AGX_ISA_ENCODING_AGX3B : AGX_ISA_ENCODING_AGX3C;
      const unsigned data_reg = g17p ? 24 : 0;
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = enc;
      key.dev.chip = g17p ? AGX_CHIP_G17P : AGX_CHIP_G17S;
      key.g17p_emrt = g17p;
      key.secondary = true;
      key.no_preamble = true;
      key.fs.nr_samples = 4;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_FRAGMENT;
      b->shader->agx3_mrt_tile_ops_ordered = true;
      for (unsigned rt = 0; rt < targets; ++rt)
         BITSET_SET(key.g17p_tile_partitions, 4 * rt);
      agx_wait_pix(b, 12);
      std::vector<std::vector<uint8_t>> expected;
      for (unsigned sample = 0; sample < 4; ++sample) {
         for (unsigned target = targets; target-- > 0;) {
            const bool last = sample == 3 && target == 0;
            agx_index data = agx_register(data_reg, AGX_SIZE_16);
            data.channels_m1 = 1;
            data.discard = last;
            const agx_index coverage = agx_register(16 + sample, AGX_SIZE_16);
            agx_st_tile(b, data, coverage, agx_null(), AGX_FORMAT_U8NORM,
                        15, target * 4, false, target);
            uint8_t bytes[AGX3_TILE_MAX_LENGTH], length;
            ASSERT_TRUE(agx3_tile_store_f13(enc, data_reg,
               AGX3_TILE_W8, 15, target * 4,
               agx3_tile_sample_reg(16 + sample, false), 0, last,
               bytes, &length));
            expected.emplace_back(bytes, bytes + length);
         }
      }
      agx_stop(b);
      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
      const uint8_t *start = (const uint8_t *)binary.data;
      const uint8_t *end = start + binary.size;
      for (const auto &bytes : expected) {
         auto found = std::search(start, end, bytes.begin(), bytes.end());
         ASSERT_NE(found, end);
         start = found + bytes.size();
      }
      ralloc_free(mem);
   }
}

TEST(NativeForms, M5EmptyMeshTopology)
{
   for (auto primitive : {MESA_PRIM_POINTS, MESA_PRIM_LINES}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
         &agx_nir_options, "empty mesh with memory work");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.primitive_type = primitive;
      b.shader->info.mesh.max_vertices_out = 1;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.io_lowered = true;
      nir_def *lane = nir_load_local_invocation_index(&b);
      nir_store_global(&b, lane, nir_iadd(&b, nir_imm_int64(&b, 0x2000000),
                       nir_u2u64(&b, nir_imul_imm(&b, lane, 4))),
                       .write_mask=1, .align_mul=4);
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), nir_undef(&b, 1, 32));
      bool canonicalized = false;
      NIR_PASS(canonicalized, b.shader, agx_nir_m5_canonicalize_empty_mesh);
      ASSERT_TRUE(canonicalized);
      EXPECT_EQ(b.shader->info.mesh.primitive_type, MESA_PRIM_TRIANGLES);
      ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
      agx_preprocess_nir(b.shader);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17S;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.no_preamble = true;
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5EmptyMeshRejectsOutput)
{
   for (unsigned nonzero = 0; nonzero < 3; ++nonzero) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
         &agx_nir_options, "nonempty mesh");
      b.shader->info.mesh.primitive_type = MESA_PRIM_POINTS;
      if (nonzero == 2)
         b.shader->info.outputs_written = VARYING_BIT_POS;
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, nonzero == 0),
         nir_imm_int(&b, nonzero == 1), nir_undef(&b, 1, 32));
      EXPECT_FALSE(agx_nir_m5_canonicalize_empty_mesh(b.shader));
      EXPECT_EQ(b.shader->info.mesh.primitive_type, MESA_PRIM_POINTS);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5EmptyMeshRejectsConditionalCount)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "conditional mesh count");
   b.shader->info.mesh.primitive_type = MESA_PRIM_POINTS;
   EXPECT_FALSE(agx_nir_m5_canonicalize_empty_mesh(b.shader));
   nir_push_if(&b, nir_ieq_imm(&b, nir_load_local_invocation_index(&b), 0));
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0),
      nir_imm_int(&b, 0), nir_undef(&b, 1, 32));
   nir_pop_if(&b, NULL);
   EXPECT_FALSE(agx_nir_m5_canonicalize_empty_mesh(b.shader));
   EXPECT_EQ(b.shader->info.mesh.primitive_type, MESA_PRIM_POINTS);
   ralloc_free(b.shader);
}

TEST(NativeForms, M5MeshActiveSubgroupIndexReachesPacker)
{
   for (unsigned half : {0u, 1u, 14u, 63u, 254u}) {
      void *mem = ralloc_context(NULL);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      agx_builder *b = agx_test_builder(mem);
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_MESH;
      agx_get_sr_coverage_to(b, agx_register(half, AGX_SIZE_16),
                             AGX_SR_ACTIVE_THREAD_INDEX_IN_SUBGROUP);
      agx_stop(b);
      util_dynarray bytes;
      util_dynarray_init(&bytes, mem);
      EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
      EXPECT_GT(bytes.size, 4u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, M5MeshLargeWorkgroupsCompile)
{
   const unsigned sizes[][3] = {
      {129,1,1}, {256,1,1}, {512,1,1}, {1024,1,1},
      {1,1024,1}, {1,1,1024}, {3,5,7}, {1025,1,1},
   };
   for (const auto &size : sizes) {
      SCOPED_TRACE(size[0] * size[1] * size[2]);
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
         &agx_nir_options, "mesh workgroup bounds");
      for (unsigned i = 0; i < 3; ++i)
         b.shader->info.workgroup_size[i] = size[i];
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      b.shader->info.mesh.max_vertices_out = 1;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.io_lowered = true;
      nir_def *lane = nir_load_local_invocation_index(&b);
      nir_def *id = nir_load_local_invocation_id(&b);
      nir_def *value = nir_iadd(&b, lane, nir_channel(&b, id, 2));
      nir_store_global(&b, value, nir_iadd(&b, nir_imm_int64(&b, 0x2000000),
                       nir_u2u64(&b, nir_imul_imm(&b, lane, 4))),
                       .write_mask=1, .align_mul=4);
      nir_barrier(&b, .execution_scope=SCOPE_WORKGROUP,
         .memory_scope=SCOPE_WORKGROUP, .memory_semantics=NIR_MEMORY_ACQ_REL,
         .memory_modes=nir_var_mem_global);
      nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, 0),
         nir_imm_int(&b, 0), nir_undef(&b, 1, 32));
      bool valid = size[0] * size[1] * size[2] <= 1024;
      ASSERT_EQ(agx_nir_lower_m5_mesh_triangle_probe(b.shader), valid);
      if (!valid) {
         EXPECT_EQ(b.shader->info.workgroup_size[0], size[0]);
         ralloc_free(b.shader);
         continue;
      }
      EXPECT_EQ(b.shader->info.workgroup_size[2], size[2] > 1023 ? 1u : size[2]);
      agx_preprocess_nir_m5(b.shader);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17S;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.no_preamble = true;
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      free(part.binary);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5MeshImageReadStagesCoordinates)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_MESH;
   agx_instr *read = agx_alloc_instr(b, AGX_OPCODE_IMAGE_LOAD, 1, 6);
   read->dest[0] = agx_register(24, AGX_SIZE_32);
   read->dest[0].channels_m1 = 3;
   read->src[0] = agx_register(12, AGX_SIZE_32);
   read->src[0].channels_m1 = 1;
   read->src[1] = agx_immediate(0);
   read->src[2] = agx_immediate(0);
   read->src[3] = agx_register(18, AGX_SIZE_64);
   read->src[4] = agx_immediate(0);
   read->src[5] = agx_null();
   read->dim = AGX_DIM_2D;
   read->lod_mode = AGX_LOD_MODE_AUTO_LOD;
   read->mask = 1;
   agx_builder_insert(&b->cursor, read);
   agx_stop(b);
   util_dynarray bytes;
   util_dynarray_init(&bytes, mem);
   EXPECT_TRUE(agx3_pack_binary(b->shader, &bytes));
   EXPECT_GT(b->shader->agx3_staging_halfs, 0u);
   ralloc_free(mem);
}

TEST(NativeForms, M5MeshImageReadUsesNativeHeapIndex)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "mesh native image descriptor");
   nir_def *handle = agx_test_bindless_image_agx(&b, nir_imm_int(&b, 64), 2);
   nir_def *coord = nir_imm_ivec4(&b, 0, 0, 0, 0);
   nir_def *value = agx_test_bindless_image_load(&b, 4, 32, handle, coord,
      nir_imm_int(&b, 0), nir_imm_int(&b, 0), GLSL_SAMPLER_DIM_2D,
      PIPE_FORMAT_R32_UINT, ACCESS_COHERENT, nir_type_uint32);
   nir_store_global(&b, value, nir_imm_int64(&b, 0x2000000),
                    .write_mask=15, .align_mul=4);
   ASSERT_TRUE(agx_nir_lower_g17g_indexed_reads(b.shader));
   unsigned reads = 0;
   nir_foreach_block(block, b.impl) nir_foreach_instr(in, block) {
      if (in->type != nir_instr_type_intrinsic) continue;
      nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
      if (i->intrinsic != nir_intrinsic_bindless_image_load) continue;
      auto image = nir_src_as_intrinsic(i->src[0]);
      ASSERT_NE(image, nullptr);
      EXPECT_EQ(image->intrinsic, nir_intrinsic_bindless_image_agx);
      EXPECT_EQ(image->src[0].ssa->bit_size, 64u);
      EXPECT_EQ(nir_intrinsic_desc_set(image), 0u);
      reads++;
   }
   EXPECT_EQ(reads, 1u);
   nir_validate_shader(b.shader, "native mesh image index");
   ralloc_free(b.shader);
}

TEST(NativeForms, M5Independent3DSamplesUseSeparateStaging)
{
   for (auto stage : {MESA_SHADER_VERTEX, MESA_SHADER_COMPUTE}) {
      for (bool gradient : {false, true}) {
         const unsigned span = gradient ? (stage == MESA_SHADER_VERTEX ? 40u : 32u) : 8u;
         for (unsigned count : {2u, std::min(8u, 256u / span)}) {
         SCOPED_TRACE(count);
         SCOPED_TRACE(stage);
         SCOPED_TRACE(gradient);
         void *mem = ralloc_context(NULL);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.dev.chip = AGX_CHIP_G17S;
         agx_builder *b = agx_test_builder(mem);
         b->shader->key = &key;
         b->shader->stage = stage;
         b->shader->max_reg = 63;
         agx_index coords = agx_register(64, AGX_SIZE_32);
         coords.channels_m1 = 2;
         agx_index lod = agx_immediate(0);
         if (gradient) {
            lod = agx_register(72, AGX_SIZE_32);
            lod.channels_m1 = 5;
         }
         for (unsigned n = 0; n < count; ++n) {
            unsigned dest = 8 * n;
            agx_index value = agx_register(dest, AGX_SIZE_32);
            value.channels_m1 = 3;
            agx_texture_sample_to(b, value, coords, lod,
               agx_immediate(0), agx_immediate(0), agx_immediate(0),
               agx_null(), AGX_DIM_3D,
               gradient ? AGX_LOD_MODE_LOD_GRAD : AGX_LOD_MODE_LOD_MIN,
               15, false, false, false, AGX_GATHER_NONE, false);
         }
         agx_fadd_to(b, agx_register(96, AGX_SIZE_32),
                     agx_register(0, AGX_SIZE_32),
                     agx_register(8, AGX_SIZE_32));
         for (unsigned n = 2; n < count; ++n)
            agx_fadd_to(b, agx_register(96, AGX_SIZE_32),
                        agx_register(96, AGX_SIZE_32),
                        agx_register(8 * n, AGX_SIZE_32));
         agx_stop(b);
         util_dynarray binary;
         util_dynarray_init(&binary, mem);
         ASSERT_TRUE(agx3_pack_binary(b->shader, &binary));
         EXPECT_GE(b->shader->agx3_staging_halfs, count * span);
         uint8_t expected[AGX3_TEX_COPY_MAX_LENGTH], len;
         ASSERT_TRUE(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3C,
            (count - 1) * span + (gradient ? 8 : 0),
            gradient ? 72 : 64, count, 0,
            true, false, expected, &len));
         const auto *data = static_cast<const uint8_t *>(binary.data);
         bool found = false;
         for (unsigned i = 0; i + len <= binary.size; ++i)
            found |= memcmp(data + i, expected, len) == 0;
         EXPECT_TRUE(found);
         util_dynarray_fini(&binary);
         ralloc_free(mem);
         }
      }
   }
}

TEST(NativeForms, M5MeshWithoutOutputsKeepsMemoryWrites)
{
   for (unsigned threads : {1u, 32u}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
         &agx_nir_options, "memory-only mesh");
      b.shader->info.workgroup_size[0] = threads;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
      b.shader->info.mesh.max_vertices_out = 3;
      b.shader->info.mesh.max_primitives_out = 1;
      b.shader->info.io_lowered = true;
      nir_store_global(&b, nir_load_local_invocation_index(&b),
                       nir_imm_int64(&b, 0x2000000),
                       .write_mask=1, .align_mul=4);
      ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
      unsigned memory_writes = 0;
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_global)
                  memory_writes++;
            }
         }
      }
      EXPECT_EQ(memory_writes, 1u);
      ralloc_free(b.shader);
   }
}

TEST(NativeForms, M5MeshOutputStillRequiresCount)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "mesh output without count");
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.primitive_type = MESA_PRIM_TRIANGLES;
   b.shader->info.mesh.max_vertices_out = 3;
   b.shader->info.mesh.max_primitives_out = 1;
   b.shader->info.io_lowered = true;
   b.shader->info.outputs_written = VARYING_BIT_POS;
   EXPECT_FALSE(agx_nir_lower_m5_mesh_triangle_probe(b.shader));
   ralloc_free(b.shader);
}

/* Two triangles sharing vertices 1 and 2, a smooth per-vertex VAR1 and
 * per-primitive VAR0/viewport (and optionally layer) words. */
static nir_shader *
m5_mesh_per_primitive_shader(bool layer, unsigned max_primitives,
                             gl_varying_slot extra_primitive_slot)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_MESH,&agx_nir_options,"mesh per-primitive words");
   b.shader->info.workgroup_size[0]=4;b.shader->info.workgroup_size[1]=b.shader->info.workgroup_size[2]=1;
   b.shader->info.mesh.max_vertices_out=4;b.shader->info.mesh.max_primitives_out=max_primitives;
   b.shader->info.mesh.primitive_type=MESA_PRIM_TRIANGLES;b.shader->info.io_lowered=true;
   b.shader->info.outputs_written=VARYING_BIT_POS|VARYING_BIT_PRIMITIVE_INDICES|VARYING_BIT_VAR(0)|
      VARYING_BIT_VAR(1)|VARYING_BIT_VIEWPORT|(layer?VARYING_BIT_LAYER:0);
   b.shader->info.per_primitive_outputs=VARYING_BIT_PRIMITIVE_INDICES|VARYING_BIT_VAR(0)|
      VARYING_BIT_VIEWPORT|(layer?VARYING_BIT_LAYER:0);
   nir_def *lane=nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b,nir_imm_int(&b,4),nir_imm_int(&b,2),nir_imm_int(&b,0));
   nir_store_per_vertex_output(&b,nir_vec4(&b,nir_u2f32(&b,lane),nir_imm_float(&b,0),nir_imm_float(&b,0),
      nir_imm_float(&b,1)),lane,nir_imm_int(&b,0),.write_mask=15,.src_type=nir_type_float32,
      .io_semantics={.location=VARYING_SLOT_POS,.num_slots=1});
   nir_store_per_vertex_output(&b,nir_imm_vec2(&b,.5,.25),lane,nir_imm_int(&b,0),.write_mask=3,
      .src_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_VAR1,.num_slots=1});
   nir_push_if(&b,nir_ult_imm(&b,lane,2));
   nir_def *first=nir_imul_imm(&b,lane,2);
   nir_store_per_primitive_output(&b,nir_vec3(&b,first,nir_iadd_imm(&b,first,1),nir_iadd_imm(&b,first,2)),
      lane,nir_imm_int(&b,0),.write_mask=7,.src_type=nir_type_uint32,
      .io_semantics={.location=VARYING_SLOT_PRIMITIVE_INDICES,.num_slots=1});
   nir_store_per_primitive_output(&b,nir_vec4(&b,nir_u2f32(&b,lane),nir_imm_float(&b,1),nir_imm_float(&b,0),
      nir_imm_float(&b,1)),lane,nir_imm_int(&b,0),.write_mask=15,.src_type=nir_type_float32,
      .io_semantics={.location=VARYING_SLOT_VAR0,.num_slots=1});
   nir_store_per_primitive_output(&b,lane,lane,nir_imm_int(&b,0),.write_mask=1,.src_type=nir_type_int32,
      .io_semantics={.location=VARYING_SLOT_VIEWPORT,.num_slots=1});
   if (layer)
      nir_store_per_primitive_output(&b,nir_iadd_imm(&b,lane,3),lane,nir_imm_int(&b,0),.write_mask=1,
         .src_type=nir_type_int32,.io_semantics={.location=VARYING_SLOT_LAYER,.num_slots=1});
   if (extra_primitive_slot != VARYING_SLOT_POS)
      nir_store_per_primitive_output(&b,lane,lane,nir_imm_int(&b,0),.write_mask=1,.src_type=nir_type_int32,
         .io_semantics={.location=extra_primitive_slot,.num_slots=1});
   nir_pop_if(&b,NULL);
   return b.shader;
}

TEST(NativeForms, M5MeshPerPrimitiveWordsUsePrivateVertices)
{
   for (bool layer : {false, true}) {
      nir_shader *s=m5_mesh_per_primitive_shader(layer,2,VARYING_SLOT_POS);
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));
      /* Slot order: layer, viewport, VAR0, VAR1; then the packed
       * layer/viewport word, as the vertex stage's UVS layout. */
      EXPECT_TRUE(v.lv);EXPECT_TRUE(v.dynamic);
      EXPECT_EQ(v.components[VARYING_SLOT_LAYER],1u);EXPECT_EQ(v.first[VARYING_SLOT_LAYER],4u);
      EXPECT_EQ(v.primitive_comps[VARYING_SLOT_LAYER],layer?1u:0u);
      EXPECT_EQ(v.components[VARYING_SLOT_VIEWPORT],1u);EXPECT_EQ(v.first[VARYING_SLOT_VIEWPORT],5u);
      EXPECT_EQ(v.components[VARYING_SLOT_VAR0],4u);EXPECT_EQ(v.primitive_comps[VARYING_SLOT_VAR0],4u);
      EXPECT_EQ(v.first[VARYING_SLOT_VAR0],6u);
      EXPECT_EQ(v.components[VARYING_SLOT_VAR1],2u);EXPECT_EQ(v.primitive_comps[VARYING_SLOT_VAR1],0u);
      EXPECT_EQ(v.first[VARYING_SLOT_VAR1],10u);
      EXPECT_EQ(v.lv_first,12u);EXPECT_EQ(v.size,13u);EXPECT_EQ(v.primitive_stride_B,0u);
      /* Linked (declaration) order, so every plane has a fixed UVS index. */
      v.dynamic=false;
      ASSERT_TRUE(agx_nir_lower_m5_mesh(s,&v));
      /* Three private vertices per primitive. */
      EXPECT_EQ(s->info.mesh.max_vertices_out,6u);
      /* Index bytes 16, counts 16, export 13 planes x 32, vertex sources
       * 6 x 16, primitive sources (VAR0 4 + viewport 1 + layer) x 16. */
      const unsigned primitive_planes=5+(layer?1:0);
      EXPECT_EQ(s->info.shared_size,(32+13*32+6*16+primitive_planes*16+63)&~63u);
      unsigned index_words=0,vertex_planes=0,index_loads=0,u8_stores=0,pack=0,rendezvous=0,loops=0;
      nir_foreach_block(block,nir_shader_get_entrypoint(s))nir_foreach_instr(in,block) {
         if(in->type==nir_instr_type_alu&&nir_instr_as_alu(in)->op==nir_op_pack_32_2x16_split)pack++;
         if(in->type==nir_instr_type_intrinsic&&nir_instr_as_intrinsic(in)->intrinsic==nir_intrinsic_barrier&&
            nir_intrinsic_execution_scope(nir_instr_as_intrinsic(in))==SCOPE_WORKGROUP)rendezvous++;
         if(in->type!=nir_instr_type_intrinsic)continue;
         nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
         EXPECT_NE(i->intrinsic,nir_intrinsic_store_per_primitive_output);
         EXPECT_NE(i->intrinsic,nir_intrinsic_load_mesh_primitive_storage_agx);
         if(i->intrinsic==nir_intrinsic_load_shared&&i->def.bit_size==8)index_loads++;
         if(i->intrinsic==nir_intrinsic_store_shared&&i->src[0].ssa->bit_size==8)u8_stores++;
         if(i->intrinsic!=nir_intrinsic_store_mesh_uvb_agx)continue;
         if(nir_intrinsic_flags(i)){index_words++;continue;}
         vertex_planes++;
         ASSERT_TRUE(nir_src_is_const(i->src[0]));
         EXPECT_EQ(nir_src_comp_as_uint(i->src[0],1),32+32*(nir_src_comp_as_uint(i->src[0],0)&65535));
      }
      /* One 32-vertex batch of 13 planes and one index batch. SIMD group 0
       * copies between two rendezvous, without a loop. */
      foreach_list_typed(nir_cf_node,node,node,&nir_shader_get_entrypoint(s)->body)
         loops+=node->type==nir_cf_node_loop;
      EXPECT_EQ(index_words,1u);EXPECT_EQ(vertex_planes,13u);
      EXPECT_EQ(rendezvous,2u);EXPECT_EQ(loops,0u);
      /* The copy reads every index byte once and rewrites it. */
      EXPECT_EQ(index_loads,1u);EXPECT_GE(u8_stores,2u);EXPECT_EQ(pack,1u);
      nir_validate_shader(s,"mesh per-primitive private vertices");agx_preprocess_nir_m5(s);
      agx_shader_key key={};key.dev.chip=AGX_CHIP_G17S;key.dev.encoding=AGX_ISA_ENCODING_AGX3C;key.no_preamble=true;
      agx_shader_part part={};ASSERT_TRUE(agx_compile_shader_nir(s,&key,&part))<<agx_last_compile_refusal();
      free(part.binary);ralloc_free(s);
   }
}

/* gl_CullPrimitiveEXT: one byte per primitive, cleared before the body and
 * compacted away before the export. */
TEST(NativeForms, M5MeshCullPrimitiveCompacts)
{
   for (unsigned primitives : {2u, 85u}) {
      nir_shader *s=m5_mesh_per_primitive_shader(false,primitives,VARYING_SLOT_CULL_PRIMITIVE);
      s->info.outputs_written|=VARYING_BIT_CULL_PRIMITIVE;
      s->info.per_primitive_outputs|=VARYING_BIT_CULL_PRIMITIVE;
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));
      EXPECT_TRUE(v.cull_primitive);
      EXPECT_EQ(v.components[VARYING_SLOT_CULL_PRIMITIVE],0u);
      ASSERT_TRUE(agx_nir_lower_m5_mesh(s,&v));
      nir_validate_shader(s,"cull primitive");
      EXPECT_LE(s->info.shared_size,32768u);
      ralloc_free(s);
   }
}

TEST(NativeForms, M5MeshPerPrimitiveRefusalsLeaveShaderUntouched)
{
   /* No private-vertex form beyond the native 256 output vertices. */
   const struct { unsigned primitives; gl_varying_slot extra; bool gather; } rows[] = {
      {86, VARYING_SLOT_POS, true},
   };
   for (const auto &r : rows) {
      nir_shader *s=m5_mesh_per_primitive_shader(false,r.primitives,r.extra);
      struct agx_m5_mesh_varyings v;
      EXPECT_EQ(agx_m5_mesh_gather_varyings(s,&v),r.gather);
      if (r.gather) {
         EXPECT_FALSE(agx_nir_lower_m5_mesh(s,&v));
      }
      EXPECT_EQ(s->info.shared_size,0u);EXPECT_EQ(s->info.mesh.max_vertices_out,4u);
      ralloc_free(s);
   }
   /* 85 primitives fit: 255 private vertices. */
   nir_shader *s=m5_mesh_per_primitive_shader(false,85,VARYING_SLOT_POS);
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));
   ASSERT_TRUE(agx_nir_lower_m5_mesh(s,&v));
   EXPECT_EQ(s->info.mesh.max_vertices_out,255u);
   nir_validate_shader(s,"mesh 85 primitives");ralloc_free(s);
}

/* A mesh of `locations` per-primitive uvec4 words whose private vertices
 * do not fit: the shapes of mesh_shader.ext.smoke.*_lib (256 triangles),
 * misc.max_lines (255 lines) and properties.max_mesh_output_size_*_per_primitive
 * (96 points, 19 locations). */
static nir_shader *
m5_mesh_provoking_shader(mesa_prim primitive, unsigned vertices,
                         unsigned primitives, unsigned locations, bool cull)
{
   const unsigned corners =
      primitive == MESA_PRIM_POINTS ? 1 : primitive == MESA_PRIM_LINES ? 2 : 3;
   const bool points = primitive == MESA_PRIM_POINTS;
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_MESH, &agx_nir_options, "mesh provoking words");
   b.shader->info.io_lowered = true;
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
   b.shader->info.mesh.max_vertices_out = vertices;
   b.shader->info.mesh.max_primitives_out = primitives;
   b.shader->info.mesh.primitive_type = primitive;
   const uint64_t words = BITFIELD64_RANGE(VARYING_SLOT_VAR0, locations);
   b.shader->info.outputs_written = VARYING_BIT_POS |
      VARYING_BIT_PRIMITIVE_INDICES | words | (points ? VARYING_BIT_PSIZ : 0) |
      (cull ? VARYING_BIT_CULL_PRIMITIVE : 0);
   b.shader->info.per_primitive_outputs = VARYING_BIT_PRIMITIVE_INDICES | words |
      (cull ? VARYING_BIT_CULL_PRIMITIVE : 0);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_set_vertex_and_primitive_count(&b, nir_imm_int(&b, vertices),
                                      nir_imm_int(&b, primitives), nir_imm_int(&b, 0));
   nir_store_per_vertex_output(&b, nir_imm_vec4(&b, 0, 0, 0, 1), lane,
      nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_float32,
      .io_semantics = {.location = VARYING_SLOT_POS, .num_slots = 1});
   if (points)
      nir_store_per_vertex_output(&b, nir_imm_float(&b, 1), lane,
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_float32,
         .io_semantics = {.location = VARYING_SLOT_PSIZ, .num_slots = 1});
   nir_def *index[3] = {lane, nir_iadd_imm(&b, lane, 1), nir_iadd_imm(&b, lane, 2)};
   nir_store_per_primitive_output(&b, nir_vec(&b, index, corners), lane,
      nir_imm_int(&b, 0), .write_mask = BITFIELD_MASK(corners),
      .src_type = nir_type_uint32,
      .io_semantics = {.location = VARYING_SLOT_PRIMITIVE_INDICES, .num_slots = 1});
   for (unsigned l = 0; l < locations; ++l) {
      nir_store_per_primitive_output(&b, nir_imm_ivec4(&b, l, 1, 2, 3), lane,
         nir_imm_int(&b, 0), .write_mask = 15, .src_type = nir_type_uint32,
         .io_semantics = {.location = VARYING_SLOT_VAR0 + l, .num_slots = 1});
   }
   if (cull)
      nir_store_per_primitive_output(&b, nir_imm_false(&b), lane,
         nir_imm_int(&b, 0), .write_mask = 1, .src_type = nir_type_bool1,
         .io_semantics = {.location = VARYING_SLOT_CULL_PRIMITIVE, .num_slots = 1});
   return b.shader;
}

TEST(NativeForms, Gen1MeshExportFencesCoverGenericShapesAndPartialCounts)
{
   const struct {
      mesa_prim topology;
      unsigned x, y, z, vertices, primitives;
   } rows[] = {
      {MESA_PRIM_TRIANGLES, 3,1,1,3,1},
      {MESA_PRIM_LINES, 16,1,1,32,16},
      {MESA_PRIM_LINES, 32,1,1,64,32},
      {MESA_PRIM_LINES, 10,4,1,256,256},
      {MESA_PRIM_POINTS, 1,1,128,128,128},
      {MESA_PRIM_LINES, 128,1,1,256,128},
      {MESA_PRIM_POINTS, 32,1,1,33,1},
   };
   for (const auto &r : rows) {
      for (unsigned count : {0u, 1u, r.vertices}) {
         nir_shader *s = m5_mesh_provoking_shader(r.topology, r.vertices,
                                                  r.primitives, 0, false);
         s->info.workgroup_size[0]=r.x; s->info.workgroup_size[1]=r.y;
         s->info.workgroup_size[2]=r.z;
         nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
            nir_foreach_instr(in, block) {
               if (in->type != nir_instr_type_intrinsic ||
                   nir_instr_as_intrinsic(in)->intrinsic !=
                      nir_intrinsic_set_vertex_and_primitive_count)
                  continue;
               nir_builder b = nir_builder_at(nir_before_instr(in));
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
               nir_src_rewrite(&i->src[0], nir_imm_int(&b, count));
               nir_src_rewrite(&i->src[1], nir_imm_int(&b, MIN2(count, r.primitives)));
            }
         }
         struct agx_m5_mesh_varyings v;
         ASSERT_TRUE(agx_gen1_mesh_gather_varyings(s, &v));
         ASSERT_TRUE(agx_nir_lower_gen1_mesh(s, &v));
         nir_validate_shader(s, "gen1 mesh export phase scopes");
         unsigned phase=0, index_sites=0, vertex_sites=0;
         nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
            nir_foreach_instr(in, block) {
               if (in->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
               if (i->intrinsic == nir_intrinsic_fence_mesh_uvb_agx) {
                  EXPECT_EQ(nir_intrinsic_base(i), phase ?
                     AGX_MESH_EXPORT_FENCE_VERTEX_BATCH : AGX_MESH_EXPORT_FENCE_INDICES);
                  /* Each fence is in SIMD0's then-list, after the prior
                   * index/batch IF and before the NEXT vertex IF. This
                   * remains true at count0, count1 and the full limit. */
                  ASSERT_NE(block->cf_node.parent, nullptr);
                  EXPECT_EQ(block->cf_node.parent->type, nir_cf_node_if);
                  EXPECT_EQ(nir_cf_node_prev(&block->cf_node)->type, nir_cf_node_if);
                  EXPECT_EQ(nir_cf_node_next(&block->cf_node)->type, nir_cf_node_if);
                  ++phase;
               } else if (i->intrinsic == nir_intrinsic_store_mesh_uvb_agx) {
                  if (nir_intrinsic_flags(i)) {
                     EXPECT_EQ(phase, 0u); ++index_sites;
                  } else {
                     ASSERT_GT(phase, 0u); ++vertex_sites;
                     ASSERT_TRUE(nir_src_is_const(i->src[0]));
                     EXPECT_EQ(nir_src_comp_as_uint(i->src[0],0) >> 16,
                               32 * (phase - 1));
                  }
               }
            }
         }
         EXPECT_EQ(phase, DIV_ROUND_UP(r.vertices,32));
         const unsigned corners = r.topology == MESA_PRIM_POINTS ? 1 :
                                  r.topology == MESA_PRIM_LINES ? 2 : 3;
         EXPECT_EQ(index_sites, DIV_ROUND_UP(DIV_ROUND_UP(corners*r.primitives,4),32));
         EXPECT_GT(vertex_sites, 0u);
         ralloc_free(s);
      }
   }
   /* The separate G17S exporter has no new fence or changed protocol. */
   nir_shader *s = m5_mesh_provoking_shader(MESA_PRIM_POINTS,128,128,0,false);
   ASSERT_TRUE(agx_nir_lower_m5_mesh_triangle_probe(s));
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(in, block) {
         if (in->type == nir_instr_type_intrinsic)
            EXPECT_NE(nir_instr_as_intrinsic(in)->intrinsic, nir_intrinsic_fence_mesh_uvb_agx);
      }
   }
   ralloc_free(s);
}

static bool
m5_test_provoking_last(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   if (i->intrinsic != nir_intrinsic_load_provoking_last)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def_replace(&i->def, nir_imm_int(b, *(const bool *)data ? -1 : 0));
   return true;
}

/* Per-primitive words on the provoking vertex: the export slots are the API
 * vertices plus private provoking copies, the threadgroup memory holds each
 * per-primitive word once, and a line or triangle mesh reads the provoking
 * selector from the draw (load_provoking_last). */
TEST(NativeForms, M5MeshProvokingVertexCarriesPrimitiveWords)
{
   const struct {
      mesa_prim primitive;
      unsigned vertices, primitives, locations;
      bool cull;
      unsigned slots, shared;
   } rows[] = {
      /* 784 index/count bytes, 8 planes x 256 slots, 1 KiB slot map. */
      {MESA_PRIM_TRIANGLES, 256, 256, 1, false, 256, 10048},
      /* The same, then 256 cull bytes. */
      {MESA_PRIM_TRIANGLES, 256, 256, 1, true, 256, 10304},
      /* 528 index/count bytes, 8 planes x 256 slots, slot map. */
      {MESA_PRIM_LINES, 256, 255, 1, false, 256, 9792},
      /* 112 index/count bytes, 81 planes x 96 slots, slot map; private
       * vertices would need 62336 bytes. No room for private copies. */
      {MESA_PRIM_POINTS, 96, 96, 19, false, 96, 31616},
   };
   for (const auto &r : rows) {
      for (bool last : {false, true}) {
         nir_shader *s = m5_mesh_provoking_shader(r.primitive, r.vertices,
            r.primitives, r.locations, r.cull);
         struct agx_m5_mesh_varyings v;
         ASSERT_TRUE(agx_m5_mesh_gather_varyings(s, &v));
         EXPECT_FALSE(v.lv);
         EXPECT_EQ(v.cull_primitive, r.cull);
         v.dynamic = false;
         ASSERT_TRUE(agx_nir_lower_m5_mesh(s, &v)) << r.primitive;
         EXPECT_EQ(s->info.mesh.max_vertices_out, r.slots);
         EXPECT_EQ(s->info.shared_size, r.shared);
         EXPECT_EQ(s->info.workgroup_size[0], 32u);
         nir_validate_shader(s, "mesh provoking words");
         unsigned provoking = 0, uvst = 0, uvb = 0;
         nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
            nir_foreach_instr(in, block) {
               if (in->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
               EXPECT_NE(i->intrinsic, nir_intrinsic_store_per_primitive_output);
               EXPECT_NE(i->intrinsic, nir_intrinsic_store_per_vertex_output);
               EXPECT_NE(i->intrinsic, nir_intrinsic_load_mesh_primitive_storage_agx);
               provoking += i->intrinsic == nir_intrinsic_load_provoking_last;
               uvb += i->intrinsic == nir_intrinsic_allocate_mesh_uvb_agx;
               uvst += i->intrinsic == nir_intrinsic_store_mesh_uvb_agx &&
                       !nir_intrinsic_flags(i);
            }
         }
         /* A point is its own provoking vertex. */
         EXPECT_EQ(provoking, r.primitive == MESA_PRIM_POINTS ? 0u : 1u);
         EXPECT_EQ(uvb, 1u);
         EXPECT_EQ(uvst, DIV_ROUND_UP(r.slots, 32) * v.size);
         NIR_PASS(_, s, nir_shader_intrinsics_pass, m5_test_provoking_last,
                  nir_metadata_control_flow, &last);
         agx_preprocess_nir_m5(s);
         agx_shader_key key = {};
         key.dev.chip = AGX_CHIP_G17S;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.no_preamble = true;
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(s, &key, &part))
            << agx_last_compile_refusal();
         free(part.binary);
         ralloc_free(s);
      }
   }
}

/* Shapes whose private vertices fit keep them, with no provoking selector:
 * 85 triangles (255 private vertices) of per-primitive words. */
TEST(NativeForms, M5MeshFittingPrimitiveWordsKeepPrivateVertices)
{
   nir_shader *s = m5_mesh_provoking_shader(MESA_PRIM_TRIANGLES, 256, 85, 1, false);
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s, &v));
   ASSERT_TRUE(agx_nir_lower_m5_mesh(s, &v));
   EXPECT_EQ(s->info.mesh.max_vertices_out, 255u);
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(in, block) {
         if (in->type == nir_instr_type_intrinsic) {
            EXPECT_NE(nir_instr_as_intrinsic(in)->intrinsic,
                      nir_intrinsic_load_provoking_last);
         }
      }
   }
   nir_validate_shader(s, "mesh fitting primitive words");
   ralloc_free(s);
}

static bool
m5_test_root_address(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   (void)data;
   if (i->intrinsic != nir_intrinsic_load_root_agx)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def_replace(&i->def, nir_imm_int64(b, 0x100000));
   return true;
}

struct m5_line_bias_counts {
   unsigned roots, root_B, fdiv, ffma, rendezvous;
};

static m5_line_bias_counts
m5_test_line_bias_counts(nir_shader *s)
{
   m5_line_bias_counts n = {};
   nir_foreach_block(block, nir_shader_get_entrypoint(s)) {
      nir_foreach_instr(in, block) {
         if (in->type == nir_instr_type_alu) {
            n.fdiv += nir_instr_as_alu(in)->op == nir_op_fdiv;
            n.ffma += nir_instr_as_alu(in)->op == nir_op_ffma;
            continue;
         }
         if (in->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(in);
         n.rendezvous += i->intrinsic == nir_intrinsic_barrier &&
                         nir_intrinsic_execution_scope(i) == SCOPE_WORKGROUP;
         if (i->intrinsic != nir_intrinsic_load_root_agx)
            continue;
         n.roots++;
         nir_foreach_use(use, &i->def) {
            nir_alu_instr *add = nir_instr_as_alu(nir_src_use_instr(use));
            EXPECT_EQ(add->op, nir_op_iadd);
            for (unsigned k = 0; k < 2; ++k) {
               if (nir_src_is_const(add->src[k].src))
                  n.root_B = nir_src_as_uint(add->src[k].src);
            }
         }
      }
   }
   return n;
}

/* Lines drawn toward +y move by the root's row step (m5_mesh_line_bias) in
 * every output form, which stays as it is: the same slot count, and the
 * same threadgroup memory but for the mark bytes. The welded and
 * private-vertex forms append one mark byte per API vertex; the
 * provoking-vertex form of misc.max_lines (255 lines of a per-primitive
 * colour) keeps them in its slot map. */
TEST(NativeForms, M5MeshLineRowBias)
{
   const unsigned root_B = 0x2f4;
   const struct {
      unsigned vertices, primitives, locations;
      bool cull;
      unsigned slots, marks, shared;
   } rows[] = {
      /* Welded: no per-primitive words. */
      {256, 255, 0, false, 256, 256, 0},
      /* 64 private vertices, cull bytes, then the marks. */
      {64, 32, 1, true, 64, 64, 0},
      /* misc.max_lines: 9792 bytes with or without the bias. */
      {256, 255, 1, false, 256, 0, 9792},
   };
   for (const auto &r : rows) {
      for (bool last : {false, true}) {
         nir_shader *plain = m5_mesh_provoking_shader(MESA_PRIM_LINES,
            r.vertices, r.primitives, r.locations, r.cull);
         nir_shader *s = m5_mesh_provoking_shader(MESA_PRIM_LINES,
            r.vertices, r.primitives, r.locations, r.cull);
         struct agx_m5_mesh_varyings v;
         ASSERT_TRUE(agx_m5_mesh_gather_varyings(plain, &v));
         v.dynamic = false;
         ASSERT_TRUE(agx_nir_lower_m5_mesh(plain, &v));
         ASSERT_TRUE(agx_m5_mesh_gather_varyings(s, &v));
         v.dynamic = false;
         v.line_bias_root_B = root_B;
         ASSERT_TRUE(agx_nir_lower_m5_mesh(s, &v)) << r.locations;
         nir_validate_shader(s, "mesh line row bias");
         EXPECT_EQ(s->info.mesh.max_vertices_out, r.slots);
         EXPECT_EQ(plain->info.mesh.max_vertices_out, r.slots);
         EXPECT_EQ(s->info.shared_size, plain->info.shared_size + r.marks);
         if (r.shared) {
            EXPECT_EQ(s->info.shared_size, r.shared);
         }
         EXPECT_EQ(s->info.workgroup_size[0], plain->info.workgroup_size[0]);
         /* One root load at the given offset and two more rendezvous. The
          * passes are unrolled per workgroup-sized chunk, as gen1's are per
          * 32 lanes: two NDC divisions per chunk of primitives and one move
          * per chunk of vertices. */
         const unsigned lanes = s->info.workgroup_size[0] *
                                s->info.workgroup_size[1] *
                                s->info.workgroup_size[2];
         ASSERT_GT(lanes, 0u);
         const m5_line_bias_counts a = m5_test_line_bias_counts(plain);
         const m5_line_bias_counts n = m5_test_line_bias_counts(s);
         EXPECT_EQ(a.roots, 0u);
         EXPECT_EQ(n.roots, 1u);
         EXPECT_EQ(n.root_B, root_B);
         EXPECT_EQ(n.fdiv, a.fdiv + 2 * DIV_ROUND_UP(r.primitives, lanes));
         EXPECT_EQ(n.ffma, a.ffma + DIV_ROUND_UP(r.vertices, lanes));
         EXPECT_EQ(n.rendezvous, a.rendezvous + 2);
         ralloc_free(plain);
         NIR_PASS(_, s, nir_shader_intrinsics_pass, m5_test_root_address,
                  nir_metadata_control_flow, NULL);
         NIR_PASS(_, s, nir_shader_intrinsics_pass, m5_test_provoking_last,
                  nir_metadata_control_flow, &last);
         agx_preprocess_nir_m5(s);
         agx_shader_key key = {};
         key.dev.chip = AGX_CHIP_G17S;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.no_preamble = true;
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(s, &key, &part))
            << agx_last_compile_refusal();
         free(part.binary);
         ralloc_free(s);
      }
   }
   /* Triangles and points take no row bias: they lower as without it. */
   for (auto primitive : {MESA_PRIM_TRIANGLES, MESA_PRIM_POINTS}) {
      nir_shader *plain = m5_mesh_provoking_shader(primitive, 96, 32, 1, false);
      nir_shader *s = m5_mesh_provoking_shader(primitive, 96, 32, 1, false);
      struct agx_m5_mesh_varyings v;
      ASSERT_TRUE(agx_m5_mesh_gather_varyings(plain, &v));
      ASSERT_TRUE(agx_nir_lower_m5_mesh(plain, &v));
      ASSERT_TRUE(agx_m5_mesh_gather_varyings(s, &v));
      v.line_bias_root_B = root_B;
      ASSERT_TRUE(agx_nir_lower_m5_mesh(s, &v)) << primitive;
      EXPECT_EQ(s->info.shared_size, plain->info.shared_size);
      EXPECT_EQ(s->info.mesh.max_vertices_out, plain->info.mesh.max_vertices_out);
      EXPECT_EQ(m5_test_line_bias_counts(s).roots, 0u);
      EXPECT_EQ(m5_test_line_bias_counts(s).rendezvous,
                m5_test_line_bias_counts(plain).rendezvous);
      ralloc_free(plain);
      ralloc_free(s);
   }
}

TEST(NativeForms, M5MeshPrimitiveIdIsAPrimitiveWord)
{
   /* A mesh-written gl_PrimitiveID is a flat user word of the primitive's
    * private vertices, which the fragment shader's gl_PrimitiveID links to. */
   nir_shader *s=m5_mesh_per_primitive_shader(false,2,VARYING_SLOT_PRIMITIVE_ID);
   s->info.outputs_written|=VARYING_BIT_PRIMITIVE_ID;
   s->info.per_primitive_outputs|=VARYING_BIT_PRIMITIVE_ID;
   struct agx_m5_mesh_varyings v;
   ASSERT_TRUE(agx_m5_mesh_gather_varyings(s,&v));
   EXPECT_EQ(v.primitive_comps[VARYING_SLOT_PRIMITIVE_ID],1u);
   EXPECT_EQ(v.components[VARYING_SLOT_PRIMITIVE_ID],1u);
   ASSERT_TRUE(agx_nir_lower_m5_mesh(s,&v));
   EXPECT_EQ(s->info.mesh.max_vertices_out,6u);
   nir_validate_shader(s,"mesh primitive id");ralloc_free(s);
}

TEST(NativeForms, M5MeshFragmentReadsPrimitiveWordsFlat)
{
   nir_builder b=nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT,&agx_nir_options,"fs per-primitive inputs");
   b.shader->info.io_lowered=true;
   b.shader->info.per_primitive_inputs=VARYING_BIT_VAR(0)|VARYING_BIT_VAR(1)|VARYING_BIT_VIEWPORT|
                                       VARYING_BIT_PRIMITIVE_ID;
   nir_def *color=nir_load_per_primitive_input(&b,4,32,nir_imm_int(&b,0),.component=0,
      .dest_type=nir_type_float32,.io_semantics={.location=VARYING_SLOT_VAR0,.num_slots=1});
   nir_def *half=nir_load_per_primitive_input(&b,2,16,nir_imm_int(&b,0),.component=1,
      .dest_type=nir_type_float16,.io_semantics={.location=VARYING_SLOT_VAR1,.num_slots=1});
   nir_def *viewport=nir_load_per_primitive_input(&b,1,32,nir_imm_int(&b,0),
      .dest_type=nir_type_int32,.io_semantics={.location=VARYING_SLOT_VIEWPORT,.num_slots=1});
   nir_def *id=nir_load_per_primitive_input(&b,1,32,nir_imm_int(&b,0),
      .dest_type=nir_type_int32,.io_semantics={.location=VARYING_SLOT_PRIMITIVE_ID,.num_slots=1});
   nir_store_global(&b,nir_vec4(&b,nir_channel(&b,color,0),nir_f2f32(&b,nir_channel(&b,half,1)),
      nir_i2f32(&b,viewport),nir_i2f32(&b,id)),nir_imm_int64(&b,0x1000),.write_mask=15,.align_mul=16);
   ASSERT_TRUE(agx_nir_lower_m5_mesh_fragment(b.shader));
   unsigned flat=0,primitive=0,narrowed=0;
   nir_foreach_block(block,b.impl)nir_foreach_instr(in,block) {
      if(in->type==nir_instr_type_alu&&nir_instr_as_alu(in)->op==nir_op_f2f16)narrowed++;
      if(in->type!=nir_instr_type_intrinsic)continue;
      nir_intrinsic_instr *i=nir_instr_as_intrinsic(in);
      if(i->intrinsic==nir_intrinsic_load_per_primitive_input)
         primitive++;
      if(i->intrinsic==nir_intrinsic_load_input) {
         flat++;EXPECT_EQ(i->def.bit_size,32u);
         EXPECT_EQ(nir_alu_type_get_type_size(nir_intrinsic_dest_type(i)),32u);
         if(nir_intrinsic_io_semantics(i).location==VARYING_SLOT_VAR1) {
            EXPECT_EQ(nir_intrinsic_component(i),1u);
         }
      }
   }
   EXPECT_EQ(flat,4u);EXPECT_EQ(primitive,0u);EXPECT_EQ(narrowed,1u);
   EXPECT_EQ(b.shader->info.per_primitive_inputs,0u);
   nir_validate_shader(b.shader,"fs per-primitive inputs");
   EXPECT_FALSE(agx_nir_lower_m5_mesh_fragment(b.shader));
   ralloc_free(b.shader);
}

TEST(NativeForms, M5UniformDifferenceHonorsNIRSignedZero)
{
   std::vector<uint8_t> programs[2];
   for (unsigned strict = 0; strict < 2; ++strict) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "uniform difference zero");
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = b.shader->info.workgroup_size[2] = 1;
      b.fp_math_ctrl = nir_fp_exact |
         (strict ? nir_fp_preserve_signed_zero : 0);
      nir_def *u = agx_test_load_preamble(&b, 1, 32, 12);
      nir_def *r = nir_u2f32(&b, nir_load_local_invocation_index(&b));
      nir_def *difference = nir_fsub(&b, u, r);
      nir_def *scale = agx_test_load_preamble(&b, 1, 32, 14);
      nir_def *product = nir_fmul(&b, difference, scale);
      nir_store_global(&b, product, nir_imm_int64(&b, 0x2000000),
                       .write_mask = 1, .align_mul = 4);
      agx_preprocess_nir_m5(b.shader);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17S;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
         << agx_last_compile_refusal();
      const uint8_t *bytes = static_cast<const uint8_t *>(part.binary);
      programs[strict].assign(bytes, bytes + part.info.main_size);
      free(part.binary);
      ralloc_free(b.shader);
   }
   EXPECT_NE(programs[1], programs[0]);
}

TEST(NativeForms, M5OrSelectTracksVectorConversionComponent)
{
   for (unsigned bits : {16u, 32u}) {
      for (unsigned conditions = 0; conditions < 8; ++conditions) {
         for (unsigned component = 0; component < 2; ++component) {
            for (unsigned selected = 0; selected < 2; ++selected) {
               for (bool swapped : {false, true}) {
                  for (bool inverted : {false, true}) {
                     SCOPED_TRACE(::testing::Message() << bits << ":" << conditions
                        << ":" << component << ":" << selected << ":"
                        << swapped << ":" << inverted);
                     nir_builder b = nir_builder_init_simple_shader(
                        MESA_SHADER_COMPUTE, &agx_nir_options, "vector b2f component");
                     nir_def *a = nir_ieq(&b,
                        nir_imm_ivec2(&b, conditions & 1, (conditions >> 1) & 1),
                        nir_imm_int(&b, 1));
                     nir_def *other = nir_ieq(&b, nir_imm_int(&b, conditions >> 2),
                                              nir_imm_int(&b, 1));
                     nir_def *condition = nir_ior(&b, nir_channel(&b, a, component), other);
                     nir_alu_instr *ior = nir_def_as_alu(condition);
                     nir_src_rewrite(&ior->src[0].src, a);
                     ior->src[0].swizzle[0] = component;
                     nir_def *boolean = inverted ? nir_inot(&b, a) : a;
                     nir_def *converted = bits == 16 ? nir_b2f16(&b, boolean)
                                                     : nir_b2f32(&b, boolean);
                     nir_alu_instr *b2f = nir_def_as_alu(converted);
                     b2f->src[0].swizzle[0] = swapped ? 1 : 0;
                     b2f->src[0].swizzle[1] = swapped ? 0 : 1;
                     nir_def *value = nir_bcsel(&b, condition,
                        nir_channel(&b, converted, selected), nir_imm_floatN_t(&b, 3.0, bits));
                     nir_alu_instr *sel = nir_def_as_alu(value);
                     nir_src_rewrite(&sel->src[1].src, converted);
                     sel->src[1].swizzle[0] = selected;
                     nir_intrinsic_instr *store = nir_store_global(&b,
                        bits == 16 ? nir_f2f32(&b, value) : value,
                        nir_imm_int64(&b, 0x2000000), .write_mask = 1, .align_mul = 4);
                     nir_validate_shader(b.shader, "vector conversion before OR select");
                     unsigned source_component = selected ^ unsigned(swapped);
                     EXPECT_EQ(agx_nir_m5_split_or_selects(b.shader),
                               source_component == component);
                     nir_validate_shader(b.shader, "vector conversion after OR select");
                     while (nir_opt_constant_folding(b.shader)) {}
                     ASSERT_TRUE(nir_src_is_const(store->src[0]));
                     bool selected_bool = ((conditions >> source_component) & 1) ^ inverted;
                     bool taken = ((conditions >> component) & 1) || (conditions & 4);
                     EXPECT_FLOAT_EQ(nir_src_as_float(store->src[0]),
                                     taken ? float(selected_bool) : 3.0f);
                     ralloc_free(b.shader);
                  }
               }
            }
         }
      }
   }
}

TEST(NativeForms, G17PHalfImmediateMultiplyReachesPacker)
{
   for (unsigned multiplier : {6u, 7u, 9u, 11u, 17u, 255u}) {
      for (bool constant_first : {false, true}) {
         SCOPED_TRACE(multiplier);
         SCOPED_TRACE(constant_first);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
         key.dev.chip = AGX_CHIP_G17P;
         b->shader->key = &key;
         b->shader->stage = MESA_SHADER_COMPUTE;
         b->shader->max_reg = 15;
         agx_index input = agx_register(2, AGX_SIZE_16);
         agx_index imm = agx_immediate(multiplier);
         agx_instr *mul = agx_imad_to(
            b, agx_register(4, AGX_SIZE_16), constant_first ? imm : input,
            constant_first ? input : imm, agx_zero(), 0);
         agx_materialize_agx3_alu_constants(b->shader);
         ASSERT_TRUE(agx_is_equiv(mul->src[0], input));
         ASSERT_EQ(mul->src[1].type, AGX_INDEX_NORMAL);
         agx_instr *constant = NULL;
         agx_foreach_instr_global(b->shader, I) {
            if (I->nr_dests && agx_is_equiv(I->dest[0], mul->src[1]))
               constant = I;
         }
         ASSERT_NE(constant, nullptr);
         ASSERT_EQ(constant->op, AGX_OPCODE_MOV_IMM);
         EXPECT_EQ(constant->imm, multiplier);
         EXPECT_EQ(constant->dest[0].size, AGX_SIZE_16);
         /* Stand in for allocation, then check the real register-multiply
          * encoding rather than accepting only a change in the IR. */
         constant->dest[0] = agx_register(6, AGX_SIZE_16);
         mul->src[1] = constant->dest[0];
         agx_stop(b);
         util_dynarray binary;
         util_dynarray_init(&binary, mem);
         EXPECT_TRUE(agx3_pack_binary(b->shader, &binary));
         util_dynarray_fini(&binary);
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, HalfMultiplyMaterializationPreservesOtherDomains)
{
   for (unsigned control = 0; control < 3; ++control) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = control == 0 ? AGX_ISA_ENCODING_AGX2 : AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = control == 0 ? AGX_CHIP_G13G : AGX_CHIP_G17P;
      b->shader->key = &key;
      agx_size size = control == 1 ? AGX_SIZE_32 : AGX_SIZE_16;
      agx_instr *mul = agx_imad_to(b, agx_register(4, size),
         agx_register(2, size), agx_immediate(7),
         control == 2 ? agx_immediate(1) : agx_zero(), 0);
      agx_materialize_agx3_alu_constants(b->shader);
      EXPECT_EQ(mul->src[1].type, AGX_INDEX_IMMEDIATE);
      EXPECT_EQ(mul->src[1].value, 7u);
      ralloc_free(mem);
   }
}

TEST(NativeForms, G17PSignExtendRetainsZeroExtendedWord)
{
   for (bool multiply : {false, true}) {
      for (bool negate : {false, true}) {
         for (agx_size size : {AGX_SIZE_32, AGX_SIZE_64}) {
            SCOPED_TRACE(multiply);
            SCOPED_TRACE(negate);
            SCOPED_TRACE(size);
            void *mem = ralloc_context(NULL);
            agx_builder *b = agx_test_builder(mem);
            agx_shader_key key = {};
            key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
            key.dev.chip = AGX_CHIP_G17P;
            b->shader->key = &key;
            agx_index word = agx_temp(b->shader, AGX_SIZE_32);
            agx_signext_to(b, word, agx_register(0, AGX_SIZE_16));
            agx_index source = agx_abs(word);
            source.neg = negate;
            agx_index out = agx_temp(b->shader, size);
            agx_instr *use = multiply
               ? agx_imad_to(b, out, source, agx_register(2, AGX_SIZE_32),
                             agx_zero(), 0)
               : agx_iadd_to(b, out, source, agx_register(2, size), 0);
            agx_unit_test(b, out);
            agx_optimizer_forward(b->shader);
            EXPECT_TRUE(agx_is_equiv(use->src[0], word));
            EXPECT_EQ(use->src[0].size, AGX_SIZE_32);
            EXPECT_TRUE(use->src[0].abs);
            EXPECT_EQ(use->src[0].neg, negate);
            ralloc_free(mem);
         }
      }
   }
}

/* The class-12 wait that begin_invocation_interlock emits orders the critical
 * section, so it survives in a fragment shader with no tile access. */
TEST(NativeForms, InterlockWaitSurvivesUnreadPixelWaitMerge)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_interlock = true;
   agx_instr *wait = agx_wait_pix(b, 0xc);
   agx_stop(b);
   EXPECT_FALSE(agx3_merge_unread_pixel_waits(b->shader));
   unsigned waits = 0;
   agx_foreach_instr_global(b->shader, I)
      waits += I == wait && I->writeout == 0xc;
   EXPECT_EQ(waits, 1u);
   ralloc_free(mem);
}

/* G13 and G14 hit the same fold: before #232 an abs (zero-extend) source on a
 * SIGNEXT def was narrowed to the half, an assert in debug builds and a wrong
 * value for negative halves in release builds. */
TEST(NativeForms, AGX2SignExtendRetainsZeroExtendedWord)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G14G}) {
      for (bool multiply : {false, true}) {
         SCOPED_TRACE(chip);
         SCOPED_TRACE(multiply);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX2;
         key.dev.chip = chip;
         b->shader->key = &key;
         agx_index word = agx_temp(b->shader, AGX_SIZE_32);
         agx_signext_to(b, word, agx_register(0, AGX_SIZE_16));
         agx_index out = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *use =
            multiply ? agx_imad_to(b, out, agx_abs(word),
                                   agx_register(2, AGX_SIZE_32), agx_zero(), 0)
                     : agx_iadd_to(b, out, agx_abs(word),
                                   agx_register(2, AGX_SIZE_32), 0);
         agx_unit_test(b, out);
         agx_optimizer_forward(b->shader);
         EXPECT_TRUE(agx_is_equiv(use->src[0], word));
         EXPECT_EQ(use->src[0].size, AGX_SIZE_32);
         EXPECT_TRUE(use->src[0].abs);
         ralloc_free(mem);
      }
   }
}

TEST(NativeForms, SignExtendSignedConsumersStillFold)
{
   for (enum agx_chip chip : {AGX_CHIP_G13G, AGX_CHIP_G17P}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = chip == AGX_CHIP_G13G ? AGX_ISA_ENCODING_AGX2 : AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = chip;
      b->shader->key = &key;
      agx_index input = agx_register(0, AGX_SIZE_16);
      agx_index word = agx_temp(b->shader, AGX_SIZE_32);
      agx_signext_to(b, word, input);
      agx_index out = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *use = agx_iadd_to(b, out, word, agx_register(2, AGX_SIZE_32), 0);
      agx_unit_test(b, out);
      agx_optimizer_forward(b->shader);
      EXPECT_TRUE(agx_is_equiv(use->src[0], input));
      EXPECT_FALSE(use->src[0].abs);
      ralloc_free(mem);
   }
}

/* The VDM-launched G17P mesh program reads textures with the indexed forms of
 * the other pre-raster stages (mesh_shader.ext.synchronization.*_to_mesh):
 * an explicit-level sample packs as in a vertex program, and an image read
 * stages its coordinates as in a kernel. */
static std::vector<uint8_t>
g17p_pack_texture_read(mesa_shader_stage stage, bool sample,
                       bool register_lod = false)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = stage;
   agx_index coords = agx_register(12, AGX_SIZE_32);
   coords.channels_m1 = 1;
   agx_index result = agx_register(24, AGX_SIZE_32);
   result.channels_m1 = 3;
   if (sample) {
      agx_index lod = register_lod ? agx_register(14, AGX_SIZE_16)
                                   : agx_immediate(0);
      agx_texture_sample_to(b, result, coords, lod, agx_zero(),
                            agx_register(16, AGX_SIZE_64),
                            agx_register(20, AGX_SIZE_64), agx_null(),
                            AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 1, false,
                            false, false, AGX_GATHER_NONE, false);
   } else {
      agx_instr *read = agx_alloc_instr(b, AGX_OPCODE_IMAGE_LOAD, 1, 6);
      read->dest[0] = result;
      read->src[0] = coords;
      read->src[1] = agx_immediate(0);
      read->src[2] = agx_immediate(0);
      read->src[3] = agx_register(18, AGX_SIZE_64);
      read->src[4] = agx_immediate(0);
      read->src[5] = agx_null();
      read->dim = AGX_DIM_2D;
      read->lod_mode = AGX_LOD_MODE_AUTO_LOD;
      read->mask = 1;
      agx_builder_insert(&b->cursor, read);
   }
   agx_stop(b);
   util_dynarray bytes;
   util_dynarray_init(&bytes, mem);
   std::vector<uint8_t> out;
   if (agx3_pack_binary(b->shader, &bytes)) {
      const uint8_t *data = static_cast<const uint8_t *>(bytes.data);
      out.assign(data, data + bytes.size);
   }
   ralloc_free(mem);
   return out;
}

TEST(NativeForms, G17PMeshTextureReadsUsePreRasterForms)
{
   for (bool sample : {false, true}) {
      SCOPED_TRACE(sample);
      const std::vector<uint8_t> mesh =
         g17p_pack_texture_read(MESA_SHADER_MESH, sample);
      ASSERT_FALSE(mesh.empty()) << agx_last_compile_refusal();
      EXPECT_EQ(mesh, g17p_pack_texture_read(
         sample ? MESA_SHADER_VERTEX : MESA_SHADER_COMPUTE, sample));
   }
}

/* A per-lane explicit level in a register (Astro Bot's mesh shaders) packs
 * in a mesh program as in a vertex program. */
TEST(NativeForms, G17PMeshRegisterLodSampleUsesVertexForm)
{
   const std::vector<uint8_t> mesh =
      g17p_pack_texture_read(MESA_SHADER_MESH, true, true);
   ASSERT_FALSE(mesh.empty()) << agx_last_compile_refusal();
   EXPECT_EQ(mesh, g17p_pack_texture_read(MESA_SHADER_VERTEX, true, true));
}

/* G17P has no task stage, so a mesh payload read is never executed
 * (mesh_shader.ext.misc.payload_not_accessed) and needs no payload memory. */
TEST(NativeForms, G17PMeshPayloadReadIsUndefined)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_MESH,
      &agx_nir_options, "gen1 mesh payload read");
   b.shader->info.task_payload_size = 16;
   nir_def *value = nir_load_task_payload(&b, 1, 32, nir_imm_int(&b, 12));
   nir_store_global(&b, value, nir_imm_int64(&b, 0x2000000),
                    .write_mask = 1, .align_mul = 4);
   ASSERT_TRUE(agx_nir_lower_gen1_task_payload_probe(b.shader));
   nir_foreach_block(block, b.impl) nir_foreach_instr(in, block) {
      if (in->type != nir_instr_type_intrinsic)
         continue;
      const nir_intrinsic_op op = nir_instr_as_intrinsic(in)->intrinsic;
      EXPECT_NE(op, nir_intrinsic_load_task_payload);
      EXPECT_NE(op, nir_intrinsic_load_mesh_payload_address_agx);
   }
   nir_validate_shader(b.shader, "gen1 mesh payload read");
   ralloc_free(b.shader);
}

/*
 * The 8x8 cooperative-matrix multiply-add exists on the AGX3B ISA (G15/G16 and
 * G17P) as well as on M5's AGX3C. These four f32matmad8x8 witnesses are real
 * Apple G16G captures carried verbatim in the AGX3B half of the compose corpus
 * (agx3_compose_corpus_vectors.c entries by74/by95/by96/by98), spanning the
 * c-is-zero and c-present forms with both the destination cache hint and the
 * operand discard hints. agx3_matrix_encode_mma8() must reproduce each one
 * byte-for-byte when asked for the AGX3B encoding.
 *
 * by74 and by95 carry a nonzero scoreboard wait: their field 74 (wait slot 5)
 * is set, i.e. wait_mask 0x20. Field 74 is the sixth wait bit, not a c-is-zero
 * marker, so the descriptor must request that wait explicitly; by96/by98 have
 * field 74 clear (wait_mask 0).
 */
TEST(NativeForms, Agx3bMatmad8x8MatchesAppleG16G)
{
   const struct {
      const char *tag;
      struct agx3_matrix_mma8_desc desc;
      std::vector<uint8_t> bytes; /* real Apple G16G capture */
   } cases[] = {
      {"by74 c_zero dest4 wait_sb5",
       {.dest = 4, .a = 4, .b = 8, .c_zero = true,
        .discard = {true, true, false}, .wait_mask = 0x20,
        .input_type = AGX3_MATRIX_INPUT_F32},
       {207, 2, 86, 2, 0, 4, 8, 0, 212, 65, 36, 0}},
      {"by95 c_zero dest12 cache wait_sb5",
       {.dest = 12, .a = 4, .b = 8, .c_zero = true, .dest_cache = true,
        .cache = {true, true, false}, .wait_mask = 0x20,
        .input_type = AGX3_MATRIX_INPUT_F32},
       {207, 2, 86, 6, 1, 132, 9, 0, 84, 64, 36, 0}},
      {"by96 c_present dest12 cache+discardC",
       {.dest = 12, .a = 4, .b = 8, .c = 12, .dest_cache = true,
        .cache = {true, true, false}, .discard = {false, false, true},
        .input_type = AGX3_MATRIX_INPUT_F32},
       {207, 2, 84, 6, 1, 132, 9, 9, 84, 66, 36, 1}},
      {"by98 c_present dest4 discardAll",
       {.dest = 4, .a = 4, .b = 8, .c = 12, .discard = {true, true, true},
        .input_type = AGX3_MATRIX_INPUT_F32},
       {207, 2, 84, 2, 0, 4, 8, 9, 212, 67, 36, 1}},
   };
   for (const auto &c : cases) {
      uint8_t out[14], length = 0;
      ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &c.desc, out,
                                          &length)) << c.tag;
      EXPECT_EQ(length, c.bytes.size()) << c.tag;
      EXPECT_EQ(std::vector<uint8_t>(out, out + length), c.bytes) << c.tag;
   }
}

/*
 * AGX3B f16matmad8x8 (f16 A, B, C and result), in the form the compiler emits
 * (C in a register): the form the f16 x f16 + f16 8x8x8 products took on G17P
 * hardware, where every result matched an f16 FMA chain (C first, round to
 * nearest even per step). Decode the encoder's bytes with the AGX3B table and
 * check each field by name. The f16 form has no operand type fields; each
 * operand is a register (kind 2) of one half per element. The f16 form refuses
 * other input types and the AGX3C encoding (its M5 layout is not mapped).
 */
TEST(NativeForms, Agx3bF16Matmad8x8Fields)
{
   enum {
      DEST = 2, DEST_KIND = 9, DEST_SIZE = 10, DEST_CACHE = 11,
      A = 17, A_KIND = 24, A_SIZE = 25, A_CACHE = 26, A_DISCARD = 28,
      B = 34, B_KIND = 41, B_SIZE = 42, B_CACHE = 43, B_DISCARD = 45,
      C = 51, C_KIND = 58, C_SIZE = 59, C_CACHE = 60, C_DISCARD = 62,
      WAIT = 68, /* slots 0..5 in 68..73 */
      KIND_REGISTER = 2, SIZE_HALF = 1,
   };
   struct agx3_matrix_mma8_desc d = {
      .dest = 16, .a = 8, .b = 4, .c = 12,
      .cache = {false, false, true}, .discard = {true, false, false},
      .wait_mask = 0x21,
      .input_type = AGX3_MATRIX_INPUT_F16, .f16_accumulate = true,
   };
   uint8_t out[14], length = 0;
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &d, out, &length));

   uint64_t f[AGX3_MAX_FIELDS];
   const int32_t form =
      agx3_decode(&agx3b_table, out, length, AGX3_MN_NONE, f);
   ASSERT_GE(form, 0);
   EXPECT_EQ(agx3b_table.forms[form].mnemonic, AGX3_MN_f16matmad8x8);
   EXPECT_EQ(agx3b_table.forms[form].size, length);

   EXPECT_EQ(f[DEST], 16u);
   EXPECT_EQ(f[DEST_KIND], (uint64_t)KIND_REGISTER);
   EXPECT_EQ(f[DEST_SIZE], (uint64_t)SIZE_HALF);
   EXPECT_EQ(f[DEST_CACHE], 0u);
   EXPECT_EQ(f[A], 8u);
   EXPECT_EQ(f[A_KIND], (uint64_t)KIND_REGISTER);
   EXPECT_EQ(f[A_SIZE], (uint64_t)SIZE_HALF);
   EXPECT_EQ(f[A_CACHE], 0u);
   EXPECT_EQ(f[A_DISCARD], 1u);
   EXPECT_EQ(f[B], 4u);
   EXPECT_EQ(f[B_KIND], (uint64_t)KIND_REGISTER);
   EXPECT_EQ(f[B_SIZE], (uint64_t)SIZE_HALF);
   EXPECT_EQ(f[B_CACHE], 0u);
   EXPECT_EQ(f[B_DISCARD], 0u);
   EXPECT_EQ(f[C], 12u);
   EXPECT_EQ(f[C_KIND], (uint64_t)KIND_REGISTER);
   EXPECT_EQ(f[C_SIZE], (uint64_t)SIZE_HALF);
   EXPECT_EQ(f[C_CACHE], 1u);
   EXPECT_EQ(f[C_DISCARD], 0u);
   for (unsigned i = 0; i < 6; ++i)
      EXPECT_EQ(f[WAIT + i], (uint64_t)((0x21 >> i) & 1)) << "wait slot " << i;

   /* Without f16_accumulate the same product is the f32-accumulator form */
   d.f16_accumulate = false;
   d.dest = d.c = 16;
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
   const int32_t f32_form =
      agx3_decode(&agx3b_table, out, length, AGX3_MN_NONE, f);
   ASSERT_GE(f32_form, 0);
   EXPECT_EQ(agx3b_table.forms[f32_form].mnemonic, AGX3_MN_f32matmad8x8);
   d.f16_accumulate = true;

   EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &d, out, &length));
   d.input_type = AGX3_MATRIX_INPUT_BF16;
   EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &d, out, &length));
}

/*
 * Adding the AGX3B path must not perturb the M5 (AGX3C) encoding. The AGX3C
 * form is 14 bytes (its two extra selector fields f67/f68 and the wider
 * scoreboard-wait range are absent on the 12-byte AGX3B form), and its output
 * for the same descriptors is exactly what the hardware-qualified M5 encoder
 * emitted before this change. The two encodings therefore never collide.
 */
TEST(NativeForms, Agx3cMatmad8x8IsUnchangedByAgx3b)
{
   const struct agx3_matrix_mma8_desc c_present = {
      .dest = 12, .a = 4, .b = 8, .c = 12, .dest_cache = true,
      .cache = {true, true, false}, .discard = {false, false, true},
      .input_type = AGX3_MATRIX_INPUT_F32,
   };
   const struct agx3_matrix_mma8_desc c_zero = {
      .dest = 4, .a = 4, .b = 8, .c_zero = true,
      .discard = {true, true, false}, .input_type = AGX3_MATRIX_INPUT_F32,
   };
   const std::vector<uint8_t> gold_present =
      {0x27, 0x80, 0x05, 0x9a, 0x28, 0x20, 0xaf, 0x1a,
       0x10, 0x08, 0x10, 0x00, 0x90, 0x18};
   const std::vector<uint8_t> gold_zero =
      {0x27, 0x00, 0x05, 0x9a, 0x20, 0x20, 0xaf, 0x0a,
       0x30, 0x06, 0x40, 0x00, 0x10, 0x01};
   uint8_t out[14], length = 0;
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &c_present, out,
                                       &length));
   EXPECT_EQ(std::vector<uint8_t>(out, out + length), gold_present);
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &c_zero, out,
                                       &length));
   EXPECT_EQ(std::vector<uint8_t>(out, out + length), gold_zero);
}

/*
 * The AGX3B form carries six scoreboard-wait bits (slots 0..5, fields 69..74);
 * a full 0x3f mask still encodes, while bit 6 (slot 6) has no witness field, so
 * the encoder refuses rather than emit an unverified encoding.
 */
TEST(NativeForms, Agx3bMatmad8x8RefusesUnwitnessedWait)
{
   struct agx3_matrix_mma8_desc desc = {
      .dest = 4, .a = 4, .b = 8, .c = 12, .discard = {true, true, true},
      .input_type = AGX3_MATRIX_INPUT_F32,
   };
   uint8_t out[14], length = 0;
   desc.wait_mask = 0x3f;
   EXPECT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &desc, out,
                                       &length));
   desc.wait_mask = 0x40;
   EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &desc, out,
                                        &length));
}

/*
 * Field 74 (packed bit 17, the second-least bit of byte 2) is AGX3B wait slot
 * 5, not a c-is-zero marker: it must track wait_mask bit 5 regardless of the
 * accumulate/c-is-zero choice (which lives in field 59). A c-is-zero op with no
 * wait leaves field 74 clear; an accumulating op that waits slot 5 sets it.
 * This guards against the field reverting to a c_zero marker.
 */
TEST(NativeForms, Agx3bMatmad8x8Field74IsWaitSlot5NotCZero)
{
   uint8_t out[14], length = 0;
   /* c-is-zero, no wait: field 74 must be 0 (c_zero alone must not set it). */
   struct agx3_matrix_mma8_desc c_zero_nowait = {
      .dest = 4, .a = 4, .b = 8, .c_zero = true,
      .discard = {true, true, false}, .wait_mask = 0,
      .input_type = AGX3_MATRIX_INPUT_F32,
   };
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &c_zero_nowait,
                                       out, &length));
   EXPECT_EQ((out[2] >> 1) & 1, 0);
   /* accumulate (c present), wait slot 5: field 74 must be 1. */
   struct agx3_matrix_mma8_desc accum_wait5 = {
      .dest = 12, .a = 4, .b = 8, .c = 12, .dest_cache = true,
      .cache = {true, true, false}, .discard = {false, false, true},
      .wait_mask = 0x20, .input_type = AGX3_MATRIX_INPUT_F32,
   };
   ASSERT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &accum_wait5,
                                       out, &length));
   EXPECT_EQ((out[2] >> 1) & 1, 1);
}

/* agx_lower_pseudo lowers AGX_OPCODE_NOT to AGX_OPCODE_BITOP with a dummy
 * immediate-zero second source. Ensure the AGX3 selector drops that ignored
 * second source so agx3_emit_alu composes a single wait-carrying AGX3_ALU_BNOT
 * instruction rather than emitting a separate scoreboard retire before NOT. */
TEST(NativeForms, UnaryBitopNotFoldsScoreboardWait)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;

   agx_device_load_to(b, agx_register(4, AGX_SIZE_32),
                      agx_register(28, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I32, 1, 0, false);
   agx_index src0 = agx_register(4, AGX_SIZE_32);
   agx_bitop_to(b, agx_register(0, AGX_SIZE_32), src0, agx_zero(),
                AGX3_SEL_TT_NOT);
   agx_stop(b);

   util_dynarray bytes;
   util_dynarray_init(&bytes, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &bytes));

   /* 14-byte device_load + 10-byte wait-carrying AGX3B inv/BNOT + 4-byte stop
    * = 28 bytes total (no separate scoreboard retire instruction before NOT). */
   uint8_t expected_bnot[AGX3_ALU_MAX_LENGTH] = {};
   uint8_t bnot_len = 0;
   ASSERT_TRUE(agx3_alu_compose(AGX_ISA_ENCODING_AGX3B, AGX3_ALU_BNOT, 0, 4, 0,
                                0, 0, 1u << 0, expected_bnot, &bnot_len));
   ASSERT_EQ(bnot_len, 10u);
   ASSERT_EQ(bytes.size, 14u + bnot_len + 4u);
   EXPECT_EQ(memcmp((const uint8_t *)bytes.data + 14, expected_bnot, bnot_len),
             0);
   ralloc_free(mem);
}

/* Truth-table inversion in agx_optimizer can leave an unread register in src[1]
 * of a unary MOV (AGX3_SEL_TT_MOV). Ensure agx3_translate_instr() drops src[1]
 * so agx3_emit_alu()'s synchronous-prelude overlap check does not refuse a
 * multi-slot waited MOV whose unread src[1] aliases dest. */
TEST(NativeForms, UnaryBitopMovIgnoresUnreadSecondSourceOnMultiSlotWait)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   agx_builder *b = agx_test_builder(mem);
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->max_reg = 31;

   agx_device_load_to(b, agx_register(4, AGX_SIZE_16),
                      agx_register(28, AGX_SIZE_64), agx_immediate(0),
                      AGX_FORMAT_I16, 1, 0, false);
   agx_device_load_to(b, agx_register(5, AGX_SIZE_16),
                      agx_register(28, AGX_SIZE_64), agx_immediate(2),
                      AGX_FORMAT_I16, 1, 0, false);
   agx_bitop_to(b, agx_register(0, AGX_SIZE_32), agx_register(4, AGX_SIZE_32),
                agx_register(0, AGX_SIZE_32), AGX3_SEL_TT_MOV);
   agx_stop(b);

   util_dynarray bytes;
   util_dynarray_init(&bytes, mem);
   ASSERT_TRUE(agx3_pack_binary(b->shader, &bytes));
   EXPECT_GT(bytes.size, 0u);
   ralloc_free(mem);
}

/* The implemented cooperative-matrix path is enabled for every AGX3B
 * encoding and for M5. Earlier AGX2 devices do not gain the extension. */
TEST(NativeForms, CooperativeMatrixDefaultsFollowTheImplementedEncoding)
{
   /* G16S and the other AGX3B parts use the same 8x8 path. */
   EXPECT_TRUE(agx_chip_cooperative_matrix_supported(AGX_CHIP_G16S));
   EXPECT_EQ(agx_chip_isa_encoding(AGX_CHIP_G16S), AGX_ISA_ENCODING_AGX3B);

   /* All AGX3B parts advertise the implemented path by default. */
   for (enum agx_chip chip : {AGX_CHIP_G17P, AGX_CHIP_G17A, AGX_CHIP_G16P,
                              AGX_CHIP_G16G, AGX_CHIP_G16C, AGX_CHIP_G15G,
                              AGX_CHIP_G15S, AGX_CHIP_G15C, AGX_CHIP_G15D}) {
      EXPECT_EQ(agx_chip_isa_encoding(chip), AGX_ISA_ENCODING_AGX3B);
      EXPECT_TRUE(agx_chip_cooperative_matrix_supported(chip));
   }

   /* M5 has the AGX3C path; AGX2 has neither implementation. */
   EXPECT_TRUE(agx_chip_cooperative_matrix_supported(AGX_CHIP_G17G));
   EXPECT_FALSE(agx_chip_cooperative_matrix_supported(AGX_CHIP_G13G));
}

/* A G17P 2D-array comparison keeps a nonzero constant level or bias in a
 * register, since agx3_emit_texture_sample() refuses that immediate. Every
 * form the packer already encoded keeps its immediate, so no shader that
 * compiled before compiles differently. */
TEST(NativeForms, G17PArrayShadowConstantLodKeepsEncodableImmediates)
{
   struct {
      mesa_shader_stage stage;
      enum agx_lod_mode lod_mode;
      uint16_t value; /* fp16 */
      bool offset;
      enum agx_chip chip;
      bool immediate;
   } cases[] = {
      /* The refused forms. */
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, 0x4000 /* 2.0 */, false,
       AGX_CHIP_G17P, false},
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD_BIAS, 0x3800 /* 0.5 */,
       false, AGX_CHIP_G17P, false},
      {MESA_SHADER_COMPUTE, AGX_LOD_MODE_LOD_MIN, 0x4000, false, AGX_CHIP_G17P,
       false},
      /* Forms the packer encodes as immediates. */
      {MESA_SHADER_VERTEX, AGX_LOD_MODE_LOD_MIN, 0x4000, false, AGX_CHIP_G17P,
       true},
      {MESA_SHADER_MESH, AGX_LOD_MODE_LOD_MIN, 0x4000, false, AGX_CHIP_G17P,
       true},
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, 0x0000 /* 0.0 */, false,
       AGX_CHIP_G17P, true},
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, 0x8000 /* -0.0 */, false,
       AGX_CHIP_G17P, true},
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_AUTO_LOD_BIAS, 0x8000, false,
       AGX_CHIP_G17P, true},
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, 0x4000, true,
       AGX_CHIP_G17P, true},
      /* Other chips are unchanged. */
      {MESA_SHADER_FRAGMENT, AGX_LOD_MODE_LOD_MIN, 0x4000, false,
       AGX_CHIP_G16G, true},
   };
   for (const auto &c : cases) {
      SCOPED_TRACE(::testing::Message()
                   << "stage=" << c.stage << " lod_mode=" << c.lod_mode
                   << " value=" << c.value << " offset=" << c.offset
                   << " chip=" << c.chip);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.chip = c.chip;
      key.dev.encoding = agx_chip_isa_encoding(c.chip);
      b->shader->key = &key;
      b->shader->stage = c.stage;
      agx_index coords = agx_register(8, AGX_SIZE_32);
      coords.channels_m1 = 2;
      agx_index lod = agx_mov_imm(b, 16, c.value);
      agx_instr *I = agx_texture_sample_to(
         b, agx_temp(b->shader, AGX_SIZE_16), coords, lod, agx_immediate(0),
         agx_immediate(0), agx_immediate(0), agx_register(16, AGX_SIZE_32),
         AGX_DIM_2D_ARRAY, c.lod_mode, 1, c.offset, true, false,
         AGX_GATHER_NONE, false);
      agx_optimizer_forward(b->shader);
      EXPECT_EQ(I->src[1].type == AGX_INDEX_IMMEDIATE, c.immediate);
      ralloc_free(mem);
   }
}

/*
 * AGX3B's 8-bit register fields accept halves 192..255 in the compose table,
 * but its general register file ends at half 191 (192 halves). Bound 4-half
 * (vec2 f32) operands to 188 and 2-half (vec2 f16) operands to 190 on AGX3B
 * while preserving AGX3C's 252/254 bounds.
 */
TEST(NativeForms, Agx3bMatmad8x8EnforcesRegisterBounds)
{
   uint8_t out[14], length = 0;

   struct agx3_matrix_mma8_desc f32_max = {
      .dest = 188, .a = 188, .b = 188, .c = 188,
      .input_type = AGX3_MATRIX_INPUT_F32,
   };
   EXPECT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &f32_max, out,
                                       &length));

   for (unsigned field = 0; field < 4; ++field) {
      struct agx3_matrix_mma8_desc over = f32_max;
      if (field == 0)
         over.dest = 190;
      else if (field == 1)
         over.a = 190;
      else if (field == 2)
         over.b = 190;
      else
         over.c = 190;
      EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &over, out,
                                           &length)) << field;
      EXPECT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &over, out,
                                          &length)) << field;
   }

   struct agx3_matrix_mma8_desc f16_max = {
      .dest = 188, .a = 190, .b = 190, .c = 188,
      .input_type = AGX3_MATRIX_INPUT_F16,
   };
   EXPECT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &f16_max, out,
                                       &length));

   struct agx3_matrix_mma8_desc f16_over = f16_max;
   f16_over.a = 192;
   EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &f16_over, out,
                                        &length));
   f16_over = f16_max;
   f16_over.b = 192;
   EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &f16_over, out,
                                        &length));
   f16_over = f16_max;
   f16_over.dest = 192;
   EXPECT_FALSE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3B, &f16_over, out,
                                        &length));

   struct agx3_matrix_mma8_desc agx3c_max = {
      .dest = 252, .a = 254, .b = 254, .c = 252,
      .input_type = AGX3_MATRIX_INPUT_F16,
   };
   EXPECT_TRUE(agx3_matrix_encode_mma8(AGX_ISA_ENCODING_AGX3C, &agx3c_max, out,
                                       &length));
}

/*
 * Unmodified all-scalar-register F32 FFMA on AGX3B (non-G18G) preserves
 * f16 GPR sources (both SSA normal and precoloured register indices) across
 * all operand slots and packs them via agx3_falu_fma_mod_runtime(). Modified,
 * uniform-mixed, and G18G forms keep explicit f16->f32 widening.
 */
TEST(NativeForms, Agx3bFfmaKeepsUnmodifiedHalfGprNative)
{
   struct LowerCase {
      const char *name;
      enum agx_isa_encoding enc;
      bool emit_agx3d;
      bool precoloured;
      unsigned half_mask;
      bool src_neg;
      bool src_abs;
      bool addend_uniform;
      bool expect_native;
   };
   const LowerCase cases[] = {
      {"agx3b_src0_normal", AGX_ISA_ENCODING_AGX3B, false, false, 1u, false,
       false, false, true},
      {"agx3b_src1_normal", AGX_ISA_ENCODING_AGX3B, false, false, 2u, false,
       false, false, true},
      {"agx3b_addend_normal", AGX_ISA_ENCODING_AGX3B, false, false, 4u, false,
       false, false, true},
      {"agx3b_dual_normal", AGX_ISA_ENCODING_AGX3B, false, false, 5u, false,
       false, false, true},
      {"agx3b_src0_precoloured", AGX_ISA_ENCODING_AGX3B, false, true, 1u, false,
       false, false, true},
      {"agx3b_neg_widened", AGX_ISA_ENCODING_AGX3B, false, false, 1u, true,
       false, false, false},
      {"agx3b_abs_widened", AGX_ISA_ENCODING_AGX3B, false, false, 1u, false,
       true, false, false},
      {"agx3b_uniform_mixed_widened", AGX_ISA_ENCODING_AGX3B, false, false, 1u,
       false, false, true, false},
      {"agx3c_src0_widened", AGX_ISA_ENCODING_AGX3C, false, false, 1u, false,
       false, false, false},
      {"g18g_src0_widened", AGX_ISA_ENCODING_AGX3C, true, false, 1u, false,
       false, false, false},
   };

   for (const LowerCase &c : cases) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      struct agx_shader_key key = {};
      key.dev.encoding = c.enc;
      key.dev.emit_agx3d = c.emit_agx3d;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;

      agx_index srcs[3];
      for (unsigned s = 0; s < 3; ++s) {
         if (s == 2 && c.addend_uniform) {
            srcs[s] = agx_uniform(0, AGX_SIZE_32);
            continue;
         }
         const bool half = (c.half_mask & (1u << s)) != 0;
         agx_index reg = agx_register(2 * s + (half ? 1 : 0),
                                      half ? AGX_SIZE_16 : AGX_SIZE_32);
         srcs[s] = c.precoloured ? reg : agx_preload(b, reg);
      }
      if (c.src_neg)
         srcs[1] = agx_neg(srcs[1]);
      if (c.src_abs)
         srcs[1] = agx_abs(srcs[1]);

      agx_index out = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *fma = agx_ffma_to(b, out, srcs[0], srcs[1], srcs[2]);
      agx_unit_test(b, out);

      agx_lower_agx3_fma_sources(b->shader);

      for (unsigned s = 0; s < 3; ++s) {
         if (!(c.half_mask & (1u << s)))
            continue;
         EXPECT_EQ(fma->src[s].size, c.expect_native ? AGX_SIZE_16 : AGX_SIZE_32)
            << c.name << " src" << s;
      }
      ralloc_free(mem);
   }

   /* End-to-end AGX3B packing: independently verify the physical instruction
    * words deposited by agx3_pack_binary() against agx3b_tables.txt without
    * calling agx3_falu_fma_mod_runtime().
    *
    * In agx3b_tables.txt:
    *   form f32fma 2 (6B, addend=f32, regs < 128):
    *     w0&0x0007==1, w1&0x0007==6, w2&0x0003==0
    *   form f32fma 3 (8B, addend=f16 or high reg):
    *     w0&0x0007==1, w1&0x0007==6, w2&0x0003==1
    *     group 270 (addend width field 60): w2&0x0004==0, w3&0x0002==0 (f16)
    *     or w3&0x0002==2 (f32)
    *   Both forms deposit width fields via LUT (f16->0, f32->1):
    *     dest width (field 10) -> w0 bit 3
    *     src1 width (field 26) -> w0 bit 8
    *     src0 width (field 43) -> w1 bit 8 */
   struct PackCase {
      unsigned mask;
      uint16_t regs[3];
      unsigned expect_len;
   };
   const PackCase pack_cases[] = {
      {1u, {5, 4, 6}, 6},      /* form 2: src0=f16, src1=f32, addend=f32 */
      {2u, {2, 7, 6}, 6},      /* form 2: src0=f32, src1=f16, addend=f32 */
      {3u, {5, 7, 6}, 6},      /* form 2: src0=f16, src1=f16, addend=f32 */
      {1u, {5, 4, 130}, 8},    /* form 3: src0=f16, addend=f32 (high reg) */
      {4u, {2, 4, 9}, 8},      /* form 3: addend=f16 */
      {5u, {5, 4, 9}, 8},      /* form 3: src0=f16, addend=f16 */
      {7u, {5, 7, 9}, 8},      /* form 3: all three sources f16 */
   };
   for (const PackCase &pc : pack_cases) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      struct agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->max_reg = 140;

      const bool src0_16 = (pc.mask & 1u) != 0;
      const bool src1_16 = (pc.mask & 2u) != 0;
      const bool add_16 = (pc.mask & 4u) != 0;
      agx_index s0 = agx_register(pc.regs[0], src0_16 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index s1 = agx_register(pc.regs[1], src1_16 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_index s2 = agx_register(pc.regs[2], add_16 ? AGX_SIZE_16 : AGX_SIZE_32);
      agx_ffma_to(b, agx_register(0, AGX_SIZE_32), s0, s1, s2);
      agx_stop(b);

      util_dynarray binary;
      util_dynarray_init(&binary, mem);
      ASSERT_TRUE(agx3_pack_binary(b->shader, &binary)) << "mask=" << pc.mask;
      /* Stop is 4 bytes on AGX3B; first instruction must be exact form size. */
      ASSERT_EQ(binary.size, pc.expect_len + 4u) << "mask=" << pc.mask;

      const uint8_t *raw = (const uint8_t *)binary.data;
      const uint16_t w0 = (uint16_t)raw[0] | ((uint16_t)raw[1] << 8);
      const uint16_t w1 = (uint16_t)raw[2] | ((uint16_t)raw[3] << 8);
      const uint16_t w2 = (uint16_t)raw[4] | ((uint16_t)raw[5] << 8);
      EXPECT_EQ(w0 & 0x0007u, 0x0001u) << "mask=" << pc.mask;
      EXPECT_EQ(w1 & 0x0007u, 0x0006u) << "mask=" << pc.mask;
      EXPECT_EQ((w0 >> 3) & 1u, 1u) << "dest f32 width bit, mask=" << pc.mask;
      EXPECT_EQ((w0 >> 8) & 1u, src1_16 ? 0u : 1u) << "src1 width bit, mask=" << pc.mask;
      EXPECT_EQ((w1 >> 8) & 1u, src0_16 ? 0u : 1u) << "src0 width bit, mask=" << pc.mask;
      if (pc.expect_len == 6) {
         EXPECT_EQ(w2 & 0x0003u, 0x0000u) << "form 2 selector, mask=" << pc.mask;
      } else {
         const uint16_t w3 = (uint16_t)raw[6] | ((uint16_t)raw[7] << 8);
         EXPECT_EQ(w2 & 0x0003u, 0x0001u) << "form 3 selector, mask=" << pc.mask;
         EXPECT_EQ(w2 & 0x0004u, 0x0000u) << "g270 high bit, mask=" << pc.mask;
         EXPECT_EQ(w3 & 0x0002u, add_16 ? 0x0000u : 0x0002u)
            << "g270 addend width bit, mask=" << pc.mask;
      }

      util_dynarray_fini(&binary);
      ralloc_free(mem);
   }
}

/* On AGX3C (M5), fence_mem_to_tex_agx must emit the non-rendezvous device and
 * texture fences rather than the workgroup-rendezvous DEVICE/TEXTURE barriers:
 * the intrinsic sits inside per-invocation bounds checks and must make device
 * image atomics visible to subsequent texture-cache reads in the same thread. */
TEST(NativeForms, M5FenceMemToTexUsesNonRendezvousFences)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "m5 fence_mem_to_tex");
   b.shader->info.workgroup_size[0] = 8;
   b.shader->info.workgroup_size[1] = 16;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *id = nir_load_local_invocation_index(&b);
   nir_def *addr = nir_iadd(&b, base, nir_u2u64(&b, nir_ishl_imm(&b, id, 2)));
   nir_global_atomic(&b, 32, addr, nir_imm_int(&b, 1),
                     .atomic_op = nir_atomic_op_iadd);
   nir_fence_mem_to_tex_agx(&b);
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G17S);

   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   key.no_preamble = true;
   key.reserved_preamble = 8;
   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   ASSERT_NE(part.binary, nullptr);

   agx3_encoded_scoped_barrier dev_fence = {}, tex_fence = {};
   agx3_encoded_scoped_barrier dev_bar = {}, tex_bar = {};
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                        AGX3_BARRIER_SCOPE_DEVICE_FENCE,
                                        &dev_fence));
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                        AGX3_BARRIER_SCOPE_TEXTURE_FENCE_WAIT,
                                        &tex_fence));
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                        AGX3_BARRIER_SCOPE_DEVICE, &dev_bar));
   ASSERT_TRUE(agx3_pack_scoped_barrier(AGX_ISA_ENCODING_AGX3C,
                                        AGX3_BARRIER_SCOPE_TEXTURE, &tex_bar));

   auto count_pattern = [&](const agx3_encoded_scoped_barrier &pat) {
      unsigned count = 0;
      const uint8_t *code = (const uint8_t *)part.binary;
      for (unsigned i = 0; i + pat.length <= part.info.binary_size; ++i)
         count += !memcmp(code + i, pat.bytes, pat.length);
      return count;
   };

   EXPECT_EQ(count_pattern(dev_fence), 1u);
   EXPECT_EQ(count_pattern(tex_fence), 1u);
   EXPECT_EQ(count_pattern(dev_bar), 0u);
   EXPECT_EQ(count_pattern(tex_bar), 0u);

   free(part.binary);
   ralloc_free(b.shader);
}
