/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"

/* try_kill_early_sources() checks whether the free tail [region_end, dest_end)
 * after a killed source region is unoccupied so the destination can reuse the
 * source's aligned slot in place. BITSET_TEST_RANGE() is inclusive of its end
 * bit, so passing dest_end erroneously tested the register immediately after
 * the destination (and read one bit past the 256-bit GPR bitset when
 * dest_end == AGX_NUM_REGS).
 */
TEST(RAEarlyKill, DoesNotTestBitAtDestEnd)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   key.dev.chip = AGX_CHIP_G13G;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* Killed 2x32-bit coordinates in r0..r1 (halves 0..3), r2..r3 (4..7) free,
    * live 32-bit guard at r4 (halves 8..9 == dest_end).
    */
   agx_index phys_coords = agx_register(0, AGX_SIZE_32);
   phys_coords.channels_m1 = 1;
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, phys_coords);

   agx_index guard = agx_preload(b, agx_register(8, AGX_SIZE_32));

   agx_index dst = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_texture_sample_to(b, dst, coords, agx_immediate(0), agx_immediate(0),
                         agx_immediate(0), agx_immediate(0), agx_null(),
                         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0xf, false, false,
                         false, AGX_GATHER_NONE, false);

   agx_unit_test(b, dst);
   agx_unit_test(b, guard);

   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);

   unsigned samples = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
         ++samples;
         EXPECT_EQ(I->dest[0].type, AGX_INDEX_REGISTER);
         EXPECT_EQ(I->dest[0].value, 0u);
      }
   }
   EXPECT_EQ(samples, 1u);
   ralloc_free(mem);
}

TEST(RAEarlyKill, DestEndingAtRegFileBoundDoesNotReadOOB)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   key.dev.chip = AGX_CHIP_G13G;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_FRAGMENT;

   /* Fill r2..r123 (halves 4..247) with live values, leaving only a 4-half gap
    * at r0..r1 (too small for the 8-half destination), place killed 2x32-bit
    * coordinates at r124..r125 (halves 248..251), leave r126..r127 (252..255)
    * free, and sample into a 4x32-bit destination spanning halves [248, 256).
    * Total demand is 244 + 4 + 8 = 256 (no spilling), and dest_end ==
    * AGX_NUM_REGS (256) sits right at the end of used_regs_gpr.
    */
   agx_index live[122];
   for (unsigned i = 0; i < 122; ++i)
      live[i] = agx_preload(b, agx_register(4 + 2 * i, AGX_SIZE_32));

   agx_index phys_coords = agx_register(248, AGX_SIZE_32);
   phys_coords.channels_m1 = 1;
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, phys_coords);

   agx_index dst = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_texture_sample_to(b, dst, coords, agx_immediate(0), agx_immediate(0),
                         agx_immediate(0), agx_immediate(0), agx_null(),
                         AGX_DIM_2D, AGX_LOD_MODE_AUTO_LOD, 0xf, false, false,
                         false, AGX_GATHER_NONE, false);

   agx_unit_test(b, dst);
   for (unsigned i = 0; i < 122; ++i)
      agx_unit_test(b, live[i]);

   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);

   unsigned samples = 0;
   unsigned moves = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_TEXTURE_SAMPLE) {
         ++samples;
         EXPECT_EQ(I->dest[0].type, AGX_INDEX_REGISTER);
         EXPECT_EQ(I->dest[0].value, 248u);
      } else if (I->op == AGX_OPCODE_MOV) {
         ++moves;
      }
   }
   EXPECT_EQ(samples, 1u);
   EXPECT_EQ(moves, 0u);
   ralloc_free(mem);
}
