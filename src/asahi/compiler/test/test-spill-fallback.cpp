/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"

TEST(SpillFallback, EdgeStoresReportScratchDuringRematerialization)
{
   for (unsigned mode = 0; mode < 4; ++mode) {
      SCOPED_TRACE(mode);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
      key.has_scratch = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->any_cf = true;
      b->shader->is_preamble = true;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *left = agx_test_block(b->shader);
      agx_block *right = agx_test_block(b->shader);
      agx_block *join = agx_test_block(b->shader);
      agx_block_add_successor(entry, left);
      agx_block_add_successor(entry, right);
      agx_block_add_successor(left, join);
      agx_block_add_successor(right, join);
      agx_index incoming[2];
      unsigned side = 0;
      for (agx_block *block : {left, right}) {
         b->cursor = agx_after_block(block);
         if (mode == 0)
            incoming[side] = agx_mov(b, agx_uniform(2 * side, AGX_SIZE_32));
         else if (mode == 1)
            incoming[side] = agx_mov_imm(b, 32, 13 + side);
         else if (mode == 2) {
            incoming[side] = agx_immediate(13 + side);
         } else
            incoming[side] = agx_uniform(2 * side, AGX_SIZE_32);
         ++side;
      }
      b->cursor = agx_after_block(join);
      agx_index values[16];
      for (agx_index &value : values) {
         value = agx_temp(b->shader, AGX_SIZE_32);
         agx_instr *phi = agx_phi_to(b, value, 2);
         phi->src[0] = incoming[0];
         phi->src[1] = incoming[1];
      }
      for (agx_index value : values)
         agx_unit_test(b, value);
      agx_compute_liveness(b->shader);
      agx_spill(b->shader, 24, true);
      EXPECT_TRUE(b->shader->has_spill_pcopy_reserved);
      EXPECT_TRUE(b->shader->remat_fallback_spilled);
      unsigned stores = 0;
      agx_foreach_instr_global(b->shader, I) {
         stores += I->op == AGX_OPCODE_MOV && I->dest[0].memory;
      }
      EXPECT_GT(stores, 0u);
      ralloc_free(mem);
   }
}

/* A value live through a G17P loop that the loop never reads stays spilled
 * across the loop when the loop body needs every register, instead of being
 * spilled in the body and reloaded on the back edge on every iteration. */
TEST(SpillLoopHeader, LiveThroughValueStaysSpilledAcrossLoop)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);

   /* None of the values is rematerializable (rcp is excluded from
    * repeatable ALU rematerialization on AGX3). */
   b->cursor = agx_after_block(entry);
   agx_index through = agx_rcp(b, agx_uniform(0, AGX_SIZE_32));

   /* Eight 32-bit values fill the sixteen halves left to values */
   b->cursor = agx_after_block(body);
   agx_index values[8];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] = agx_rcp(b, agx_uniform(4 + 2 * i, AGX_SIZE_32));
   }
   for (agx_index value : values)
      agx_unit_test(b, value);

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, through);

   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 16 + 8, false);

   unsigned in_loop = 0, stores = 0, fills = 0;
   agx_foreach_block(b->shader, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op != AGX_OPCODE_MOV ||
             !(I->dest[0].memory || I->src[0].memory))
            continue;

         in_loop += block == header || block == body;
         stores += block == entry && I->dest[0].memory;
         fills += block == exit && I->src[0].memory;
      }
   }

   EXPECT_EQ(in_loop, 0u);
   EXPECT_EQ(stores, 1u);
   EXPECT_EQ(fills, 1u);
   ralloc_free(mem);
}

/* One lane of a split device load is evicted and every reload repeats the
 * load, so the spill store is dead after spilling and with it the only read
 * of that split destination. Partial DCE after spilling must null the lane,
 * or RA counts it until the end of the block and exceeds the budget. */
TEST(SpillFallback, RepeatedSplitLaneLeavesNoDeadDestination)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17C;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;

   agx_index vector = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *load = agx_device_load_to(b, vector, agx_uniform(0, AGX_SIZE_64),
                                       agx_immediate(0), AGX_FORMAT_I32, 3, 0,
                                       false);
   load->memory_can_reorder = true;
   agx_instr *split = agx_alloc_instr(b, AGX_OPCODE_SPLIT, 2, 1);
   split->src[0] = vector;
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_builder_insert(&b->cursor, split);
   agx_unit_test(b, split->dest[0]);

   /* Fill the 16 halves left after the spiller's parallel-copy reserve. */
   agx_index pressure[8];
   for (unsigned i = 0; i < ARRAY_SIZE(pressure); ++i)
      pressure[i] = agx_fadd(b, agx_uniform(4 + 2 * i, AGX_SIZE_32),
                            agx_uniform(2, AGX_SIZE_32));

   for (agx_index value : pressure)
      agx_unit_test(b, value);

   agx_unit_test(b, split->dest[1]);
   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 24, false);
   agx_compute_liveness(b->shader);

   EXPECT_FALSE(agx_is_null(split->dest[0]));
   EXPECT_TRUE(agx_is_null(split->dest[1]));
   EXPECT_LE(agx_calc_register_demand(b->shader, false), 24u);
   unsigned repeated = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_DEVICE_LOAD && I->mask == 1 &&
          I->memory_byte_offset == 4)
         ++repeated;
   }
   EXPECT_EQ(repeated, 1u);
   ralloc_free(mem);
}

/* Exercise next_uses growth (>16 live-in values) and multi-block propagation
 * with spilling across a diamond and loop. */
TEST(SpillLoopHeader, ManyLiveValuesGrowNextUsesAndValidate)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);

   /* RCP is not rematerializable, so the spiller must store and fill these
    * values rather than repeat them at their uses. */
   constexpr unsigned N = 24;
   agx_index live[N];
   b->cursor = agx_after_block(entry);
   /* rcp is excluded from repeatable ALU rematerialization on AGX3, so the
    * values spill instead of rematerializing. */
   for (unsigned i = 0; i < N; ++i)
      live[i] = agx_rcp(b, agx_uniform(2 * i, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index acc = agx_fadd(b, live[0], live[1]);
   agx_unit_test(b, acc);

   b->cursor = agx_after_block(exit);
   for (unsigned i = 0; i < N; ++i)
      agx_unit_test(b, live[i]);

   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 24, false);

   unsigned stores = 0, fills = 0;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_MOV && I->dest[0].memory)
         stores++;
      if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
         fills++;
   }
   EXPECT_GT(stores, 0u);
   EXPECT_GT(fills, 0u);
   ralloc_free(mem);
}

/* On AGX3B, cheap pure ALU instructions whose operands are uniforms can be
 * rematerialized across a loop instead of spilling to scratch memory. */
TEST(SpillLoopHeader, Agx3bRematerializesRepeatableAlu)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);

   b->cursor = agx_after_block(entry);
   agx_index through =
      agx_fadd(b, agx_uniform(0, AGX_SIZE_32), agx_uniform(2, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index values[8];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] = agx_rcp(b, agx_uniform(4 + 2 * i, AGX_SIZE_32));
   }
   for (agx_index value : values)
      agx_unit_test(b, value);

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, through);

   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 16 + 8, false);

   unsigned mem_ops = 0, exit_fadds = 0;
   agx_foreach_block(b->shader, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_MOV &&
             (I->dest[0].memory || I->src[0].memory))
            mem_ops++;
         if (block == exit && I->op == AGX_OPCODE_FADD)
            exit_fadds++;
      }
   }

   EXPECT_EQ(mem_ops, 0u);
   EXPECT_EQ(exit_fadds, 1u);
   ralloc_free(mem);
}

/* When repeatable ALU rematerialization eliminates all fills of a value that
 * was live into a loop header and evicted inside the loop, its spill store is
 * removed. Post-spill DCE must also remove the resulting dead loop-header PHI
 * and its dead backedge reload so RA does not leak registers in the loop. */
TEST(SpillLoopHeader, RematRemovesDeadLoopHeaderPhi)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G16G;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_FRAGMENT;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);

   b->cursor = agx_after_block(entry);
   agx_index through =
      agx_fadd(b, agx_uniform(0, AGX_SIZE_32), agx_uniform(2, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index values[8];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
      values[i] = agx_rcp(b, agx_uniform(4 + 2 * i, AGX_SIZE_32));
   }
   for (agx_index value : values)
      agx_unit_test(b, value);
   /* Use through after the pressure spike so it is live-in to header, evicted
    * in body, and reloaded via repeatable ALU instead of a scratch fill. With
    * no post-loop use, the loop-header PHI only feeds the dead spill store and
    * must be eliminated by post-spill DCE. */
   agx_unit_test(b, through);

   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 16 + 8, false);

   unsigned header_phis = 0, mem_ops = 0;
   agx_foreach_block(b->shader, block) {
      agx_foreach_instr_in_block(block, I) {
         if (block == header && I->op == AGX_OPCODE_PHI)
            header_phis++;
         if (I->op == AGX_OPCODE_MOV &&
             (I->dest[0].memory || I->src[0].memory))
            mem_ops++;
      }
   }

   EXPECT_EQ(header_phis, 0u);
   EXPECT_EQ(mem_ops, 0u);
   ralloc_free(mem);
}

/* Positive AGX3B rematerialization tests: uniform fadd and reorderable
 * device loads live through a full-pressure loop are repeated at exit
 * instead of spilling to scratch memory. */
TEST(SpillLoopHeader, AGX3BRepeatsUniformFaddInsteadOfSpilling)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);

   b->cursor = agx_after_block(entry);
   agx_index through =
      agx_fadd(b, agx_uniform(0, AGX_SIZE_32), agx_uniform(2, AGX_SIZE_32));

   b->cursor = agx_after_block(body);
   agx_index values[8];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_rcp(b, agx_uniform(4 + 2 * i, AGX_SIZE_32));
   for (agx_index value : values)
      agx_unit_test(b, value);

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, through);

   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 16 + 8, false);

   unsigned stores = 0, fills = 0, exit_fadds = 0;
   agx_foreach_block(b->shader, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_MOV && I->dest[0].memory)
            stores++;
         if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
            fills++;
         if (block == exit && I->op == AGX_OPCODE_FADD)
            exit_fadds++;
      }
   }

   EXPECT_EQ(stores, 0u);
   EXPECT_EQ(fills, 0u);
   EXPECT_EQ(exit_fadds, 1u);
   ralloc_free(mem);
}

TEST(SpillLoopHeader, AGX3BRepeatsReorderableDeviceLoadInsteadOfSpilling)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key.dev.chip = AGX_CHIP_G17P;
   key.has_scratch = true;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *header = agx_test_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   header->loop_header = true;
   agx_block_add_successor(entry, header);
   agx_block_add_successor(header, body);
   agx_block_add_successor(header, exit);
   agx_block_add_successor(body, header);

   b->cursor = agx_after_block(entry);
   agx_index through = agx_temp(b->shader, AGX_SIZE_32);
   agx_instr *ld = agx_device_load_to(
      b, through, agx_uniform(0, AGX_SIZE_64), agx_immediate(0),
      AGX_FORMAT_I32, 1, 0, false);
   ld->memory_can_reorder = true;

   b->cursor = agx_after_block(body);
   agx_index values[8];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = agx_rcp(b, agx_uniform(4 + 2 * i, AGX_SIZE_32));
   for (agx_index value : values)
      agx_unit_test(b, value);

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, through);

   agx_compute_liveness(b->shader);
   agx_spill(b->shader, 16 + 8, false);

   unsigned stores = 0, fills = 0, exit_loads = 0;
   agx_foreach_block(b->shader, block) {
      agx_foreach_instr_in_block(block, I) {
         if (I->op == AGX_OPCODE_MOV && I->dest[0].memory)
            stores++;
         if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
            fills++;
         if (block == exit && I->op == AGX_OPCODE_DEVICE_LOAD)
            exit_loads++;
      }
   }

   EXPECT_EQ(stores, 0u);
   EXPECT_EQ(fills, 0u);
   EXPECT_EQ(exit_loads, 1u);
   ralloc_free(mem);
}
