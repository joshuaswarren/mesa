/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"

/* Loop-invariant motion of late-materialized operands (agx_opt_licm). */

namespace {

struct Loop {
   void *mem;
   agx_builder *b;
   agx_shader_key key;
   agx_block *entry, *body, *exit;

   Loop()
   {
      mem = ralloc_context(NULL);
      b = agx_test_builder(mem);
      key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->any_cf = true;
      entry = agx_start_block(b->shader);
      body = agx_test_block(b->shader);
      exit = agx_test_block(b->shader);
      body->loop_header = true;
      agx_block_add_successor(entry, body);
      agx_block_add_successor(body, body);
      agx_block_add_successor(body, exit);
   }

   ~Loop() { ralloc_free(mem); }
};

unsigned
count(agx_block *block, enum agx_opcode op)
{
   unsigned n = 0;
   agx_foreach_instr_in_block(block, I)
      n += I->op == op;
   return n;
}

/* Four values live at once before the loop: the shader's peak. */
agx_index
busy_preheader(agx_builder *b)
{
   agx_index p = agx_mov_imm(b, 32, 1), q = agx_mov_imm(b, 32, 2);
   agx_index r = agx_mov_imm(b, 32, 3), s = agx_mov_imm(b, 32, 4);
   return agx_iadd(b, agx_iadd(b, p, q, 0), agx_iadd(b, r, s, 0), 0);
}

/* As register allocation calls it: bounded by the demand before and, unless
 * given a smaller file, by the registers of that demand's occupancy step. */
bool
licm(agx_context *ctx, unsigned file_regs = 0)
{
   agx_compute_liveness(ctx);
   unsigned demand = agx_calc_register_demand(ctx, false);
   if (!file_regs)
      file_regs = agx_occupancy_for_register_count(demand).max_registers;
   return agx_opt_licm(ctx, demand, file_regs);
}

} // namespace

/* Without a static register split (AGX3B/AGX3C), loop-invariant motion may
 * use the halves below the next file alignment boundary and no more: the
 * assignment file is sized from the aligned demand. */
TEST(OptLicm, DemandBoundStopsAtTheAlignedFile)
{
   Loop l;
   agx_builder *b = l.b;
   l.key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   b->cursor = agx_after_block(l.entry);
   agx_index w = busy_preheader(b);
   b->cursor = agx_after_block(l.body);
   agx_index out = agx_iadd(b, agx_mov_imm(b, 32, 0x1234), w, 0);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, out);

   /* 32-bit values only: the file aligns to 8 halves. */
   EXPECT_EQ(agx_licm_demand_bound(b->shader, 21, 256), 24u);
   EXPECT_EQ(agx_licm_demand_bound(b->shader, 24, 256), 24u);
   EXPECT_EQ(agx_licm_demand_bound(b->shader, 250, 248), 248u);
}

/* With a static register split the bound is the demand itself, so hoisting
 * keeps the occupancy step. */
TEST(OptLicm, DemandBoundKeepsStaticSplitDemand)
{
   Loop l;
   agx_builder *b = l.b;
   l.key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   b->cursor = agx_after_block(l.entry);
   agx_index w = busy_preheader(b);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, w);

   EXPECT_EQ(agx_licm_demand_bound(b->shader, 21, 256), 21u);
   EXPECT_EQ(agx_licm_demand_bound(b->shader, 24, 256), 24u);
}

/* Two equal immediates materialized in the loop become one definition in the
 * preheader, and every use reads it. */
TEST(OptLicm, HoistsAndMergesLoopImmediates)
{
   Loop l;
   agx_builder *b = l.b;
   b->cursor = agx_after_block(l.entry);
   agx_index w = busy_preheader(b);
   b->cursor = agx_after_block(l.body);
   agx_index k1 = agx_mov_imm(b, 32, 0x1234);
   agx_instr *use1 = agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32), k1, w, 0);
   agx_index k2 = agx_mov_imm(b, 32, 0x1234);
   agx_instr *use2 = agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32), k2, w, 0);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, use1->dest[0]);
   agx_unit_test(b, use2->dest[0]);

   EXPECT_TRUE(licm(b->shader, 64));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV_IMM), 0u);
   EXPECT_EQ(count(l.entry, AGX_OPCODE_MOV_IMM), 5u);
   EXPECT_EQ(use1->src[0].value, k1.value);
   EXPECT_EQ(use2->src[0].value, k1.value);
}

/* With no free register in the file for the loop's widest value, a value
 * live through the loop would have to move out of its way: the immediates
 * stay in the loop. */
TEST(OptLicm, KeepsImmediatesWhenTheFileIsFull)
{
   Loop l;
   agx_builder *b = l.b;
   b->cursor = agx_after_block(l.entry);
   agx_index w = busy_preheader(b);
   b->cursor = agx_after_block(l.body);
   agx_index k = agx_mov_imm(b, 32, 0x1234);
   agx_index out = agx_iadd(b, k, w, 0);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, out);

   agx_compute_liveness(b->shader);
   unsigned demand = agx_calc_register_demand(b->shader, false);
   EXPECT_FALSE(licm(b->shader, demand + 1));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV_IMM), 1u);
   EXPECT_EQ(count(l.entry, AGX_OPCODE_MOV_IMM), 4u);

   /* The same loop with a register to spare. */
   EXPECT_TRUE(licm(b->shader, demand + 2));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV_IMM), 0u);
}

/* Hoisting an immediate that is not live where the loop peaks would raise
 * register demand: it stays in the loop. */
TEST(OptLicm, KeepsImmediatesThatWouldRaiseDemand)
{
   Loop l;
   agx_builder *b = l.b;
   b->cursor = agx_after_block(l.body);
   agx_index g[4];
   for (unsigned i = 0; i < 4; ++i)
      g[i] = agx_get_sr(b, 32, AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_index sum = agx_iadd(b, agx_iadd(b, g[0], g[1], 0),
                            agx_iadd(b, g[2], g[3], 0), 0);
   agx_index k = agx_mov_imm(b, 32, 0x1234);
   agx_index out = agx_iadd(b, sum, k, 0);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, out);

   EXPECT_FALSE(licm(b->shader));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV_IMM), 1u);
   EXPECT_EQ(count(l.entry, AGX_OPCODE_MOV_IMM), 0u);
   /* The kept immediate is back before its use. */
   agx_instr *last = list_last_entry(&l.body->instructions, agx_instr, link);
   EXPECT_EQ(last->src[1].value, k.value);
}

/* A loop whose hoisting is reverted after merging equal immediates gets both
 * back, each before its use. */
TEST(OptLicm, RevertRestoresMergedImmediates)
{
   Loop l;
   agx_builder *b = l.b;
   b->cursor = agx_after_block(l.body);
   agx_index g[4];
   for (unsigned i = 0; i < 4; ++i)
      g[i] = agx_get_sr(b, 32, AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_index sum = agx_iadd(b, agx_iadd(b, g[0], g[1], 0),
                            agx_iadd(b, g[2], g[3], 0), 0);
   agx_index k1 = agx_mov_imm(b, 32, 0x1234);
   agx_instr *use1 = agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32), sum, k1, 0);
   agx_index k2 = agx_mov_imm(b, 32, 0x1234);
   agx_instr *use2 =
      agx_iadd_to(b, agx_temp(b->shader, AGX_SIZE_32), use1->dest[0], k2, 0);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, use2->dest[0]);

   EXPECT_FALSE(licm(b->shader));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV_IMM), 2u);
   EXPECT_EQ(count(l.entry, AGX_OPCODE_MOV_IMM), 0u);
   EXPECT_EQ(use1->src[1].value, k1.value);
   EXPECT_EQ(use2->src[1].value, k2.value);
   agx_instr *before1 = list_entry(use1->link.prev, agx_instr, link);
   agx_instr *before2 = list_entry(use2->link.prev, agx_instr, link);
   EXPECT_EQ(before1->op, AGX_OPCODE_MOV_IMM);
   EXPECT_EQ(before2->op, AGX_OPCODE_MOV_IMM);
}

/* Adjacent immediates of a reverted loop come back in their order. */
TEST(OptLicm, RevertKeepsAdjacentOrder)
{
   Loop l;
   agx_builder *b = l.b;
   b->cursor = agx_after_block(l.body);
   agx_index g[4];
   for (unsigned i = 0; i < 4; ++i)
      g[i] = agx_get_sr(b, 32, AGX_SR_THREAD_POSITION_IN_GRID_X);
   agx_index sum = agx_iadd(b, agx_iadd(b, g[0], g[1], 0),
                            agx_iadd(b, g[2], g[3], 0), 0);
   agx_index k1 = agx_mov_imm(b, 32, 0x1234);
   agx_index k2 = agx_mov_imm(b, 32, 0x5678);
   agx_index out = agx_iadd(b, agx_iadd(b, sum, k1, 0), k2, 0);
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, out);

   EXPECT_FALSE(licm(b->shader));
   agx_instr *first = NULL, *second = NULL;
   agx_foreach_instr_in_block(l.body, I) {
      if (I->op == AGX_OPCODE_MOV_IMM) {
         if (first)
            second = I;
         else
            first = I;
      }
   }
   ASSERT_NE(second, nullptr);
   EXPECT_EQ(first->dest[0].value, k1.value);
   EXPECT_EQ(second->dest[0].value, k2.value);
   EXPECT_EQ(list_entry(second->link.prev, agx_instr, link), first);
}

/* An immediate that becomes a vector component keeps its place: allocated
 * right there it lands in the vector, while a hoisted one would be copied in
 * every iteration. */
TEST(OptLicm, KeepsVectorComponentsInPlace)
{
   Loop l;
   agx_builder *b = l.b;
   b->cursor = agx_after_block(l.entry);
   agx_index w = busy_preheader(b);
   b->cursor = agx_after_block(l.body);
   agx_index k = agx_mov_imm(b, 32, 0x1234);
   agx_index v = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, v, 2);
   collect->src[0] = k;
   collect->src[1] = w;
   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, v);

   EXPECT_FALSE(licm(b->shader));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV_IMM), 1u);
}

/* A uniform copy is invariant unless the program writes uniforms. */
TEST(OptLicm, UniformCopiesOnlyWhenUniformsAreReadOnly)
{
   for (int written = 0; written < 2; ++written) {
      Loop l;
      agx_builder *b = l.b;
      b->shader->uniforms_written = written;
      b->cursor = agx_after_block(l.entry);
      agx_index w = busy_preheader(b);
      b->cursor = agx_after_block(l.body);
      agx_index u = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
      agx_index out = agx_iadd(b, u, w, 0);
      b->cursor = agx_after_block(l.exit);
      agx_unit_test(b, out);

      EXPECT_EQ(licm(b->shader), !written);
      EXPECT_EQ(count(l.body, AGX_OPCODE_MOV), written ? 1u : 0u);
   }
}

/* Straight-line code has nothing to hoist. */
TEST(OptLicm, NoLoopNoChange)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
   b->shader->key = &key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   agx_unit_test(b, agx_mov_imm(b, 32, 7));
   EXPECT_FALSE(licm(b->shader));
   ralloc_free(mem);
}

/* An M5 unpacked 2D/3D scalar LOD copy must stay inside its loop so its
 * consumer keeps a killed half-float source. */
TEST(OptLicm, KeepsM5UnpackedScalarLodInLoop)
{
   Loop l;
   agx_builder *b = l.b;
   l.key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   l.key.dev.chip = AGX_CHIP_G17C;
   b->cursor = agx_after_block(l.entry);
   agx_index coords = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_preload_to(b, coords, agx_register(8, AGX_SIZE_32));
   agx_index p[8];
   for (unsigned i = 0; i < 8; ++i)
      p[i] = agx_mov_imm(b, 32, i + 1);
   agx_index keep = agx_iadd(b, agx_iadd(b, p[0], p[1], 0),
                             agx_iadd(b, p[2], p[3], 0), 0);
   keep = agx_iadd(b, keep,
                   agx_iadd(b, agx_iadd(b, p[4], p[5], 0),
                            agx_iadd(b, p[6], p[7], 0), 0),
                   0);

   b->cursor = agx_after_block(l.body);
   agx_index lod = agx_mov(b, agx_uniform(14, AGX_SIZE_16));
   agx_index plain = agx_mov(b, agx_uniform(8, AGX_SIZE_32));
   agx_index sum = agx_iadd(b, keep, plain, 0);
   agx_index sample = agx_vec_temp(b->shader, AGX_SIZE_32, 4);
   agx_instr *tex = agx_texture_sample_to(
      b, sample, coords, lod, agx_zero(), agx_immediate(0), agx_zero(),
      agx_null(), AGX_DIM_2D, AGX_LOD_MODE_LOD_MIN, 0, false, false, false,
      AGX_GATHER_NONE, false);
   tex->mask = 15;

   b->cursor = agx_after_block(l.exit);
   agx_unit_test(b, sum);
   agx_unit_test(b, sample);

   /* The plain ALU uniform copy hoists while the scalar LOD copy stays in body. */
   EXPECT_TRUE(licm(b->shader));
   EXPECT_EQ(count(l.body, AGX_OPCODE_MOV), 1u);
   EXPECT_EQ(count(l.entry, AGX_OPCODE_MOV), 1u);
}

/* Uniform copies consumed by scalar branch compares (IF_FCMP) hoist and merge
 * with identical ALU copies in the loop; written uniforms stay pinned. */
TEST(OptLicm, HoistsUniformUsedByBranchCompare)
{
   for (bool uniforms_written : {false, true}) {
      Loop l;
      agx_builder *b = l.b;
      b->shader->uniforms_written = uniforms_written;
      b->cursor = agx_after_block(l.entry);
      agx_index w = busy_preheader(b);

      b->cursor = agx_after_block(l.body);
      agx_index u1 = agx_mov(b, agx_uniform(22, AGX_SIZE_32));
      agx_index u2 = agx_mov(b, agx_uniform(22, AGX_SIZE_32));
      agx_instr *alu = agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32), w, u2);
      agx_instr *br = agx_if_fcmp(b, w, u1, 1, AGX_FCOND_GE, true, l.exit);

      b->cursor = agx_after_block(l.exit);
      agx_unit_test(b, alu->dest[0]);

      EXPECT_EQ(licm(b->shader, 64), !uniforms_written);
      EXPECT_EQ(count(l.body, AGX_OPCODE_MOV), uniforms_written ? 2u : 0u);
      EXPECT_EQ(count(l.entry, AGX_OPCODE_MOV), uniforms_written ? 0u : 1u);
      if (!uniforms_written) {
         EXPECT_EQ(br->src[1].value, alu->src[1].value);
      }
   }
}
