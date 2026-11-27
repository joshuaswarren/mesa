/* SPDX-License-Identifier: MIT */
#include <gtest/gtest.h>
#include "agx_test.h"

/*
 * The spiller reloads a value by repeating its arithmetic together with one
 * operand that left the register file, when that operand is arithmetic over
 * registers still in the file: v = ffma(c, k1, -(s * k2)) is computed again
 * from s and c instead of being filled from scratch.
 */

namespace {

struct counts {
   unsigned stores, fills, fmul, ffma;
};

class SpillRepeatPair : public testing::Test {
 protected:
   SpillRepeatPair()
   {
      mem = ralloc_context(NULL);
      b = agx_test_builder(mem);
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = AGX_CHIP_G17S;
      key.has_scratch = true;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
   }

   ~SpillRepeatPair()
   {
      ralloc_free(mem);
   }

   agx_index u(unsigned i)
   {
      return agx_uniform(2 * i, AGX_SIZE_32);
   }

   /* A value that cannot be computed again */
   agx_index leaf(unsigned i)
   {
      return agx_rcp(b, u(i));
   }

   /* Spill to 16 halves: k = 24 less the 8 kept for memory swaps */
   void spill()
   {
      agx_compute_liveness(b->shader);
      agx_spill(b->shader, 24, false);
      agx_compute_liveness(b->shader);
      EXPECT_LE(agx_calc_register_demand(b->shader, false), 24u);
   }

   counts count()
   {
      counts c = {};
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_MOV && I->dest[0].memory)
            c.stores++;
         if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
            c.fills++;
         c.fmul += I->op == AGX_OPCODE_FMUL;
         c.ffma += I->op == AGX_OPCODE_FFMA;
      }
      return c;
   }

   agx_instr *def(agx_index v)
   {
      agx_foreach_instr_global(b->shader, I) {
         agx_foreach_ssa_dest(I, d) {
            if (I->dest[d].value == v.value)
               return I;
         }
      }
      return NULL;
   }

   /* Evict v at a peak of npeak more values where s, c and the q values stay
    * live, then read s, c and q before v and again after it, so v is the
    * value furthest from its next use at the peak and every other value is
    * in a register when v is reloaded.
    */
   void use_after_peak(agx_index v, agx_index s, agx_index c, agx_index *q,
                       unsigned nq, unsigned npeak = 2)
   {
      agx_index peak[4];
      for (unsigned i = 0; i < npeak; ++i)
         peak[i] = leaf(30 + i);
      for (unsigned i = 0; i < npeak; ++i)
         agx_unit_test(b, peak[i]);
      agx_unit_test(b, s);
      agx_unit_test(b, c);
      for (unsigned i = 0; i < nq; ++i)
         agx_unit_test(b, q[i]);

      agx_unit_test(b, v);

      agx_unit_test(b, s);
      agx_unit_test(b, c);
      for (unsigned i = 0; i < nq; ++i)
         agx_unit_test(b, q[i]);
   }

   void *mem;
   agx_builder *b;
   agx_shader_key key = {};
};

} // namespace

TEST_F(SpillRepeatPair, RepeatsOperandWithModifiers)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_instr *t =
      agx_fmul_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_abs(s), u(2));
   t->saturate = true;
   agx_index v = agx_ffma(b, c, u(3), agx_neg(t->dest[0]));
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 0u);
   EXPECT_EQ(n.fills, 0u);

   /* The originals feed only the dead store, so only the repeats are left */
   EXPECT_EQ(n.fmul, 1u);
   EXPECT_EQ(n.ffma, 1u);

   agx_instr *F = NULL;
   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_FFMA)
         F = I;
   }
   ASSERT_NE(F, nullptr);
   EXPECT_TRUE(F->src[2].neg);
   EXPECT_FALSE(F->src[2].abs);
   EXPECT_EQ(F->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(F->src[1].value, 6u);

   agx_instr *M = def(F->src[2]);
   ASSERT_NE(M, nullptr);
   EXPECT_EQ(M->op, AGX_OPCODE_FMUL);
   EXPECT_TRUE(M->saturate);
   EXPECT_TRUE(M->src[0].abs);
   EXPECT_FALSE(M->src[0].neg);
   EXPECT_EQ(M->src[1].type, AGX_INDEX_UNIFORM);
   EXPECT_EQ(M->src[1].value, 4u);

   /* The operand is a value of its own read only by the repeat */
   EXPECT_NE(M->dest[0].value, t->dest[0].value);
}

TEST_F(SpillRepeatPair, SameOperandInTwoSlots)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_fmul(b, s, u(2));
   agx_index v = agx_ffma(b, t, t, c);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 0u);
   EXPECT_EQ(n.fills, 0u);
   EXPECT_EQ(n.fmul, 1u);

   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_FFMA) {
         EXPECT_EQ(I->src[0].value, I->src[1].value);
         EXPECT_EQ(def(I->src[0])->op, AGX_OPCODE_FMUL);
      }
   }
}

/* A constant operand that left the file is rematerialized for the repeat */
TEST_F(SpillRepeatPair, RematerializesConstantOperand)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index k = agx_mov_imm(b, 32, 0x3f7d70a4);
   agx_index v = agx_ffma(b, c, k, s);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 0u);
   EXPECT_EQ(n.fills, 0u);
   EXPECT_EQ(n.ffma, 1u);

   agx_foreach_instr_global(b->shader, I) {
      if (I->op == AGX_OPCODE_FFMA) {
         agx_instr *K = def(I->src[1]);
         ASSERT_NE(K, nullptr);
         EXPECT_EQ(K->op, AGX_OPCODE_MOV_IMM);
         EXPECT_EQ(K->imm, 0x3f7d70a4u);
         EXPECT_NE(I->src[1].value, k.value);
      }
   }
}

/* The leaf s is dead at the reload: computing the operand again would extend
 * its live range, so v is filled.
 */
TEST_F(SpillRepeatPair, DeadLeafIsFilled)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_fmul(b, s, u(2));
   agx_index v = agx_ffma(b, c, u(3), t);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   /* s dies before the peak */
   agx_unit_test(b, s);
   agx_index p = leaf(8), r = leaf(9), w = leaf(10);
   agx_unit_test(b, p);
   agx_unit_test(b, r);
   agx_unit_test(b, w);
   agx_unit_test(b, c);
   for (unsigned i = 0; i < 4; ++i)
      agx_unit_test(b, q[i]);
   agx_unit_test(b, v);
   agx_unit_test(b, c);
   for (unsigned i = 0; i < 4; ++i)
      agx_unit_test(b, q[i]);

   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

/* Only one level: an operand whose own operand also left the file is not
 * computed again.
 */
TEST_F(SpillRepeatPair, ThreeLevelsAreFilled)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index r = agx_fadd(b, s, u(9));
   agx_index t = agx_fmul(b, r, u(2));
   agx_index v = agx_ffma(b, c, u(3), t);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

TEST_F(SpillRepeatPair, UnrepeatableOperandIsFilled)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_rcp(b, s);
   agx_index v = agx_ffma(b, c, u(3), t);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

TEST_F(SpillRepeatPair, TwoMissingOperandsAreFilled)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index t0 = agx_fmul(b, s, u(2));
   agx_index t1 = agx_fmul(b, c, u(3));
   agx_index v = agx_ffma(b, t0, u(4), t1);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(5 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

/* With every register taken at the reload there is no room for the operand,
 * so v is filled.
 */
TEST_F(SpillRepeatPair, NoRoomForOperandIsFilled)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_fmul(b, s, u(2));
   agx_index v = agx_ffma(b, c, u(3), t);
   agx_index q[5];
   for (unsigned i = 0; i < 5; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 5, 1);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

/* An operand's leaf that is itself reloaded by repeating its instruction in
 * front of the same instruction is not in a register yet, so the pair is not
 * used: v is filled and w is repeated.
 */
TEST_F(SpillRepeatPair, LeafReloadedByRepeatIsNotRead)
{
   agx_index s = leaf(0), c = leaf(1);
   agx_index w = agx_fadd(b, s, u(9));
   agx_index t = agx_fmul(b, w, u(2));
   agx_index v = agx_ffma(b, c, u(3), t);
   agx_index q[3];
   for (unsigned i = 0; i < 3; ++i)
      q[i] = leaf(4 + i);

   /* v and w are both evicted at the peak and both reloaded for x */
   agx_index peak[3] = {leaf(30), leaf(31), leaf(32)};
   for (agx_index p : peak)
      agx_unit_test(b, p);
   agx_unit_test(b, s);
   agx_unit_test(b, c);
   for (unsigned i = 0; i < 3; ++i)
      agx_unit_test(b, q[i]);

   agx_index x = agx_fadd(b, v, w);

   agx_unit_test(b, x);
   agx_unit_test(b, s);
   agx_unit_test(b, c);
   for (unsigned i = 0; i < 3; ++i)
      agx_unit_test(b, q[i]);

   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

/* The deferred-lighting shape: s and c are read in a loop that needs every
 * other register, and v, live through the loop but read only after it, is
 * computed again after the loop instead of being spilled across it.
 */
TEST_F(SpillRepeatPair, LiveThroughLoopIsComputedAfterIt)
{
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
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_fmul(b, s, u(2));
   agx_index v = agx_ffma(b, c, u(3), agx_neg(t));

   b->cursor = agx_after_block(body);
   agx_unit_test(b, s);
   agx_unit_test(b, c);
   agx_index values[6];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = leaf(4 + i);
   for (agx_index value : values)
      agx_unit_test(b, value);

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, v);
   agx_unit_test(b, s);
   agx_unit_test(b, c);

   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 0u);
   EXPECT_EQ(n.fills, 0u);

   unsigned exit_fmul = 0, exit_ffma = 0;
   agx_foreach_instr_in_block(exit, I) {
      exit_fmul += I->op == AGX_OPCODE_FMUL;
      exit_ffma += I->op == AGX_OPCODE_FFMA;
   }
   EXPECT_EQ(exit_fmul, 1u);
   EXPECT_EQ(exit_ffma, 1u);
}

/* A join needs v in a register on entry, but one predecessor evicted it. The
 * edge from that predecessor computes v again from s and c, which both
 * predecessors keep in registers, instead of filling it.
 */
TEST_F(SpillRepeatPair, JoinEdgeComputesValueAgain)
{
   b->shader->any_cf = true;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *left = agx_test_block(b->shader);
   agx_block *right = agx_test_block(b->shader);
   agx_block *join = agx_test_block(b->shader);
   agx_block_add_successor(entry, left);
   agx_block_add_successor(entry, right);
   agx_block_add_successor(left, join);
   agx_block_add_successor(right, join);

   b->cursor = agx_after_block(entry);
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_fmul(b, s, u(2));
   agx_index v = agx_ffma(b, c, u(3), agx_neg(t));

   b->cursor = agx_after_block(left);
   agx_index values[6];
   for (unsigned i = 0; i < ARRAY_SIZE(values); ++i)
      values[i] = leaf(4 + i);
   for (agx_index value : values)
      agx_unit_test(b, value);
   agx_unit_test(b, s);
   agx_unit_test(b, c);

   b->cursor = agx_after_block(join);
   agx_unit_test(b, v);
   agx_unit_test(b, s);
   agx_unit_test(b, c);

   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 0u);
   EXPECT_EQ(n.fills, 0u);

   unsigned left_fmul = 0, left_ffma = 0;
   agx_foreach_instr_in_block(left, I) {
      left_fmul += I->op == AGX_OPCODE_FMUL;
      left_ffma += I->op == AGX_OPCODE_FFMA;
   }
   EXPECT_EQ(left_fmul, 1u);
   EXPECT_EQ(left_ffma, 1u);
}

/* Before AGX3 nothing is repeated, pairs included */
TEST_F(SpillRepeatPair, Agx2IsFilled)
{
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   key.dev.chip = AGX_CHIP_G14G;
   agx_index s = leaf(0), c = leaf(1);
   agx_index t = agx_fmul(b, s, u(2));
   agx_index v = agx_ffma(b, c, u(3), t);
   agx_index q[4];
   for (unsigned i = 0; i < 4; ++i)
      q[i] = leaf(4 + i);

   use_after_peak(v, s, c, q, 4);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 1u);
   EXPECT_EQ(n.fills, 1u);
}

TEST_F(SpillRepeatPair, RepeatsHalfFloatAndExtract)
{
   /* AGX3 repeats FP16 ALU chains (HMUL/HFMA) instead of spilling to scratch. */
   agx_index s = agx_rcp(b, agx_uniform(0, AGX_SIZE_16));
   agx_index c = agx_rcp(b, agx_uniform(2, AGX_SIZE_16));
   agx_index t = agx_hmul(b, s, agx_uniform(4, AGX_SIZE_16));
   agx_index v = agx_hfma(b, c, agx_uniform(6, AGX_SIZE_16), t);
   agx_index q[5];
   for (unsigned i = 0; i < 5; ++i)
      q[i] = leaf(10 + i);

   /* Keep original v live so DCE cannot sink it; spilling must repeat it. */
   agx_unit_test(b, v);
   use_after_peak(v, s, c, q, 5, 2);
   spill();

   counts n = count();
   EXPECT_EQ(n.stores, 0u);
   EXPECT_EQ(n.fills, 0u);

   unsigned hmul = 0, hfma = 0;
   agx_foreach_instr_global(b->shader, I) {
      hmul += I->op == AGX_OPCODE_HMUL;
      hfma += I->op == AGX_OPCODE_HFMA;
   }
   EXPECT_GE(hmul, 2u);
   EXPECT_GE(hfma, 2u);
}

TEST_F(SpillRepeatPair, LoopExitPenaltyProtectsHotLoopValues)
{
   /* A value used immediately after a loop exit has a larger effective
    * next-use distance inside the loop than a value used on the next loop
    * iteration, so the spiller evicts the cold post-loop value instead of
    * filling the hot loop-carried value every iteration. */
   agx_block *entry = b->cursor.block;
   agx_block *hdr = agx_test_block(b->shader);
   agx_block *exit_blk = agx_test_block(b->shader);
   hdr->loop_header = true;
   agx_block_add_successor(entry, hdr);
   agx_block_add_successor(hdr, hdr);
   agx_block_add_successor(hdr, exit_blk);

   b->cursor = agx_after_block(entry);
   agx_index hot = leaf(0);
   agx_index cold = leaf(1);
   agx_index keep[5];
   for (unsigned i = 0; i < 5; ++i)
      keep[i] = leaf(2 + i);

   b->cursor = agx_after_block(hdr);
   /* Use `keep` first, then trigger pressure peak before using `hot` so that
    * without loop-exit weighting `cold` (distance 0 in exit_blk) looks closer
    * than `hot` on the back-edge. */
   for (unsigned i = 0; i < 5; ++i)
      agx_unit_test(b, keep[i]);
   agx_index spike[2] = { leaf(20), leaf(21) };
   agx_unit_test(b, spike[0]);
   agx_unit_test(b, spike[1]);
   agx_unit_test(b, hot);

   b->cursor = agx_after_block(exit_blk);
   agx_unit_test(b, cold);

   spill();

   /* No memory fills inside the hot loop body; the cold post-loop value is
    * filled after exiting the loop instead. */
   unsigned loop_fills = 0, exit_fills = 0;
   agx_foreach_instr_in_block(hdr, I) {
      if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
         loop_fills++;
   }
   agx_foreach_instr_in_block(exit_blk, I) {
      if (I->op == AGX_OPCODE_MOV && I->src[0].memory)
         exit_fills++;
   }
   EXPECT_EQ(loop_fills, 0u);
   EXPECT_GE(exit_fills, 1u);
}
