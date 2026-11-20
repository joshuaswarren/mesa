/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"

/* Register cache hints (agx_opt_register_cache). On AGX3 a last-use hint
 * releases the register for the whole SIMD-group, so a register that any
 * thread reads again never gets one. */

namespace {

struct Shader {
   void *mem;
   agx_builder *b;
   agx_shader_key key;

   Shader(enum agx_isa_encoding encoding = AGX_ISA_ENCODING_AGX3B)
   {
      mem = ralloc_context(NULL);
      b = agx_test_builder(mem);
      key = {};
      key.dev.encoding = encoding;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->any_cf = true;
   }

   ~Shader() { ralloc_free(mem); }
};

agx_index
reg(unsigned r)
{
   return agx_register(r, AGX_SIZE_32);
}

/* The first source of the first instruction in `block` that reads `r`. */
agx_index *
read_of(agx_block *block, unsigned r)
{
   agx_foreach_instr_in_block(block, I) {
      agx_foreach_reg_src(I, s) {
         if (I->src[s].value == r)
            return &I->src[s];
      }
   }
   return NULL;
}

} // namespace

/* A uniform loop whose `if (c) break;` agx_opt_break_if fused into the block
 * after the if: the edge to the exit stays on the emptied then block. The
 * break's compare source is read again after the loop, so it is not a last
 * use, in a uniform loop and in a divergent one. */
TEST(RegisterCache, FusedBreakKeepsTheExitsValue)
{
   for (int divergent = 0; divergent < 2; ++divergent) {
      Shader s;
      agx_builder *b = s.b;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *header = agx_test_block(b->shader);
      agx_block *then_ = agx_test_block(b->shader);
      agx_block *else_ = agx_test_block(b->shader);
      agx_block *after = agx_test_block(b->shader);
      agx_block *exit = agx_test_block(b->shader);
      agx_block_add_successor(entry, header);
      agx_block_add_successor(header, then_);
      agx_block_add_successor(header, else_);
      agx_block_add_successor(then_, exit);
      agx_block_add_successor(else_, after);
      agx_block_add_successor(after, header);
      header->loop_header = true;
      /* A divergent loop takes AGX3's physical liveness. */
      header->cf_divergent = then_->cf_divergent = else_->cf_divergent =
         after->cf_divergent = divergent;

      b->cursor = agx_after_block(entry);
      agx_mov_imm_to(b, reg(2), 0);
      b->cursor = agx_after_block(header);
      agx_iadd_to(b, reg(0), reg(2), reg(2), 0);
      b->cursor = agx_after_block(after);
      agx_instr *brk = agx_break_if_icmp(b, reg(0), agx_immediate(9), 1, false,
                                         AGX_ICOND_UEQ, exit);
      agx_iadd_to(b, reg(2), reg(2), agx_immediate(1), 0);
      b->cursor = agx_after_block(exit);
      agx_unit_test(b, reg(0));

      agx_opt_register_cache(b->shader);
      EXPECT_FALSE(brk->src[0].discard) << "divergent " << divergent;
   }
}

/* Last-use hints in divergent control flow, on AGX3. A last use releases the
 * register for the whole SIMD-group, so a register that a masked-off thread
 * still needs never gets one. */

/* entry: v -> r0. A divergent if: the then side reuses r0 for its own
 * temporaries, the else side reads v. The then side's reads of r0 must keep
 * it (the else threads, masked off, still need v); the else side's read is
 * the last one for every thread. */
TEST(DivergentLastUse, ThenSideKeepsTheElseSidesRegister)
{
   Shader s;
   agx_builder *b = s.b;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *then_ = agx_test_block(b->shader);
   agx_block *else_ = agx_test_block(b->shader);
   agx_block *merge = agx_test_block(b->shader);
   agx_block_add_successor(entry, then_);
   agx_block_add_successor(entry, else_);
   agx_block_add_successor(then_, merge);
   agx_block_add_successor(else_, merge);
   then_->cf_divergent = else_->cf_divergent = true;

   b->cursor = agx_after_block(entry);
   agx_mov_imm_to(b, reg(0), 7);

   b->cursor = agx_after_block(then_);
   agx_mov_imm_to(b, reg(0), 3);                 /* u, then only */
   agx_iadd_to(b, reg(2), reg(0), reg(0), 0);    /* last read of u */
   agx_mov_imm_to(b, reg(0), 5);                 /* t, then only */
   agx_iadd_to(b, reg(4), reg(0), reg(2), 0);    /* last read of t, r2 */

   b->cursor = agx_after_block(else_);
   agx_iadd_to(b, reg(4), reg(0), agx_immediate(1), 0); /* reads v */

   b->cursor = agx_after_block(merge);
   agx_unit_test(b, reg(4));

   agx_opt_register_cache(b->shader);

   /* u's last read: r0 still holds v for the else threads. */
   EXPECT_FALSE(read_of(then_, 0)->discard);
   /* r2 is read nowhere else: its last use in a divergent block is marked. */
   EXPECT_TRUE(read_of(then_, 2)->discard);
   /* v's read on the else side is the last read of r0 for every thread. */
   EXPECT_TRUE(read_of(else_, 0)->discard);
}

/* A loop whose break is divergent: threads that broke out wait at the exit
 * with their value of r6 while the others keep iterating and reuse r6 for a
 * temporary. The loop body must not release r6. */
TEST(DivergentLastUse, LoopKeepsTheExitedThreadsRegister)
{
   Shader s;
   agx_builder *b = s.b;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *body = agx_test_block(b->shader);
   agx_block *brk = agx_test_block(b->shader);
   agx_block *latch = agx_test_block(b->shader);
   agx_block *exit = agx_test_block(b->shader);
   agx_block_add_successor(entry, body);
   agx_block_add_successor(body, brk);
   agx_block_add_successor(body, latch);
   agx_block_add_successor(brk, exit);
   agx_block_add_successor(latch, body);
   body->loop_header = true;
   body->cf_divergent = brk->cf_divergent = latch->cf_divergent = true;

   b->cursor = agx_after_block(entry);
   agx_mov_imm_to(b, reg(8), 1);

   b->cursor = agx_after_block(brk);
   agx_iadd_to(b, reg(6), reg(8), agx_immediate(2), 0); /* y, needed at exit */

   b->cursor = agx_after_block(latch);
   agx_mov_imm_to(b, reg(6), 9);                        /* temporary in r6 */
   agx_iadd_to(b, reg(8), reg(6), reg(8), 0);           /* its last read */

   b->cursor = agx_after_block(exit);
   agx_unit_test(b, reg(6));

   agx_opt_register_cache(b->shader);

   /* The temporary's read must not release r6: exited threads hold y in it. */
   EXPECT_FALSE(read_of(latch, 6)->discard);
}

/* A register reused on one side of a divergent if, which no masked-off
 * thread needs (the else side and the code after never read it): each value
 * it holds gets its last use, as in uniform control flow. */
TEST(DivergentLastUse, ReuseNoMaskedThreadNeedsIsMarked)
{
   Shader s;
   agx_builder *b = s.b;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *then_ = agx_test_block(b->shader);
   agx_block *else_ = agx_test_block(b->shader);
   agx_block *merge = agx_test_block(b->shader);
   agx_block_add_successor(entry, then_);
   agx_block_add_successor(entry, else_);
   agx_block_add_successor(then_, merge);
   agx_block_add_successor(else_, merge);
   then_->cf_divergent = else_->cf_divergent = true;

   b->cursor = agx_after_block(then_);
   agx_mov_imm_to(b, reg(0), 3);                     /* u */
   agx_instr *u_read = agx_iadd_to(b, reg(2), reg(0), agx_immediate(1), 0);
   agx_mov_imm_to(b, reg(0), 5);                     /* t, same register */
   agx_iadd_to(b, reg(4), reg(0), reg(2), 0);

   b->cursor = agx_after_block(else_);
   agx_mov_imm_to(b, reg(4), 1);

   b->cursor = agx_after_block(merge);
   agx_unit_test(b, reg(4));

   agx_opt_register_cache(b->shader);

   EXPECT_TRUE(u_read->src[0].discard);
}

/* The rule follows the instruction set: both AGX3 encodings mark the last use
 * of ReuseNoMaskedThreadNeedsIsMarked, and AGX2, whose hint only discards the
 * register-cache copy, keeps no hints in divergent control flow. */
TEST(DivergentLastUse, FollowsTheInstructionSet)
{
   const struct {
      enum agx_isa_encoding encoding;
      bool marked;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX3B, true},
      {AGX_ISA_ENCODING_AGX3C, true},
      {AGX_ISA_ENCODING_AGX2, false},
   };

   for (const auto &c : cases) {
      Shader s(c.encoding);
      agx_builder *b = s.b;
      agx_block *entry = agx_start_block(b->shader);
      agx_block *then_ = agx_test_block(b->shader);
      agx_block *else_ = agx_test_block(b->shader);
      agx_block *merge = agx_test_block(b->shader);
      agx_block_add_successor(entry, then_);
      agx_block_add_successor(entry, else_);
      agx_block_add_successor(then_, merge);
      agx_block_add_successor(else_, merge);
      then_->cf_divergent = else_->cf_divergent = true;

      b->cursor = agx_after_block(then_);
      agx_mov_imm_to(b, reg(0), 3);
      agx_instr *u_read = agx_iadd_to(b, reg(2), reg(0), agx_immediate(1), 0);
      agx_mov_imm_to(b, reg(0), 5);
      agx_iadd_to(b, reg(4), reg(0), reg(2), 0);

      b->cursor = agx_after_block(else_);
      agx_mov_imm_to(b, reg(4), 1);

      b->cursor = agx_after_block(merge);
      agx_unit_test(b, reg(4));

      agx_opt_register_cache(b->shader);

      EXPECT_EQ(u_read->src[0].discard, c.marked) << "encoding " << c.encoding;
   }
}

/* subgroupShuffle in divergent control flow: each register lane index goes
 * through the scratch register of agx_lower_divergent_shuffle, which must
 * stay zero in every masked-off thread (the hardware adds the index up over
 * the whole quad). No read of it may release it. */
TEST(DivergentLastUse, ShuffleScratchKeepsMaskedLanesZero)
{
   Shader s;
   agx_builder *b = s.b;
   agx_block *entry = agx_start_block(b->shader);
   agx_block *then_ = agx_test_block(b->shader);
   agx_block *else_ = agx_test_block(b->shader);
   agx_block *merge = agx_test_block(b->shader);
   agx_block_add_successor(entry, then_);
   agx_block_add_successor(entry, else_);
   agx_block_add_successor(then_, merge);
   agx_block_add_successor(else_, merge);
   then_->cf_divergent = else_->cf_divergent = true;
   b->shader->any_quad_divergent_shuffle = true;

   const agx_index scratch = reg(2);

   b->cursor = agx_after_block(entry);
   agx_mov_imm_to(b, scratch, 0);
   agx_mov_imm_to(b, reg(4), 1);
   agx_mov_imm_to(b, reg(6), 0);
   agx_mov_imm_to(b, reg(8), 1);

   b->cursor = agx_after_block(then_);
   agx_mov_to(b, scratch, reg(6));
   agx_instr *s0 = agx_shuffle_to(b, reg(10), reg(4), scratch);
   agx_mov_to(b, scratch, reg(8));
   agx_instr *s1 = agx_shuffle_to(b, reg(12), reg(4), scratch);
   agx_mov_imm_to(b, scratch, 0);
   agx_iadd_to(b, reg(14), reg(10), reg(12), 0);

   b->cursor = agx_after_block(else_);
   agx_mov_imm_to(b, reg(14), 0);

   b->cursor = agx_after_block(merge);
   agx_unit_test(b, reg(14));

   agx_opt_register_cache(b->shader);

   EXPECT_FALSE(s0->src[1].discard);
   EXPECT_FALSE(s1->src[1].discard);
}
