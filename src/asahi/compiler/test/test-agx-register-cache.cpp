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

   Shader()
   {
      mem = ralloc_context(NULL);
      b = agx_test_builder(mem);
      key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.dev.chip = AGX_CHIP_G17P;
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

} // namespace

/* A uniform loop whose `if (c) break;` agx_opt_break_if fused into the block
 * after the if: the edge to the exit stays on the emptied then block. The
 * break's compare source is read again after the loop, so it is not a last
 * use. */
TEST(RegisterCache, FusedBreakKeepsTheExitsValue)
{
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
   EXPECT_FALSE(brk->src[0].discard);
}
