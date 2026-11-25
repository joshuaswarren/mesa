/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx3_copy_core.h"
#include "agx_test.h"

/* A phi reading the uniform file becomes a move from it on that edge. On G17P
 * the 16-bit uniform copy names only the lower 128 halves, so the phi must be
 * allocated there even when its other source sits above them. */

namespace {

/* entry -> {then, else} -> merge, with `live` 32-bit values live throughout and
 * a phi of `size` taking a uniform from `then`. The phi's other source is
 * defined in `else` after values that are dead by the merge, so it sits just
 * above the lower 128 halves while they are free again for the phi. Returns
 * the program. */
agx_context *
diamond(void *mem, agx_shader_key *key, enum agx_size size, unsigned live)
{
   agx_builder *b = agx_test_builder(mem);
   *key = {};
   key->dev.encoding = AGX_ISA_ENCODING_AGX3B;
   key->dev.chip = AGX_CHIP_G17P;
   b->shader->key = key;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   b->shader->any_cf = true;

   agx_block *entry = agx_start_block(b->shader);
   agx_block *then = agx_test_block(b->shader);
   agx_block *els = agx_test_block(b->shader);
   agx_block *merge = agx_test_block(b->shader);
   agx_block_add_successor(entry, then);
   agx_block_add_successor(entry, els);
   agx_block_add_successor(then, merge);
   agx_block_add_successor(els, merge);

   b->cursor = agx_after_block(entry);
   agx_index values[96];
   assert(live <= ARRAY_SIZE(values));
   for (unsigned i = 0; i < live; ++i)
      values[i] = agx_mov_imm(b, 32, 0x1000 + i);

   b->cursor = agx_after_block(els);
   agx_index scratch[4];
   for (unsigned i = 0; i < ARRAY_SIZE(scratch); ++i)
      scratch[i] = agx_mov_imm(b, 32, 0x2000 + i);
   /* And the free half beside the reserved r0l. */
   agx_index scratch16 = agx_mov_imm(b, 16, 0x30);
   agx_index other = agx_mov_imm(b, agx_size_align_16(size) * 16, 7);
   for (unsigned i = 0; i < ARRAY_SIZE(scratch); ++i)
      agx_unit_test(b, scratch[i]);
   agx_unit_test(b, scratch16);

   b->cursor = agx_after_block(merge);
   agx_index phi = agx_temp(b->shader, size);
   agx_instr *I = agx_phi_to(b, phi, 2);
   I->src[0] = agx_uniform(40, size);
   I->src[1] = other;
   agx_unit_test(b, phi);
   for (unsigned i = 0; i < live; ++i)
      agx_unit_test(b, values[i]);

   return b->shader;
}

/* The highest destination of a move from the uniform file, or -1. */
int
highest_uniform_move(agx_context *ctx, enum agx_size size)
{
   int highest = -1;
   agx_foreach_instr_global(ctx, I) {
      if (I->op == AGX_OPCODE_MOV && I->src[0].type == AGX_INDEX_UNIFORM &&
          I->dest[0].size == size)
         highest = MAX2(highest, (int)I->dest[0].value);
   }
   return highest;
}

} // namespace

TEST(RAUniformPhi, HalfPhiFromUniformStaysLow)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key;
   agx_context *ctx = diamond(mem, &key, AGX_SIZE_16, 60);

   agx_ra(ctx);
   ASSERT_FALSE(ctx->ra_failed);

   const int dest = highest_uniform_move(ctx, AGX_SIZE_16);
   ASSERT_GE(dest, 0) << "the phi's edge copy";
   EXPECT_LT(dest, (int)AGX3_COPY_REG16_DEST_HALVES);
   ralloc_free(mem);
}

/* G17P's 32-bit uniform copy names every register, so such a phi keeps the
 * whole file. */
TEST(RAUniformPhi, WordPhiFromUniformMayUseTheUpperFile)
{
   void *mem = ralloc_context(NULL);
   agx_shader_key key;
   agx_context *ctx = diamond(mem, &key, AGX_SIZE_32, 60);

   agx_ra(ctx);
   ASSERT_FALSE(ctx->ra_failed);

   const int dest = highest_uniform_move(ctx, AGX_SIZE_32);
   ASSERT_GE(dest, 0) << "the phi's edge copy";
   EXPECT_GE(dest, (int)AGX3_COPY_REG16_DEST_HALVES);
   ralloc_free(mem);
}
