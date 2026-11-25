/* SPDX-License-Identifier: MIT */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "agx_compile.h"

TEST(NativeForms, AGX3CCollectSplitFoldingFeedsRegisterAllocator)
{
   void *mem = ralloc_context(NULL);
   agx_builder *b = agx_test_builder(mem);
   agx_shader_key key = {};
   agx_shader_info info = {};
   key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
   key.dev.chip = AGX_CHIP_G17S;
   b->shader->key = &key;
   b->shader->out = &info;
   b->shader->stage = MESA_SHADER_COMPUTE;
   b->shader->nir = rzalloc(mem, nir_shader);
   b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
   b->shader->nir->info.workgroup_size[0] = 32;
   b->shader->nir->info.workgroup_size[1] = 1;
   b->shader->nir->info.workgroup_size[2] = 1;

   agx_index x = agx_mov_imm(b, 32, 0x11111111u);
   agx_index y = agx_mov_imm(b, 32, 0x22222222u);
   agx_index pair = agx_vec_temp(b->shader, AGX_SIZE_32, 2);
   agx_instr *collect = agx_collect_to(b, pair, 2);
   collect->src[0] = x;
   collect->src[1] = y;
   agx_instr *split = agx_split(b, 2, pair);
   split->dest[0] = agx_temp(b->shader, AGX_SIZE_32);
   split->dest[1] = agx_temp(b->shader, AGX_SIZE_32);
   agx_index sum = agx_iadd(b, split->dest[0], split->dest[1], 0);
   agx_unit_test(b, sum);

   ASSERT_TRUE(agx_opt_collect_split(b->shader));
   agx_dce(b->shader, true);
   agx_foreach_instr_global(b->shader, I) {
      EXPECT_NE(I->op, AGX_OPCODE_COLLECT);
      EXPECT_NE(I->op, AGX_OPCODE_SPLIT);
   }
   agx_ra(b->shader);
   ASSERT_FALSE(b->shader->ra_failed);
   ralloc_free(mem);
}
