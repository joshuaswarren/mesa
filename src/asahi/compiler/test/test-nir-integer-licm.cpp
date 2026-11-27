/* SPDX-License-Identifier: MIT */

#include <gtest/gtest.h>
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "agx_compiler.h"
#include "agx_nir.h"
}

TEST(NirIntegerLICM, GuardsMemoryAndBranchLocalWork)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "licm");
   nir_def *x = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *ptr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *take = nir_ine_imm(&b, x, 0);
   nir_loop *loop = nir_push_loop(&b);
   nir_push_if(&b, take);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *limit = nir_load_preamble(&b, 1, 32, .base = 4);
   nir_def *offset = nir_umin(&b, x, limit);
   nir_def *load = nir_load_global(&b, 1, 32, ptr, .access = ACCESS_VOLATILE,
                                   .align_mul = 4);
   nir_block *body = nir_def_instr(load)->block;
   nir_push_if(&b, nir_ine_imm(&b, load, 0));
   nir_def *product = nir_imul(&b, offset, limit);
   nir_intrinsic_instr *store = nir_store_global(&b, product, ptr, .align_mul = 4);
   nir_block *branch = store->instr.block;
   nir_pop_if(&b, NULL);
   nir_pop_loop(&b, loop);

   EXPECT_TRUE(agx_nir_opt_licm_agx2_integer(b.shader, 256));
   nir_block *pre = nir_loop_predecessor_block(loop);
   EXPECT_EQ(nir_def_instr(limit)->block, pre);
   EXPECT_EQ(nir_def_instr(offset)->block, pre);
   EXPECT_EQ(nir_def_instr(load)->block, body);
   EXPECT_EQ(nir_def_instr(product)->block, branch);
   EXPECT_EQ(store->instr.block, branch);
   nir_validate_shader(b.shader, "AGX integer LICM");
   ralloc_free(b.shader);
}

TEST(NirIntegerLICM, RespectsRegisterBudget)
{
   for (unsigned budget : {8u, 128u}) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "licm");
      nir_def *x = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
      nir_def *ptr = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *take = nir_ine_imm(&b, x, 0);
      nir_loop *loop = nir_push_loop(&b);
      nir_push_if(&b, take);
      nir_jump(&b, nir_jump_break);
      nir_pop_if(&b, NULL);

      /* Sequential stores reuse registers until their operands are hoisted
       * and become live together in the preheader. */
      nir_def *values[12];
      for (unsigned i = 0; i < ARRAY_SIZE(values); ++i) {
         values[i] = nir_iadd_imm(&b, x, i + 1);
         nir_store_global(&b, values[i], ptr, .access = ACCESS_VOLATILE,
                          .align_mul = 4);
      }
      nir_block *body = nir_def_instr(values[0])->block;
      nir_pop_loop(&b, loop);

      bool fits = budget == 128;
      EXPECT_EQ(agx_nir_opt_licm_agx2_integer(b.shader, budget), fits);
      for (nir_def *value : values)
         EXPECT_EQ(nir_def_instr(value)->block,
                   fits ? nir_loop_predecessor_block(loop) : body);
      nir_validate_shader(b.shader, "AGX integer LICM pressure");
      ralloc_free(b.shader);
   }
}

TEST(NirIntegerLICM, HoistsWorkgroupIntrinsicsAndBitAddressOps)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "licm-ops");
   nir_def *ptr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *cond = nir_ine_imm(&b, nir_load_preamble(&b, 1, 32, .base = 8), 0);
   nir_loop *loop = nir_push_loop(&b);
   nir_push_if(&b, cond);
   nir_jump(&b, nir_jump_break);
   nir_pop_if(&b, NULL);

   nir_def *lid = nir_load_local_invocation_index(&b);
   nir_def *sg = nir_load_subgroup_invocation(&b);
   nir_def *wg = nir_channel(&b, nir_load_workgroup_id(&b), 0);
   nir_def *bits = nir_ubitfield_extract_imm(&b, lid, 2, 4);
   nir_def *shifted = nir_ior(&b, nir_ishl_imm(&b, bits, 4),
                              nir_iand_imm(&b, sg, 15));
   nir_def *addr =
      nir_imadshl_agx(&b, wg, nir_imm_int(&b, 64), shifted, nir_imm_int(&b, 2));
   nir_store_global(&b, addr, ptr, .access = ACCESS_VOLATILE, .align_mul = 4);
   nir_pop_loop(&b, loop);

   EXPECT_TRUE(agx_nir_opt_licm_agx2_integer(b.shader, 256));
   nir_block *pre = nir_loop_predecessor_block(loop);
   EXPECT_EQ(nir_def_instr(lid)->block, pre);
   EXPECT_EQ(nir_def_instr(sg)->block, pre);
   EXPECT_EQ(nir_def_instr(wg)->block, pre);
   EXPECT_EQ(nir_def_instr(bits)->block, pre);
   EXPECT_EQ(nir_def_instr(shifted)->block, pre);
   EXPECT_EQ(nir_def_instr(addr)->block, pre);
   nir_validate_shader(b.shader, "AGX integer LICM bit/address ops");
   ralloc_free(b.shader);
}
