/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

class ShrinkMemoryLoadsTest : public testing::Test {
 protected:
   ShrinkMemoryLoadsTest()
   {
      glsl_type_singleton_init_or_ref();
   }

   ~ShrinkMemoryLoadsTest()
   {
      glsl_type_singleton_decref();
   }

   nir_builder make_cs(const char *name)
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "%s",
                                                     name);
      b.shader->info.io_lowered = true;
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      return b;
   }

   nir_intrinsic_instr *find_intrinsic(nir_shader *s, nir_intrinsic_op op)
   {
      nir_foreach_function_impl(impl, s) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type == nir_instr_type_intrinsic) {
                  nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
                  if (intr->intrinsic == op)
                     return intr;
               }
            }
         }
      }
      return NULL;
   }
};

TEST_F(ShrinkMemoryLoadsTest, TrimsTrailingUnusedScalarAluChannels)
{
   nir_builder b = make_cs("trim_scalar_alu");
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);

   nir_def *ld_ubo = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
                                  nir_imm_int(&b, 0), .range = 16);
   nir_def *ld_ssbo = nir_load_ssbo(&b, 4, 32, nir_imm_int(&b, 1),
                                    nir_imm_int(&b, 0));
   nir_def *ld_glob = nir_load_global(&b, 4, 16, addr, .align_mul = 8);
   nir_def *ld_gc = nir_load_global_constant(&b, 4, 32,
                                             nir_iadd_imm(&b, addr, 64),
                                             .align_mul = 16);

   nir_def *s_ubo = nir_fadd(&b, nir_channel(&b, ld_ubo, 0),
                             nir_channel(&b, ld_ubo, 2));
   nir_def *s_ssbo = nir_iadd(&b, nir_channel(&b, ld_ssbo, 0),
                              nir_channel(&b, ld_ssbo, 1));
   nir_def *s_glob = nir_fadd(&b, nir_channel(&b, ld_glob, 0),
                              nir_channel(&b, ld_glob, 2));
   nir_def *s_gc = nir_fadd(&b, s_ubo,
                            nir_fadd(&b, s_ssbo,
                                     nir_fadd(&b, nir_f2f32(&b, s_glob),
                                              nir_channel(&b, ld_gc, 0))));
   nir_store_global(&b, s_gc, addr, .write_mask = 1, .align_mul = 4);

   EXPECT_TRUE(agx_nir_shrink_memory_loads(b.shader));

   EXPECT_EQ(find_intrinsic(b.shader, nir_intrinsic_load_ubo)->def.num_components, 3u);
   EXPECT_EQ(find_intrinsic(b.shader, nir_intrinsic_load_ssbo)->def.num_components, 2u);
   EXPECT_EQ(find_intrinsic(b.shader, nir_intrinsic_load_global)->def.num_components, 3u);
   EXPECT_EQ(find_intrinsic(b.shader, nir_intrinsic_load_global_constant)->def.num_components, 1u);

   ralloc_free(b.shader);
}

TEST_F(ShrinkMemoryLoadsTest, SkipsVectorPackingAndNonAluUses)
{
   nir_builder b = make_cs("skip_vec_and_non_alu");
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);

   /* Feeding nir_vec3 must leave vec4 load intact. */
   nir_def *ld_vec = nir_load_global(&b, 4, 32, addr, .align_mul = 16);
   nir_def *v3 = nir_vec3(&b, nir_channel(&b, ld_vec, 0),
                          nir_channel(&b, ld_vec, 1),
                          nir_channel(&b, ld_vec, 2));
   nir_store_global(&b, v3, addr, .write_mask = 0x7, .align_mul = 4);

   /* Feeding pack_64_2x32_split must leave vec4 load intact. */
   nir_def *ld_pack = nir_load_global_constant(&b, 4, 32,
                                               nir_iadd_imm(&b, addr, 32),
                                               .align_mul = 16);
   nir_def *p64 = nir_pack_64_2x32_split(&b, nir_channel(&b, ld_pack, 0),
                                         nir_channel(&b, ld_pack, 1));
   nir_store_global(&b, p64, nir_iadd_imm(&b, addr, 48), .write_mask = 1,
                    .align_mul = 8);

   /* Non-ALU intrinsic use must leave vec4 load intact. */
   nir_def *ld_intr = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 0),
                                   nir_imm_int(&b, 0), .range = 16);
   nir_store_ssbo(&b, ld_intr, nir_imm_int(&b, 0), nir_imm_int(&b, 0),
                  .write_mask = 0x3);

   /* Scratch loads, descriptors, and volatile loads stay intact. */
   nir_def *ld_scratch = nir_load_scratch(&b, 4, 16, nir_imm_int(&b, 0),
                                          .align_mul = 8);
   nir_store_global(&b, nir_channel(&b, ld_scratch, 0),
                    nir_iadd_imm(&b, addr, 64), .write_mask = 1,
                    .align_mul = 2);

   nir_def *ld_desc = nir_load_ubo(&b, 4, 32, nir_imm_int(&b, 1),
                                   nir_imm_int(&b, 0),
                                   .access = ACCESS_BUFFER_DESCRIPTOR_AGX,
                                   .range = 16);
   nir_store_global(&b, nir_channel(&b, ld_desc, 0),
                    nir_iadd_imm(&b, addr, 80), .write_mask = 1,
                    .align_mul = 4);

   nir_def *ld_vol = nir_load_global(&b, 4, 32, nir_iadd_imm(&b, addr, 96),
                                     .access = ACCESS_VOLATILE,
                                     .align_mul = 16);
   nir_store_global(&b, nir_channel(&b, ld_vol, 0),
                    nir_iadd_imm(&b, addr, 112), .write_mask = 1,
                    .align_mul = 4);

   EXPECT_FALSE(agx_nir_shrink_memory_loads(b.shader));

   ralloc_free(b.shader);
}

TEST_F(ShrinkMemoryLoadsTest, CompilesVectorConstructingLoadWithoutSpillingOnG16GAndG17P)
{
   for (unsigned chip_idx = 0; chip_idx < 2; ++chip_idx) {
      const bool g17p = chip_idx == 1;
      nir_builder b = make_cs("rt_vec4_no_spill");
      nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);

      nir_def *accum[8];
      for (unsigned i = 0; i < 8; ++i) {
         nir_def *hdr = nir_load_global(&b, 4, 32,
                                        nir_iadd_imm(&b, base, i * 16),
                                        .align_mul = 16);
         accum[i] = nir_vec4(&b, nir_channel(&b, hdr, 0),
                             nir_channel(&b, hdr, 1),
                             nir_channel(&b, hdr, 2),
                             nir_imm_float(&b, (float)i));
      }

      nir_def *sum = accum[0];
      for (unsigned i = 1; i < 8; ++i)
         sum = nir_fadd(&b, sum, accum[i]);

      nir_store_global(&b, sum, base, .write_mask = 0xf, .align_mul = 16);

      struct agx_shader_key key = {};
      key.dev.chip = g17p ? AGX_CHIP_G17P : AGX_CHIP_G16G;
      key.dev.encoding = agx_chip_isa_encoding(key.dev.chip);
      if (g17p) {
         key.agx3_native_cprog = true;
         key.g17p_native_uniforms = true;
      }

      struct agx_shader_part out = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));
      EXPECT_EQ(out.info.stats.spills, 0u);
      EXPECT_EQ(out.info.stats.fills, 0u);
      free(out.binary);
      ralloc_free(b.shader);
   }
}
