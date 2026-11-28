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

   EXPECT_TRUE(agx_nir_shrink_memory_loads(b.shader, false));

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

   EXPECT_FALSE(agx_nir_shrink_memory_loads(b.shader, false));

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

TEST_F(ShrinkMemoryLoadsTest, TrimsLoweredAgxLoadsAndSkipsPackedWords)
{
   nir_builder b = make_cs("trim_lowered_agx");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);

   nir_def *ld_agx =
      nir_load_agx(&b, 4, 32, base, nir_imm_int(&b, 0),
                   .access = ACCESS_NON_WRITEABLE, .base = 0,
                   .format = PIPE_FORMAT_R32_UINT);
   nir_def *ld_const =
      nir_load_constant_agx(&b, 4, 32, base, nir_imm_int(&b, 16),
                            .base = 0, .format = PIPE_FORMAT_R32_UINT);
   nir_def *ld_packed =
      nir_load_agx(&b, 4, 16, base, nir_imm_int(&b, 32),
                   .access = ACCESS_NON_WRITEABLE,
                   .base = (2u << AGX_NIR_MEMORY_PACKED_COUNT_SHIFT),
                   .format = PIPE_FORMAT_R16_UINT);
   nir_def *ld_w8 =
      nir_load_agx(&b, 4, 8, base, nir_imm_int(&b, 64),
                   .access = ACCESS_NON_WRITEABLE,
                   .base = AGX_NIR_MEMORY_UNPACK_W8_WORD,
                   .format = PIPE_FORMAT_R8_UINT);

   nir_def *sum = nir_fadd(
      &b,
      nir_fadd(&b, nir_channel(&b, ld_agx, 0), nir_channel(&b, ld_agx, 2)),
      nir_fadd(&b, nir_channel(&b, ld_const, 0),
               nir_fadd(&b, nir_u2f32(&b, nir_channel(&b, ld_packed, 0)),
                        nir_u2f32(&b, nir_channel(&b, ld_w8, 0)))));
   nir_store_global(&b, sum, base, .write_mask = 1, .align_mul = 4);

   EXPECT_TRUE(agx_nir_shrink_memory_loads(b.shader, false));
   EXPECT_EQ(nir_def_as_intrinsic(ld_agx)->def.num_components, 3u);
   EXPECT_EQ(nir_def_as_intrinsic(ld_const)->def.num_components, 1u);
   EXPECT_EQ(nir_def_as_intrinsic(ld_packed)->def.num_components, 4u);
   EXPECT_EQ(nir_def_as_intrinsic(ld_w8)->def.num_components, 4u);

   ralloc_free(b.shader);
}

TEST_F(ShrinkMemoryLoadsTest, SkipsDivergentVec4ToVec3Shrink)
{
   nir_builder b = make_cs("skip_divergent_vec3");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);

   nir_def *ld_xyz =
      nir_load_agx(&b, 4, 32, base, gid,
                   .access = ACCESS_NON_WRITEABLE, .base = 0,
                   .format = PIPE_FORMAT_R32_UINT);
   nir_def *ld_xy =
      nir_load_agx(&b, 4, 32, base, nir_iadd_imm(&b, gid, 16),
                   .access = ACCESS_NON_WRITEABLE, .base = 0,
                   .format = PIPE_FORMAT_R32_UINT);

   nir_def *s_xyz = nir_fadd(&b, nir_channel(&b, ld_xyz, 0),
                             nir_fadd(&b, nir_channel(&b, ld_xyz, 1),
                                      nir_channel(&b, ld_xyz, 2)));
   nir_def *s_xy = nir_fadd(&b, nir_channel(&b, ld_xy, 0),
                            nir_channel(&b, ld_xy, 1));
   nir_store_global(&b, nir_fadd(&b, s_xyz, s_xy),
                    nir_iadd(&b, base, nir_u2u64(&b, nir_ishl_imm(&b, gid, 2))),
                    .write_mask = 1, .align_mul = 4);

   EXPECT_TRUE(agx_nir_shrink_memory_loads(b.shader, false));
   EXPECT_EQ(nir_def_as_intrinsic(ld_xyz)->def.num_components, 4u);
   EXPECT_EQ(nir_def_as_intrinsic(ld_xy)->def.num_components, 2u);

   ralloc_free(b.shader);
}

TEST_F(ShrinkMemoryLoadsTest, PreservesSharedAddressLoadWidthForCse)
{
   nir_builder b = make_cs("preserve_shared_cse");
   nir_def *base = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *gid = nir_channel(&b, nir_load_global_invocation_id(&b, 32), 0);
   nir_def *cond = nir_ieq_imm(&b, nir_iand_imm(&b, gid, 1), 0);

   nir_def *res_then = NULL;
   nir_def *res_else = NULL;
   nir_def *ld_then = NULL;
   nir_def *ld_else = NULL;

   nir_push_if(&b, cond);
   {
      /* Scalar ALU uses only .x, but sibling block needs .xyzw. */
      ld_then = nir_load_agx(&b, 4, 32, base, nir_imm_int(&b, 0),
                             .access = ACCESS_NON_WRITEABLE, .base = 0,
                             .format = PIPE_FORMAT_R32_UINT);
      res_then = nir_fadd_imm(&b, nir_channel(&b, ld_then, 0), 1.0f);
   }
   nir_push_else(&b, NULL);
   {
      ld_else = nir_load_agx(&b, 4, 32, base, nir_imm_int(&b, 0),
                             .access = ACCESS_NON_WRITEABLE, .base = 0,
                             .format = PIPE_FORMAT_R32_UINT);
      res_else = nir_fadd(
         &b, nir_fadd(&b, nir_channel(&b, ld_else, 0), nir_channel(&b, ld_else, 1)),
         nir_fadd(&b, nir_channel(&b, ld_else, 2), nir_channel(&b, ld_else, 3)));
   }
   nir_pop_if(&b, NULL);

   nir_def *phi = nir_if_phi(&b, res_then, res_else);
   nir_store_global(&b, phi, base, .write_mask = 1, .align_mul = 4);

   EXPECT_FALSE(agx_nir_shrink_memory_loads(b.shader, false));
   EXPECT_EQ(nir_def_as_intrinsic(ld_then)->def.num_components, 4u);
   EXPECT_EQ(nir_def_as_intrinsic(ld_else)->def.num_components, 4u);

   ralloc_free(b.shader);
}

TEST_F(ShrinkMemoryLoadsTest, TrimsLeadingUnusedChannelsWithByteOffset)
{
   nir_builder b = make_cs("trim_leading");
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *idx = nir_load_local_invocation_index(&b);

   /* Only .z of a divergent vec4: becomes one element 8 bytes further on. */
   nir_def *ld = nir_load_agx(&b, 4, 32, addr, idx, .base = 2,
                              .format = PIPE_FORMAT_R32_UINT);
   nir_def *v = nir_iadd_imm(&b, nir_channel(&b, ld, 2), 1);
   nir_store_global(&b, v, addr, .write_mask = 1, .align_mul = 4);

   EXPECT_TRUE(agx_nir_shrink_memory_loads(b.shader, true));
   nir_validate_shader(b.shader, "after leading trim");

   nir_intrinsic_instr *load =
      find_intrinsic(b.shader, nir_intrinsic_load_agx);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->def.num_components, 1);
   EXPECT_EQ(nir_intrinsic_base(load),
             2u | (8u << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT));
}

TEST_F(ShrinkMemoryLoadsTest, KeepsLeadingChannelsWithoutByteOffsets)
{
   nir_builder b = make_cs("keep_leading");
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *idx = nir_load_local_invocation_index(&b);

   nir_def *ld = nir_load_agx(&b, 4, 32, addr, idx, .base = 2,
                              .format = PIPE_FORMAT_R32_UINT);
   nir_def *v = nir_iadd(&b, nir_channel(&b, ld, 1), nir_channel(&b, ld, 3));
   nir_store_global(&b, v, addr, .write_mask = 1, .align_mul = 4);

   agx_nir_shrink_memory_loads(b.shader, false);

   nir_intrinsic_instr *load =
      find_intrinsic(b.shader, nir_intrinsic_load_agx);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->def.num_components, 4);
   EXPECT_EQ(nir_intrinsic_base(load), 2u);
}

TEST_F(ShrinkMemoryLoadsTest, TrimsLeadingAndTrailingChannels)
{
   nir_builder b = make_cs("trim_both");
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *idx = nir_load_local_invocation_index(&b);

   /* .y and .z of a vec4 at byte offset 16: a vec2 at byte offset 20. */
   nir_def *ld = nir_load_agx(&b, 4, 32, addr, idx,
                              .base = 2u | (16u << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT),
                              .format = PIPE_FORMAT_R32_UINT);
   nir_def *v = nir_iadd(&b, nir_channel(&b, ld, 1), nir_channel(&b, ld, 2));
   nir_store_global(&b, v, addr, .write_mask = 1, .align_mul = 4);

   EXPECT_TRUE(agx_nir_shrink_memory_loads(b.shader, true));
   nir_validate_shader(b.shader, "after leading and trailing trim");

   nir_intrinsic_instr *load =
      find_intrinsic(b.shader, nir_intrinsic_load_agx);
   ASSERT_NE(load, nullptr);
   EXPECT_EQ(load->def.num_components, 2);
   EXPECT_EQ(nir_intrinsic_base(load),
             2u | (20u << AGX_NIR_MEMORY_BYTE_OFFSET_SHIFT));
}

TEST_F(ShrinkMemoryLoadsTest, KeepsLeadingChannelsOfSharedAddressLoads)
{
   nir_builder b = make_cs("keep_shared_leading");
   nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *idx = nir_load_local_invocation_index(&b);

   /* Two loads of one address: one reads .z, the other .x and .w. */
   nir_def *a = nir_load_agx(&b, 4, 32, addr, idx, .base = 2,
                             .format = PIPE_FORMAT_R32_UINT);
   nir_def *c = nir_load_agx(&b, 4, 32, addr, idx, .base = 2,
                             .format = PIPE_FORMAT_R32_UINT);
   nir_def *v = nir_iadd(&b, nir_channel(&b, a, 2),
                         nir_iadd(&b, nir_channel(&b, c, 0),
                                  nir_channel(&b, c, 3)));
   nir_store_global(&b, v, addr, .write_mask = 1, .align_mul = 4);

   agx_nir_shrink_memory_loads(b.shader, true);

   nir_foreach_function_impl(impl, b.shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_agx)
               EXPECT_EQ(nir_intrinsic_base(intr), 2u);
         }
      }
   }
}
