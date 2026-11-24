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

class PreambleHoistTest : public testing::Test {
 protected:
   PreambleHoistTest()
   {
      glsl_type_singleton_init_or_ref();
   }

   ~PreambleHoistTest()
   {
      glsl_type_singleton_decref();
   }

   struct agx_shader_key make_agx3_vs_key()
   {
      struct agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G16G;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      key.agx3_native_cprog = true;
      key.g17p_native_uniforms = true;
      key.reserved_preamble = 64;
      return key;
   }

   nir_builder make_vs(const char *name)
   {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX,
                                                     &agx_nir_options, "%s",
                                                     name);
      b.shader->info.io_lowered = true;
      return b;
   }

   void emit_pos_store(nir_builder *b, nir_def *v4)
   {
      for (unsigned c = 0; c < 4; ++c)
         nir_store_uvs_agx(b, nir_channel(b, v4, c), nir_imm_int(b, c));
      b->shader->info.outputs_written |= VARYING_BIT_POS;
   }
};

TEST_F(PreambleHoistTest, RootOnlyPreambleIsHoistable)
{
   nir_builder b = make_vs("hoist_root_only");

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 60);
   nir_def *ubo_ptr = nir_load_global_constant(&b, 1, 64, root, .align_mul = 8);
   nir_def *m0 = nir_load_global_constant(&b, 4, 32, ubo_ptr, .align_mul = 16);
   nir_def *m1 = nir_load_global_constant(&b, 4, 32,
                                          nir_iadd_imm(&b, ubo_ptr, 16),
                                          .align_mul = 16);
   nir_def *sum = nir_fadd(&b, m0, m1);

   nir_def *vid = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *pos = nir_fmul(&b, sum, nir_vec4(&b, vid, vid, vid, vid));
   emit_pos_store(&b, pos);

   struct agx_shader_key key = make_agx3_vs_key();
   struct agx_shader_part out = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));

   EXPECT_TRUE(out.info.has_preamble);
   EXPECT_TRUE(out.info.agx3_preamble_hoistable);
   EXPECT_EQ(out.info.agx3_preamble_in_base_half, 60u);
   EXPECT_EQ(out.info.agx3_preamble_in_size_halfs, 4u);
   EXPECT_GE(out.info.agx3_preamble_out_start_half, 64u);
   EXPECT_GT(out.info.agx3_preamble_out_size_halfs, 0u);
   EXPECT_LE(out.info.agx3_preamble_out_size_halfs, 128u);
   EXPECT_EQ(out.info.agx3_preamble_out_start_half % 2u, 0u);
   EXPECT_EQ(out.info.agx3_preamble_out_size_halfs % 2u, 0u);

   free(out.binary);
   ralloc_free(b.shader);
}

TEST_F(PreambleHoistTest, ExtraPerDrawInputIsNotHoistable)
{
   nir_builder b = make_vs("hoist_reject_base_instance");

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 60);
   nir_def *base_inst = nir_load_preamble(&b, 1, 32, .base = 46);
   nir_def *ubo_ptr = nir_load_global_constant(&b, 1, 64, root, .align_mul = 8);
   nir_def *off = nir_u2u64(&b, nir_imul_imm(&b, base_inst, 16));
   nir_def *v = nir_load_global_constant(&b, 4, 32, nir_iadd(&b, ubo_ptr, off),
                                         .align_mul = 16);

   nir_def *vid = nir_u2f32(&b, nir_load_vertex_id(&b));
   emit_pos_store(&b, nir_fadd(&b, v, nir_vec4(&b, vid, vid, vid, vid)));

   struct agx_shader_key key = make_agx3_vs_key();
   struct agx_shader_part out = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));

   EXPECT_TRUE(out.info.has_preamble);
   EXPECT_FALSE(out.info.agx3_preamble_hoistable);

   free(out.binary);
   ralloc_free(b.shader);
}

TEST_F(PreambleHoistTest, OversizedPreambleOutputIsNotHoistable)
{
   nir_builder b = make_vs("hoist_reject_oversized");

   nir_def *root = nir_load_preamble(&b, 1, 64, .base = 60);
   nir_def *ubo_ptr = nir_load_global_constant(&b, 1, 64, root, .align_mul = 8);
   nir_def *vid = nir_load_vertex_id(&b);
   nir_def *acc = nir_imm_vec4(&b, 0.0f, 0.0f, 0.0f, 0.0f);

   /* 17 distinct vec4s = 68 dwords = 136 halfs > 128-half hoist cap. */
   for (unsigned i = 0; i < 17; ++i) {
      nir_def *v = nir_load_global_constant(&b, 4, 32,
                                            nir_iadd_imm(&b, ubo_ptr, i * 16),
                                            .align_mul = 16);
      nir_def *s = nir_fadd(&b, v, nir_imm_float(&b, (float)(i + 1)));
      nir_def *cond = nir_ieq_imm(&b, vid, i);
      acc = nir_bcsel(&b, cond, s, acc);
   }

   emit_pos_store(&b, acc);

   struct agx_shader_key key = make_agx3_vs_key();
   struct agx_shader_part out = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));

   EXPECT_TRUE(out.info.has_preamble);
   EXPECT_FALSE(out.info.agx3_preamble_hoistable);

   free(out.binary);
   ralloc_free(b.shader);
}

TEST_F(PreambleHoistTest, RejectsNonHoistableOpInPreamble)
{
   nir_builder b = make_vs("hoist_reject_side_effect");

   nir_def *pre_val = nir_load_preamble(&b, 1, 32, .base = 64);
   nir_def *vid = nir_u2f32(&b, nir_load_vertex_id(&b));
   nir_def *f = nir_fadd(&b, pre_val, vid);
   emit_pos_store(&b, nir_vec4(&b, f, f, f, f));

   /* Manual preamble with an external memory store (non-hoistable side effect). */
   nir_function *pre_fn = nir_function_create(b.shader, "preamble");
   pre_fn->is_preamble = true;
   nir_function_impl *pre_impl = nir_function_impl_create(pre_fn);
   nir_builder pb = nir_builder_at(nir_before_impl(pre_impl));
   nir_def *p_root = nir_load_preamble(&pb, 1, 64, .base = 60);
   nir_def *v = nir_load_global_constant(&pb, 1, 32, p_root, .align_mul = 4);
   nir_intrinsic_instr *st = nir_store_global(&pb, v, p_root);
   nir_intrinsic_set_align(st, 4, 0);
   nir_intrinsic_set_write_mask(st, 1);
   nir_store_preamble(&pb, v, .base = 64);
   nir_progress(true, pre_impl, nir_metadata_none);
   nir_progress(true, b.impl, nir_metadata_none);

   struct agx_shader_key key = make_agx3_vs_key();
   struct agx_shader_part out = {};
   ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &out));

   EXPECT_TRUE(out.info.has_preamble);
   EXPECT_FALSE(out.info.agx3_preamble_hoistable);

   free(out.binary);
   ralloc_free(b.shader);
}
