/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

#include "agx_compiler.h"
#include "agx_test.h"

extern "C" {
#include "asahi/isa/disasm.h"
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

namespace {

static const struct glsl_type *
make_cmat_type(enum glsl_base_type base, enum glsl_cmat_use use)
{
   struct glsl_cmat_description desc = {};
   desc.rows = 8;
   desc.cols = 8;
   desc.element_type = base;
   desc.scope = SCOPE_SUBGROUP;
   desc.use = use;
   return glsl_cmat_type(&desc);
}

static nir_shader *
build_cmat_muladd_shader(enum glsl_base_type a_type,
                         enum glsl_base_type c_type,
                         unsigned local_x,
                         nir_deref_instr **ptr_out = nullptr)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "agx2_cmat_test");
   b.shader->info.cs.has_cooperative_matrix = true;
   b.shader->info.workgroup_size[0] = local_x;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.workgroup_size_variable = false;

   nir_variable *va = nir_local_variable_create(
      b.impl, make_cmat_type(a_type, GLSL_CMAT_USE_A), "a");
   nir_variable *vb = nir_local_variable_create(
      b.impl, make_cmat_type(a_type, GLSL_CMAT_USE_B), "b");
   nir_variable *vc = nir_local_variable_create(
      b.impl, make_cmat_type(c_type, GLSL_CMAT_USE_ACCUMULATOR), "c");
   nir_variable *vd = nir_local_variable_create(
      b.impl, make_cmat_type(c_type, GLSL_CMAT_USE_ACCUMULATOR), "d");

   nir_deref_instr *da = nir_build_deref_var(&b, va);
   nir_deref_instr *db = nir_build_deref_var(&b, vb);
   nir_deref_instr *dc = nir_build_deref_var(&b, vc);
   nir_deref_instr *dd = nir_build_deref_var(&b, vd);

   nir_def *addr = nir_pack_64_2x32(
      &b, nir_load_preamble(&b, 2, 32));
   nir_deref_instr *ptr_a = nir_build_deref_cast(
      &b, addr, nir_var_mem_global, glsl_scalar_type(a_type),
      glsl_base_type_bit_size(a_type) / 8);
   nir_deref_instr *ptr_b = nir_build_deref_cast(
      &b, nir_iadd_imm(&b, addr, 256), nir_var_mem_global,
      glsl_scalar_type(a_type), glsl_base_type_bit_size(a_type) / 8);
   nir_deref_instr *ptr_c = nir_build_deref_cast(
      &b, nir_iadd_imm(&b, addr, 512), nir_var_mem_global,
      glsl_scalar_type(c_type), glsl_base_type_bit_size(c_type) / 8);
   nir_deref_instr *ptr_d = nir_build_deref_cast(
      &b, nir_iadd_imm(&b, addr, 1024), nir_var_mem_global,
      glsl_scalar_type(c_type), glsl_base_type_bit_size(c_type) / 8);
   if (ptr_out)
      *ptr_out = ptr_d;

   nir_def *stride = nir_imm_int(&b, 8);
   nir_cmat_load(&b, &da->def, &ptr_a->def, stride,
                 GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_load(&b, &db->def, &ptr_b->def, stride,
                 GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_load(&b, &dc->def, &ptr_c->def, stride,
                 GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   nir_cmat_muladd(&b, &dd->def, &da->def, &db->def, &dc->def);
   nir_cmat_store(&b, &ptr_d->def, &dd->def, stride,
                  GLSL_MATRIX_LAYOUT_ROW_MAJOR);
   return b.shader;
}

static bool
ranges_overlap(unsigned start_a, unsigned len_a, unsigned start_b, unsigned len_b)
{
   return start_a < start_b + len_b && start_b < start_a + len_a;
}

} /* namespace */

TEST(AGX2Cmat, LowerWidensBf16AndKeepsF16Accum)
{
   glsl_type_singleton_init_or_ref();

   struct Case {
      enum glsl_base_type ab;
      enum glsl_base_type c;
      nir_alu_type expect_src_type;
      unsigned expect_def_bits;
   } cases[] = {
      {GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT, nir_type_float32, 32},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT, nir_type_float16, 32},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, nir_type_float16, 16},
      {GLSL_TYPE_BFLOAT16, GLSL_TYPE_FLOAT, nir_type_float32, 32},
      {GLSL_TYPE_BFLOAT16, GLSL_TYPE_BFLOAT16, nir_type_float32, 32},
   };

   for (const auto &tc : cases) {
      nir_shader *nir = build_cmat_muladd_shader(tc.ab, tc.c, 32);
      ASSERT_TRUE(agx_nir_lower_cmat(nir, true, true));
      nir_validate_shader(nir, "after agx_nir_lower_cmat");

      unsigned muladds = 0;
      nir_foreach_function_impl(impl, nir) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_cmat_muladd_agx)
                  continue;
               ++muladds;
               EXPECT_EQ(nir_intrinsic_src_type(intr), tc.expect_src_type);
               EXPECT_EQ(nir_intrinsic_src_type2(intr), tc.expect_src_type);
               EXPECT_EQ(intr->def.bit_size, tc.expect_def_bits);
               EXPECT_EQ(intr->def.num_components, 2u);
            }
         }
      }
      EXPECT_EQ(muladds, 1u);
      ralloc_free(nir);
   }

   glsl_type_singleton_decref();
}

TEST(AGX2Cmat, CompileAllAdvertised8x8Shapes)
{
   glsl_type_singleton_init_or_ref();

   struct Shape {
      enum glsl_base_type ab;
      enum glsl_base_type c;
      const char *expect_mnemonic;
   } shapes[] = {
      {GLSL_TYPE_FLOAT, GLSL_TYPE_FLOAT, "simd_matrix_fmadd32"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT, "simd_matrix_fmadd32"},
      {GLSL_TYPE_FLOAT16, GLSL_TYPE_FLOAT16, "simd_matrix_fmadd16"},
      {GLSL_TYPE_BFLOAT16, GLSL_TYPE_FLOAT, "simd_matrix_fmadd32"},
      {GLSL_TYPE_BFLOAT16, GLSL_TYPE_BFLOAT16, "simd_matrix_fmadd32"},
   };

   for (const auto &shape : shapes) {
      nir_shader *nir = build_cmat_muladd_shader(shape.ab, shape.c, 32);
      ASSERT_TRUE(agx_nir_lower_cmat(nir, true, true));
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
               nir_address_format_64bit_global);
      agx_preprocess_nir(nir);

      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G13G;
      key.dev.encoding = AGX_ISA_ENCODING_AGX2;
      key.no_preamble = true;

      agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(nir, &key, &part));
      ASSERT_GT(part.info.main_size, 0u);

      char *disasm = nullptr;
      size_t disasm_len = 0;
      FILE *fp = open_memstream(&disasm, &disasm_len);
      ASSERT_NE(fp, nullptr);
      bool disasm_err = agx2_disassemble(
         static_cast<uint8_t *>(part.binary) + part.info.main_offset,
         part.info.main_size, fp);
      fclose(fp);
      EXPECT_FALSE(disasm_err);
      ASSERT_NE(disasm, nullptr);
      EXPECT_NE(strstr(disasm, shape.expect_mnemonic), nullptr) << disasm;

      free(disasm);
      free(part.binary);
      ralloc_free(nir);
   }

   glsl_type_singleton_decref();
}

TEST(AGX2Cmat, RANeverAliasesDestWithAOrB)
{
   for (enum agx_size size : {AGX_SIZE_32, AGX_SIZE_16}) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX2;
      key.dev.chip = AGX_CHIP_G13G;
      b->shader->key = &key;
      b->shader->stage = MESA_SHADER_COMPUTE;
      b->shader->nir = rzalloc(mem, nir_shader);
      b->shader->nir->info.stage = MESA_SHADER_COMPUTE;
      b->shader->nir->info.workgroup_size[0] = 32;
      b->shader->nir->info.workgroup_size[1] = 1;
      b->shader->nir->info.workgroup_size[2] = 1;

      const unsigned step = size == AGX_SIZE_32 ? 4 : 2;
      agx_index phys_a = agx_register(0, size);
      phys_a.channels_m1 = 1;
      agx_index phys_b = agx_register(step, size);
      phys_b.channels_m1 = 1;
      agx_index phys_c = agx_register(2 * step, size);
      phys_c.channels_m1 = 1;

      agx_index va = agx_vec_temp(b->shader, size, 2);
      agx_index vb = agx_vec_temp(b->shader, size, 2);
      agx_index vc = agx_vec_temp(b->shader, size, 2);
      agx_preload_to(b, va, phys_a);
      agx_preload_to(b, vb, phys_b);
      agx_preload_to(b, vc, phys_c);

      agx_index dst = agx_vec_temp(b->shader, size, 2);
      if (size == AGX_SIZE_16)
         agx_simd_matrix_fmadd16_to(b, dst, va, vb, vc);
      else
         agx_simd_matrix_fmadd32_to(b, dst, va, vb, vc);
      agx_unit_test(b, dst);

      agx_ra(b->shader);
      ASSERT_FALSE(b->shader->ra_failed);

      unsigned found = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op != AGX_OPCODE_SIMD_MATRIX_FMADD32 &&
             I->op != AGX_OPCODE_SIMD_MATRIX_FMADD16)
            continue;
         ++found;
         unsigned d_start = I->dest[0].value;
         unsigned d_len = agx_index_size_16(I->dest[0]);
         unsigned a_start = I->src[0].value;
         unsigned a_len = agx_index_size_16(I->src[0]);
         unsigned b_start = I->src[1].value;
         unsigned b_len = agx_index_size_16(I->src[1]);
         EXPECT_FALSE(ranges_overlap(d_start, d_len, a_start, a_len));
         EXPECT_FALSE(ranges_overlap(d_start, d_len, b_start, b_len));
      }
      EXPECT_EQ(found, 1u);
      ralloc_free(mem);
   }
}

TEST(AGX2Cmat, SoftwareFallbackPartialSubgroupPSBAndBf16)
{
   glsl_type_singleton_init_or_ref();

   nir_shader *nir = build_cmat_muladd_shader(
      GLSL_TYPE_BFLOAT16, GLSL_TYPE_BFLOAT16, 16);
   ASSERT_TRUE(agx_nir_lower_cmat_sw(nir));
   nir_validate_shader(nir, "after agx_nir_lower_cmat_sw");

   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global,
            nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_lower_compute_system_values, NULL);
   agx_preprocess_nir(nir);

   agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G13G;
   key.dev.encoding = AGX_ISA_ENCODING_AGX2;
   key.no_preamble = true;
   key.has_scratch = true;

   agx_shader_part part = {};
   ASSERT_TRUE(agx_compile_shader_nir(nir, &key, &part));
   ASSERT_GT(part.info.main_size, 0u);

   char *disasm = nullptr;
   size_t disasm_len = 0;
   FILE *fp = open_memstream(&disasm, &disasm_len);
   ASSERT_NE(fp, nullptr);
   bool disasm_err = agx2_disassemble(
      static_cast<uint8_t *>(part.binary) + part.info.main_offset,
      part.info.main_size, fp);
   fclose(fp);
   EXPECT_FALSE(disasm_err);
   ASSERT_NE(disasm, nullptr);
   EXPECT_EQ(strstr(disasm, "simd_matrix_fmadd"), nullptr) << disasm;

   free(disasm);
   free(part.binary);
   ralloc_free(nir);
   glsl_type_singleton_decref();
}
