/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_nir.h"
}

static nir_intrinsic_instr *
find_agx_load(nir_shader *s)
{
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic == nir_intrinsic_load_agx ||
                intr->intrinsic == nir_intrinsic_load_constant_agx)
               return intr;
         }
      }
   }
   return NULL;
}

class NarrowSubwordLoads : public ::testing::Test {
 protected:
   NarrowSubwordLoads() { glsl_type_singleton_init_or_ref(); }
   ~NarrowSubwordLoads() { glsl_type_singleton_decref(); }
};

TEST_F(NarrowSubwordLoads, ByteUbitfieldExtract)
{
   for (unsigned variant = 0; variant < 4; ++variant) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "narrow byte bfe");
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *off = nir_load_subgroup_invocation(&b);

      const int load_base =
         variant == 3 ? (int)AGX_NIR_MEMORY_INDEX_BYTES : 0;
      nir_def *idx = variant == 3 ? nir_iand_imm(&b, off, 0xfffffffcu)
                                  : nir_ushr_imm(&b, off, 2);
      nir_def *word =
         variant == 2
            ? nir_load_constant_agx(&b, 1, 32, base, idx,
                                    .access = ACCESS_NON_WRITEABLE,
                                    .base = load_base,
                                    .format = PIPE_FORMAT_R32_UINT,
                                    .sign_extend = false)
            : nir_load_agx(&b, 1, 32, base, idx, .access = ACCESS_NON_WRITEABLE,
                           .base = load_base, .format = PIPE_FORMAT_R32_UINT,
                           .sign_extend = false);

      nir_def *shift;
      if (variant == 0)
         shift = nir_ishl_imm(&b, nir_iand_imm(&b, off, 3), 3);
      else if (variant == 1)
         shift = nir_iand_imm(&b, nir_ishl_imm(&b, off, 3), 31);
      else if (variant == 2)
         shift = nir_iand_imm(&b, nir_imul_imm(&b, off, 8), 24);
      else
         shift = nir_imul_imm(&b, nir_iand_imm(&b, off, 3), 8);

      nir_def *val = nir_ubitfield_extract(&b, word, shift, nir_imm_int(&b, 8));
      nir_store_global(&b, val, base, .align_mul = 4);

      ASSERT_TRUE(agx_nir_opt_narrow_subword_loads(b.shader)) << variant;
      nir_validate_shader(b.shader, "after narrow byte bfe");

      nir_intrinsic_instr *load = find_agx_load(b.shader);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(load->intrinsic, variant == 2 ? nir_intrinsic_load_constant_agx
                                              : nir_intrinsic_load_agx);
      EXPECT_EQ(load->def.bit_size, 16u);
      EXPECT_EQ(nir_intrinsic_format(load), PIPE_FORMAT_R8_UINT);
      EXPECT_EQ(nir_intrinsic_base(load), 0u);
      EXPECT_EQ(load->src[1].ssa, off);
      ralloc_free(b.shader);
   }
}

TEST_F(NarrowSubwordLoads, ByteAndHalfShiftMaskAndConvert)
{
   for (unsigned variant = 0; variant < 4; ++variant) {
      const bool half = variant >= 2;
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "narrow shift mask");
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *off = nir_load_subgroup_invocation(&b);
      nir_def *idx = nir_ushr_imm(&b, off, half ? 1 : 2);
      nir_def *word = nir_load_agx(&b, 1, 32, base, idx,
                                   .access = ACCESS_NON_WRITEABLE, .base = 0,
                                   .format = PIPE_FORMAT_R32_UINT,
                                   .sign_extend = false);

      /* ushr masks shift count mod 32, so bare off << 3 / off << 4 is valid. */
      nir_def *shift =
         (variant & 1) ? nir_ishl_imm(&b, off, half ? 4 : 3)
                       : nir_imul_imm(&b, nir_iand_imm(&b, off, half ? 1 : 3),
                                      half ? 16 : 8);
      nir_def *shifted = nir_ushr(&b, word, shift);
      nir_def *res;
      if (variant == 0)
         res = nir_iand_imm(&b, shifted, 0xff);
      else if (variant == 1)
         res = nir_u2u32(&b, nir_u2u8(&b, shifted));
      else if (variant == 2)
         res = nir_iand_imm(&b, shifted, 0xffff);
      else
         res = nir_u2u32(&b, nir_u2u16(&b, shifted));

      nir_store_global(&b, res, base, .align_mul = 4);

      ASSERT_TRUE(agx_nir_opt_narrow_subword_loads(b.shader)) << variant;
      nir_validate_shader(b.shader, "after narrow shift mask");

      nir_intrinsic_instr *load = find_agx_load(b.shader);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(load->def.bit_size, 16u);
      EXPECT_EQ(nir_intrinsic_format(load),
                half ? PIPE_FORMAT_R16_UINT : PIPE_FORMAT_R8_UINT);
      EXPECT_EQ(nir_intrinsic_base(load), 0u);
      EXPECT_EQ(load->src[1].ssa, off);
      ralloc_free(b.shader);
   }
}

TEST_F(NarrowSubwordLoads, RefusesUnsafeOrMismatchedShapes)
{
   for (unsigned variant = 0; variant < 9; ++variant) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "narrow negative");
      nir_def *base = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *off = nir_load_subgroup_invocation(&b);
      nir_def *other_off = nir_iadd_imm(&b, off, 1);

      nir_def *idx =
         variant == 3 ? nir_ishr_imm(&b, off, 2) : nir_ushr_imm(&b, off, 2);
      enum gl_access_qualifier access = ACCESS_NON_WRITEABLE;
      if (variant == 0)
         access = (enum gl_access_qualifier)(access | ACCESS_COHERENT);
      if (variant == 1)
         access = (enum gl_access_qualifier)(access | ACCESS_VOLATILE);

      nir_def *word = nir_load_agx(
         &b, 1, 32, base, idx, .access = access,
         .base = variant == 8 ? 1 : 0, .format = PIPE_FORMAT_R32_UINT,
         .sign_extend = variant == 2);

      nir_def *val;
      if (variant == 4) {
         /* Mismatched offset between load index and extract shift. */
         nir_def *shift = nir_ishl_imm(&b, nir_iand_imm(&b, other_off, 3), 3);
         val = nir_ubitfield_extract(&b, word, shift, nir_imm_int(&b, 8));
      } else if (variant == 5) {
         /* Unmasked shift into ubitfield_extract (does not mask mod 32). */
         nir_def *shift = nir_ishl_imm(&b, off, 3);
         val = nir_ubitfield_extract(&b, word, shift, nir_imm_int(&b, 8));
      } else if (variant == 6) {
         /* Extracting 16 bits from a byte-shifted word is not a 1-byte load. */
         nir_def *shift = nir_ishl_imm(&b, nir_iand_imm(&b, off, 3), 3);
         val = nir_ubitfield_extract(&b, word, shift, nir_imm_int(&b, 16));
      } else if (variant == 7) {
         /* Multi-use load must stay 32-bit. */
         nir_def *shift = nir_ishl_imm(&b, nir_iand_imm(&b, off, 3), 3);
         nir_def *ext =
            nir_ubitfield_extract(&b, word, shift, nir_imm_int(&b, 8));
         val = nir_iadd(&b, ext, word);
      } else {
         nir_def *shift = nir_ishl_imm(&b, nir_iand_imm(&b, off, 3), 3);
         val = nir_ubitfield_extract(&b, word, shift, nir_imm_int(&b, 8));
      }
      nir_store_global(&b, val, base, .align_mul = 4);

      EXPECT_FALSE(agx_nir_opt_narrow_subword_loads(b.shader)) << variant;
      nir_intrinsic_instr *load = find_agx_load(b.shader);
      ASSERT_NE(load, nullptr);
      EXPECT_EQ(nir_intrinsic_format(load), PIPE_FORMAT_R32_UINT) << variant;
      ralloc_free(b.shader);
   }
}
