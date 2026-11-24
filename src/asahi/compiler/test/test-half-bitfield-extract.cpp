/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"

extern "C" {
#include "agx_nir.h"
}

static unsigned
count_alu(nir_shader *shader, nir_op op)
{
   unsigned n = 0;
   nir_foreach_function_impl(impl, shader) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_alu &&
                nir_instr_as_alu(instr)->op == op)
               n++;
         }
      }
   }
   return n;
}

TEST(HalfBitfieldExtract, SharesShiftAcrossByte1AndByte3AndLowersByte2)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "rgba8-unpack");

   nir_def *p0 = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_def *p1 = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 4), .align_mul = 4);

   nir_def *g0 = nir_ubitfield_extract(&b, p0, nir_imm_int(&b, 8), nir_imm_int(&b, 8));
   nir_def *b0 = nir_iand_imm(&b, nir_extract_u16(&b, p0, nir_imm_int(&b, 1)), 255);
   nir_def *a0 = nir_ushr_imm(&b, p0, 24);

   nir_def *g1 = nir_ubitfield_extract(&b, p1, nir_imm_int(&b, 8), nir_imm_int(&b, 8));
   nir_def *b1 = nir_iand_imm(&b, nir_extract_u16(&b, p1, nir_imm_int(&b, 1)), 255);
   nir_def *a1 = nir_ushr_imm(&b, p1, 24);

   nir_def *sum = nir_iadd(&b, nir_iadd(&b, nir_iadd(&b, g0, g1), nir_iadd(&b, b0, b1)),
                           nir_iadd(&b, a0, a1));
   nir_store_global(&b, sum, nir_imm_int64(&b, 8), .align_mul = 4);

   EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
   nir_opt_dce(b.shader);

   /* Each packed word now needs only 1 ushr (p >> 8) instead of 2 shifts + 1 bfe. */
   EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 0u);
   EXPECT_EQ(count_alu(b.shader, nir_op_ushr), 2u);
   EXPECT_EQ(count_alu(b.shader, nir_op_unpack_32_2x16_split_x), 2u);
   EXPECT_EQ(count_alu(b.shader, nir_op_unpack_32_2x16_split_y), 4u);

   nir_validate_shader(b.shader, "rgba8-unpack result");
   ralloc_free(b.shader);
}

/* Non-widening consumers (e.g. direct store) must keep the original form so
 * agx_nir_fold_extract16_bfe emits a single bfeil instead of u2u32 + copy/clr. */
TEST(HalfBitfieldExtract, LeavesNonWideningConsumersUntouched)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "non-widening");

   nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_def *hi = nir_iand_imm(&b, nir_extract_u16(&b, p, nir_imm_int(&b, 1)), 255);
   nir_store_global(&b, hi, nir_imm_int64(&b, 4), .align_mul = 4);

   EXPECT_FALSE(agx_nir_opt_half_bitfield_extract(b.shader));
   nir_validate_shader(b.shader, "non-widening result");
   ralloc_free(b.shader);
}

/* Full end-to-end AGX3 compile across G17P, G17S, and G18G must succeed with
 * zero refusal and eliminate bfeil for byte 2. */
TEST(HalfBitfieldExtract, CompilesCleanlyAcrossAGX3)
{
   const enum agx_chip chips[] = {AGX_CHIP_G17P, AGX_CHIP_G17S, AGX_CHIP_G18G};
   for (enum agx_chip chip : chips) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "rgba8-compile");
      b.shader->info.workgroup_size[0] = 16;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *addr = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *p0 = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
      nir_def *p1 = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4), .align_mul = 4);

      nir_def *g = nir_iadd(&b,
                            nir_iand_imm(&b, nir_ushr_imm(&b, p0, 8), 255),
                            nir_iand_imm(&b, nir_ushr_imm(&b, p1, 8), 255));
      nir_def *bl = nir_iadd(&b,
                             nir_iand_imm(&b, nir_ushr_imm(&b, p0, 16), 255),
                             nir_iand_imm(&b, nir_ushr_imm(&b, p1, 16), 255));
      nir_def *a = nir_iadd(&b, nir_ushr_imm(&b, p0, 24), nir_ushr_imm(&b, p1, 24));
      nir_def *acc = nir_iadd(&b, nir_imul_imm(&b, g, 86),
                              nir_iadd(&b, nir_imul_imm(&b, bl, 342),
                                       nir_imul_imm(&b, a, 1052)));
      nir_store_global(&b, acc, nir_iadd_imm(&b, addr, 8), .align_mul = 4);

      agx_preprocess_nir_native(b.shader, chip);
      struct agx_shader_key key = {};
      key.dev.chip = chip;
      key.dev.encoding = (chip == AGX_CHIP_G18G) ? AGX_ISA_ENCODING_AGX3C
                                                 : agx_chip_isa_encoding(chip);
      key.dev.emit_agx3d = (chip == AGX_CHIP_G18G);
      key.reserved_preamble = 4;
      key.no_preamble = true;

      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part)) << chip;
      EXPECT_GT(part.info.binary_size, 0u) << chip;
      free(part.binary);
      ralloc_free(b.shader);
   }
}

static uint32_t
eval_global_u32(const nir_shader *orig, uint32_t p0, uint32_t p1,
                bool *is_undef_out = nullptr)
{
   nir_shader *s = nir_shader_clone(NULL, orig);
   nir_foreach_function_impl(impl, s) {
      nir_builder b = nir_builder_create(impl);
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_load_global)
               continue;
            uint64_t addr = nir_src_as_uint(intr->src[0]);
            b.cursor = nir_before_instr(instr);
            /* A vector load reads p0, p1 and two words derived from them. */
            uint32_t words[4] = {p0, p1, p0 ^ 0x13579bdfu, p1 ^ 0xeca86420u};
            nir_def *v = intr->def.num_components == 1
                            ? nir_imm_int(&b, addr == 0 ? p0 : p1)
                            : nir_build_imm(&b, intr->def.num_components, 32,
                                            (nir_const_value[4]){
                                               nir_const_value_for_uint(words[0], 32),
                                               nir_const_value_for_uint(words[1], 32),
                                               nir_const_value_for_uint(words[2], 32),
                                               nir_const_value_for_uint(words[3], 32)});
            nir_def_replace(&intr->def, v);
         }
      }
   }

   bool progress;
   do {
      progress = false;
      progress |= nir_opt_constant_folding(s);
      progress |= nir_opt_copy_prop(s);
      progress |= nir_opt_dce(s);
   } while (progress);

   uint32_t out = 0;
   bool found = false;
   if (is_undef_out)
      *is_undef_out = false;
   nir_foreach_function_impl(impl, s) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
            if (intr->intrinsic != nir_intrinsic_store_global)
               continue;
            if (nir_src_is_undef(intr->src[0])) {
               if (is_undef_out)
                  *is_undef_out = true;
               found = true;
            } else {
               EXPECT_TRUE(nir_src_is_const(intr->src[0]));
               out = nir_src_as_uint(intr->src[0]);
               found = true;
            }
         }
      }
   }
   EXPECT_TRUE(found);
   ralloc_free(s);
   return out;
}

TEST(HalfBitfieldExtract, ValueEquivalenceK15Rem1)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "bfe-k15-rem1");

   nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_def *lo = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 15),
                                       nir_imm_int(&b, 8));
   nir_def *hi_ushr = nir_ushr_imm(&b, p, 31);
   nir_def *hi_ubfe = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 31),
                                            nir_imm_int(&b, 1));
   nir_def *acc = nir_iadd(&b, nir_imul_imm(&b, lo, 7),
                           nir_iadd(&b, nir_imul_imm(&b, hi_ushr, 13),
                                    nir_imul_imm(&b, hi_ubfe, 29)));
   nir_store_global(&b, acc, nir_imm_int64(&b, 8), .align_mul = 4);

   nir_shader *orig = nir_shader_clone(NULL, b.shader);
   EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 0u);
   EXPECT_EQ(count_alu(b.shader, nir_op_ushr), 1u);

   const uint32_t vals[] = {
      0x00000000u, 0xffffffffu, 0x80000000u, 0x7fffffffu,
      0x007f8000u, 0xfedcba98u, 0x80018001u, 0xa55a5aa5u,
   };
   for (uint32_t p0 : vals) {
      EXPECT_EQ(eval_global_u32(b.shader, p0, 0), eval_global_u32(orig, p0, 0));
   }

   nir_validate_shader(b.shader, "bfe-k15-rem1");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

TEST(HalfBitfieldExtract, ValueEquivalenceHiUbfeWidthCases)
{
   const uint32_t vals[] = {
      0x00000000u, 0xffffffffu, 0xfedcba98u, 0x80018001u,
      0x12345678u, 0x0f0f0f0fu, 0xa55a5aa5u,
   };

   /* 1. w == rem (k = 12, rem = 4, hi = ubfe(p, 28, 4)). */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "bfe-hi-w-eq-rem");
      nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
      nir_def *lo = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 12),
                                          nir_imm_int(&b, 8));
      nir_def *hi = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 28),
                                          nir_imm_int(&b, 4));
      nir_def *sum = nir_iadd(&b, nir_imul_imm(&b, lo, 3),
                              nir_imul_imm(&b, hi, 11));
      nir_store_global(&b, sum, nir_imm_int64(&b, 8), .align_mul = 4);

      nir_shader *orig = nir_shader_clone(NULL, b.shader);
      EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
      nir_opt_dce(b.shader);
      EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 0u);
      EXPECT_EQ(count_alu(b.shader, nir_op_iand), 1u);

      for (uint32_t p0 : vals)
         EXPECT_EQ(eval_global_u32(b.shader, p0, 0), eval_global_u32(orig, p0, 0));

      nir_validate_shader(b.shader, "bfe-hi-w-eq-rem");
      ralloc_free(orig);
      ralloc_free(b.shader);
   }

   /* 2. w < rem (k = 8, rem = 8, hi = ubfe(p, 24, 5)): hi needs a 5-bit mask. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "bfe-hi-w-lt-rem");
      nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
      nir_def *lo = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 8),
                                          nir_imm_int(&b, 8));
      nir_def *hi = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 24),
                                          nir_imm_int(&b, 5));
      nir_def *sum = nir_iadd(&b, nir_imul_imm(&b, lo, 5),
                              nir_imul_imm(&b, hi, 17));
      nir_store_global(&b, sum, nir_imm_int64(&b, 8), .align_mul = 4);

      nir_shader *orig = nir_shader_clone(NULL, b.shader);
      EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
      nir_opt_dce(b.shader);
      EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 0u);
      EXPECT_EQ(count_alu(b.shader, nir_op_iand), 2u);

      for (uint32_t p0 : vals)
         EXPECT_EQ(eval_global_u32(b.shader, p0, 0), eval_global_u32(orig, p0, 0));

      nir_validate_shader(b.shader, "bfe-hi-w-lt-rem");
      ralloc_free(orig);
      ralloc_free(b.shader);
   }

   /* 3. w > rem (k = 12, rem = 4, hi = ubfe(p, 28, 8)): offset+bits > 32 is
    * poison in NIR, which the pass refines cleanly to the remaining high bits
    * (p >> 28) without an extra mask. */
   {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "bfe-hi-w-gt-rem");
      nir_def *p = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
      nir_def *lo = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 12),
                                          nir_imm_int(&b, 8));
      nir_def *hi = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 28),
                                          nir_imm_int(&b, 8));
      nir_def *sum = nir_iadd(&b, nir_imul_imm(&b, lo, 3),
                              nir_imul_imm(&b, hi, 11));
      nir_store_global(&b, sum, nir_imm_int64(&b, 8), .align_mul = 4);

      nir_shader *orig = nir_shader_clone(NULL, b.shader);
      EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
      nir_opt_dce(b.shader);
      EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 0u);
      EXPECT_EQ(count_alu(b.shader, nir_op_iand), 1u);

      for (uint32_t p0 : vals) {
         bool opt_undef = false;
         uint32_t actual = eval_global_u32(b.shader, p0, 0, &opt_undef);
         uint32_t ref = (((p0 >> 12) & 0xffu) * 3u) + ((p0 >> 28) * 11u);
         EXPECT_FALSE(opt_undef);
         EXPECT_EQ(actual, ref);
      }

      nir_validate_shader(b.shader, "bfe-hi-w-gt-rem");
      ralloc_free(orig);
      ralloc_free(b.shader);
   }
}

TEST(HalfBitfieldExtract, ValueEquivalenceRGBA8AndStandaloneByte2)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "rgba8-unpack-val");

   nir_def *p0 = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 0), .align_mul = 4);
   nir_def *p1 = nir_load_global(&b, 1, 32, nir_imm_int64(&b, 4), .align_mul = 4);

   nir_def *g0 = nir_ubitfield_extract(&b, p0, nir_imm_int(&b, 8), nir_imm_int(&b, 8));
   nir_def *b0 = nir_iand_imm(&b, nir_extract_u16(&b, p0, nir_imm_int(&b, 1)), 255);
   nir_def *a0 = nir_ushr_imm(&b, p0, 24);

   nir_def *g1 = nir_ubitfield_extract(&b, p1, nir_imm_int(&b, 8), nir_imm_int(&b, 8));
   nir_def *b1 = nir_iand_imm(&b, nir_extract_u16(&b, p1, nir_imm_int(&b, 1)), 255);
   nir_def *a1 = nir_ushr_imm(&b, p1, 24);

   nir_def *sum = nir_iadd(&b, nir_imul_imm(&b, nir_iadd(&b, g0, g1), 86),
                           nir_iadd(&b, nir_imul_imm(&b, nir_iadd(&b, b0, b1), 342),
                                    nir_imul_imm(&b, nir_iadd(&b, a0, a1), 1052)));
   nir_store_global(&b, sum, nir_imm_int64(&b, 8), .align_mul = 4);

   nir_shader *orig = nir_shader_clone(NULL, b.shader);
   EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
   nir_opt_dce(b.shader);

   const uint32_t pairs[][2] = {
      {0x00000000u, 0x00000000u},
      {0xffffffffu, 0xffffffffu},
      {0xfedcba98u, 0x12345678u},
      {0x8001ff00u, 0x00ff0180u},
   };
   for (const auto &pr : pairs) {
      EXPECT_EQ(eval_global_u32(b.shader, pr[0], pr[1]),
                eval_global_u32(orig, pr[0], pr[1]));
   }

   nir_validate_shader(b.shader, "rgba8-unpack-val");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

/* The words of a vectorized load are channels of one vector def: with the
 * shared-memory row of a separable RGBA8 filter, nir_opt_load_store_vectorize
 * merges four adjacent word loads into one vec4 load, and every byte extract
 * then reads its word through a swizzle. Those must take the same shared
 * shift and 16-bit masks as scalar words. */
TEST(HalfBitfieldExtract, VectorLoadComponents)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "rgba8-vec4-unpack");

   nir_def *v = nir_load_global(&b, 4, 32, nir_imm_int64(&b, 0), .align_mul = 16);
   const uint32_t w[4] = {86, 342, 1052, 2554};
   nir_def *acc = nir_imm_int(&b, 32768);
   for (unsigned c = 0; c < 4; ++c) {
      nir_def *p = nir_channel(&b, v, c);
      nir_def *g = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 8), nir_imm_int(&b, 8));
      nir_def *bl = nir_ubitfield_extract(&b, p, nir_imm_int(&b, 16), nir_imm_int(&b, 8));
      nir_def *a = nir_ushr_imm(&b, p, 24);
      acc = nir_iadd(&b, acc, nir_imul_imm(&b, nir_iadd(&b, g, a), w[c]));
      acc = nir_iadd(&b, acc, nir_imul_imm(&b, nir_iadd(&b, bl, g), w[3 - c]));
   }
   nir_store_global(&b, acc, nir_imm_int64(&b, 64), .align_mul = 4);

   /* Read each channel straight from the load, as the vectorizer leaves it. */
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader *orig = nir_shader_clone(NULL, b.shader);

   EXPECT_TRUE(agx_nir_opt_half_bitfield_extract(b.shader));
   nir_opt_dce(b.shader);

   /* One shift per word (p >> 8) and no IC-pipe bitfield extract. */
   EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 0u);
   EXPECT_EQ(count_alu(b.shader, nir_op_ushr), 4u);
   EXPECT_EQ(count_alu(b.shader, nir_op_unpack_32_2x16_split_x), 4u);
   EXPECT_EQ(count_alu(b.shader, nir_op_unpack_32_2x16_split_y), 8u);

   const uint32_t pairs[][2] = {
      {0x00000000u, 0x00000000u},
      {0xffffffffu, 0xffffffffu},
      {0xfedcba98u, 0x12345678u},
      {0x8001ff00u, 0x00ff0180u},
   };
   for (const auto &pr : pairs) {
      EXPECT_EQ(eval_global_u32(b.shader, pr[0], pr[1]),
                eval_global_u32(orig, pr[0], pr[1]));
   }

   nir_validate_shader(b.shader, "rgba8-vec4-unpack");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

/* The extracts of one component never borrow another component's shift: a
 * byte-1 extract of channel 0 pairs with a byte-3 extract of channel 0 only. */
TEST(HalfBitfieldExtract, VectorComponentsDoNotPairAcrossChannels)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "rgba8-vec2-cross");

   nir_def *v = nir_load_global(&b, 2, 32, nir_imm_int64(&b, 0), .align_mul = 8);
   nir_def *g0 = nir_ubitfield_extract(&b, nir_channel(&b, v, 0), nir_imm_int(&b, 8),
                                       nir_imm_int(&b, 8));
   nir_def *a1 = nir_ushr_imm(&b, nir_channel(&b, v, 1), 24);
   nir_def *sum = nir_iadd(&b, nir_imul_imm(&b, g0, 86), nir_imul_imm(&b, a1, 342));
   nir_store_global(&b, sum, nir_imm_int64(&b, 64), .align_mul = 4);
   nir_opt_copy_prop(b.shader);
   nir_opt_dce(b.shader);
   nir_shader *orig = nir_shader_clone(NULL, b.shader);

   /* No pair exists; nothing about channel 0 may be rewritten from channel 1. */
   agx_nir_opt_half_bitfield_extract(b.shader);
   nir_opt_dce(b.shader);
   EXPECT_EQ(count_alu(b.shader, nir_op_ubitfield_extract), 1u);

   const uint32_t pairs[][2] = {
      {0xfedcba98u, 0x12345678u},
      {0x00ff00ffu, 0xff00ff00u},
   };
   for (const auto &pr : pairs) {
      EXPECT_EQ(eval_global_u32(b.shader, pr[0], pr[1]),
                eval_global_u32(orig, pr[0], pr[1]));
   }

   nir_validate_shader(b.shader, "rgba8-vec2-cross");
   ralloc_free(orig);
   ralloc_free(b.shader);
}

/* End to end on G16G (AGX3B): a separable filter's shared-memory row. Adjacent
 * words are merged into vector loads and words 32 bytes apart are not (and
 * keep their offsets within the local load's immediate); the byte unpack must
 * cost the same integer-pipe work either way. */
static unsigned
g16g_shared_unpack_ic(unsigned stride_B)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "rgba8-shared-row");
   b.shader->info.internal = false;
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   b.shader->info.shared_size = 4096;

   nir_def *lid = nir_channel(&b, nir_load_local_invocation_id(&b), 0);
   nir_def *offs = nir_u2u16(&b, nir_imul_imm(&b, lid, 4));
   nir_def *out_base = nir_load_preamble(&b, 1, 64, .base = 0);

   const uint32_t w[8] = {16, 86, 342, 1052, 2554, 4996, 7914, 10242};
   nir_def *acc[4];
   for (unsigned c = 0; c < 4; ++c)
      acc[c] = nir_imm_int(&b, 32768);
   for (unsigned t = 0; t < 8; ++t) {
      nir_def *p = nir_load_shared(&b, 1, 32, offs, .base = (int)(t * stride_B),
                                   .align_mul = 4);
      nir_def *ch[4] = {
         nir_iand_imm(&b, p, 0xff),
         nir_ubitfield_extract(&b, p, nir_imm_int(&b, 8), nir_imm_int(&b, 8)),
         nir_ubitfield_extract(&b, p, nir_imm_int(&b, 16), nir_imm_int(&b, 8)),
         nir_ushr_imm(&b, p, 24),
      };
      for (unsigned c = 0; c < 4; ++c)
         acc[c] = nir_iadd(&b, acc[c], nir_imul_imm(&b, ch[c], w[t]));
   }
   nir_def *res = nir_ior(&b, nir_ior(&b, nir_ushr_imm(&b, acc[0], 16),
                                      nir_ishl_imm(&b, nir_ushr_imm(&b, acc[1], 16), 8)),
                          nir_ior(&b, nir_ishl_imm(&b, nir_ushr_imm(&b, acc[2], 16), 16),
                                  nir_ishl_imm(&b, nir_ushr_imm(&b, acc[3], 16), 24)));
   nir_store_global(&b, res, nir_iadd(&b, out_base, nir_u2u64(&b, nir_imul_imm(&b, lid, 4))),
                    .align_mul = 4);

   struct agx_shader_key key = {};
   key.dev.chip = AGX_CHIP_G16G;
   key.dev.encoding = agx_chip_isa_encoding(AGX_CHIP_G16G);
   key.reserved_preamble = 4;
   key.no_preamble = true;
   agx_preprocess_nir_native(b.shader, AGX_CHIP_G16G);
   struct agx_shader_part part = {};
   EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
   const unsigned ic = part.info.stats.ic;
   free(part.binary);
   ralloc_free(b.shader);
   return ic;
}

TEST(HalfBitfieldExtract, G16GVectorizedRowMatchesStridedRow)
{
   const unsigned vectorized = g16g_shared_unpack_ic(4);
   const unsigned strided = g16g_shared_unpack_ic(32);
   EXPECT_GT(strided, 0u);
   EXPECT_EQ(vectorized, strided);
}
