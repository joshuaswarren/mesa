/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_compile.h"
#include "nir_builder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t
lowered_add64(uint64_t x, uint64_t y)
{
   const uint32_t lo = (uint32_t)x + (uint32_t)y;
   const uint32_t hi = (uint32_t)(x >> 32) + (uint32_t)(y >> 32) +
                       (lo < (uint32_t)x);
   return ((uint64_t)hi << 32) | lo;
}

static uint64_t
lowered_mul64(uint64_t x, uint64_t y)
{
   const uint32_t xl = x, xh = x >> 32;
   const uint32_t yl = y, yh = y >> 32;
   const uint32_t xll = xl & 0xffff, xlh = xl >> 16;
   const uint32_t yll = yl & 0xffff, ylh = yl >> 16;
   const uint32_t lo = xll * yll;
   const uint32_t m1 = xll * ylh;
   const uint32_t m2 = xlh * yll;
   const uint32_t hi = xlh * ylh;
   const uint32_t t = m2 + (lo >> 16);
   const uint32_t w1 = m1 + (t & 0xffff);
   const uint32_t high32 = hi + (w1 >> 16) + (t >> 16) +
                           xl * yh + xh * yl;

   return ((uint64_t)high32 << 32) | (uint32_t)(xl * yl);
}

static bool
check_host_oracle(void)
{
   static const uint64_t values[] = {
      0, 1, UINT64_MAX, UINT64_C(0xffffffff), UINT64_C(0x100000000),
      UINT64_C(0x0123456789abcdef), UINT64_C(0xfedcba9876543210),
      UINT64_C(0x8000000000000000),
   };

   for (unsigned x = 0; x < ARRAY_SIZE(values); ++x) {
      for (unsigned y = 0; y < ARRAY_SIZE(values); ++y) {
         if (lowered_add64(values[x], values[y]) != values[x] + values[y] ||
             lowered_mul64(values[x], values[y]) != values[x] * values[y]) {
            fprintf(stderr, "int64 arithmetic oracle mismatch at %u,%u\n", x,
                    y);
            return false;
         }
      }
   }

   return true;
}

static bool
check_g17p_compile(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options,
      "G17P int64 arithmetic and dynamic shifts");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;

   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *x = nir_load_global(&b, 1, 64, input, .align_mul = 8);
   nir_def *y = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, input, 8), .align_mul = 8);
   nir_def *shift = nir_load_preamble(&b, 1, 32, .base = 8);
   nir_store_global(&b, nir_iadd(&b, x, y), output, .align_mul = 8);
   nir_store_global(&b, nir_imul(&b, x, y),
                    nir_iadd_imm(&b, output, 8), .align_mul = 8);
   nir_store_global(&b, nir_imul_imm(&b, x, UINT64_C(0xfedcba9876543211)),
                    nir_iadd_imm(&b, output, 16), .align_mul = 8);
   nir_store_global(&b, nir_ishl(&b, x, shift),
                    nir_iadd_imm(&b, output, 24), .align_mul = 8);
   nir_store_global(&b, nir_ushr(&b, x, shift),
                    nir_iadd_imm(&b, output, 32), .align_mul = 8);
   nir_store_global(&b, nir_ishr(&b, x, shift),
                    nir_iadd_imm(&b, output, 40), .align_mul = 8);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid =
      compiled && part.binary && part.info.binary_size > sizeof(end) &&
      part.info.stats.instrs > 10 && part.info.scratch_size == 0 &&
      !part.info.has_preamble &&
      !memcmp((uint8_t *)part.binary + part.info.binary_size - sizeof(end),
              end, sizeof(end));

   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

/* agx_nir_fuse_algebraic_late carries strength-reduction rules
 *
 *    (('imul', a, 1 + (1 << s)), iaddshl(a, a, s))
 *    (('imul', a, 1 - (1 << s)), isubshl(a, a, s))
 *
 * for s in 1..4, and neither is width-qualified.  On G17P nir_lower_imul64 is
 * deliberately cleared so agx_nir_lower_imul64 can use the grounded 32x32->64
 * multiply, which used to leave the 64-bit imul standing right through
 * agx_optimize_nir -- so those rules fired on it and produced a 64-bit
 * imadshl_agx, which agx_emit_imadshl_agx() emits as a 64-bit iadd.  G17P has
 * no grounded 64-bit add, so the packer refused the whole shader:
 *
 *    no grounded AGX3 encoding for iadd
 *      [d64:reg4 s0=64:reg4.discard s1=64:reg4.discard]
 *
 * check_g17p_compile() above does not reach it: 0xfedcba9876543211 is not one
 * of the eight strength-reducible constants.  These are, and this is ordinary
 * shaderInt64 code, which the driver advertises. */
static bool
check_g17p_strength_reduced_mul(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   static const uint64_t multipliers[] = {3, 5, 9, 17};

   for (unsigned i = 0; i < ARRAY_SIZE(multipliers); ++i) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options,
         "G17P int64 strength-reduced multiply then add");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_global(&b, 1, 64, input, .align_mul = 8);
      nir_def *product = nir_imul_imm(&b, x, multipliers[i]);
      nir_def *sum =
         nir_iadd_imm(&b, product, UINT64_C(0x1122334455667788));
      nir_store_global(&b, sum, output, .align_mul = 8);
      agx_preprocess_nir(b.shader);

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .reserved_preamble = 8,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
      const bool valid =
         compiled && part.binary && part.info.binary_size > sizeof(end) &&
         part.info.scratch_size == 0 && !part.info.has_preamble &&
         !memcmp((uint8_t *)part.binary + part.info.binary_size - sizeof(end),
                 end, sizeof(end));

      free(part.binary);
      ralloc_free(b.shader);

      if (!valid) {
         fprintf(stderr,
                 "G17P refused int64 multiply by %llu followed by an add\n",
                 (unsigned long long)multipliers[i]);
         return false;
      }
   }

   return true;
}

static bool
check_g17g_add64(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "M5 full-width address addition");
   b.shader->info.workgroup_size[0] = 1;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
   nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
   nir_def *x = nir_load_global(&b, 1, 64, input, .align_mul = 8);
   nir_def *y = nir_load_global(
      &b, 1, 64, nir_iadd_imm(&b, input, 8), .align_mul = 8);
   nir_store_global(&b, nir_iadd(&b, x, y), output, .align_mul = 8);
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3C,
      .reserved_preamble = 8,
      .no_preamble = true,
   };
   struct agx_shader_part part = {0};
   const bool compiled = agx_compile_shader_nir(b.shader, &key, &part);
   const bool valid = compiled && part.binary &&
      part.info.binary_size > sizeof(end) && part.info.scratch_size == 0 &&
      !memcmp((uint8_t *)part.binary + part.info.binary_size - sizeof(end),
              end, sizeof(end)) && b.shader->options == &agx_nir_options;
   free(part.binary);
   ralloc_free(b.shader);
   return valid;
}

static bool
check_g17p_widened_mul(void)
{
   for (unsigned is_signed = 0; is_signed < 2; ++is_signed) {
      nir_builder b = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options,
         "G17P widened 32-bit multiply");
      b.shader->info.workgroup_size[0] = 1;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;
      nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
      nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
      nir_def *x = nir_load_global(&b, 1, 32, input, .align_mul = 4);
      nir_def *y = nir_load_global(
         &b, 1, 32, nir_iadd_imm(&b, input, 4), .align_mul = 4);
      nir_def *product = is_signed ? nir_imul_2x32_64(&b, x, y)
                                  : nir_umul_2x32_64(&b, x, y);
      nir_store_global(&b, product, output, .align_mul = 8);
      agx_preprocess_nir(b.shader);
      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .reserved_preamble = 8,
         .no_preamble = true,
      };
      struct agx_shader_part part = {0};
      const bool valid = agx_compile_shader_nir(b.shader, &key, &part) &&
                         part.binary && part.info.binary_size > 4 &&
                         part.info.scratch_size == 0;
      free(part.binary);
      ralloc_free(b.shader);
      if (!valid) {
         fprintf(stderr, "G17P refused %s 32x32-to-64 multiply\n",
                 is_signed ? "signed" : "unsigned");
         return false;
      }
   }
   return true;
}

/* M5 multiplies 32x32->64 in one instruction, which takes a word register and
 * a word register, a half register or a small constant. Every other pair of
 * factors has to be legalized: two halves (16-bit or 8-bit values), a half
 * and a constant, a half extended the other way than the product extends its
 * factors, and constants with the high bit set. */
enum m5_factor {
   M5_WORD_X,
   M5_WORD_Y,
   M5_ZEXT_HALF_A,
   M5_ZEXT_HALF_B,
   M5_SEXT_HALF_A,
   M5_SEXT_HALF_B,
   M5_ZEXT_BYTE,
   M5_SEXT_BYTE,
   M5_CONST_37,
   M5_CONST_200,
   M5_CONST_HIGH_BIT,
   M5_CONST_MINUS_5,
   M5_NUM_FACTORS,
};

static const char *const m5_factor_names[M5_NUM_FACTORS] = {
   "x", "y", "zext(a16)", "zext(b16)", "sext(a16)", "sext(b16)", "zext(c8)",
   "sext(c8)", "37", "200", "0x80000001", "-5",
};

static nir_def *
m5_factor(nir_builder *b, nir_def *input, enum m5_factor factor)
{
   switch (factor) {
   case M5_WORD_X:
      return nir_load_global(b, 1, 32, input, .align_mul = 4);
   case M5_WORD_Y:
      return nir_load_global(b, 1, 32, nir_iadd_imm(b, input, 4),
                             .align_mul = 4);
   case M5_ZEXT_HALF_A:
   case M5_SEXT_HALF_A:
   case M5_ZEXT_HALF_B:
   case M5_SEXT_HALF_B: {
      const bool second = factor == M5_ZEXT_HALF_B || factor == M5_SEXT_HALF_B;
      nir_def *half = nir_load_global(
         b, 1, 16, nir_iadd_imm(b, input, second ? 10 : 8), .align_mul = 2);
      return factor == M5_ZEXT_HALF_A || factor == M5_ZEXT_HALF_B
                ? nir_u2u32(b, half)
                : nir_i2i32(b, half);
   }
   case M5_ZEXT_BYTE:
   case M5_SEXT_BYTE: {
      nir_def *byte = nir_load_global(b, 1, 8, nir_iadd_imm(b, input, 12),
                                      .align_mul = 1);
      return factor == M5_ZEXT_BYTE ? nir_u2u32(b, byte) : nir_i2i32(b, byte);
   }
   case M5_CONST_37:
      return nir_imm_int(b, 37);
   case M5_CONST_200:
      return nir_imm_int(b, 200);
   case M5_CONST_HIGH_BIT:
      return nir_imm_int(b, 0x80000001);
   case M5_CONST_MINUS_5:
      return nir_imm_int(b, -5);
   default:
      UNREACHABLE("invalid factor");
   }
}

static bool
check_m5_wide_product(void)
{
   static const uint8_t end[] = {0x0e, 0x00, 0x00, 0x00};
   bool ok = true;

   /* 0, 1: umul_2x32_64, imul_2x32_64; 2, 3: umul_high, imul_high, which
    * are the same instruction with the low word dropped. */
   for (unsigned op = 0; op < 4; ++op) {
      for (unsigned p = 0; p < M5_NUM_FACTORS; ++p) {
         for (unsigned q = 0; q < M5_NUM_FACTORS; ++q) {
            /* Two constants fold, and a factor pairs with itself as x * x. */
            if (p >= M5_CONST_37 && q >= M5_CONST_37)
               continue;

            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options, "M5 32x32->64 product");
            b.shader->info.workgroup_size[0] = 1;
            b.shader->info.workgroup_size[1] = 1;
            b.shader->info.workgroup_size[2] = 1;
            nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
            nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
            nir_def *x = m5_factor(&b, input, p);
            nir_def *y = m5_factor(&b, input, q);
            nir_def *product =
               op == 0   ? nir_umul_2x32_64(&b, x, y)
               : op == 1 ? nir_imul_2x32_64(&b, x, y)
               : op == 2 ? nir_umul_high(&b, x, y)
                         : nir_imul_high(&b, x, y);
            nir_store_global(&b, product, output, .align_mul = 8);
            agx_preprocess_nir(b.shader);

            struct agx_shader_key key = {
               .dev.encoding = AGX_ISA_ENCODING_AGX3C,
               .reserved_preamble = 8,
               .no_preamble = true,
            };
            struct agx_shader_part part = {0};
            const bool valid =
               agx_compile_shader_nir(b.shader, &key, &part) && part.binary &&
               part.info.binary_size > sizeof(end) &&
               !memcmp((uint8_t *)part.binary + part.info.binary_size -
                          sizeof(end),
                       end, sizeof(end));
            free(part.binary);
            ralloc_free(b.shader);
            if (!valid) {
               static const char *const ops[] = {
                  "umul_2x32_64", "imul_2x32_64", "umul_high", "imul_high"};
               fprintf(stderr, "M5 refused %s(%s, %s)\n", ops[op],
                       m5_factor_names[p], m5_factor_names[q]);
               ok = false;
            }
         }
      }
   }

   return ok;
}

static bool
check_mul_extended_64bit(void)
{
   static const struct {
      enum agx_isa_encoding encoding;
      const char *name;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX2, "AGX2"},
      {AGX_ISA_ENCODING_AGX3B, "AGX3B"},
      {AGX_ISA_ENCODING_AGX3C, "AGX3C"},
   };
   static const unsigned bit_sizes[] = {32, 64};

   for (unsigned t = 0; t < ARRAY_SIZE(targets); ++t) {
      for (unsigned bs_i = 0; bs_i < ARRAY_SIZE(bit_sizes); ++bs_i) {
         const unsigned bs = bit_sizes[bs_i];
         const unsigned bytes = bs / 8;

         for (unsigned is_signed = 0; is_signed < 2; ++is_signed) {
            nir_builder b = nir_builder_init_simple_shader(
               MESA_SHADER_COMPUTE, &agx_nir_options,
               "mul_extended (imul + mul_high)");
            b.shader->info.workgroup_size[0] = 1;
            b.shader->info.workgroup_size[1] = 1;
            b.shader->info.workgroup_size[2] = 1;

            nir_def *input = nir_load_preamble(&b, 1, 64, .base = 0);
            nir_def *output = nir_load_preamble(&b, 1, 64, .base = 4);
            nir_def *x = nir_load_global(&b, 1, bs, input, .align_mul = bytes);
            nir_def *y = nir_load_global(&b, 1, bs,
                                         nir_iadd_imm(&b, input, bytes),
                                         .align_mul = bytes);
            nir_def *lo = nir_imul(&b, x, y);
            nir_def *hi =
               is_signed ? nir_imul_high(&b, x, y) : nir_umul_high(&b, x, y);
            nir_store_global(&b, lo, output, .align_mul = bytes);
            nir_store_global(&b, hi, nir_iadd_imm(&b, output, bytes),
                             .align_mul = bytes);
            agx_preprocess_nir(b.shader);

            struct agx_shader_key key = {
               .dev.encoding = targets[t].encoding,
               .reserved_preamble = 8,
               .no_preamble = true,
            };
            struct agx_shader_part part = {0};
            const bool valid = agx_compile_shader_nir(b.shader, &key, &part) &&
                               part.binary && part.info.binary_size > 4;
            free(part.binary);
            ralloc_free(b.shader);

            if (!valid) {
               fprintf(stderr, "%s failed %u-bit %s mul_extended compile\n",
                       targets[t].name, bs,
                       is_signed ? "signed" : "unsigned");
               return false;
            }
         }
      }
   }

   return true;
}

int
main(void)
{
   if (!check_host_oracle() || !check_g17p_compile() ||
       !check_g17p_strength_reduced_mul() || !check_g17g_add64() ||
       !check_g17p_widened_mul() || !check_m5_wide_product() ||
       !check_mul_extended_64bit())
      return 1;

   puts("agx3_int64: arithmetic, dynamic shifts and strength-reduced "
        "multiplies compile for G17P; full-width addition and 32x32->64 "
        "products compile for M5; 64-bit mul_extended compiles across AGX2/AGX3");
   return 0;
}
