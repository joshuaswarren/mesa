/* Copyright 2026 Aurora Silicon; SPDX-License-Identifier: MIT */
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_builder.h"
extern "C" {
#include "asahi/layout/layout.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

static float
as_float(uint32_t x)
{
   float f;
   memcpy(&f, &x, sizeof(f));
   return f;
}

static uint32_t
as_bits(float f)
{
   uint32_t x;
   memcpy(&x, &f, sizeof(x));
   return x;
}

/* Lower one constant fp32 ffma for DenormPreserve and fold it. The native ffma
 * folds to the host's fused result, so the exact path is what is checked
 * wherever an operand, the product or the result is tiny. */
static uint32_t
fold_fma(uint32_t x, uint32_t y, uint32_t z)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "denorm preserve fma");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_DENORM_PRESERVE_FP32 |
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   nir_def *r = nir_ffma(&b, nir_imm_int(&b, x), nir_imm_int(&b, y),
                         nir_imm_int(&b, z));
   nir_intrinsic_instr *store = agx_test_store_output(
      &b, r, nir_imm_int(&b, 0), nir_type_float32, FRAG_RESULT_DATA0);
   EXPECT_TRUE(agx_nir_lower_denorm_preserve(b.shader));

   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
      progress |= nir_opt_remove_phis(b.shader);
   } while (progress);
   nir_validate_shader(b.shader, "denorm preserve fma");

   uint32_t result = 0;
   EXPECT_TRUE(nir_src_is_const(store->src[0]));
   if (nir_src_is_const(store->src[0]))
      result = nir_src_as_uint(store->src[0]);
   ralloc_free(b.shader);
   return result;
}

static void
check(uint32_t x, uint32_t y, uint32_t z)
{
   const uint32_t expected =
      as_bits(std::fma(as_float(x), as_float(y), as_float(z)));
   const uint32_t got = fold_fma(x, y, z);
   if (std::isnan(as_float(expected)))
      EXPECT_TRUE(std::isnan(as_float(got)))
         << std::hex << x << " " << y << " " << z << " -> " << got;
   else
      EXPECT_EQ(got, expected) << std::hex << x << " " << y << " " << z;
}

TEST(DenormPreserveFMA, SpecialValues)
{
   static const uint32_t values[] = {
      0x00000000, /* 0 */
      0x00000001, /* smallest denormal */
      0x000ffe00, /* denormal */
      0x007fffff, /* largest denormal */
      0x00800000, /* smallest normal */
      0x00800001, /* smallest normal + 1 ulp */
      0x0c000000, /* 2^-103: products below 2^-126 */
      0x34000000, /* 2^-23 */
      0x3f7fffff, /* 1 - 1 ulp */
      0x3f800000, /* 1 */
      0x3f800001, /* 1 + 1 ulp */
      0x7f7fffff, /* largest normal */
      0x7f800000, /* infinity */
   };
   for (uint32_t x : values) {
      for (uint32_t y : values) {
         for (uint32_t z : values) {
            /* The sign of y only reaches the product through x's. */
            for (uint32_t signs = 0; signs < 4; signs++) {
               check(x ^ (signs & 1 ? 0x80000000u : 0), y,
                     z ^ (signs & 2 ? 0x80000000u : 0));
            }
         }
      }
   }
}

/* x * y - round(x * y) is the rounding error of the product, which only a
 * single rounding returns; separate multiply and add rounding returns zero.
 * These keep the native fma, so they check that ffma is not split. */
TEST(DenormPreserveFMA, StaysFused)
{
   static const float pairs[][2] = {
      {0x1.000002p0f, 0x1.000002p0f},
      {1.25f, 0x1.000004p0f},
      {3.0f, 0x1.555556p-2f},
   };
   for (const auto &p : pairs) {
      const float product = p[0] * p[1];
      check(as_bits(p[0]), as_bits(p[1]), as_bits(-product));
   }
   /* Correctly rounded with one rounding, but not with two. */
   check(as_bits(1.25f), as_bits(0x1.000004p0f), as_bits(0x1.000002p-126f));
}

/* A denormal operand takes the exact path. Each result differs from a
 * separately rounded product plus z. */
TEST(DenormPreserveFMA, ExactPathStaysFused)
{
   static const uint32_t cases[][3] = {
      {0x0075fbdeu, 0x4357ae2cu, 0x8446cd98u},
      {0x00704277u, 0x46e9c561u, 0x87cd061bu},
      {0x001c82cau, 0x4c357993u, 0x8c21b01bu},
      {0x000a022bu, 0x4e8fffbau, 0x8db426acu},
      {0x001364d9u, 0x48dbdf8bu, 0x888541a8u},
   };
   for (const auto &c : cases) {
      /* volatile: no contraction into an fma */
      volatile float product = as_float(c[0]) * as_float(c[1]);
      const float split = product + as_float(c[2]);
      EXPECT_NE(as_bits(split), as_bits(std::fma(as_float(c[0]), as_float(c[1]),
                                                 as_float(c[2]))));
      check(c[0], c[1], c[2]);
   }
}

static uint32_t
fold_alu_denorm(nir_op op, uint32_t x, uint32_t y)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "denorm preserve minmax");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_DENORM_PRESERVE_FP32 |
      FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
   nir_def *r;
   if (op == nir_op_fsat)
      r = nir_fsat(&b, nir_imm_int(&b, x));
   else
      r = nir_build_alu2(&b, op, nir_imm_int(&b, x), nir_imm_int(&b, y));
   nir_intrinsic_instr *store = agx_test_store_output(
      &b, r, nir_imm_int(&b, 0), nir_type_float32, FRAG_RESULT_DATA0);
   EXPECT_TRUE(agx_nir_lower_denorm_preserve(b.shader));

   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
      progress |= nir_opt_remove_phis(b.shader);
   } while (progress);
   nir_validate_shader(b.shader, "denorm preserve minmax");

   uint32_t result = 0;
   EXPECT_TRUE(nir_src_is_const(store->src[0]));
   if (nir_src_is_const(store->src[0]))
      result = nir_src_as_uint(store->src[0]);
   ralloc_free(b.shader);
   return result;
}

static uint32_t
ref_minmax_bits(uint32_t x, uint32_t y, bool is_max)
{
   const bool nan_x = (x & 0x7fffffffu) > 0x7f800000u;
   const bool nan_y = (y & 0x7fffffffu) > 0x7f800000u;
   if (nan_x)
      return y;
   if (nan_y)
      return x;
   const uint32_t kx = (x & 0x80000000u) ? ~x : (x ^ 0x80000000u);
   const uint32_t ky = (y & 0x80000000u) ? ~y : (y ^ 0x80000000u);
   return (is_max ? (kx >= ky) : (kx <= ky)) ? x : y;
}

TEST(DenormPreserveMinMax, DirectMinMaxClampAndSat)
{
   static const uint32_t vals[] = {
      0x00000000u, /* +0 */
      0x80000000u, /* -0 */
      0x00000001u, /* +min denorm */
      0x80000001u, /* -min denorm */
      0x000ffe00u, /* +mid denorm */
      0x800ffe00u, /* -mid denorm */
      0x007fffffu, /* +max denorm */
      0x807fffffu, /* -max denorm */
      0x00800000u, /* +min normal */
      0x80800000u, /* -min normal */
      0x3f800000u, /* +1.0 */
      0xbf800000u, /* -1.0 */
      0x7f800000u, /* +inf */
      0xff800000u, /* -inf */
      0x7fc00000u, /* qNaN */
   };
   for (uint32_t x : vals) {
      for (uint32_t y : vals) {
         const bool has_denorm =
            ((x & 0x7f800000u) == 0 && (x & 0x007fffffu) != 0) ||
            ((y & 0x7f800000u) == 0 && (y & 0x007fffffu) != 0);
         if (!has_denorm)
            continue;
         for (nir_op op : {nir_op_fmin, nir_op_fmax, nir_op_fmin_agx, nir_op_fmax_agx}) {
            const bool is_max = op == nir_op_fmax || op == nir_op_fmax_agx;
            const uint32_t got = fold_alu_denorm(op, x, y);
            const uint32_t exp = ref_minmax_bits(x, y, is_max);
            EXPECT_EQ(got, exp) << std::hex << x << " " << y << " op=" << op;
         }
      }
      if ((x & 0x7f800000u) == 0 && (x & 0x007fffffu) != 0) {
         const uint32_t got = fold_alu_denorm(nir_op_fsat, x, 0);
         const uint32_t exp = (x & 0x80000000u) ? 0x00000000u : x;
         EXPECT_EQ(got, exp) << std::hex << x;
      }
   }

   /* Verify clamp(denorm, lo, hi) via fmin(fmax(x, lo), hi) preserves denormals. */
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "denorm preserve clamp");
   b.shader->info.float_controls_execution_mode =
      FLOAT_CONTROLS_DENORM_PRESERVE_FP32;
   nir_def *x = nir_imm_int(&b, 0x00000007u);
   nir_def *lo = nir_imm_int(&b, 0x00000003u);
   nir_def *hi = nir_imm_int(&b, 0x00000010u);
   nir_def *clamped = nir_fmin(&b, nir_fmax(&b, x, lo), hi);
   nir_intrinsic_instr *store = agx_test_store_output(
      &b, clamped, nir_imm_int(&b, 0), nir_type_float32, FRAG_RESULT_DATA0);
   EXPECT_TRUE(agx_nir_lower_denorm_preserve(b.shader));
   bool progress;
   do {
      progress = nir_opt_constant_folding(b.shader);
      progress |= nir_opt_copy_prop(b.shader);
      progress |= nir_opt_dead_cf(b.shader);
      progress |= nir_opt_dce(b.shader);
      progress |= nir_opt_remove_phis(b.shader);
   } while (progress);
   ASSERT_TRUE(nir_src_is_const(store->src[0]));
   EXPECT_EQ(nir_src_as_uint(store->src[0]), 0x00000007u);
   ralloc_free(b.shader);
}

TEST(DenormPreserveLowering, CapabilityStagesAndEncodings)
{
   EXPECT_FALSE(agx_chip_has_fp32_denorm_preserve(AGX_CHIP_G13G));
   EXPECT_FALSE(agx_chip_has_fp32_denorm_preserve(AGX_CHIP_G14G));
   EXPECT_FALSE(agx_chip_has_fp32_denorm_preserve(AGX_CHIP_G18G));

   const struct {
      enum agx_chip chip;
      enum agx_isa_encoding enc;
      bool m5;
   } targets[] = {
      {AGX_CHIP_G15G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17G, AGX_ISA_ENCODING_AGX3C, true},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C, true},
      {AGX_CHIP_G17C, AGX_ISA_ENCODING_AGX3C, true},
   };
   for (const auto &t : targets) {
      EXPECT_TRUE(agx_chip_has_fp32_denorm_preserve(t.chip));
   }

   /* Compile runtime denormal-input and denormal-result operations (fmin,
    * fmax, fclamp, fsat, fadd, fmul) across VS, FS, and CS on both AGX3B
    * (G15G, G16G, G17P) and AGX3C (G17S). */
   const struct {
      enum agx_chip chip;
      enum agx_isa_encoding enc;
      bool m5;
   } compile_targets[] = {
      {AGX_CHIP_G15G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G16G, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B, false},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C, true},
   };
   const mesa_shader_stage stages[] = {
      MESA_SHADER_VERTEX,
      MESA_SHADER_FRAGMENT,
      MESA_SHADER_COMPUTE,
   };
   for (const auto &t : compile_targets) {
      for (mesa_shader_stage stage : stages) {
         SCOPED_TRACE(::testing::Message()
                      << agx_chip_name(t.chip) << ":" << mesa_shader_stage_name(stage));
         nir_builder b = nir_builder_init_simple_shader(
            stage, &agx_nir_options, "denorm preserve stage");
         b.shader->info.float_controls_execution_mode =
            FLOAT_CONTROLS_DENORM_PRESERVE_FP32 |
            FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32;
         nir_def *a = agx_test_load_preamble(&b, 1, 32, 4);
         nir_def *bv = agx_test_load_preamble(&b, 1, 32, 6);
         nir_def *cv = agx_test_load_preamble(&b, 1, 32, 8);
         nir_def *mn = nir_fmin(&b, a, bv);
         nir_def *mx = nir_fmax(&b, mn, cv);
         nir_def *sat = nir_fsat(&b, mx);
         nir_def *prod = nir_fmul(&b, sat, nir_imm_float(&b, 0x1p-120f));
         nir_def *res = nir_fadd(&b, prod, a);
         if (stage == MESA_SHADER_COMPUTE || stage == MESA_SHADER_VERTEX) {
            if (stage == MESA_SHADER_COMPUTE) {
               b.shader->info.workgroup_size[0] = 1;
               b.shader->info.workgroup_size[1] = 1;
               b.shader->info.workgroup_size[2] = 1;
            }
            nir_def *addr = agx_test_load_preamble(&b, 1, 64, 10);
            nir_store_global(&b, res, addr, .write_mask = 1, .align_mul = 4);
         } else {
            agx_test_store_local_pixel_agx(
               &b, nir_vec4(&b, res, res, res, res), nir_imm_intN_t(&b, 0xff, 16),
               nir_undef(&b, 2, 16), 0, 0xf,
               (enum pipe_format)AIL_ISA_FORMAT_I32);
            b.shader->info.outputs_written = BITFIELD64_BIT(FRAG_RESULT_DATA0);
         }
         if (t.m5)
            agx_preprocess_nir_m5(b.shader);
         else
            agx_preprocess_nir_native(b.shader, t.chip);
         agx_shader_key key = {};
         key.dev.chip = t.chip;
         key.dev.encoding = t.enc;
         key.no_preamble = true;
         key.reserved_preamble = 16;
         if (stage == MESA_SHADER_FRAGMENT)
            key.fs.nr_samples = 1;
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         EXPECT_GT(part.info.main_size, 0u);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}
