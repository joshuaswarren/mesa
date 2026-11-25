#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include "agx_test.h"
#include "compiler/nir/nir_constant_expressions.h"
extern "C" {
#include "agx_nir.h"
}

static bool
valid_signed_conversion(uint32_t word)
{
   uint32_t magnitude = word & 0x7fffffff;
   return magnitude < 0x4f000000 ||
          (magnitude == 0x4f000000 && (word & 0x80000000));
}

static uint32_t
truncate_signed_bits(uint32_t word)
{
   uint32_t exponent = (word >> 23) & 255;
   if (exponent < 127)
      return 0;
   uint64_t significand = (word & 0x7fffff) | 0x800000;
   unsigned power = exponent - 127;
   uint64_t magnitude = power >= 23 ? significand << (power - 23)
                                    : significand >> (23 - power);
   return word >> 31 ? 0u - uint32_t(magnitude) : uint32_t(magnitude);
}

static nir_const_value
evaluate_selected_conversion(nir_scalar scalar, nir_def *a, nir_def *b,
                             nir_def *choice, uint32_t x, uint32_t y,
                             bool selected, unsigned mode)
{
   scalar = nir_scalar_chase_movs(scalar);
   if (scalar.def == a || scalar.def == b || scalar.def == choice)
      return nir_const_value_for_uint(scalar.def == a ? x : scalar.def == b ? y
                                                                      : selected,
                                       scalar.def->bit_size);
   if (nir_scalar_is_const(scalar))
      return nir_const_value_for_uint(nir_scalar_as_uint(scalar),
                                       scalar.def->bit_size);
   if (!nir_scalar_is_alu(scalar)) {
      ADD_FAILURE() << "Unexpected conversion instruction";
      return {};
   }
   nir_alu_instr *alu = nir_scalar_as_alu(scalar);
   const nir_op_info *info = &nir_op_infos[alu->op];
   nir_const_value values[NIR_ALU_MAX_INPUTS] = {};
   nir_const_value *sources[NIR_ALU_MAX_INPUTS] = {};
   unsigned bits = nir_alu_type_get_type_size(info->output_type) == 0
                   ? alu->def.bit_size : 0;
   for (unsigned i = 0; i < info->num_inputs; ++i) {
      nir_scalar input = nir_scalar_chase_alu_src(scalar, i);
      values[i] = evaluate_selected_conversion(input, a, b, choice, x, y,
                                               selected, mode);
      sources[i] = &values[i];
      if (!bits && nir_alu_type_get_type_size(info->input_types[i]) == 0)
         bits = input.def->bit_size;
   }
   nir_const_value result = {};
   uint16_t poison = 0;
   nir_eval_const_opcode(alu->op, &result, &poison, 1, bits ? bits : 32,
                         sources, mode);
   EXPECT_FALSE(poison);
   return result;
}

TEST(SelectedF2I, PreservesTruncationForDefinedInputsAndControls)
{
   const std::array<uint32_t, 24> edges = {
      0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x807fffff,
      0x00800000, 0x80800000, 0x3f000000, 0xbf000000,
      0x3f7fffff, 0xbf7fffff, 0x3f800000, 0xbf800000,
      0x3fc00000, 0xbfc00000, 0x40200000, 0xc0200000,
      0x4b800000, 0xcb800000, 0x4effffff, 0xceffffff,
      0xcf000000, 0x3f800001};
   for (unsigned mode : {0u, unsigned(FLOAT_CONTROLS_ROUNDING_MODE_RTE_FP32),
                          unsigned(FLOAT_CONTROLS_ROUNDING_MODE_RTZ_FP32),
                          unsigned(FLOAT_CONTROLS_DENORM_FLUSH_TO_ZERO_FP32)}) {
      for (unsigned flags : {unsigned(nir_fp_fast_math), unsigned(nir_fp_exact)}) {
         SCOPED_TRACE(mode);
         SCOPED_TRACE(flags);
         nir_builder nb = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "selected FP32 integer conversion");
         nb.shader->info.float_controls_execution_mode = mode;
         nb.fp_math_ctrl = flags;
         nir_def *a = agx_test_load_preamble(&nb, 1, 32, 0);
         nir_def *b = agx_test_load_preamble(&nb, 1, 32, 4);
         nir_def *choice = agx_test_load_preamble(&nb, 1, 32, 8);
         nir_def *keep = nir_mov(&nb, nir_bcsel(&nb, nir_ine_imm(&nb, choice, 0),
                                   nir_f2i32(&nb, a), nir_f2i32(&nb, b)));
         ASSERT_TRUE(agx_nir_fuse_algebraic_late(nb.shader));
         nir_scalar result = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
         ASSERT_EQ(nir_scalar_alu_op(result), nir_op_f2i32);
         EXPECT_EQ(nir_scalar_as_alu(result)->fp_math_ctrl, flags);
         EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(result, 0)), nir_op_bcsel);
         auto check = [&](uint32_t x, uint32_t y) {
            ASSERT_TRUE(valid_signed_conversion(x));
            ASSERT_TRUE(valid_signed_conversion(y));
            for (bool selected : {false, true}) {
               nir_const_value actual = evaluate_selected_conversion(
                  result, a, b, choice, x, y, selected, mode);
               EXPECT_EQ(nir_const_value_as_uint(actual, 32),
                         truncate_signed_bits(selected ? x : y))
                  << x << " " << y << " " << selected;
            }
         };
         for (uint32_t x : edges)
            for (uint32_t y : edges)
               check(x, y);
         uint32_t state = 0x83274615;
         auto next = [&]() {
            do {
               state ^= state << 13;
               state ^= state >> 17;
               state ^= state << 5;
            } while (!valid_signed_conversion(state));
            return state;
         };
         for (unsigned i = 0; i < 10000; ++i) {
            uint32_t x = next(), y = next();
            check(x, y);
         }
         nir_validate_shader(nb.shader, "selected FP32 integer conversion");
         ralloc_free(nb.shader);
      }
   }
}

TEST(SelectedF2I, CollapsesThreeWaySelection)
{
   nir_builder nb = nir_builder_init_simple_shader(
      MESA_SHADER_COMPUTE, &agx_nir_options, "three coordinate alternatives");
   nir_def *a = agx_test_load_preamble(&nb, 1, 32, 0);
   nir_def *b = agx_test_load_preamble(&nb, 1, 32, 4);
   nir_def *c = agx_test_load_preamble(&nb, 1, 32, 8);
   nir_def *choice = agx_test_load_preamble(&nb, 1, 32, 12);
   nir_def *inner = nir_bcsel(&nb, nir_ine_imm(&nb, choice, 0),
                             nir_f2i32(&nb, a), nir_f2i32(&nb, b));
   nir_def *out = nir_bcsel(&nb, nir_ine_imm(&nb, choice, 1),
                           nir_f2i32(&nb, c), inner);
   nir_def *keep = nir_mov(&nb, out);
   unsigned rounds = 0;
   while (agx_nir_fuse_algebraic_late(nb.shader))
      ASSERT_LT(++rounds, 4u);
   EXPECT_GT(rounds, 0u);
   nir_scalar result = nir_scalar_chase_alu_src(nir_get_scalar(keep, 0), 0);
   ASSERT_EQ(nir_scalar_alu_op(result), nir_op_f2i32);
   nir_scalar select = nir_scalar_chase_alu_src(result, 0);
   ASSERT_EQ(nir_scalar_alu_op(select), nir_op_bcsel);
   EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(select, 2)), nir_op_bcsel);
   nir_validate_shader(nb.shader, "three coordinate alternatives");
   ralloc_free(nb.shader);
}

TEST(SelectedF2I, LeavesSharedMixedAndOtherWidthsUnchanged)
{
   for (unsigned control = 0; control < 7; ++control) {
      SCOPED_TRACE(control);
      nir_builder nb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "unrelated conversions");
      nb.fp_math_ctrl = nir_fp_fast_math;
      unsigned components = control == 6 ? 2 : 1;
      nir_def *a = agx_test_load_preamble(&nb, components, control == 4 ? 16 : 32, 0);
      nir_def *b = agx_test_load_preamble(&nb, components, 32, 4);
      nir_def *choice = agx_test_load_preamble(&nb, 1, 32, 8);
      nir_def *left = control == 3 ? nir_f2i16(&nb, a) : nir_f2i32(&nb, a);
      nir_def *right = control == 2 ? nir_f2i32_rtne(&nb, b)
                     : control == 3 ? nir_f2i16(&nb, b) : nir_f2i32(&nb, b);
      if (control == 0)
         agx_test_store_preamble(&nb, left, 20);
      if (control == 1)
         agx_test_store_preamble(&nb, right, 20);
      if (control == 5)
         nir_scalar_as_alu(nir_get_scalar(left, 0))->fp_math_ctrl = nir_fp_exact;
      nir_mov(&nb, nir_bcsel(&nb, nir_ine_imm(&nb, choice, 0), left, right));
      EXPECT_FALSE(agx_nir_fuse_algebraic_late(nb.shader));
      nir_validate_shader(nb.shader, "unrelated conversions");
      ralloc_free(nb.shader);
   }
}

static nir_def *
build_d3d_f2i_chain(nir_builder *b, nir_def *a)
{
   nir_def *nan = nir_fneu(b, a, a);
   nir_def *lo = nir_flt(b, a, nir_imm_float(b, -2147483648.0f));
   nir_def *hi = nir_flt(b, nir_imm_float(b, 2147483520.0f), a);
   return nir_bcsel(
      b, nan, nir_imm_int(b, 0),
      nir_bcsel(b, lo, nir_imm_int(b, INT32_MIN),
                nir_bcsel(b, hi, nir_imm_int(b, INT32_MAX), nir_f2i32(b, a))));
}

static nir_def *
build_d3d_f2u_chain(nir_builder *b, nir_def *a, bool explicit_nan)
{
   nir_def *clamped = nir_fmax(b, a, nir_imm_float(b, 0.0f));
   if (explicit_nan)
      clamped = nir_bcsel(b, nir_fneu(b, a, a), nir_imm_float(b, 0.0f), clamped);
   return nir_bcsel(b, nir_flt(b, nir_imm_float(b, 4294967040.0f), a),
                    nir_imm_int(b, 0xffffffffu), nir_f2u32(b, clamped));
}

TEST(D3DFloatToInt, FoldsDynamicClampsToSaturatingOpsAndCompilesOnAgx3)
{
   const enum agx_isa_encoding encodings[] = {
      AGX_ISA_ENCODING_AGX3B,
      AGX_ISA_ENCODING_AGX3C,
   };
   for (enum agx_isa_encoding enc : encodings) {
      SCOPED_TRACE(enc);
      nir_builder nb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "d3d f2i/f2u dynamic");
      nb.shader->info.workgroup_size[0] = 1;
      nb.shader->info.workgroup_size[1] = 1;
      nb.shader->info.workgroup_size[2] = 1;
      nir_def *out = nir_load_preamble(&nb, 1, 64, .base = 0);
      nir_def *a = agx_test_load_preamble(&nb, 1, 32, 4);
      nir_def *s = nir_mov(&nb, build_d3d_f2i_chain(&nb, a));
      nir_def *u0 = nir_mov(&nb, build_d3d_f2u_chain(&nb, a, true));
      nir_def *u1 = nir_mov(&nb, build_d3d_f2u_chain(&nb, a, false));
      nir_store_global(&nb, s, out, .align_mul = 4);
      nir_store_global(&nb, u0, nir_iadd_imm(&nb, out, 4), .align_mul = 4);
      nir_store_global(&nb, u1, nir_iadd_imm(&nb, out, 8), .align_mul = 4);

      ASSERT_TRUE(agx_nir_fold_d3d_float_to_int(nb.shader));
      EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(nir_get_scalar(s, 0), 0)),
                nir_op_f2i32_sat);
      EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(nir_get_scalar(u0, 0), 0)),
                nir_op_f2u32_sat);
      EXPECT_EQ(nir_scalar_alu_op(nir_scalar_chase_alu_src(nir_get_scalar(u1, 0), 0)),
                nir_op_f2u32_sat);
      nir_validate_shader(nb.shader, "d3d f2i/f2u dynamic folded");

      if (enc == AGX_ISA_ENCODING_AGX3B)
         agx_preprocess_nir_native(nb.shader, AGX_CHIP_G16G);
      else
         agx_preprocess_nir_m5(nb.shader);

      struct agx_shader_key key = {};
      key.dev.encoding = enc;
      key.dev.chip = enc == AGX_ISA_ENCODING_AGX3B ? AGX_CHIP_G16G : AGX_CHIP_G17S;
      key.reserved_preamble = 8;
      key.no_preamble = true;
      struct agx_shader_part part = {};
      ASSERT_TRUE(agx_compile_shader_nir(nb.shader, &key, &part));
      EXPECT_NE(part.binary, nullptr);
      EXPECT_GT(part.info.binary_size, 0u);
      free(part.binary);
      ralloc_free(nb.shader);
   }
}

TEST(D3DFloatToInt, ConstantFoldingPreservesSaturationAndNaNToZero)
{
   struct Case {
      uint32_t fbits;
      uint32_t want_i32;
      uint32_t want_u32;
   };
   const Case cases[] = {
      /* Out-of-range positive / negative floats that bare f2i32/f2u32 poison-fold to 0 */
      {0x4f32d05eu, 0x7fffffffu, 3000000000u}, /* +3e9f */
      {0xcf32d05eu, 0x80000000u, 0x00000000u}, /* -3e9f */
      {0x4f9502f9u, 0x7fffffffu, 0xffffffffu}, /* +5e9f */
      {0xbf800000u, 0xffffffffu, 0x00000000u}, /* -1.0f */
      /* Exact saturation boundaries, infinities, and both signs of NaN */
      {0x4effffffu, 2147483520u, 2147483520u}, /* max float < 2^31 */
      {0x4f000000u, 0x7fffffffu, 0x80000000u}, /* 2^31 */
      {0xcf000000u, 0x80000000u, 0x00000000u}, /* -2^31 */
      {0x4f7fffffu, 0x7fffffffu, 4294967040u}, /* max float < 2^32 */
      {0x4f800000u, 0x7fffffffu, 0xffffffffu}, /* 2^32 */
      {0x7f800000u, 0x7fffffffu, 0xffffffffu}, /* +inf */
      {0xff800000u, 0x80000000u, 0x00000000u}, /* -inf */
      {0x7fc00000u, 0x00000000u, 0x00000000u}, /* +qNaN */
      {0xffc00000u, 0x00000000u, 0x00000000u}, /* -qNaN */
      {0x422b999au, 42u, 42u},                 /* 42.9f -> 42 */
      {0xbfe00000u, 0xffffffffu, 0x00000000u}, /* -1.75f -> -1 / 0 */
   };

   for (const Case &c : cases) {
      SCOPED_TRACE(c.fbits);
      nir_builder nb = nir_builder_init_simple_shader(
         MESA_SHADER_COMPUTE, &agx_nir_options, "d3d f2i/f2u const");
      nir_def *a = nir_imm_int(&nb, c.fbits);
      nir_intrinsic_instr *st_s =
         agx_test_store_preamble(&nb, build_d3d_f2i_chain(&nb, a), 0);
      nir_intrinsic_instr *st_u0 =
         agx_test_store_preamble(&nb, build_d3d_f2u_chain(&nb, a, true), 4);
      nir_intrinsic_instr *st_u1 =
         agx_test_store_preamble(&nb, build_d3d_f2u_chain(&nb, a, false), 8);

      ASSERT_TRUE(agx_nir_fold_d3d_float_to_int(nb.shader));
      EXPECT_TRUE(nir_opt_constant_folding(nb.shader));
      nir_opt_dce(nb.shader);

      ASSERT_TRUE(nir_src_is_const(st_s->src[0]));
      ASSERT_TRUE(nir_src_is_const(st_u0->src[0]));
      ASSERT_TRUE(nir_src_is_const(st_u1->src[0]));
      EXPECT_EQ(nir_src_as_uint(st_s->src[0]), c.want_i32);
      EXPECT_EQ(nir_src_as_uint(st_u0->src[0]), c.want_u32);
      EXPECT_EQ(nir_src_as_uint(st_u1->src[0]), c.want_u32);

      nir_validate_shader(nb.shader, "d3d f2i/f2u const folded");
      ralloc_free(nb.shader);
   }
}
