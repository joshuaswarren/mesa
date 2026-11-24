/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

/* On AGX3C a 32-bit FADD of a half and an immediate is legalized by widening
 * the half with f32add(half, -0.0). That conversion itself must not be widened
 * again: an FMA of float(half) costs one conversion per half, not two. A
 * saturated add and an add of a nonzero immediate keep their legalization.
 */

namespace {

enum kind {
   FMA_OF_HALF,  /* ffma(f2f32(h_i), x, y) */
   FMA_OF_FLOAT, /* ffma(f_i, x, y), the same with f32 loads */
   SAT_ADD,      /* fsat(f2f32(h) + 1.5) */
   NEG_ABS,      /* ffma(-|f2f32(h)|, x, y) */
};

static unsigned
compile(enum kind k, agx_isa_encoding encoding, agx_chip chip, bool emit_agx3d,
        bool *ok)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "widen");
   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *xy = nir_load_global(&b, 2, 32, addr, .align_mul = 8);
   nir_def *x = nir_channel(&b, xy, 0), *y = nir_channel(&b, xy, 1);
   nir_def *src = k == FMA_OF_FLOAT
                     ? nir_load_global(&b, 4, 32, nir_iadd_imm(&b, addr, 16),
                                       .align_mul = 16)
                     : nir_load_global(&b, 4, 16, nir_iadd_imm(&b, addr, 16),
                                       .align_mul = 8);
   nir_def *out[4];
   for (unsigned i = 0; i < 4; ++i) {
      nir_def *v = nir_channel(&b, src, i);
      nir_def *f = k == FMA_OF_FLOAT ? v : nir_f2f32(&b, v);
      switch (k) {
      case SAT_ADD:
         out[i] = nir_fsat(&b, nir_fadd_imm(&b, f, 1.5));
         break;
      case NEG_ABS:
         out[i] = nir_ffma(&b, nir_fneg(&b, nir_fabs(&b, f)), x, y);
         break;
      default:
         out[i] = nir_ffma(&b, f, x, y);
         break;
      }
   }
   nir_store_global(&b, nir_vec(&b, out, 4), nir_iadd_imm(&b, addr, 32),
                    .align_mul = 16);
   b.shader->info.workgroup_size[0] = 32;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   agx_shader_key key = {};
   key.dev.encoding = encoding;
   key.dev.chip = chip;
   key.dev.emit_agx3d = emit_agx3d;
   key.no_preamble = true;
   key.reserved_preamble = 16;
   agx_preprocess_nir(b.shader);
   agx_shader_part part = {};
   *ok = agx_compile_shader_nir(b.shader, &key, &part);
   const unsigned instrs = part.info.stats.instrs;
   free(part.binary);
   ralloc_free(b.shader);
   return instrs;
}

/* One conversion per half on M5. */
TEST(M5HalfWidenOnce, OneConversionPerHalf)
{
   bool ok;
   const unsigned halves =
      compile(FMA_OF_HALF, AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, &ok);
   ASSERT_TRUE(ok);
   const unsigned floats =
      compile(FMA_OF_FLOAT, AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, &ok);
   ASSERT_TRUE(ok);
   EXPECT_LE(halves, floats + 4);
}

/* The other legalizations still compile, on every target. */
TEST(M5HalfWidenOnce, OtherFormsAndTargetsCompile)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false},
   };
   for (auto t : targets) {
      for (enum kind k : {FMA_OF_HALF, SAT_ADD, NEG_ABS}) {
         bool ok;
         compile(k, t.encoding, t.chip, t.agx3d, &ok);
         EXPECT_TRUE(ok) << "chip " << t.chip << " kind " << k;
      }
   }
}


/* Subtracting a shifted zero- or sign-extended 16-bit integer from a 32-bit
 * value folds into a 32-bit IADD with a 16-bit negated source and nonzero
 * shift. After agx_lower_agx3_widening_int() widens the source to 32-bit,
 * AGX3C must still lower the shifted negated operand so packing succeeds. */
TEST(M5HalfWidenOnce, WidenedShiftedNegatedIaddPacksOnAGX3C)
{
   for (bool is_signed : {false, true}) {
      for (unsigned shift : {1u, 2u}) {
         nir_builder b = nir_builder_init_simple_shader(
            MESA_SHADER_COMPUTE, &agx_nir_options, "widened_shifted_sub");
         nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
         nir_def *base = nir_load_global(&b, 1, 32, addr, .align_mul = 4);
         nir_def *half = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 16),
                                         .align_mul = 2);
         nir_def *wide = is_signed ? nir_i2i32(&b, half) : nir_u2u32(&b, half);
         nir_def *res = nir_isub(&b, base, nir_ishl_imm(&b, wide, shift));
         nir_store_global(&b, res, nir_iadd_imm(&b, addr, 32), .align_mul = 4);

         b.shader->info.workgroup_size[0] = 32;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;

         agx_shader_key key = {};
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         key.dev.chip = AGX_CHIP_G17S;
         key.no_preamble = true;
         key.reserved_preamble = 16;

         agx_preprocess_nir_m5(b.shader);
         agx_shader_part part = {};
         ASSERT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << "is_signed=" << is_signed << " shift=" << shift;
         ASSERT_NE(part.binary, nullptr);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

} // namespace
