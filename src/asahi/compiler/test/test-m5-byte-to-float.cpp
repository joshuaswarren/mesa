/* Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */
#include <gtest/gtest.h>
#include "agx_test.h"
extern "C" {
#include "compiler/nir/nir_builder.h"
#include "util/half_float.h"
#include "agx_compile.h"
#include "agx_nir.h"
}

/* agx_nir_fold_byte_to_float and agx_nir_fold_ubyte_to_float rewrite byte
 * extracts converted to float into conversions of a u2u8 of the half that
 * holds the byte. The rewritten shader must compute the same values (checked
 * here by evaluating both forms for bytes 0x00, 0x7f, 0x80 and 0xff beside
 * nonzero neighbours), and a whole compute shader using them must compile on
 * M5 and on the targets that keep the earlier lowering.
 */

namespace {

enum pattern {
   S_H_LO,  /* i2f(ishr(ishl(u2u32(h), 24), 24)) */
   S_H_HI,  /* i2f(ishr(ishl(u2u32(h), 16), 24)) */
   S_W_B0,  /* i2f(ishr(ishl(w, 24), 24)) */
   S_W_B1,  /* i2f(ishr(ishl(w, 16), 24)) */
   S_W_B2,  /* i2f(ishr(ishl(w, 8), 24)) */
   S_BFE_8, /* i2f(ibitfield_extract(w, 8, 8)) */
   S_BFE_16,
   U_H_LO, /* u2f32(iand(u2u32(h), 0xff)) */
   U_W_B0, /* u2f32(iand(w, 0xff)) */
   U_W_B2, /* u2f32(iand(ushr(w, 16), 0xff)) */
   U_BFE_16,
   NUM_PATTERNS,
};

static bool
is_signed(enum pattern p)
{
   return p < U_H_LO;
}

static nir_def *
extract(nir_builder *b, enum pattern p, nir_def *h, nir_def *w)
{
   nir_def *h32 = nir_u2u32(b, h);
   switch (p) {
   case S_H_LO:
      return nir_ishr_imm(b, nir_ishl_imm(b, h32, 24), 24);
   case S_H_HI:
      return nir_ishr_imm(b, nir_ishl_imm(b, h32, 16), 24);
   case S_W_B0:
      return nir_ishr_imm(b, nir_ishl_imm(b, w, 24), 24);
   case S_W_B1:
      return nir_ishr_imm(b, nir_ishl_imm(b, w, 16), 24);
   case S_W_B2:
      return nir_ishr_imm(b, nir_ishl_imm(b, w, 8), 24);
   case S_BFE_8:
      return nir_ibitfield_extract(b, w, nir_imm_int(b, 8), nir_imm_int(b, 8));
   case S_BFE_16:
      return nir_ibitfield_extract(b, w, nir_imm_int(b, 16), nir_imm_int(b, 8));
   case U_H_LO:
      return nir_iand_imm(b, h32, 0xff);
   case U_W_B0:
      return nir_iand_imm(b, w, 0xff);
   case U_W_B2:
      return nir_iand_imm(b, nir_ushr_imm(b, w, 16), 0xff);
   case U_BFE_16:
      return nir_ubitfield_extract(b, w, nir_imm_int(b, 16), nir_imm_int(b, 8));
   default:
      return NULL;
   }
}

static int
expected_byte(enum pattern p, uint16_t h, uint32_t w)
{
   switch (p) {
   case S_H_LO:
      return (int8_t)h;
   case S_H_HI:
      return (int8_t)(h >> 8);
   case S_W_B0:
      return (int8_t)w;
   case S_W_B1:
      return (int8_t)(w >> 8);
   case S_W_B2:
      return (int8_t)(w >> 16);
   case S_BFE_8:
      return (int8_t)(w >> 8);
   case S_BFE_16:
      return (int8_t)(w >> 16);
   case U_H_LO:
      return (uint8_t)h;
   case U_W_B0:
      return (uint8_t)w;
   case U_W_B2:
      return (uint8_t)(w >> 16);
   case U_BFE_16:
      return (uint8_t)(w >> 16);
   default:
      return 0;
   }
}

/* Builds conv(extract), runs both passes, substitutes h and w and folds.
 * Returns the float the shader computes; *progress tells whether a pass
 * rewrote it. */
static float
evaluate(enum pattern p, unsigned dest_bits, uint16_t hv, uint32_t wv,
         bool *progress)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "bytes");
   nir_def *h = agx_test_load_preamble(&b, 1, 16, 0);
   nir_def *w = agx_test_load_preamble(&b, 1, 32, 2);
   nir_def *v = extract(&b, p, h, w);
   nir_def *f =
      is_signed(p)
         ? nir_type_convert(&b, v, nir_type_int32,
                            (nir_alu_type)(nir_type_float | dest_bits),
                            nir_rounding_mode_undef)
         : nir_type_convert(&b, v, nir_type_uint32,
                            (nir_alu_type)(nir_type_float | dest_bits),
                            nir_rounding_mode_undef);
   agx_test_store_preamble(&b, f, 8);

   *progress = false;
   NIR_PASS(*progress, b.shader, agx_nir_fold_byte_to_float);
   NIR_PASS(*progress, b.shader, agx_nir_fold_ubyte_to_float);

   b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
   nir_def_rewrite_uses(h, nir_imm_intN_t(&b, hv, 16));
   nir_def_rewrite_uses(w, nir_imm_int(&b, wv));
   nir_opt_constant_folding(b.shader);

   float result = NAN;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic !=
                nir_intrinsic_store_preamble)
            continue;
         nir_src *src = &nir_instr_as_intrinsic(instr)->src[0];
         EXPECT_TRUE(nir_src_is_const(*src));
         const uint64_t bits = nir_src_as_uint(*src);
         if (dest_bits == 16) {
            result = _mesa_half_to_float((uint16_t)bits);
         } else {
            uint32_t u = (uint32_t)bits;
            memcpy(&result, &u, 4);
         }
      }
   }
   ralloc_free(b.shader);
   return result;
}

TEST(M5ByteToFloat, RewritesKeepEveryValue)
{
   const uint16_t hs[] = {0x0000, 0x007f, 0x0080, 0x00ff, 0x7f00,
                          0x8000, 0xff00, 0x12ff, 0xab80, 0x807f};
   const uint32_t ws[] = {0x00000000u, 0x0000007fu, 0x00008000u,
                          0x00ff0000u, 0xff000000u, 0x12ab80ffu,
                          0x7f80ff01u, 0x80ff7f00u, 0xdeadbeefu};
   for (unsigned p = 0; p < NUM_PATTERNS; ++p) {
      /* u2f16 of an unsigned byte is not rewritten: f32 only. */
      for (unsigned dest : {16u, 32u}) {
         for (uint16_t h : hs) {
            for (uint32_t w : ws) {
               bool progress;
               const float got =
                  evaluate((enum pattern)p, dest, h, w, &progress);
               EXPECT_EQ(got, (float)expected_byte((enum pattern)p, h, w))
                  << "pattern " << p << " dest " << dest << std::hex << " h 0x"
                  << h << " w 0x" << w;
               if (is_signed((enum pattern)p) || dest == 32)
                  EXPECT_TRUE(progress) << "pattern " << p << " dest " << dest;
            }
         }
      }
   }
}

/* A compute shader that loads a word and a half and stores, as floats, both
 * bytes of the half signed and the low one unsigned, and bytes 0, 1 and 2 of
 * the word signed and 0 and 2 unsigned. */
static unsigned
compile_bytes(agx_isa_encoding encoding, agx_chip chip, bool emit_agx3d,
              bool *ok)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "bytes");
   nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
   nir_def *w = nir_load_global(&b, 1, 32, addr);
   nir_def *h = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 4));
   nir_def *out[8];
   const enum pattern ps[8] = {S_H_LO, S_H_HI, S_W_B0, S_W_B1,
                               S_W_B2, U_H_LO, U_W_B0, U_W_B2};
   for (unsigned i = 0; i < 8; ++i) {
      nir_def *v = extract(&b, ps[i], h, w);
      out[i] = is_signed(ps[i]) ? nir_i2f32(&b, v) : nir_u2f32(&b, v);
   }
   nir_store_global(&b, nir_vec4(&b, out[0], out[1], out[2], out[3]),
                    nir_iadd_imm(&b, addr, 16), .align_mul = 16);
   nir_store_global(&b, nir_vec4(&b, out[4], out[5], out[6], out[7]),
                    nir_iadd_imm(&b, addr, 32), .align_mul = 16);
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

TEST(M5ByteToFloat, CompilesOnEachTarget)
{
   bool ok;
   const unsigned m5 =
      compile_bytes(AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, &ok);
   EXPECT_TRUE(ok);
   const unsigned g18 =
      compile_bytes(AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true, &ok);
   EXPECT_TRUE(ok);
   compile_bytes(AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, &ok);
   EXPECT_TRUE(ok);
   compile_bytes(AGX_ISA_ENCODING_AGX2, AGX_CHIP_G13G, false, &ok);
   EXPECT_TRUE(ok);
   /* M5 converts each of the eight bytes natively; G18 keeps its halfword
    * forms, at least one more instruction each. */
   EXPECT_GE(g18, m5 + 8);
}

} // namespace
