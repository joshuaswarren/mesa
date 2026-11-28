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

/* Paired low/high u8-to-f32 unpacks on AGX3B and AGX3C (not AGX3D) fold into
 * normflt.f16 + hmul + mixed-width ffma when consumed by all-register f32 FMAs,
 * and cleanly fall back to normflt.f32 + fmul (or widen) when consumed by
 * stores or uniforms. */
TEST(M5ByteToFloat, AGX3NormfltBytePairsCompile)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };
   for (auto t : targets) {
      for (bool fma_consumer : {false, true}) {
         nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                        &agx_nir_options,
                                                        "normflt-pairs");
         nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
         nir_def *w = nir_load_global(&b, 1, 32, addr);
         nir_def *weight = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 4));
         nir_def *acc0 = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 8));
         nir_def *acc1 = nir_load_global(&b, 1, 32, nir_iadd_imm(&b, addr, 12));
         nir_def *b0 = nir_u2f32(&b, nir_iand_imm(&b, w, 0xff));
         nir_def *b1 = nir_u2f32(&b, nir_iand_imm(&b, nir_ushr_imm(&b, w, 8), 0xff));
         nir_def *b2 = nir_u2f32(&b, nir_iand_imm(&b, nir_ushr_imm(&b, w, 16), 0xff));
         nir_def *b3 = nir_u2f32(&b, nir_ushr_imm(&b, w, 24));
         nir_def *out;
         if (fma_consumer) {
            nir_def *r0 = nir_ffma(&b, b0, weight, acc0);
            nir_def *r1 = nir_ffma(&b, b1, weight, acc1);
            nir_def *r2 = nir_ffma(&b, b2, weight, acc0);
            nir_def *r3 = nir_ffma(&b, b3, weight, acc1);
            out = nir_vec4(&b, r0, r1, r2, r3);
         } else {
            out = nir_vec4(&b, b0, b1, b2, b3);
         }
         nir_store_global(&b, out, nir_iadd_imm(&b, addr, 16), .align_mul = 16);
         b.shader->info.workgroup_size[0] = 32;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;

         agx_shader_key key = {};
         key.dev.encoding = t.encoding;
         key.dev.chip = t.chip;
         key.dev.emit_agx3d = t.emit_agx3d;
         key.no_preamble = true;
         key.reserved_preamble = 16;
         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part))
            << "chip=" << t.chip << " fma_consumer=" << fma_consumer;
         EXPECT_GT(part.info.stats.instrs, 0u);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

/* Verify AGX IR selection directly: paired u8->f32 unpacks emit
 * normflt.f16 + hmul when consumed by all-register f32 FFMA, emit
 * normflt.f32 + fmul when consumed by non-FMA users (e.g. stores), and emit
 * no normflt at all when emit_agx3d is set. */
TEST(M5ByteToFloat, AGX3NormfltBytePairsSelectPath)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
      bool expect_normflt;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true,  false},
   };
   for (auto t : targets) {
      for (bool fma_consumer : {false, true}) {
         SCOPED_TRACE(t.chip);
         SCOPED_TRACE(fma_consumer);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.encoding = t.encoding;
         key.dev.chip = t.chip;
         key.dev.emit_agx3d = t.emit_agx3d;
         b->shader->key = &key;

         agx_index half = agx_preload(b, agx_register(0, AGX_SIZE_16));
         agx_index weight = agx_preload(b, agx_register(2, AGX_SIZE_32));
         agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));
         agx_index zero = agx_mov_imm(b, 16, 0);
         agx_index sh8 = agx_mov_imm(b, 16, 8);
         agx_index hi_half = agx_bfeil(b, zero, half, sh8, 0);
         agx_index lo = agx_temp(b->shader, AGX_SIZE_32);
         agx_index hi = agx_temp(b->shader, AGX_SIZE_32);
         agx_convert_to(b, lo, agx_immediate(AGX_CONVERT_U8_TO_F), half,
                        AGX_ROUND_RTE);
         agx_convert_to(b, hi, agx_immediate(AGX_CONVERT_U16_TO_F), hi_half,
                        AGX_ROUND_RTE);

         if (fma_consumer) {
            agx_index r0 = agx_ffma(b, lo, weight, acc);
            agx_index r1 = agx_ffma(b, hi, weight, acc);
            agx_unit_test(b, r0);
            agx_unit_test(b, r1);
         } else {
            agx_unit_test(b, lo);
            agx_unit_test(b, hi);
         }

         agx_opt_agx3_normflt_byte_pairs(b->shader);

         unsigned normflt_f16 = 0, normflt_f32 = 0, hmul = 0, fmul = 0;
         unsigned ffma_f16_src0 = 0;
         agx_foreach_instr_global(b->shader, I) {
            if (I->op == AGX_OPCODE_NORMFLT_AGX3) {
               EXPECT_EQ(I->format, AGX_FORMAT_U8NORM);
               EXPECT_EQ(I->dest[0].channels_m1, 1u);
               if (I->dest[0].size == AGX_SIZE_16)
                  normflt_f16++;
               else if (I->dest[0].size == AGX_SIZE_32)
                  normflt_f32++;
            } else if (I->op == AGX_OPCODE_HMUL) {
               hmul++;
            } else if (I->op == AGX_OPCODE_FMUL) {
               fmul++;
            } else if (I->op == AGX_OPCODE_FFMA &&
                       I->src[0].size == AGX_SIZE_16) {
               ffma_f16_src0++;
            }
         }

         if (!t.expect_normflt) {
            EXPECT_EQ(normflt_f16, 0u);
            EXPECT_EQ(normflt_f32, 0u);
            EXPECT_EQ(hmul, 0u);
            EXPECT_EQ(fmul, 0u);
         } else if (fma_consumer) {
            EXPECT_EQ(normflt_f16, 1u);
            EXPECT_EQ(normflt_f32, 0u);
            EXPECT_EQ(hmul, 2u);
            EXPECT_EQ(fmul, 0u);
            EXPECT_EQ(ffma_f16_src0, 2u);
         } else {
            EXPECT_EQ(normflt_f16, 0u);
            EXPECT_EQ(normflt_f32, 1u);
            EXPECT_EQ(hmul, 0u);
            EXPECT_EQ(fmul, 2u);
         }
         ralloc_free(mem);
      }
   }
}

TEST(M5ByteToFloat, AGX3NormfltHalfBitopAndDedupScale)
{
   /* Match 16-bit BITOP(AND, h, 0xff) + BFEIL(0, h, 8) feeding U16_TO_F
    * (the unpack emitted for packed r8_uint vector loads) across AGX3B and
    * AGX3C, and emit at most one 255.0h scale immediate per block. */
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
      bool expect_normflt;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true,  false},
   };
   for (auto t : targets) {
      SCOPED_TRACE(t.chip);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      b->shader->key = &key;

      agx_index h0 = agx_preload(b, agx_register(0, AGX_SIZE_16));
      agx_index h1 = agx_preload(b, agx_register(1, AGX_SIZE_16));
      agx_index weight = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));
      agx_index mask = agx_mov_imm(b, 16, 0xff);
      agx_index zero = agx_mov_imm(b, 16, 0);
      agx_index sh8 = agx_mov_imm(b, 16, 8);

      agx_index b0 = agx_and(b, h0, mask);
      agx_index b1 = agx_bfeil(b, zero, h0, sh8, 0);
      agx_index b2 = agx_and(b, h1, mask);
      agx_index b3 = agx_bfeil(b, zero, h1, sh8, 0);

      agx_index f[4];
      agx_index bytes[4] = {b0, b1, b2, b3};
      for (unsigned i = 0; i < 4; ++i) {
         f[i] = agx_temp(b->shader, AGX_SIZE_32);
         agx_convert_to(b, f[i], agx_immediate(AGX_CONVERT_U16_TO_F),
                        bytes[i], AGX_ROUND_RTE);
         acc = agx_ffma(b, f[i], weight, acc);
      }
      agx_unit_test(b, acc);

      agx_opt_agx3_normflt_byte_pairs(b->shader);

      unsigned normflt_f16 = 0, hmul = 0, scale_imm_255h = 0, converts = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_NORMFLT_AGX3 &&
             I->dest[0].size == AGX_SIZE_16)
            normflt_f16++;
         else if (I->op == AGX_OPCODE_HMUL)
            hmul++;
         else if (I->op == AGX_OPCODE_MOV_IMM &&
                  I->dest[0].size == AGX_SIZE_16 && I->imm == 0x5bf8)
            scale_imm_255h++;
         else if (I->op == AGX_OPCODE_CONVERT)
            converts++;
      }

      if (t.expect_normflt) {
         EXPECT_EQ(normflt_f16, 2u);
         EXPECT_EQ(hmul, 4u);
         EXPECT_EQ(scale_imm_255h, 1u);
         EXPECT_EQ(converts, 0u);
      } else {
         EXPECT_EQ(normflt_f16, 0u);
         EXPECT_EQ(hmul, 0u);
         EXPECT_EQ(scale_imm_255h, 0u);
         EXPECT_EQ(converts, 4u);
      }
      ralloc_free(mem);
   }
}

TEST(M5ByteToFloat, AGX3BNormfltLoopKeepsScaleInGpr)
{
   /* Inside a loop on AGX3B, agx_opt_agx3_normflt_byte_pairs hoists the 255.0h
    * scale constant to the entry block and agx_opt_promote_constants keeps it
    * in a GPR instead of promoting it to a 16-bit uniform read on every HMUL. */
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool expect_gpr_hoisted;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
   };
   for (auto t : targets) {
      SCOPED_TRACE(t.chip);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.promote_constants = true;
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, struct agx_shader_info);

      agx_block *entry = agx_start_block(b->shader);
      agx_block *loop = agx_test_block(b->shader);
      agx_block *exit_blk = agx_test_block(b->shader);
      loop->loop_header = true;
      agx_block_add_successor(entry, loop);
      agx_block_add_successor(loop, loop);
      agx_block_add_successor(loop, exit_blk);

      b->cursor = agx_after_block(entry);
      agx_index h0 = agx_preload(b, agx_register(0, AGX_SIZE_16));
      agx_index weight = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));

      b->cursor = agx_after_block(loop);
      agx_index mask = agx_mov_imm(b, 16, 0xff);
      agx_index zero = agx_mov_imm(b, 16, 0);
      agx_index sh8 = agx_mov_imm(b, 16, 8);
      agx_index b0 = agx_and(b, h0, mask);
      agx_index b1 = agx_bfeil(b, zero, h0, sh8, 0);
      agx_index f0 = agx_temp(b->shader, AGX_SIZE_32);
      agx_index f1 = agx_temp(b->shader, AGX_SIZE_32);
      agx_convert_to(b, f0, agx_immediate(AGX_CONVERT_U16_TO_F), b0, AGX_ROUND_RTE);
      agx_convert_to(b, f1, agx_immediate(AGX_CONVERT_U16_TO_F), b1, AGX_ROUND_RTE);
      acc = agx_ffma(b, f0, weight, acc);
      acc = agx_ffma(b, f1, weight, acc);

      b->cursor = agx_after_block(exit_blk);
      agx_unit_test(b, acc);

      agx_opt_agx3_normflt_byte_pairs(b->shader);
      agx_opt_promote_constants(b->shader);

      unsigned entry_scale = 0, loop_scale = 0;
      unsigned hmul_uniform = 0, hmul_gpr = 0;
      agx_foreach_instr_in_block(entry, I) {
         if (I->op == AGX_OPCODE_MOV_IMM &&
             I->dest[0].size == AGX_SIZE_16 && I->imm == 0x5bf8)
            entry_scale++;
      }
      agx_foreach_instr_in_block(loop, I) {
         if (I->op == AGX_OPCODE_MOV_IMM &&
             I->dest[0].size == AGX_SIZE_16 && I->imm == 0x5bf8)
            loop_scale++;
         if (I->op == AGX_OPCODE_HMUL) {
            if (I->src[1].type == AGX_INDEX_UNIFORM)
               hmul_uniform++;
            else if (I->src[1].type == AGX_INDEX_NORMAL)
               hmul_gpr++;
         }
      }

      if (t.expect_gpr_hoisted) {
         EXPECT_EQ(entry_scale, 1u);
         EXPECT_EQ(loop_scale, 0u);
         EXPECT_EQ(hmul_uniform, 0u);
         EXPECT_EQ(hmul_gpr, 2u);
      } else {
         EXPECT_EQ(entry_scale, 0u);
         EXPECT_EQ(loop_scale, 1u);
         EXPECT_EQ(hmul_uniform, 0u);
         EXPECT_EQ(hmul_gpr, 2u);
      }
      ralloc_free(mem);
   }
}

/* On AGX3C (M5), HADD with a plain 16-bit minifloat immediate is encoded
 * natively (agx3_alu_hadd_minifloat), so agx_materialize_agx3_alu_constants()
 * should preserve it for register operands while still materializing it on
 * AGX3B and for uniform operands. */
TEST(M5ByteToFloat, MaterializeHaddMinifloatOnM5)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool uniform_src;
      bool expect_imm_kept;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, true, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, false, true},
   };

   for (auto c : cases) {
      SCOPED_TRACE(c.chip);
      SCOPED_TRACE(c.uniform_src);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = c.encoding;
      key.dev.chip = c.chip;
      key.dev.emit_agx3d = (c.chip == AGX_CHIP_G18G);
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, struct agx_shader_info);

      agx_index x = c.uniform_src ? agx_uniform(0, AGX_SIZE_16)
                                  : agx_preload(b, agx_register(0, AGX_SIZE_16));
      agx_index y = agx_hadd(b, x, agx_immediate_f(-8.0f));
      agx_unit_test(b, y);

      agx_materialize_agx3_alu_constants(b->shader);

      unsigned hadd_imm = 0, hadd_reg = 0;
      agx_foreach_instr_global(b->shader, I) {
         if (I->op != AGX_OPCODE_HADD)
            continue;
         if (I->src[1].type == AGX_INDEX_IMMEDIATE)
            hadd_imm++;
         else
            hadd_reg++;
      }

      if (c.expect_imm_kept) {
         EXPECT_EQ(hadd_imm, 1u);
         EXPECT_EQ(hadd_reg, 0u);
      } else {
         EXPECT_EQ(hadd_imm, 0u);
         EXPECT_EQ(hadd_reg, 1u);
      }
      ralloc_free(mem);
   }
}

/* Q4_0 / Q6_K dequantization chains:
 *   f2f16(fmul(fadd(u2f32(byte), -8.0f), f2f32(d_f16))) -> hmul(hadd(u2f16(byte), #-8.0), d_f16)
 *   ffma(fadd(u2f32(byte), -32.0f), w_f32, acc_f32)     -> ffma(hadd(u2f16(byte), mov_imm(-32.0h)), w_f32, acc_f32)
 * On AGX3C, both stay in binary16 with zero FADD/FMUL/f2f16 overhead; on AGX3B
 * or for non-integer biases they remain in f32. */
TEST(M5ByteToFloat, NarrowDequantF16Chain)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      float bias;
      bool narrow_mul_to_f16;
      bool expect_narrow;
      bool expect_inline_minifloat;
   } cases[] = {
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, -8.0f, true, true, true},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, -32.0f, false, true, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, -8.5f, true, false, false},
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, -8.0f, true, false, false},
   };

   for (auto c : cases) {
      SCOPED_TRACE(c.chip);
      SCOPED_TRACE(c.bias);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = c.encoding;
      key.dev.chip = c.chip;
      b->shader->key = &key;
      b->shader->out = rzalloc(mem, struct agx_shader_info);

      agx_index h0 = agx_preload(b, agx_register(0, AGX_SIZE_16));
      agx_index d16 = agx_preload(b, agx_register(1, AGX_SIZE_16));
      agx_index w32 = agx_preload(b, agx_register(2, AGX_SIZE_32));
      agx_index acc = agx_preload(b, agx_register(4, AGX_SIZE_32));

      agx_index mask = agx_mov_imm(b, 16, 0xff);
      agx_index zero = agx_mov_imm(b, 16, 0);
      agx_index sh8 = agx_mov_imm(b, 16, 8);
      agx_index b0 = agx_and(b, h0, mask);
      agx_index b1 = agx_bfeil(b, zero, h0, sh8, 0);
      agx_index f0 = agx_temp(b->shader, AGX_SIZE_32);
      agx_index f1 = agx_temp(b->shader, AGX_SIZE_32);
      agx_convert_to(b, f0, agx_immediate(AGX_CONVERT_U16_TO_F), b0, AGX_ROUND_RTE);
      agx_convert_to(b, f1, agx_immediate(AGX_CONVERT_U16_TO_F), b1, AGX_ROUND_RTE);

      agx_index bias_imm = agx_mov_imm(b, 32, fui(c.bias));
      agx_index a0 = agx_fadd(b, f0, bias_imm);
      agx_index a1 = agx_fadd(b, f1, bias_imm);

      if (c.narrow_mul_to_f16) {
         agx_index d32 = agx_fadd(b, d16, agx_negzero());
         agx_index m0 = agx_fmul(b, a0, d32);
         agx_index m1 = agx_fmul(b, a1, d32);
         agx_index o0 = agx_temp(b->shader, AGX_SIZE_16);
         agx_index o1 = agx_temp(b->shader, AGX_SIZE_16);
         agx_fadd_to(b, o0, m0, agx_negzero());
         agx_fadd_to(b, o1, m1, agx_negzero());
         agx_unit_test(b, o0);
         agx_unit_test(b, o1);
      } else {
         acc = agx_ffma(b, a0, w32, acc);
         acc = agx_ffma(b, a1, w32, acc);
         agx_unit_test(b, acc);
      }

      agx_opt_agx3_normflt_byte_pairs(b->shader);

      unsigned hadd_imm = 0, hadd_reg = 0, hmul = 0, fmul = 0, bias_mov16 = 0;
      uint16_t bias_bits = _mesa_float_to_half(c.bias);
      agx_foreach_instr_global(b->shader, I) {
         if (I->op == AGX_OPCODE_HADD) {
            if (I->src[1].type == AGX_INDEX_IMMEDIATE)
               hadd_imm++;
            else
               hadd_reg++;
         } else if (I->op == AGX_OPCODE_HMUL) {
            hmul++;
         } else if (I->op == AGX_OPCODE_FMUL) {
            fmul++;
         } else if (I->op == AGX_OPCODE_MOV_IMM &&
                    I->dest[0].size == AGX_SIZE_16 &&
                    I->imm == bias_bits) {
            bias_mov16++;
         }
      }

      if (c.expect_narrow) {
         if (c.expect_inline_minifloat) {
            EXPECT_EQ(hadd_imm, 2u);
            EXPECT_EQ(hadd_reg, 0u);
            EXPECT_EQ(bias_mov16, 0u);
         } else {
            EXPECT_EQ(hadd_imm, 0u);
            EXPECT_EQ(hadd_reg, 2u);
            EXPECT_EQ(bias_mov16, 1u); /* shared per block */
         }
         if (c.narrow_mul_to_f16) {
            EXPECT_EQ(hmul, 4u); /* 2x *255 + 2x *d_f16 */
            EXPECT_EQ(fmul, 0u);
         } else {
            EXPECT_EQ(hmul, 2u); /* 2x *255 */
         }
      } else {
         EXPECT_EQ(hadd_imm, 0u);
         EXPECT_EQ(hadd_reg, 0u);
         EXPECT_EQ(hmul, 0u);
         if (c.narrow_mul_to_f16) {
            EXPECT_EQ(fmul, 4u); /* 2x *255.0f + 2x *d32 */
         }
      }
      ralloc_free(mem);
   }
}

/* End-to-end AGX3C/G18G binary encoding check for HADD minifloat + HMUL chain. */
TEST(M5ByteToFloat, EncodeNarrowDequantF16Chain)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };

   for (auto t : targets) {
      SCOPED_TRACE(t.chip);
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options,
                                                     "dequant-q4-f16");
      nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *h = nir_load_global(&b, 1, 16, addr);
      nir_def *d = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 2));
      nir_def *d32 = nir_f2f32(&b, d);
      nir_def *lo = nir_iand_imm(&b, h, 0xff);
      nir_def *hi = nir_ushr_imm(&b, h, 8);
      nir_def *f0 = nir_f2f16(&b, nir_fmul(&b, nir_fadd_imm(&b, nir_u2f32(&b, lo), -8.0), d32));
      nir_def *f1 = nir_f2f16(&b, nir_fmul(&b, nir_fadd_imm(&b, nir_u2f32(&b, hi), -8.0), d32));
      nir_store_global(&b, nir_vec2(&b, f0, f1), nir_iadd_imm(&b, addr, 8), .align_mul = 4);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      agx_shader_key key = {};
      key.dev.encoding = t.encoding;
      key.dev.chip = t.chip;
      key.dev.emit_agx3d = t.emit_agx3d;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      EXPECT_GT(part.info.stats.instrs, 0u);
      free(part.binary);
      ralloc_free(b.shader);
   }
}


/* GLSL unpack8(w) lowers to u2u8/i2i8 of a 32-bit word and its 8/16/24-bit
 * shifts. On AGX3C (M5, !emit_agx3d), u2f32 and i2f32/i2f16 must extract each
 * byte directly from the word's 16-bit halves without 0xff masks or 32-bit
 * shifts, and every target (AGX3B, G17C, G18G) must compile cleanly. */
TEST(M5ByteToFloat, Unpack8WordToFloat)
{
   const struct {
      agx_isa_encoding encoding;
      agx_chip chip;
      bool emit_agx3d;
   } targets[] = {
      {AGX_ISA_ENCODING_AGX3B, AGX_CHIP_G17P, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G17C, false},
      {AGX_ISA_ENCODING_AGX3C, AGX_CHIP_G18G, true},
   };

   for (auto t : targets) {
      for (bool is_signed_conv : {false, true}) {
         SCOPED_TRACE(t.chip);
         SCOPED_TRACE(is_signed_conv);
         nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                        &agx_nir_options,
                                                        "unpack8-word");
         nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
         nir_def *w = nir_load_global(&b, 1, 32, addr);
         nir_def *h = nir_load_global(&b, 1, 16, nir_iadd_imm(&b, addr, 4));
         nir_def *masked = nir_iand_imm(&b, w, 0x0f0f0f0f);

         nir_def *bytes[6] = {
            nir_u2u8(&b, masked),
            nir_u2u8(&b, nir_ushr_imm(&b, masked, 8)),
            nir_u2u8(&b, nir_unpack_32_2x16_split_y(&b, masked)),
            nir_u2u8(&b, nir_ushr_imm(&b, masked, 24)),
            nir_i2i8(&b, nir_ishr_imm(&b, w, 24)),
            nir_u2u8(&b, nir_ushr_imm(&b, nir_u2u32(&b, h), 8)),
         };
         nir_def *f[6];
         for (unsigned i = 0; i < 6; ++i)
            f[i] = is_signed_conv ? nir_i2f32(&b, bytes[i])
                                  : nir_u2f32(&b, bytes[i]);

         nir_def *out0 = nir_vec4(&b, f[0], f[1], f[2], f[3]);
         nir_def *out1 = nir_vec2(&b, f[4], f[5]);
         nir_store_global(&b, out0, nir_iadd_imm(&b, addr, 16), .align_mul = 16);
         nir_store_global(&b, out1, nir_iadd_imm(&b, addr, 32), .align_mul = 8);
         b.shader->info.workgroup_size[0] = 32;
         b.shader->info.workgroup_size[1] = 1;
         b.shader->info.workgroup_size[2] = 1;

         agx_shader_key key = {};
         key.dev.encoding = t.encoding;
         key.dev.chip = t.chip;
         key.dev.emit_agx3d = t.emit_agx3d;
         key.no_preamble = true;
         key.reserved_preamble = 16;
         agx_preprocess_nir(b.shader);
         agx_shader_part part = {};
         EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
         EXPECT_GT(part.info.stats.instrs, 0u);
         free(part.binary);
         ralloc_free(b.shader);
      }
   }
}

TEST(M5ByteToFloat, Unpack8WordNoMaskOnM5)
{
   /* Compare instruction counts on M5 (G17C) vs G18G (emit_agx3d) for unpacking
    * all 4 unsigned bytes of a 32-bit word through u2u8 + u2f32 (the GLSL
    * unpack8() lowering). M5 must eliminate the 0xff masks and 32-bit shifts,
    * yielding strictly fewer instructions than the G18G fallback. */
   auto compile_for = [](agx_chip chip, bool emit_agx3d, bool is_signed_conv) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options,
                                                     "unpack8-m5-count");
      nir_def *addr = agx_test_load_preamble(&b, 1, 64, 0);
      nir_def *w = nir_load_global(&b, 1, 32, addr);
      nir_def *b0 = nir_u2u8(&b, w);
      nir_def *b1 = nir_u2u8(&b, nir_ushr_imm(&b, w, 8));
      nir_def *b2 = nir_u2u8(&b, nir_unpack_32_2x16_split_y(&b, w));
      nir_def *b3 = nir_u2u8(&b, nir_ushr_imm(&b, w, 24));
      nir_def *f0 = is_signed_conv ? nir_i2f32(&b, b0) : nir_u2f32(&b, b0);
      nir_def *f1 = is_signed_conv ? nir_i2f32(&b, b1) : nir_u2f32(&b, b1);
      nir_def *f2 = is_signed_conv ? nir_i2f32(&b, b2) : nir_u2f32(&b, b2);
      nir_def *f3 = is_signed_conv ? nir_i2f32(&b, b3) : nir_u2f32(&b, b3);
      nir_store_global(&b, nir_vec4(&b, f0, f1, f2, f3),
                       nir_iadd_imm(&b, addr, 16), .align_mul = 16);
      b.shader->info.workgroup_size[0] = 32;
      b.shader->info.workgroup_size[1] = 1;
      b.shader->info.workgroup_size[2] = 1;

      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.chip = chip;
      key.dev.emit_agx3d = emit_agx3d;
      key.no_preamble = true;
      key.reserved_preamble = 16;
      agx_preprocess_nir(b.shader);
      agx_shader_part part = {};
      EXPECT_TRUE(agx_compile_shader_nir(b.shader, &key, &part));
      unsigned instrs = part.info.stats.instrs;
      free(part.binary);
      ralloc_free(b.shader);
      return instrs;
   };

   for (bool is_signed_conv : {false, true}) {
      unsigned m5 = compile_for(AGX_CHIP_G17C, false, is_signed_conv);
      unsigned g18 = compile_for(AGX_CHIP_G18G, true, is_signed_conv);
      EXPECT_LT(m5, g18) << "is_signed_conv=" << is_signed_conv;
   }
}


TEST(M5ByteToFloat, NarrowHalfMaskShiftAndUbfeToFloat)
{
   /* agx_nir_opt_m5_narrow_to_float narrows u2f32/i2f32 of masked, shifted,
    * and top-bits ubfe of a zero-extended u16 to 16-bit ALU + u2f32(u16),
    * while leaving 0xff masks for the 1-instruction U8_TO_F path and rejecting
    * out-of-range shifts or non-top ubfe. */
   const uint16_t samples[] = {0x0000, 0x000f, 0x00f0, 0x0f0f, 0x8001, 0xf123, 0xffff};
   for (unsigned mode = 0; mode < 6; ++mode) {
      SCOPED_TRACE(mode);
      for (uint16_t hv : samples) {
         nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                        &agx_nir_options, "narrow16");
         nir_def *h = agx_test_load_preamble(&b, 1, 16, 0);
         nir_def *h32 = nir_u2u32(&b, h);
         nir_def *expr = NULL;
         bool use_i2f = (mode & 1) != 0;
         float expected = 0.0f;
         switch (mode / 2) {
         case 0: /* nibble mask 0xf */
            expr = nir_iand_imm(&b, h32, 0xf);
            expected = (float)(hv & 0xf);
            break;
         case 1: /* shift right by 4 */
            expr = nir_ushr_imm(&b, h32, 4);
            expected = (float)(hv >> 4);
            break;
         case 2: /* top-bits ubfe(h32, 12, 4) */
            expr = nir_ubitfield_extract(&b, h32, nir_imm_int(&b, 12), nir_imm_int(&b, 4));
            expected = (float)(hv >> 12);
            break;
         }
         nir_def *f = use_i2f ? nir_i2f32(&b, expr) : nir_u2f32(&b, expr);
         nir_store_global(&b, f, nir_imm_int64(&b, 0), .write_mask = 1);

         EXPECT_TRUE(agx_nir_opt_m5_narrow_to_float(b.shader));
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         NIR_PASS(_, b.shader, nir_opt_dce);
         nir_validate_shader(b.shader, "narrow16");

         /* u2u32 should be dead and conversion should take 16-bit u2f32. */
         bool found_u2u32 = false, found_u2f32_16 = false;
         nir_foreach_block(blk, nir_shader_get_entrypoint(b.shader)) {
            nir_foreach_instr(instr, blk) {
               if (instr->type != nir_instr_type_alu)
                  continue;
               nir_alu_instr *alu = nir_instr_as_alu(instr);
               if (alu->op == nir_op_u2u32)
                  found_u2u32 = true;
               if (alu->op == nir_op_u2f32 && nir_src_bit_size(alu->src[0].src) == 16)
                  found_u2f32_16 = true;
            }
         }
         EXPECT_FALSE(found_u2u32);
         EXPECT_TRUE(found_u2f32_16);

         /* Substitute constant h and fold to verify bit-exact result. */
         b.cursor = nir_before_impl(nir_shader_get_entrypoint(b.shader));
         nir_def_replace(h, nir_imm_intN_t(&b, hv, 16));
         NIR_PASS(_, b.shader, nir_opt_constant_folding);
         NIR_PASS(_, b.shader, nir_opt_dce);
         nir_intrinsic_instr *st = NULL;
         nir_foreach_block(blk, nir_shader_get_entrypoint(b.shader)) {
            nir_foreach_instr(instr, blk) {
               if (instr->type == nir_instr_type_intrinsic &&
                   nir_instr_as_intrinsic(instr)->intrinsic == nir_intrinsic_store_global)
                  st = nir_instr_as_intrinsic(instr);
            }
         }
         ASSERT_NE(st, nullptr);
         EXPECT_EQ(nir_src_as_float(st->src[0]), expected);
         ralloc_free(b.shader);
      }
   }

   /* Negative controls: 0xff mask (kept for U8_TO_F), shift >= 16, non-top ubfe. */
   for (unsigned neg = 0; neg < 3; ++neg) {
      SCOPED_TRACE(neg);
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                     &agx_nir_options, "narrow16_neg");
      nir_def *h = agx_test_load_preamble(&b, 1, 16, 0);
      nir_def *h32 = nir_u2u32(&b, h);
      nir_def *expr = NULL;
      if (neg == 0)
         expr = nir_iand_imm(&b, h32, 0xff);
      else if (neg == 1)
         expr = nir_ushr_imm(&b, h32, 16);
      else
         expr = nir_ubitfield_extract(&b, h32, nir_imm_int(&b, 4), nir_imm_int(&b, 4));
      nir_store_global(&b, nir_u2f32(&b, expr), nir_imm_int64(&b, 0), .write_mask = 1);
      EXPECT_FALSE(agx_nir_opt_m5_narrow_to_float(b.shader));
      nir_validate_shader(b.shader, "narrow16_neg");
      ralloc_free(b.shader);
   }
}

} // namespace
