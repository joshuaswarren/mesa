/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_test.h"
#include <gtest/gtest.h>

TEST(SignedZeroFMA, PreservesProductSignAndAddendModifiers)
{
   for (unsigned imm = 0; imm < 256; imm++) {
      for (unsigned zero : {0u, 128u}) {
         for (bool swap : {false, true}) {
            for (unsigned flags = 0; flags < 16; flags++) {
               if ((flags & 12) == 12)
                  continue;
               SCOPED_TRACE(imm);
               SCOPED_TRACE(zero);
               SCOPED_TRACE(swap);
               SCOPED_TRACE(flags);
               void *mem = ralloc_context(NULL);
               agx_builder *b = agx_test_builder(mem);
               agx_shader_key key = {};
               key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
               b->shader->key = &key;
               agx_index addend = agx_register(48, AGX_SIZE_32);
               addend.abs = flags & 1;
               addend.neg = (flags >> 1) & 1;
               addend.cache = (flags >> 2) & 1;
               addend.discard = (flags >> 3) & 1;
               agx_instr *I = agx_ffma_to(
                  b, agx_register(32, AGX_SIZE_32),
                  agx_immediate(swap ? zero : imm),
                  agx_immediate(swap ? imm : zero), addend);
               agx_lower_agx3_fma_sources(b->shader);
               ASSERT_EQ(I->op, AGX_OPCODE_FADD);
               EXPECT_EQ(I->nr_srcs, 2u);
               EXPECT_EQ(memcmp(&I->src[0], &addend, sizeof(addend)), 0);
               EXPECT_EQ(I->src[1].type, AGX_INDEX_IMMEDIATE);
               EXPECT_EQ(I->src[1].value, (imm ^ zero) & 128);
               EXPECT_FALSE(I->src[1].neg);
               EXPECT_EQ(I->dest[0].value, 32u);
               EXPECT_EQ(I->dest[0].size, AGX_SIZE_32);
               ralloc_free(mem);
            }
         }
      }
   }
}

TEST(SignedZeroFMA, RetainsUnsupportedForms)
{
   for (unsigned mode = 0; mode < 10; mode++) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = mode == 6 ? AGX_ISA_ENCODING_AGX2
                                  : AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      agx_instr *I = agx_ffma_to(
         b, agx_register(32, AGX_SIZE_32), agx_immediate(128),
         agx_immediate(64), agx_register(48, AGX_SIZE_32));
      switch (mode) {
      case 0: I->saturate = true; break;
      case 1: I->src[0].neg = true; break;
      case 2: I->src[0].abs = true; break;
      case 3: I->src[0].memory = true; break;
      case 4: I->src[0] = agx_immediate(0x100); break;
      case 5: I->src[0] = agx_immediate(0x38); break;
      case 6: break;
      case 7: I->src[0] = agx_register(20, AGX_SIZE_32); break;
      case 8: I->dest[0].size = AGX_SIZE_16; break;
      case 9: I->dest[0].channels_m1 = 1; break;
      }
      agx_lower_agx3_fma_sources(b->shader);
      EXPECT_EQ(I->op, AGX_OPCODE_FFMA) << mode;
      ralloc_free(mem);
   }
}

/* Mixed FMA shapes the G17P packer encodes with their modifiers fused
 * (agx3_falu_fma_addend_imm_mod / agx3_falu_fma_mul_imm_mod) keep them; other
 * mixed shapes still have their modifiers materialized as sign bits. */
TEST(SignedZeroFMA, KeepsNativeModifierShapes)
{
   for (unsigned mode = 0; mode < 6; mode++) {
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.encoding = AGX_ISA_ENCODING_AGX3B;
      b->shader->key = &key;
      agx_index x = agx_temp(b->shader, AGX_SIZE_32);
      agx_index y = agx_temp(b->shader, AGX_SIZE_32);
      agx_index h = agx_temp(b->shader, AGX_SIZE_16);
      (void)h;
      agx_instr *I = NULL;
      bool keep = true;
      switch (mode) {
      case 0: /* fma(-x, y, 1.0) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_neg(x), y,
                         agx_immediate(48));
         break;
      case 1: /* fma(x, |y|, 0.5) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), x, agx_abs(y),
                         agx_immediate(32));
         break;
      case 2: /* fma(2.0, -x, |y|) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_immediate(64),
                         agx_neg(x), agx_abs(y));
         break;
      case 3: /* fma(-x, 2.0, half) */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_neg(x),
                         agx_immediate(64), h);
         break;
      case 4: /* fma(2.0, -x, 1.0): two minifloats, still materialized */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_immediate(64),
                         agx_neg(x), agx_immediate(48));
         keep = false;
         break;
      case 5: /* fma(|x|, 2.0, 1.0): likewise */
         I = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), agx_abs(x),
                         agx_immediate(64), agx_immediate(48));
         keep = false;
         break;
      }
      agx_lower_agx3_fma_sources(b->shader);
      bool modified = false;
      for (unsigned s = 0; s < 3; ++s)
         modified |= I->src[s].abs || I->src[s].neg;
      EXPECT_EQ(modified, keep) << mode;
      ralloc_free(mem);
   }
}

/* An AGX3C FFMA takes plain half registers as they are, in any operand and
 * for either destination size, and a half value that two FFMAs read stays in
 * its half register. A value only one FFMA reads (whatever else reads it) is
 * still widened, as in a shape with a modified full register (which takes the
 * all-F32 form). AGX3B follows its own unmodified F32 scalar-register rule
 * (keeping half registers for 32-bit destinations and widening 16-bit ones). */
TEST(SignedZeroFMA, AGX3CKeepsHalfRegisterOperands)
{
   struct {
      enum agx_chip chip;
      enum agx_isa_encoding encoding;
      bool emit_agx3d;
      bool keeps;
   } parts[] = {
      {AGX_CHIP_G17C, AGX_ISA_ENCODING_AGX3C, false, true},
      {AGX_CHIP_G17S, AGX_ISA_ENCODING_AGX3C, false, true},
      {AGX_CHIP_G17G, AGX_ISA_ENCODING_AGX3C, false, true},
      {AGX_CHIP_G17P, AGX_ISA_ENCODING_AGX3B, false, false},
   };
   for (auto part : parts) {
      for (unsigned half_mask = 1; half_mask < 8; ++half_mask) {
         for (unsigned dest_size : {32u, 16u}) {
            for (unsigned full_mod = 0; full_mod < 3; ++full_mod) {
               /* 0: one FFMA; 1: two FFMAs; 2: one FFMA and an FADD */
               for (unsigned readers = 0; readers < 3; ++readers) {
                  SCOPED_TRACE(part.chip);
                  SCOPED_TRACE(half_mask);
                  SCOPED_TRACE(dest_size);
                  SCOPED_TRACE(full_mod);
                  SCOPED_TRACE(readers);
                  void *mem = ralloc_context(NULL);
                  agx_builder *b = agx_test_builder(mem);
                  agx_shader_key key = {};
                  key.dev.chip = part.chip;
                  key.dev.encoding = part.encoding;
                  key.dev.emit_agx3d = part.emit_agx3d;
                  b->shader->key = &key;
                  agx_index src[3];
                  bool modified = false;
                  for (unsigned s = 0; s < 3; ++s) {
                     const bool half = half_mask & BITFIELD_BIT(s);
                     src[s] =
                        agx_temp(b->shader, half ? AGX_SIZE_16 : AGX_SIZE_32);
                     /* full_mod 1: neg, 2: abs on the first full register */
                     if (!half && full_mod && !modified) {
                        src[s].neg = full_mod == 1;
                        src[s].abs = full_mod == 2;
                        modified = true;
                     }
                  }
                  const enum agx_size dsize =
                     dest_size == 16 ? AGX_SIZE_16 : AGX_SIZE_32;
                  agx_instr *I = agx_ffma_to(b, agx_temp(b->shader, dsize),
                                             src[0], src[1], src[2]);
                  agx_instr *J = NULL;
                  if (readers == 1) {
                     J = agx_ffma_to(b, agx_temp(b->shader, dsize), src[0],
                                     src[1], src[2]);
                  } else if (readers == 2) {
                     for (unsigned s = 0; s < 3; ++s) {
                        if (half_mask & BITFIELD_BIT(s)) {
                           agx_index plain = src[s];
                           plain.abs = plain.neg = false;
                           agx_fadd_to(b, agx_temp(b->shader, AGX_SIZE_32),
                                       plain, agx_negzero());
                        }
                     }
                  }
                  unsigned before = 0;
                  agx_foreach_instr_global(b->shader, K)
                     before++;
                  agx_lower_agx3_fma_sources(b->shader);

                  const bool keep = part.encoding == AGX_ISA_ENCODING_AGX3B
                                       ? (dest_size == 32 && !modified)
                                       : (part.keeps && !modified && readers == 1);
                  unsigned count = 0;
                  agx_foreach_instr_global(b->shader, K)
                     count++;
                  if (keep)
                     EXPECT_EQ(count, before);
                  for (agx_instr *F : {I, J}) {
                     if (!F)
                        continue;
                     ASSERT_EQ(F->op, AGX_OPCODE_FFMA);
                     for (unsigned s = 0; s < 3; ++s) {
                        const bool half = half_mask & BITFIELD_BIT(s);
                        EXPECT_EQ(F->src[s].size,
                                  half && keep ? AGX_SIZE_16 : AGX_SIZE_32)
                           << s;
                     }
                  }
                  ralloc_free(mem);
               }
            }
         }
      }
   }
}

/* An FFMA that takes one half value in several slots reads it once: alone it
 * is widened, while two such FFMAs keep it in its half register. */
TEST(SignedZeroFMA, AGX3CCountsEachFFMAOnce)
{
   for (unsigned slots : {3u, 5u, 6u, 7u}) {
      for (unsigned nr_ffma : {1u, 2u}) {
         SCOPED_TRACE(slots);
         SCOPED_TRACE(nr_ffma);
         void *mem = ralloc_context(NULL);
         agx_builder *b = agx_test_builder(mem);
         agx_shader_key key = {};
         key.dev.chip = AGX_CHIP_G17C;
         key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
         b->shader->key = &key;
         agx_index h = agx_temp(b->shader, AGX_SIZE_16);
         agx_instr *F[2] = {NULL, NULL};
         for (unsigned f = 0; f < nr_ffma; ++f) {
            agx_index src[3];
            for (unsigned s = 0; s < 3; ++s) {
               src[s] = (slots & BITFIELD_BIT(s))
                           ? h
                           : agx_temp(b->shader, AGX_SIZE_32);
            }
            F[f] = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), src[0],
                               src[1], src[2]);
         }
         agx_lower_agx3_fma_sources(b->shader);
         for (unsigned f = 0; f < nr_ffma; ++f) {
            for (unsigned s = 0; s < 3; ++s) {
               if (slots & BITFIELD_BIT(s)) {
                  EXPECT_EQ(F[f]->src[s].size,
                            nr_ffma == 2 ? AGX_SIZE_16 : AGX_SIZE_32)
                     << f << "/" << s;
               }
            }
         }
         ralloc_free(mem);
      }
   }
}

/* Legalization runs twice. A reader whose full register operand is modified
 * widens the half value, so on the second run it no longer counts: with one
 * other reader, that reader is widened too; with two, they keep the half
 * register beside the modified reader's widened copy. */
TEST(SignedZeroFMA, AGX3CHalfOperandsAfterTwoLegalizations)
{
   for (unsigned plain_readers : {1u, 2u}) {
      SCOPED_TRACE(plain_readers);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17C;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
      agx_index h = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *plain[2] = {NULL, NULL};
      for (unsigned f = 0; f < plain_readers; ++f) {
         plain[f] = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), h,
                                agx_temp(b->shader, AGX_SIZE_32),
                                agx_temp(b->shader, AGX_SIZE_32));
      }
      agx_index x = agx_temp(b->shader, AGX_SIZE_32);
      agx_instr *modified =
         agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), h, agx_neg(x),
                     agx_temp(b->shader, AGX_SIZE_32));

      agx_lower_agx3_fma_sources(b->shader);
      agx_lower_agx3_fma_sources(b->shader);

      EXPECT_EQ(modified->src[0].size, AGX_SIZE_32);
      EXPECT_TRUE(modified->src[1].neg);
      for (unsigned f = 0; f < plain_readers; ++f) {
         EXPECT_EQ(plain[f]->src[0].size,
                   plain_readers == 2 ? AGX_SIZE_16 : AGX_SIZE_32)
            << f;
         if (plain_readers == 2)
            EXPECT_EQ(plain[f]->src[0].value, h.value) << f;
      }
      ralloc_free(mem);
   }
}

/* G18G compiles as AGX3C and re-encodes as agx3d; it keeps its own half
 * widening: a 32-bit FFMA's plain half operands are widened even when two
 * FFMAs read them, and a 16-bit FFMA is left as it is. */
TEST(SignedZeroFMA, G18GKeepsItsHalfWidening)
{
   for (unsigned dest_size : {32u, 16u}) {
      SCOPED_TRACE(dest_size);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17G;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      key.dev.emit_agx3d = true;
      b->shader->key = &key;
      const enum agx_size dsize = dest_size == 16 ? AGX_SIZE_16 : AGX_SIZE_32;
      agx_index h = agx_temp(b->shader, AGX_SIZE_16);
      agx_instr *F[2];
      for (unsigned f = 0; f < 2; ++f) {
         F[f] = agx_ffma_to(b, agx_temp(b->shader, dsize), h,
                            agx_temp(b->shader, AGX_SIZE_32),
                            agx_temp(b->shader, AGX_SIZE_32));
      }
      agx_lower_agx3_fma_sources(b->shader);
      for (unsigned f = 0; f < 2; ++f) {
         EXPECT_EQ(F[f]->src[0].size,
                   dest_size == 32 ? AGX_SIZE_32 : AGX_SIZE_16)
            << f;
      }
      ralloc_free(mem);
   }
}

/* A modifier on a 32-bit uniform operand takes the all-F32 form like one on
 * a full register: the half values are widened, not kept, so the modifier
 * stays on the FFMA instead of becoming a move and a bitop. */
TEST(SignedZeroFMA, AGX3CModifiedUniformWidensHalves)
{
   for (unsigned mod = 1; mod < 3; ++mod) {
      SCOPED_TRACE(mod);
      void *mem = ralloc_context(NULL);
      agx_builder *b = agx_test_builder(mem);
      agx_shader_key key = {};
      key.dev.chip = AGX_CHIP_G17C;
      key.dev.encoding = AGX_ISA_ENCODING_AGX3C;
      b->shader->key = &key;
      agx_index h = agx_temp(b->shader, AGX_SIZE_16);
      agx_index u = agx_uniform(8, AGX_SIZE_32);
      u.neg = mod == 1;
      u.abs = mod == 2;
      agx_instr *F[2];
      for (unsigned f = 0; f < 2; ++f) {
         F[f] = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), h,
                            agx_temp(b->shader, AGX_SIZE_32), u);
      }
      agx_instr *G = agx_ffma_to(b, agx_temp(b->shader, AGX_SIZE_32), h,
                                 agx_temp(b->shader, AGX_SIZE_32),
                                 agx_temp(b->shader, AGX_SIZE_32));
      agx_lower_agx3_fma_sources(b->shader);
      unsigned bitops = 0;
      agx_foreach_instr_global(b->shader, K)
         bitops += K->op == AGX_OPCODE_BITOP;
      EXPECT_EQ(bitops, 0u);
      for (unsigned f = 0; f < 2; ++f) {
         EXPECT_EQ(F[f]->src[0].size, AGX_SIZE_32) << f;
         EXPECT_EQ(F[f]->src[2].type, AGX_INDEX_UNIFORM) << f;
         EXPECT_TRUE(F[f]->src[2].neg || F[f]->src[2].abs) << f;
      }
      /* The plain reader keeps the half: three FFMAs read it */
      EXPECT_EQ(G->src[0].size, AGX_SIZE_16);
      ralloc_free(mem);
   }
}
