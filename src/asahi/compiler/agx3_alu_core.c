/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include <stdlib.h>
#include "agx3_alu_core.h"
#include "agx3_binder.h"

#include <string.h>

/* Static G17G/G18G helpers, defined beside their encoders below. */
static bool
bit_reg_gpu1(enum agx3_mnemonic mnemonic, uint8_t ep, unsigned table, bool word,
             uint16_t dest, bool dest_cache, uint16_t src0,
             enum agx3_alu_hint hint0, uint16_t src1,
             enum agx3_alu_hint hint1, uint16_t wait_mask,
             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
static bool
agx3_alu_asr_imm32_source(enum agx_isa_encoding enc, uint16_t dest,
                   uint16_t source, bool source_half,
                   unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *length);
static bool
agx3_alu_fcmp_select_gpu1(uint16_t dest, const uint16_t src[4],
                        bool compare_zero, const bool compare_abs[2],
                        const bool immediate[2],
                        const enum agx3_alu_hint hints[4], unsigned condition,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
static bool
agx3_alu_m5_fsel_half(uint16_t dest, const uint16_t src[4],
                         const enum agx3_alu_hint hints[4], unsigned condition,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
static bool
agx3_alu_m5_fminmax_half(uint16_t dest, const uint16_t src[4],
                        const enum agx3_alu_hint hints[4], bool maximum,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);
static bool
agx3_alu_fminmax32_gpu1(uint16_t dest, uint16_t src0, uint16_t src2,
                       bool immediate, uint16_t src1, uint16_t src3,
                       const enum agx3_alu_hint hints[4], bool maximum,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len);

bool
agx3_alu_bitrev_m5(uint16_t dest, uint16_t src, bool src_half,
                   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 || (dest & 1) ||
       (!src_half && (src & 1)) || hint > AGX3_ALU_HINT_CACHE ||
       wait_mask > 255)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, src_half ? 1 : 2}, {29, 3}, {30, 143},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest},
      {11, dest_cache},
      {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){31 + i, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_rev, 0, base, 6, fields,
                            n, out, len);
}

bool
agx3_alu_popcount(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  bool half, enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len || dest >= 256 ||
       src >= 256 || (!half && ((dest | src) & 1)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;

   /* applegpu_g17p `popcount` is the eight-byte cbs point. f10/f24 are the
    * result/source widths, f11 destination cache, f25/f27 source
    * cache/discard, and f31..f36 the six scoreboard slots:
    *   2705560202045004  h2 <- popcount(^h2), wait slot 6
    *   2705540202045004  h2 <- popcount(^h2)
    *   2725540402025004  h4 <- popcount(^h1), wait slot 2
    *   2705560002024204  h0 <- popcount($h1), wait slot 6
    *   2705560302044004  h3 <- popcount(h2), h2 stored later, slot 6
    *   2705560402004c04  r2 <- popcount(r0), r0 stored later, slot 6
    *   2715540202045c04  r1 <- popcount(^r1), wait slot 1
    *   2705540403004e04  $r2 <- popcount($r0)
    * and, with enough 64-bit values live that the source lands high:
    *   2705548003005d04  $r64 <- popcount(^r64)
    *   2705549803305d04  $r76 <- popcount(^r76)
    *   2705549003005c04  $r72 <- popcount(^r0)
    * native G17G (port/cwit3 k_popc_u16, k_popc_u16w) counts a ushort the
    * same way with half widths and C's form constants f29 = 3, f30 = 23; the
    * uint result of the second clears the high half after it:
    *   3f0200003220aa020400  h2 <- popcount(^h2), wait slot 8 (f38) */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, half ? 1 : 2}, {23, 2}, {24, half ? 1 : 2}, {29,3}, {30,23},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot) {
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){31 + slot, 1};
   }
   return agx3_bind_compose(enc, AGX3_MN_cbs, 0, base, m5 ? 6 : 4, fields, n, out, len);
}

/* G18G (M6), which compiles as agx3c: the G18G-witnessed cbs form. A live
 * source is always cached and only the slot-1 and slot-8 waits are
 * admitted, so the retire carrier takes the others. agx3_alu_popcount
 * keeps the G17G-family (agx3c) form for every other GPU. */
bool
agx3_alu_popcount_g18g(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       bool half, enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      /* G17G uint4/ushort4 popcount witnesses use cbs ep0. The separate
       * live uint4 probe also witnesses cached sources and destinations.
       * The wait bank starts at f31 as on G17P: a count of the slot-8 load
       * waits in f38, and one of the slot-1 load in f31
       * (test/native/g17g/m5-libagx-reference.metal popcnt_two_loads). Only
       * those two slots are admitted; the caller retires other
       * dependencies separately.
       *
       * A source that stays live is always cached by Apple, at either width
       * and however far away its next read is: popcount(a) + a,
       * popcount(a) with a * 3u stored next, and popcount(a) with a read
       * only after six dependent fmas, for uint and ushort alike, all set
       * f25 (applegpu_g18g/g17g, g18ref-out/probe/m6-alu-misc popc_live,
       * popc_live2 and m6-alu-sel2 popc_far, popc16_far). No cbs without a
       * lifetime hint was ever emitted, so a live source takes Apple's
       * cached form rather than an unwitnessed hint-free one; the cache bit
       * only asks for the value to be kept close and changes nothing that
       * is read. */
      if (hint == AGX3_ALU_HINT_NONE)
         hint = AGX3_ALU_HINT_CACHE;
      if (!out || !len || dest >= 256 || src >= 256 ||
          (!half && ((dest | src) & 1)) || (wait_mask & ~129u) ||
          (hint != AGX3_ALU_HINT_DISCARD && hint != AGX3_ALU_HINT_CACHE))
         return false;

      const struct agx3_field_kv fields[] = {
         {9, 2}, {10, half ? 1 : 2}, {2, dest}, {11, dest_cache},
         {23, 2}, {24, half ? 1 : 2}, {16, src},
         {25, hint == AGX3_ALU_HINT_CACHE},
         {27, hint == AGX3_ALU_HINT_DISCARD},
         {29, 3}, {30, 23}, {31, wait_mask & 1}, {38, !!(wait_mask & 128)},
      };
      return agx3_bind_compose(enc, AGX3_MN_cbs, 0, fields,
                                 sizeof(fields) / sizeof(fields[0]),
                                 NULL, 0, out, len);
   }

   /* Exact Apple G17P witnesses:
    *   ushort: 27 05 56 02 02 04 50 04
    *   uint:   27 05 56 00 02 00 5c 04
    * byte3 is the destination half, byte5 is twice the source half, and the
    * source descriptor distinguishes 16 from 32 bits. A fresh subgroup-count
    * witness adds the live uint source form (descriptor 0x4e). Refuse every
    * other unwitnessed cache/wait/lifetime combination. */
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (!half && ((dest | src) & 1)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;

   /* applegpu_g17p `popcount` is the eight-byte cbs point. f10/f24 are the
    * result/source widths, f11 destination cache, f25/f27 source
    * cache/discard, and f31..f36 the six scoreboard slots:
    *   2705560202045004  h2 <- popcount(^h2), wait slot 6
    *   2705540202045004  h2 <- popcount(^h2)
    *   2725540402025004  h4 <- popcount(^h1), wait slot 2
    *   2705560002024204  h0 <- popcount($h1), wait slot 6
    *   2705560302044004  h3 <- popcount(h2), h2 stored later, slot 6
    *   2705560402004c04  r2 <- popcount(r0), r0 stored later, slot 6
    *   2715540202045c04  r1 <- popcount(^r1), wait slot 1
    *   2705540403004e04  $r2 <- popcount($r0)
    * and, with enough 64-bit values live that the source lands high:
    *   2705548003005d04  $r64 <- popcount(^r64)
    *   2705549803305d04  $r76 <- popcount(^r76)
    *   2705549003005c04  $r72 <- popcount(^r0) */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, half ? 1 : 2}, {23, 2}, {24, half ? 1 : 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 6; ++slot) {
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){31 + slot, 1};
   }
   return agx3_bind_compose(enc, AGX3_MN_cbs, 0, base, 4, fields, n, out, len);
}

bool
agx3_alu_ft1_u32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache,
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                 uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      if (!out || !len || dest >= 256 || src >= 256 || ((dest | src) & 1) ||
          (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0xffu))
         return false;
      const struct agx3_field_kv base[] = {
         {9,2},{10,2},{23,2},{24,2},{29,3},{30,79},
      };
      struct agx3_field_kv f[13] = {
         {2,dest},{11,dest_cache},{16,src},
         {25,hint == AGX3_ALU_HINT_CACHE},{27,hint == AGX3_ALU_HINT_DISCARD},
      };
      for (unsigned i=0;i<8;i++)
         f[5+i]=(struct agx3_field_kv){31+i,!!(wait_mask & (1u<<i))};
      return agx3_bind_compose(enc,AGX3_MN_ft1,0,base,6,f,13,out,len);
   }

   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Direct llvm.agx2.ft1 witness:
    *   a705560202004c04  ft1 r1, wait0
    * A live ALU chain independently grounds destination/source caching:
    *   a705540003005c04  ft1 $r0, $r0
    * Apple's clz lowering grounds independent destination/source plus wait0:
    *   a705560603044e04  ft1 $r3, ^r2, wait0 */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, 2},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest},
      {11, dest_cache},
      {25, hint == AGX3_ALU_HINT_DISCARD},
      {27, hint == AGX3_ALU_HINT_CACHE},
   };
   unsigned n = 4;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){36 + slot,
                                          !!(wait_mask & (1u << slot))};

   if (!agx3_bind_compose(enc, AGX3_MN_ft1, 0, base, 4, fields, n, out, len))
      return false;

   /* The recovered composer form fixes the independent source byte to zero,
    * but fresh Apple clz kernels vary it directly with the source half:
    * a705560603004e04 (src half0) -> a705560603044e04 (src half4).
    * The full byte is the source field and every admitted register is even. */
   out[5] = (uint8_t)src;
   return true;
}

/* f10, f24 and f38 are the result's and the two operands' widths: 1 for a
 * half, 2 for an aligned 32-bit pair. */
static bool
bit_reg_gpu1(enum agx3_mnemonic mnemonic, uint8_t ep, unsigned table, bool word,
             uint16_t dest, bool dest_cache, uint16_t src0,
             enum agx3_alu_hint hint0, uint16_t src1,
             enum agx3_alu_hint hint1, uint16_t wait_mask,
             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src0 >= 256 || src1 >= 256 ||
       (word && ((dest | src0 | src1) & 1)) ||
       (unsigned)hint0 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint1 > AGX3_ALU_HINT_CACHE || wait_mask >= (1u << 8))
      return false;
   const unsigned width = word ? 2 : 1;
   struct agx3_field_kv fields[24];
   unsigned n = 0;
   fields[n++] = (struct agx3_field_kv){9, 2};
   fields[n++] = (struct agx3_field_kv){10, width};
   fields[n++] = (struct agx3_field_kv){2, dest};
   fields[n++] = (struct agx3_field_kv){11, dest_cache};
   fields[n++] = (struct agx3_field_kv){23, 2};
   fields[n++] = (struct agx3_field_kv){24, width};
   fields[n++] = (struct agx3_field_kv){16, src0};
   fields[n++] = (struct agx3_field_kv){25, hint0 == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){27, hint0 == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){30, src1};
   fields[n++] = (struct agx3_field_kv){37, 2};
   fields[n++] = (struct agx3_field_kv){38, width};
   fields[n++] = (struct agx3_field_kv){39, hint1 == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){41, hint1 == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){42, 3};
   fields[n++] = (struct agx3_field_kv){43, table};
   for (unsigned i = 0; i < 8; ++i)
      fields[n++] = (struct agx3_field_kv){44 + i, (wait_mask >> i) & 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, mnemonic, ep, fields, n,
                            NULL, 0, out, len);
}

bool
agx3_alu_bit_andc16_gpu1(uint16_t dest, bool dest_cache, uint16_t src0,
                         enum agx3_alu_hint hint0, uint16_t src1,
                         enum agx3_alu_hint hint1, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Apple applegpu_g17g ctz: 33 03 06 01 3a 20 a2 02 84 00
    *   bit_andc h3 (cached), h3 (dead), h2 (cached) */
   return bit_reg_gpu1(AGX3_MN_bit_andc, 0, 11, false, dest, dest_cache, src0,
                       hint0, src1, hint1, wait_mask, out, len);
}

bool
agx3_alu_bit_andc32_gpu1(uint16_t dest, bool dest_cache, uint16_t src0,
                         enum agx3_alu_hint hint0, uint16_t src1,
                         enum agx3_alu_hint hint1, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return bit_reg_gpu1(AGX3_MN_bit_andc, 0, 11, true, dest, dest_cache, src0,
                       hint0, src1, hint1, wait_mask, out, len);
}

bool
agx3_alu_bit_named_gpu1(unsigned truth_table, bool word, uint16_t dest,
                        bool dest_cache, uint16_t src0,
                        enum agx3_alu_hint hint0, uint16_t src1,
                        enum agx3_alu_hint hint1, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   enum agx3_mnemonic mnemonic;
   unsigned table;
   switch (truth_table) {
   case 0x1u: mnemonic = AGX3_MN_bit_nor; table = 14; break;
   case 0x2u: mnemonic = AGX3_MN_bit_andc; table = 11; break;
   case 0x7u: mnemonic = AGX3_MN_bit_nand; table = 12; break;
   case 0x9u: mnemonic = AGX3_MN_bit_xnor; table = 18; break;
   case 0xbu: mnemonic = AGX3_MN_bit_orc; table = 17; break;
   default: return false;
   }
   for (uint8_t ep = 0; ep < 2; ++ep)
      if (bit_reg_gpu1(mnemonic, ep, table, word, dest, dest_cache, src0,
                       hint0, src1, hint1, wait_mask, out, len))
         return true;
   return false;
}

bool
agx3_alu_bit_reg16_gpu1(enum agx3_alu_op op, uint16_t dest, bool dest_cache,
                        uint16_t src0, enum agx3_alu_hint hint0,
                        uint16_t src1, enum agx3_alu_hint hint1,
                        uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                        uint8_t *len)
{
   switch (op) {
   case AGX3_ALU_BAND:
      return bit_reg_gpu1(AGX3_MN_bit_and, 1, 10, false, dest, dest_cache, src0,
                          hint0, src1, hint1, wait_mask, out, len);
   case AGX3_ALU_BOR:
      return bit_reg_gpu1(AGX3_MN_bit_or, 1, 16, false, dest, dest_cache, src0,
                          hint0, src1, hint1, wait_mask, out, len);
   case AGX3_ALU_BXOR:
      return bit_reg_gpu1(AGX3_MN_bit_xor, 1, 19, false, dest, dest_cache, src0,
                          hint0, src1, hint1, wait_mask, out, len);
   default:
      return false;
   }
}

bool
agx3_alu_ft1_gpu1(uint16_t dest, uint16_t src, bool half,
                  enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   /* Apple applegpu_g17g clz kernels:
    *   uint    2f 80 00 1a 2a 00 ab 0a 08 00   ft1 r2, r0 (cached), wait8
    *   ushort  3f 83 00 18 2a 00 ab 02 00 00   ft1 h2, h3 (cached), wait8
    * The same shape as G17G cbs with opcode f30=79 instead of 23. */
   if (!out || !len || dest >= 256 || src >= 256 ||
       (!half && ((dest | src) & 1)) || (wait_mask & ~128u) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv fields[] = {
      {9, 2}, {10, half ? 1 : 2}, {2, dest}, {11, dest_cache},
      {23, 2}, {24, half ? 1 : 2}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
      {29, 3}, {30, 79}, {38, !!wait_mask},
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_ft1, 0, fields,
                            sizeof(fields) / sizeof(fields[0]), NULL, 0, out,
                            len);
}

bool
agx3_alu_rev_gpu1(uint16_t dest, uint16_t src, enum agx3_alu_hint hint,
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Apple applegpu_g17g reverse_bits(uint)
    * (test/native/g17g/m5-gl4-reference.metal rbit32_*):
    *   27 00 00 1a 26 00 a9 02 08 00   rev r0, r0 (dead)
    *   37 80 00 1a 2a 00 a9 02 08 00   rev r1 (cached), r0 (cached)
    *   2f 00 00 1a 26 00 a9 02 08 00   rev r0, r0 (dead), wait8
    * cbs's and ft1's shape with opcode f30=143. Only the slot-8 wait is
    * witnessed. */
   if (!out || !len || dest >= 256 || src >= 256 || ((dest | src) & 1) ||
       (wait_mask & ~128u) || (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv fields[] = {
      {9, 2}, {10, 2}, {2, dest}, {11, dest_cache},
      {23, 2}, {24, 2}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
      {29, 3}, {30, 143}, {38, !!wait_mask},
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_rev, 0, fields,
                            sizeof(fields) / sizeof(fields[0]), NULL, 0, out,
                            len);
}

bool
agx3_alu_fcmp_bool16_gpu1(uint16_t dest, uint16_t src0,
                          enum agx3_alu_hint src0_hint, bool src1_is_register,
                          uint16_t src1, bool src1_kill, uint8_t if_true,
                          uint8_t if_false, unsigned condition,
                          uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len)
{
   /* Apple applegpu_g17g, `out[id] = a[id] OP K ? T : F` into a ushort:
    *   a < -1.0 ? 1 : 0   32 02 05 0a 23 00 a0 02 14 ac 4e 58 01 00 80 00
    *   a >= 2.0 ? 1 : 0   32 02 05 0a 23 00 a0 02 14 90 4e 1c 01 00 80 00
    *   a <  1.0 ? 3 : 7   32 02 05 0a 23 00 a0 02 14 8c 4e 58 27 02 80 00
    *   a <  1.0 ? 0 : 1   32 02 05 0a 23 00 a0 02 14 8c 4a 58 05 00 80 00
    *   a <  b   ? 1 : 0   32 02 05 0a 23 00 a0 02 14 01 65 58 01 00 80 00
    *   a == 1.0 ? 1 : 0   32 02 05 0a 23 00 a0 02 14 8c 4e 18 01 00 80 00
    *   a != 1.0 ? 1 : 0   32 02 05 0a 23 00 a0 02 14 8c 4a 18 05 00 80 00
    *   a >  1.0 ? 1 : 0   32 02 45 0a 23 00 a0 02 14 8c 4e 18 01 00 80 00
    *   a <= 1.0 ? 1 : 0   32 02 45 0a 23 00 a0 02 14 8c 4e 1c 01 00 80 00
    * f32 is the minifloat, f33 the register, f49/f63 the two selected
    * constants and f86 the condition in enum agx_fcond numbering (eq 0,
    * lt 1, gt 2, ge 5, le 6; `!=` is eq with the constants exchanged, as
    * agx_lower_pseudo spells an inverted compare); f85 is the slot-8 wait.
    * The only 32-bit boolean Apple emits is this half followed by a clear
    * of the upper half. */
   if (!out || !len || dest >= 256 || src0 >= 256 || (src0 & 1) ||
       (unsigned)src0_hint > AGX3_ALU_HINT_CACHE || if_true > 7 ||
       if_false > 7 ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask & ~128u))
      return false;
   if (src1_is_register ? ((src1 & 1) || src1 >= 256)
                        : (src1 > 255 || src1_kill))
      return false;

   struct agx3_field_kv fields[32];
   unsigned n = 0;
   fields[n++] = (struct agx3_field_kv){2, dest};
   fields[n++] = (struct agx3_field_kv){9, 2};
   fields[n++] = (struct agx3_field_kv){10, 1};
   fields[n++] = (struct agx3_field_kv){16, src0};
   fields[n++] = (struct agx3_field_kv){23, 2};
   fields[n++] = (struct agx3_field_kv){24, 2};
   fields[n++] = (struct agx3_field_kv){25, src0_hint == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){27, src0_hint == AGX3_ALU_HINT_DISCARD};
   if (src1_is_register) {
      fields[n++] = (struct agx3_field_kv){33, src1};
      fields[n++] = (struct agx3_field_kv){40, 2};
      fields[n++] = (struct agx3_field_kv){41, 2};
      fields[n++] = (struct agx3_field_kv){44, src1_kill};
   } else {
      fields[n++] = (struct agx3_field_kv){32, src1};
      fields[n++] = (struct agx3_field_kv){40, 1};
      fields[n++] = (struct agx3_field_kv){41, 2};
   }
   fields[n++] = (struct agx3_field_kv){49, if_true};
   fields[n++] = (struct agx3_field_kv){57, 1};
   fields[n++] = (struct agx3_field_kv){58, 1};
   fields[n++] = (struct agx3_field_kv){63, if_false};
   fields[n++] = (struct agx3_field_kv){71, 1};
   fields[n++] = (struct agx3_field_kv){72, 1};
   fields[n++] = (struct agx3_field_kv){76, 3};
   fields[n++] = (struct agx3_field_kv){77, 75};
   fields[n++] = (struct agx3_field_kv){85, !!wait_mask};
   fields[n++] = (struct agx3_field_kv){86, condition};
   for (unsigned f = 90; f <= 94; ++f)
      fields[n++] = (struct agx3_field_kv){f, 0};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fseli16, 8, fields, n,
                            NULL, 0, out, len);
}

bool
agx3_alu_sqrt_step_source_modifiers(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, bool src_abs, bool src_neg,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || (m5 && wait_mask) || !out || !len || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~UINT16_C(0x3f)))
      return false;
   /* Apple sqrt emits this step followed by FMUL with the original input.
    * Ordinary rsqrt is a distinct opcode and cannot replace this operation.
    *
    * f30 folds the source's absolute value in, so `sqrt(abs(x))` needs no
    * separate move. Witnessed as a single-field difference against the plain
    * control; see agx3_alu_core.h for the byte pair. */
   const struct agx3_field_kv base[] = {{9,2},{10,2},{25,2},{26,2}};
   const struct agx3_field_kv base_c[] = {
      {9,2},{10,2},{25,2},{26,2},{33,3},{34,68},
   };
   struct agx3_field_kv fields[13] = {
      {2,dest},{18,src},{11,dest_cache},
      {27,hint == AGX3_ALU_HINT_CACHE},{29,hint == AGX3_ALU_HINT_DISCARD},
      {30,src_abs}, {31,src_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[7+i] = (struct agx3_field_kv){35+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,AGX3_MN_f32sqrtstep,0,m5 ? base_c : base,m5 ? 6 : 4,fields,13,out,len);
}

bool
agx3_alu_sqrt_step_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Apple emits the half-input square-root step with its float32 temporary
    * cached and its half retained for the following multiply. A six-load
    * kernel witnesses the complete wait bank f35..f40 on this width pair. */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || hint != AGX3_ALU_HINT_CACHE ||
       (!m5 && !dest_cache) || (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;

   const struct agx3_field_kv base[] = {{9,2},{10,2},{25,2},{26,1}};
   struct agx3_field_kv fields[14] = {
      {2, dest},
      {18, src},
      {11, dest_cache},
      {27, 1},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[4 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   if (m5) {
      fields[10] = (struct agx3_field_kv){33, 3};
      fields[11] = (struct agx3_field_kv){34, 68};
      fields[12] = (struct agx3_field_kv){41, !!(wait_mask & 64)};
      fields[13] = (struct agx3_field_kv){42, !!(wait_mask & 128)};
   }
   return agx3_bind_compose(enc, AGX3_MN_f32sqrtstep, 0, base, 4, fields,
                            m5 ? 14 : 10, out, len);
}

bool
agx3_alu_fmul_minifloat_mod(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool src_abs,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_fmul_minifloat_narrow(enc, dest, src, immediate,
                                         dest_cache, src_abs, false, false, hint,
                                         wait_mask, out, len);
}

bool
agx3_alu_fmul_minifloat_narrow(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool src_abs, bool dest_half,
   bool saturate,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) ||
        dest >= 256 ||
       src >= 256 || (src & 1) || (!dest_half && (dest & 1)) ||
       immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(gpu1 ? UINT16_C(0xff) : UINT16_C(0x3f))) ||
       ((src_abs || saturate) && !gpu1))
      return false;

   /* f10 is the DESTINATION width and f26 the source's; they are independent,
    * exactly as on f32rcp. A 32-bit source multiplied by a minifloat straight
    * into a half destination is therefore one instruction, not a multiply
    * plus a narrowing move -- which is what NIR hands the packer for
    * `packHalf2x16(vec2(x * k, ...))`.
    *
    * Witnessed with `metal-nt -arch applegpu_g17p`, a single-field difference
    * from the all-32-bit control, and re-composed byte-exactly through this
    * encoder. G17G also independently witnesses narrowing to even and odd
    * half destinations, including the slot-eight dependency form:
    *   half4(input * 0.5f): 290005182280ac020a0020000210,
    *                       01a115038400, 11a115058000, 11a115078400.
    * G17P controls:
    *
    *   o[i] = a[i] * 0.5f        (float o)  19 a1 15 03 80 c0   f10 = 2
    *   o[i] = half(a[i] * 0.5f)  (half  o)  11 a1 15 03 80 c0   f10 = 1
    *   o[i] = half(a[i] * 2.0f)             11 c1 15 03 80 c0   f17 = 64
    *
    * The recovered six-byte endpoint fixes bits 32:33. There is no grounded
    * fused ABS form on G17P: hardware testing showed that mutating bit 33
    * executes incorrectly. G17G instead carries ABS in field47 of the longer
    * form, retaining its source lifetime and dependency wait. */

   /* Fresh Apple mul_scale: 0983150180c0 is loaded f32 * minifloat0x02
    * (1/32), wait6. mul_reuse: 0983050380c0 retains its half2 source.
    * The ordinary multiply's additional fixed fields60/74 are both2;
    * it shares add's operand routing, but its wait fields are80..85.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, dest_half ? 1 : 2}, {25, 1}, {26, 2},
      {42, 2}, {43, 2}, {60, 2}, {74, 2},
      {78, 3}, {79, 62},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {35, src}, {17, immediate}, {46, hint == AGX3_ALU_HINT_DISCARD},
      {44, hint == AGX3_ALU_HINT_CACHE}, {11, dest_cache},
   };
   unsigned n = 6;
   if (gpu1)
      fields[n++] = (struct agx3_field_kv){47, src_abs};
   /* G17G saturate(x * 1/16) differs from the plain scale only in f14;
    * ABS uses f47 and narrowing independently selects f10=1. */
   if (saturate)
      fields[n++] = (struct agx3_field_kv){14, 1};
   for (unsigned slot = 0; slot < (gpu1 ? 8 : 6); ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){80 + slot, 1};

   uint8_t encoded[AGX3_ALU_MAX_LENGTH], encoded_len = 0;
   bool composed = false;
   /* Native G17G scale kernels use ep1 for ready inputs and ep3 when
    * waiting on slot8. Preserve the G17P ep1-only admission. */
   for (unsigned ep = 1; ep <= (gpu1 ? 3 : 1); ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32mul, ep, base, gpu1 ? 10 : 8,
                             fields, n, encoded, &encoded_len)) {
         composed = true;
         break;
      }
   }
   if (!composed)
      return false;

   memcpy(out, encoded, encoded_len);
   *len = encoded_len;
   return true;
}

bool
agx3_alu_fmul_half_minifloat_to_f32(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || (dest & 1) || immediate > UINT8_MAX ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       wait_mask)
      return false;

   /* Apple u32_to_half_math multiplies the scalar half by minifloat 16
    * directly into float32: f43=1 selects the half source, while f25=1 and
    * fields17/60/74 retain the established minifloat operand routing. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {11, 1}, {25, 1}, {26, 2},
      {42, 2}, {43, 1}, {46, 1}, {60, 2}, {74, 2},
   };
   const struct agx3_field_kv fields[] = {
      {2, dest}, {17, immediate}, {35, src},
   };
   return agx3_bind_compose(enc, AGX3_MN_f32mul, 1, base, 10, fields, 3, out,
                            len);
}

bool
agx3_alu_fmul_half_minifloat_m5(
   uint16_t dest, uint16_t src, uint16_t immediate, bool dest_cache,
   bool src_abs, bool saturate, enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 || (dest & 1) ||
       immediate > UINT8_MAX || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0xff)))
      return false;

   /* G17G float(h) * k (port/hmulwit): the float-source minifloat multiply
    * with f43=1 for the half register in f35, e.g.
    *   k_hmul_half  float(h) * 0.5f      290005102280ac02028020000210
    *   k_hmul_sat   saturate(...)        ...028020020210        (f14)
    *   k_hmul_abs   float(fabs(h)) * .5  ...028020080210        (f47)
    *   k_hmul_reuse3 cached source, no wait: 19c105868000      (f44, ep1)
    * Slot-eight dependencies select ep3 with f87, as for the float form. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 1}, {60, 2}, {74, 2},
      {78, 3}, {79, 62},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {35, src}, {17, immediate},
      {46, hint == AGX3_ALU_HINT_DISCARD}, {44, hint == AGX3_ALU_HINT_CACHE},
      {11, dest_cache}, {47, src_abs}, {14, saturate},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 8; ++slot)
      fields[n++] = (struct agx3_field_kv){80 + slot, !!(wait_mask & (1u << slot))};

   for (unsigned ep = 1; ep <= 3; ++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_f32mul, ep, base, 10,
                            fields, n, out, len))
         return true;
   return false;
}

bool
agx3_alu_fadd_minifloat(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       uint16_t immediate, bool dest_cache,
                       enum agx3_alu_hint hint, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple add_half: 09a1140180c0 (f32 register + minifloat0x20, wait6).
    * add_reuse: 09a1040380c0 keeps its half2 source for a later store.
    * pair: 09a12c8580c0 independently retains its result and loaded source.
    * Fragment position uses the same add without a wait. f25=1 selects the
    * minifloat; f42=2 selects the GPR. The register is field35, not field18.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest}, {35, src}, {17, immediate}, {46, hint == AGX3_ALU_HINT_DISCARD},
      {44, hint == AGX3_ALU_HINT_CACHE}, {11, dest_cache},
   };
   unsigned n = 6;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){52 + slot, 1};

   /* This encoding point rejects unsupported wait combinations. Do not
    * silently clear waits or switch to an unwitnessed long-form variant.
    */
   return agx3_bind_compose(enc, AGX3_MN_f32add, 1, base, 6, fields, n, out, len);
}

/* G18G (M6), which compiles as agx3c: also the compact agx3c
 * register+minifloat add (f50=3/f51=48). agx3_alu_fadd_minifloat keeps
 * the G17G-family (agx3c) admission for every other GPU. */
bool
agx3_alu_fadd_minifloat_g18g(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                            uint16_t immediate, bool dest_cache,
                            enum agx3_alu_hint hint, uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple add_half: 09a1140180c0 (f32 register + minifloat0x20, wait6).
    * add_reuse: 09a1040380c0 keeps its half2 source for a later store.
    * pair: 09a12c8580c0 independently retains its result and loaded source.
    * Fragment position uses the same add without a wait. f25=1 selects the
    * minifloat; f42=2 selects the GPR. The register is field35, not field18.
    * AGX3c independently retains these compact operand/wait fields, adding
    * the GPR dispatch selector f50=3 and opcode f51=48. Keep slots7/8 gated:
    * this compact form's table only admits slots1..6 or the slots1+2 pair.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
      {50, 3}, {51, 48},
   };
   /* Six operand/hint fields plus at most six independent wait fields. */
   struct agx3_field_kv fields[12] = {
      {2, dest}, {35, src}, {17, immediate}, {46, hint == AGX3_ALU_HINT_DISCARD},
      {44, hint == AGX3_ALU_HINT_CACHE}, {11, dest_cache},
   };
   unsigned n = 6;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){52 + slot, 1};

   /* This encoding point rejects unsupported wait combinations. Do not
    * silently clear waits or switch to an unwitnessed long-form variant.
    */
   return agx3_bind_compose(enc, AGX3_MN_f32add, 1, base, gpu1 ? 8 : 6,
                           fields, n, out, len);
}

bool
agx3_alu_fadd_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint16_t immediate, bool negate_src,
                           bool saturate, enum agx3_alu_hint hint,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple sat_only: 09011c8101020082, sat(r0 + -0), wait6.
    * Native puts this -0 minifloat in field34 and its GPR in field18.
    * Apple complement: 09b1140180c8, -r0 + 1, discard/wait6.
    * sat_add and sat_neg retain that field35/field17 routing, extending
    * the instruction to ep2 for saturation and independent wait bits.
    * Commuting the add's two unmodified operands retains signed-zero
    * semantics; do not commute a negation away from its source register.
    */
   const bool swap = saturate && immediate == 0x80 && !negate_src;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, swap ? 2 : 1}, {26, 2},
      {42, swap ? 1 : 2}, {43, 2},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {swap ? 18 : 35, src}, {swap ? 34 : 17, immediate},
      {swap ? 27 : 44, hint == AGX3_ALU_HINT_CACHE},
      {swap ? 29 : 46, hint == AGX3_ALU_HINT_DISCARD},
      {swap ? 31 : 48, negate_src}, {14, saturate},
   };
   unsigned n = 7;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){52 + slot, 1};

   return agx3_bind_compose(enc, AGX3_MN_f32add, saturate ? 2 : 1,
                            base, 6, fields, n, out, len);
}

/* G18G (M6), which compiles as agx3c: also the agx3c modified
 * register+minifloat add. agx3_alu_fadd_minifloat_mod keeps the
 * G17G-family (agx3c) admission for every other GPU. */
bool
agx3_alu_fadd_minifloat_mod_g18g(enum agx_isa_encoding enc, uint16_t dest,
                                uint16_t src, uint16_t immediate, bool negate_src,
                                bool saturate, enum agx3_alu_hint hint,
                                uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > 0xff ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple sat_only: 09011c8101020082, sat(r0 + -0), wait6.
    * Native puts this -0 minifloat in field34 and its GPR in field18.
    * Apple complement: 09b1140180c8, -r0 + 1, discard/wait6.
    * sat_add and sat_neg retain that field35/field17 routing, extending
    * the instruction to ep2 for saturation and independent wait bits.
    * Commuting the add's two unmodified operands retains signed-zero
    * semantics; do not commute a negation away from its source register.
    */
   const bool swap = saturate && immediate == 0x80 && !negate_src;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, swap ? 2 : 1}, {26, 2},
      {42, swap ? 1 : 2}, {43, 2},
      {50, 3}, {51, 48},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {swap ? 18 : 35, src}, {swap ? 34 : 17, immediate},
      {swap ? 27 : 44, hint == AGX3_ALU_HINT_CACHE},
      {swap ? 29 : 46, hint == AGX3_ALU_HINT_DISCARD},
      {swap ? 31 : 48, negate_src}, {14, saturate},
   };
   unsigned n = 7;
   for (unsigned slot = 0; slot < 6; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){52 + slot, 1};

   return agx3_bind_compose(enc, AGX3_MN_f32add, saturate ? 2 : 1,
                            base, gpu1 ? 8 : 6, fields, n, out, len);
}

bool
agx3_alu_fadd_abs_minifloat(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   uint16_t immediate, bool dest_cache, bool negate_src, bool saturate,
   enum agx3_alu_hint hint, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) || immediate > UINT8_MAX ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;

   /* Fresh applegpu_g17p kernels independently cover abs(x) plus
    * minifloat {0.5, 1.5, 2.0}, negated abs, and saturation. They all select
    * ep3: f47/f48 are the absolute/negate bits and f14 is saturation. The
    * existing one-minus-abs witnesses establish destination retention,
    * source lifetime and all wait fields on this same endpoint. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[20] = {
      {2, dest}, {35, src}, {17, immediate}, {11, dest_cache},
      {47, 1}, {48, negate_src}, {14, saturate},
      {44, hint == AGX3_ALU_HINT_CACHE},
      {46, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 9;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){52 + slot,
                                          (wait_mask >> slot) & 1};

   uint8_t encoded[AGX3_ALU_MAX_LENGTH], encoded_len = 0;
   if (!agx3_bind_compose(enc, AGX3_MN_f32add, 3, base, 6, fields, n,
                          encoded, &encoded_len))
      return false;
   memcpy(out, encoded, encoded_len);
   *len = encoded_len;
   return true;
}

bool
agx3_alu_fadd_one_absneg(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src, bool dest_cache,
                         enum agx3_alu_hint hint, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;

   /* Native live/discard variants are 19b12481820800000200 and
    * 09b13401820800000200. f25 selects the minifloat input, while f47/f48
    * retain abs+neg on the GPR and f11 retains a partially live destination.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[20] = {
      {2, dest}, {35, src}, {17, 0x30}, {11, dest_cache}, {47, 1}, {48, 1},
      {44, hint == AGX3_ALU_HINT_CACHE},
      {46, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){52 + slot,
                                          (wait_mask >> slot) & 1};

   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32add, ep, base, 6, fields, n, out, len))
         return true;
   }

   return false;
}

bool
agx3_alu_fadd_abs_negzero(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src, bool dest_cache,
                          enum agx3_alu_hint hint, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;

   /* This is the exact fmov/fabs form: |src| + -0. f17 selects negative
    * zero, while f47 is the independently grounded register absolute bit. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[20] = {
      {2, dest}, {35, src}, {17, 0x80}, {11, dest_cache}, {47, 1}, {48, 0},
      {44, hint == AGX3_ALU_HINT_CACHE},
      {46, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){52 + slot,
                                          (wait_mask >> slot) & 1};

   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32add, ep, base, 6, fields, n, out, len))
         return true;
   }

   return false;
}

bool
agx3_alu_float_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                     uint16_t dest, const uint16_t src[3],
                     const enum agx3_alu_hint hints[3], uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const unsigned column = enc == AGX_ISA_ENCODING_AGX3C ? 1 : 0;
   const uint16_t slots = enc == AGX_ISA_ENCODING_AGX3C ? 0xff : 0x3f;
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       !src || !hints || !out || !len ||
       (op != AGX3_ALU_FADD && op != AGX3_ALU_FMUL && op != AGX3_ALU_FMA) ||
       (wait_mask & ~slots))
      return false;
   const unsigned count = op == AGX3_ALU_FMA ? 3 : 2;
   /* Native FADD/FMUL/FMA all encode eight-bit half-register operands.
    * TEX alone has a separate seven-bit destination restriction. */
   const unsigned limit = 256;
   if (dest >= limit || (dest & 1))
      return false;
   for (unsigned i = 0; i < count; ++i)
      if (src[i] >= limit || (src[i] & 1) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;

   /* Retain the qualified operand classes, not a specimen's liveness or
    * unconditional waits. In Mesa order src0/src1/src2 map to f35/f18/f52.
    * Fresh add_live/mul_live/fma_live witnesses retain all their operands;
    * all_ops additionally witnesses independently cached/discarded sources.
    */
   const struct agx3_alu_form *f = &agx3_alu_forms[op][column];
   struct agx3_field_kv base[20], ov[1 + 3 * 3 + 8];
   unsigned nb = 0, n = 0;
   for (unsigned i = 0; i < f->nbase; ++i) {
      unsigned field = f->base[i].f;
      bool wait_field = false;
      for (unsigned w = 0; w < f->nwait; ++w)
         wait_field |= field == f->wait[w];
      if (field != 29 && field != 46 && field != 63 && !wait_field)
         base[nb++] = (struct agx3_field_kv){field, f->base[i].v};
   }
   ov[n++] = (struct agx3_field_kv){2, dest};
   const unsigned operands[] = {35, 18, 52};
   const unsigned cache[] = {44, 27, 61};
   const unsigned discard[] = {46, 29, 63};
   for (unsigned i = 0; i < count; ++i) {
      ov[n++] = (struct agx3_field_kv){operands[i], src[i]};
      ov[n++] = (struct agx3_field_kv){cache[i], hints[i] == AGX3_ALU_HINT_CACHE};
      ov[n++] = (struct agx3_field_kv){discard[i], hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < f->nwait; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){f->wait[i], 1};

   if (enc == AGX_ISA_ENCODING_AGX3C) {
      static const uint8_t add_eps[] = {0, 1, 2, 3};
      static const uint8_t mul_eps[] = {0, 1, 2, 4, 5, 3};
      static const uint8_t fma_eps[] = {0, 1, 2, 3, 4, 5, 6, 7};
      const uint8_t *eps = op == AGX3_ALU_FMA ? fma_eps
                           : op == AGX3_ALU_FMUL ? mul_eps : add_eps;
      const unsigned count_eps = op == AGX3_ALU_FMA ? 8
                                 : op == AGX3_ALU_FMUL ? 6 : 4;
      for (unsigned i = 0; i < count_eps; ++i)
         if (agx3_bind_compose(enc, f->mnemonic, eps[i], base, nb, ov, n,
                               out, len))
            return true;
      return false;
   }

   const unsigned first = op == AGX3_ALU_FMA ? 2 : 0;
   const unsigned last = op == AGX3_ALU_FMA ? 3 : 1;
   for (unsigned ep = first; ep <= last; ++ep)
      if (agx3_bind_compose(enc, f->mnemonic, ep, base, nb, ov, n, out, len))
         return true;
   return false;
}

bool
agx3_alu_fadd_stage(enum agx_isa_encoding enc, uint16_t staging_dest,
                    uint8_t staging_slot, const uint16_t src[2],
                    const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       (staging_dest & 1) || staging_dest >= 256 || staging_slot < 1 ||
       staging_slot > 6 || (wait_mask & ~UINT16_C(0x3f)))
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      if ((src[i] & 1) || src[i] >= 256 ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* Exact native destination routing:
    *
    *   09 01 3c 09 40 c0   stage[0] = r4 + r0, slot1, wait6
    *   19 03 3c 05 40 00   stage[2] = r2 + r1, slot1
    *
    * f9=5 changes the destination from a GPR (f2) to staging (f5), while
    * f12 names the shared coordinate transaction.  Source operands and
    * dependency waits retain the ordinary f32add fields. */
   const struct agx3_field_kv base[] = {
      {9, 5}, {10, 2}, {25, 2}, {26, 2}, {42, 2}, {43, 2},
   };
   struct agx3_field_kv fields[16] = {
      {5, staging_dest}, {12, staging_slot}, {13, 0},
      {35, src[0]}, {18, src[1]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {27, hints[1] == AGX3_ALU_HINT_CACHE},
      {29, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 9;
   for (unsigned slot = 0; slot < 6; ++slot) {
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){(uint16_t)(52 + slot), 1};
   }

   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32add, ep, base, 6, fields, n, out, len))
         return true;
   }

   return false;
}

bool
agx3_alu_fma_minifloat_mod(enum agx_isa_encoding enc, uint16_t dest,
                           uint16_t src, uint8_t immediate, uint16_t addend,
                           bool negate_addend, bool saturate, enum agx3_alu_hint src_hint,
                           enum agx3_alu_hint addend_hint, bool dest_cache,
                           uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) ||
       dest >= 256 || src >= 256 || addend >= 256 ||
       ((dest | src | addend) & 1) ||
       (wait_mask & ~(gpu1 ? UINT16_C(0xff) : UINT16_C(0x3f))) ||
       (unsigned)src_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)addend_hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{25,2},{26,2},{42,1},{43,2},{59,2},{60,2},{77,2},{91,2},
      {102,1}, {95,3}, {96,55},
   };
   struct agx3_field_kv fields[19] = {
      {2,dest},{11,dest_cache},{18,src},{34,immediate},{52,addend},
      {27,src_hint==AGX3_ALU_HINT_CACHE},{29,src_hint==AGX3_ALU_HINT_DISCARD},
      {61,addend_hint==AGX3_ALU_HINT_CACHE},{63,addend_hint==AGX3_ALU_HINT_DISCARD},
      {65,negate_addend},{14,saturate},
   };
   unsigned n=11;
   /* Native G17G fma(x, 0.5, y), its two live-input variants and fma(x,
    * 0.5, -y) preserve these operand fields. The long endpoint needs its
    * own opcode fields and all eight dependency slots. Override f102 too:
    * G17P's negative-addend template must not introduce a G17G slot-6 wait. */
   for (unsigned slot = 0; slot < (gpu1 ? 8 : 6); slot++)
      if (gpu1 || (wait_mask & (1u << slot)))
         fields[n++] = (struct agx3_field_kv){97 + slot,
                                             !!(wait_mask & (1u << slot))};
   for (unsigned ep = 2; ep <= (gpu1 ? 7 : 4); ep++)
      if(agx3_bind_compose(enc,AGX3_MN_f32fma,ep,base,gpu1 ? 13 : ((negate_addend || saturate) ? 11 : 10),
                           fields,n,out,len))
         return true;
   return false;
}

static bool
agx3_alu_int16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool is_signed, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   /* AGX3c int2flt fields are 8-bit half indices (the checked-in form binds
    * every even dest and any src through 254, like the sibling encodings
    * above); the old >=128 cap failed vkCreateComputePipelines for G18G
    * llama.cpp shaders. No live capture at half >=128 yet: a G18G capture
    * or the hardware run confirms. */
   if (!out || !len || (!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) ||
       dest >= (gpu1 ? 256 : 192) || src >= (gpu1 ? 256 : 192) ||
       (dest & 1) || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(gpu1 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /* Apple ushort_cs: a70756000200a420 (r0f <- r0h, wait slot6).
    * The register fields reach half 191: hiregs.metal (96 live conversions)
    * writes destinations up to half 160 and reads sources up to half 189 in
    * the same form, the source's high bit in byte 6.
    * ushort_pair_cs additionally witnesses odd source half3 and no wait;
    * ushort_reuse_cs has f29=0 with a later16-bit store of the same source.
    * Versus U32 int2flt only f26 changes2->1. The fresh short_float and
    * short_float_live witnesses establish signedness in f30.
    * AGX3c independently retains source width f26, signedness f30 and the
    * operand fields. Its GPR destination additionally requires f31=3 and
    * opcode f32=91, as in the existing 32-bit conversion table; waits use
    * all eight fields f33..40. Its bit layout is not the AGX3b byte layout.
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 1}, {31, 3}, {32, 91},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {29, hint == AGX3_ALU_HINT_DISCARD},
      {27, hint == AGX3_ALU_HINT_CACHE}, {30, is_signed},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < (gpu1 ? 8 : 6); ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){33 + slot, 1};
   return agx3_bind_compose(enc, AGX3_MN_int2flt, 0, base, gpu1 ? 6 : 4,
                           fields, n, out, len);
}

bool
agx3_alu_u16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_int16_to_f32(enc,dest,src,hint,false,wait_mask,out,len);
}

bool
agx3_alu_s16_to_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_int16_to_f32(enc,dest,src,hint,true,wait_mask,out,len);
}

bool
agx3_alu_float_to_int32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        bool is_signed, enum agx3_alu_hint hint,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (enc != AGX_ISA_ENCODING_AGX3B && !m5) || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /* Apple's argument-evaluation witness keeps the converted float live:
    *
    *   27075404030496480300  d=r2, s=r1, signed, d.cache, s.cache
    *
    * Against the previously used fixed baseline, f11 is destination cache,
    * f25 is source cache and f27 is source discard. They are lifetime fields,
    * so derive them from post-RA state rather than baking in discard.
    *
    * metal-nt 32023.917 applegpu_g17p, 96 live uint(float) results:
    *   270754be037cb5080200  $r95 <- uint(^r95)
    * Both register fields address eight-bit half indices. Restricting them
    * to halves below 128 rejected high-pressure tessellation evaluation
    * shaders whose float-to-integer result landed at half 128.
    */
   const struct agx3_field_kv base[] = {
      {9, 2},  {10, 2}, {23, 2},  {24, 2}, {41, is_signed ? 5 : 4},
      {42, 1}, {31, 3}, {32, 71},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest},
      {11, dest_cache},
      {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      fields[n++] =
         (struct agx3_field_kv){33 + slot, !!(wait_mask & (1u << slot))};

   return agx3_bind_compose(enc, AGX3_MN_flt2int, 0, base, m5 ? 8 : 6, fields, n, out,
                            len);
}

bool
agx3_alu_float_to_int16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        bool src_16, bool is_signed, enum agx3_alu_hint hint,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (enc != AGX_ISA_ENCODING_AGX3B && !m5) || dest >= 256 ||
       src >= (m5 ? 256 : 128) || (!src_16 && (src & 1)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /*
    * `flt2int` is one form with three independent axes, each pinned by a
    * kernel pair that changes nothing else. f10 is the RESULT width, f24 the
    * SOURCE width and f41 the signed/unsigned mode:
    *
    *   short(float a)    270756020204b0c80200   f10=1 f24=2 f41=3
    *   short(half  a)    270756020204a8c80200   f10=1 f24=1 f41=3
    *   ushort(half a)    270756020204a8080300   f10=1 f24=1 f41=2
    *   ushort(float a)   270756020204b0080300   f10=1 f24=2 f41=2
    *   int(half a)       270756000200ac480300   f10=2 f24=1 f41=5
    *
    * The first two differ in exactly one field, which is what identifies f24
    * as the half-source selector rather than an unrelated instruction. This
    * used to be a hardcoded ten-byte array with the source width welded to
    * f32, so `short(half)` -- what `int(clamp(x, 1.0, 32.0))` becomes once
    * the arithmetic around it is half-precision, as it is in Hyprland's
    * motion blur -- had no encoding at all and failed the shader closed.
    *
    * A half source may sit in an odd half; a 32-bit one may not:
    *
    *   short(half a[2*i+1])  270754030206a8c80200   f2=3 f16=3
    *
    * The wait fields are f33..f38, the six producer slots, exactly as on the
    * 32-bit-result twin agx3_alu_f16_to_int32().
    */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, src_16 ? 1 : 2},
      {41, is_signed ? 3 : 2}, {42, 1}, {31,3}, {32,71},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 4;
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      fields[n++] = (struct agx3_field_kv){33 + slot,
                                           !!(wait_mask & (1u << slot))};

   return agx3_bind_compose(enc, AGX3_MN_flt2int, 0, base, m5 ? 8 : 6, fields, n, out, len);
}

bool
agx3_alu_add_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    uint16_t immediate, enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || immediate > 255 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~0x3fu))
      return false;
   /* Native short +2 and ushort2 +(2,17): result and register operand
    * are individual halves, with the immediate in slot A. Addition wraps
    * at16 bits without widening or allocating a temporary register. */
   const struct agx3_field_kv base[] = {
      {9,2}, {10,1}, {23,1}, {24,1}, {38,2}, {39,1},
   };
   struct agx3_field_kv fields[12] = {
      {2,dest}, {11,dest_cache}, {15,immediate}, {31,src},
      {40,hint == AGX3_ALU_HINT_CACHE}, {42,hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n=6;
   for(unsigned i=0;i<6;++i)
      if(wait_mask & (1u<<i))fields[n++]=(struct agx3_field_kv){45+i,1};
   return agx3_bind_compose(enc,AGX3_MN_iadd,0,base,6,fields,n,out,len);
}

bool
agx3_alu_sub_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   uint16_t immediate, enum agx3_alu_hint hint,
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       src >= 128 || immediate == 0 || immediate > 255 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   /* Apple G17P emits a distinct ten-byte subtract-immediate form rather
    * than negating the add form. Controlled Metal kernels witness immediate
    * values 1, 2, 17 and 255; half destinations 1, 2 and 3; half sources 2,
    * 3 and 4; both destination-cache states; both source lifetime hints; and
    * no wait versus wait slot 6. The immediate is an unsigned byte scaled by
    * four in bytes 6..7, exactly matching modulo-16-bit subtraction. */
   const uint16_t encoded_immediate = immediate << 2;
   const uint8_t bytes[] = {
      0x1f, 0x01, wait_mask ? 0x56 : 0x54, (uint8_t)dest,
      dest_cache ? 0x03 : 0x02, (uint8_t)(src << 1),
      (uint8_t)encoded_immediate, (uint8_t)(encoded_immediate >> 8),
      hint == AGX3_ALU_HINT_DISCARD ? 0x13 : 0x11, 0x01,
   };
   memcpy(out, bytes, sizeof(bytes));
   *len = sizeof(bytes);
   return true;
}

/* Exact native G17P witnesses include 0x1800 (1/512), adjacent literals,
 * odd destination halves and mixed multiplies with both source hints.
 * LDIMM's nonzero-wait form has not been witnessed and remains refused. */
bool
agx3_alu_mov16(enum agx_isa_encoding enc, uint16_t dest, uint32_t bits,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       bits > 0xffff || wait_mask)
      return false;
   const struct agx3_field_kv fields[] = {{9, 2}, {10, 1}, {2, dest}, {14, bits}};
   return agx3_bind_compose(enc, AGX3_MN_ldimm, 1, fields, 4, NULL, 0, out, len);
}

bool
agx3_alu_mov16_gpu1(uint16_t dest, bool dest_cache, uint32_t bits,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len)
{
   /* Apple's depth/stencil fragments load the stencil word this way:
    * ldimm r0h = 0xff00 / 0xff5a / 0xff01 are 04 80 26 00 1c 0c 07 00 with
    * the constant moved. The form is the G17G ldimm row narrowed to a half. */
   if (!out || !len || dest >= 256 || bits > 0xffff || wait_mask > 0xff)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {15, 3}, {16, 105},
   };
   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv){2, dest};
   ov[n++] = (struct agx3_field_kv){11, dest_cache ? 1 : 0};
   ov[n++] = (struct agx3_field_kv){14, bits};
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){17 + i, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_ldimm, 2, base, 4, ov, n,
                            out, len);
}

bool
agx3_alu_fmul_mixed(enum agx_isa_encoding enc, uint16_t dest,
                    uint16_t f32, uint16_t f16,
                    enum agx3_alu_hint hint32, enum agx3_alu_hint hint16,
                    bool dest_cache, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || ((dest | f32) & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3f))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {42, 2}, {43, 1}, {60, 2}, {74, 2},
   };
   struct agx3_field_kv fields[14] = {
      {2, dest}, {11, dest_cache}, {18, f32}, {35, f16},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned bit = 0; bit < 6; ++bit)
      if (wait_mask & (1u << bit))
         fields[n++] = (struct agx3_field_kv){80 + bit, 1};
   return agx3_bind_compose(enc, AGX3_MN_f32mul, 0, base, 8, fields, n, out, len) ||
          agx3_bind_compose(enc, AGX3_MN_f32mul, 1, base, 8, fields, n, out, len);
}

bool
agx3_alu_fmul_abs_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || enc != AGX_ISA_ENCODING_AGX3B || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || ((dest | f32) & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Fresh applegpu_g17p kernels use the long f32mul.ep3 form. f30 carries
    * ABS on the full source while f43 keeps the other source in its assigned
    * scalar half. The two source lifetimes, destination cache bit, and six
    * producer slots remain independent on this endpoint. */
   const struct agx3_field_kv base[] = {
      {9, 2},  {10, 2}, {25, 2}, {26, 2}, {30, 1},
      {42, 2}, {43, 1}, {60, 2}, {74, 2},
   };
   struct agx3_field_kv fields[14] = {
      {2, dest},
      {11, dest_cache},
      {18, f32},
      {35, f16},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[8 + slot] = (struct agx3_field_kv){80 + slot,
                                               !!(wait_mask & (1u << slot))};

   return agx3_bind_compose(enc, AGX3_MN_f32mul, 3, base, 9, fields, 14, out,
                            len);
}

bool
agx3_alu_fadd_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      if (!out || !len || dest >= 256 || f32 >= 256 || f16 >= 256 ||
          ((dest | f32) & 1) ||
          (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
          (unsigned)hint16 > AGX3_ALU_HINT_CACHE || (wait_mask & ~0xffu))
         return false;

      /* G17G texture-backed half values add directly to full pixel
       * coordinates. 098524820020 retains both sources and waits slot 1;
       * 29053c021000 ends both sources. C keeps B's register routing and
       * width fields, with its own required f50/f51 endpoint selectors. */
      const struct agx3_field_kv base[] = {
         {9, 2}, {10, 2}, {25, 2}, {26, 2}, {42, 2}, {43, 1},
         {50, 3}, {51, 48},
      };
      struct agx3_field_kv fields[16] = {
         {2, dest}, {11, dest_cache}, {18, f32}, {35, f16},
         {27, hint32 == AGX3_ALU_HINT_CACHE},
         {29, hint32 == AGX3_ALU_HINT_DISCARD},
         {44, hint16 == AGX3_ALU_HINT_CACHE},
         {46, hint16 == AGX3_ALU_HINT_DISCARD},
      };
      unsigned n = 8;
      for (unsigned slot = 0; slot < 8; ++slot)
         if (wait_mask & (1u << slot))
            fields[n++] = (struct agx3_field_kv){52 + slot, 1};

      const unsigned ep = (wait_mask & 0xc0) ? 3 :
                          ((wait_mask & (wait_mask - 1)) ? 2 : 1);
      return agx3_bind_compose(enc, AGX3_MN_f32add, ep, base, 8, fields, n,
                               out, len);
   }

   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || ((dest | f32) & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Apple keeps exact half constants in scalar halves when adding them to
    * full floats.  The half occupies f35 (including odd halves) while f18 is
    * the full source; f43 selects the mixed width.  Native Hyprland-shaped
    * kernels independently witness retained/dead halves, destination cache,
    * and both empty and producer-wait forms on this compact endpoint. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {42, 2}, {43, 1},
   };
   struct agx3_field_kv fields[14] = {
      {2, dest},
      {11, dest_cache},
      {18, f32},
      {35, f16},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[8 + i] = (struct agx3_field_kv){52 + i,
                                            !!(wait_mask & (1u << i))};

   return agx3_bind_compose(enc, AGX3_MN_f32add, 1, base, 6, fields, 14, out,
                            len);
}

bool
agx3_alu_fadd_abs_mixed_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t f32, uint16_t f16,
   enum agx3_alu_hint hint32, enum agx3_alu_hint hint16, bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || ((dest | f32) & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Fresh G17P Metal output uses f32add.ep3 directly: the first source is
    * absolute f32, while the second retains its physical half register. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {30, 1}, {42, 2}, {43, 1},
   };
   struct agx3_field_kv fields[15] = {
      {2, dest},
      {11, dest_cache},
      {18, f32},
      {35, f16},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[8 + i] = (struct agx3_field_kv){52 + i,
                                            !!(wait_mask & (1u << i))};

   return agx3_bind_compose(enc, AGX3_MN_f32add, 3, base, 7, fields, 14, out,
                            len);
}

bool
agx3_alu_fmul_mixed_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t f32,
                         uint16_t f16, enum agx3_alu_hint hint32,
                         enum agx3_alu_hint hint16, bool dest_cache,
                         uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                         uint8_t *len)
{
   /* This is the exact second half of Apple's scalar half sqrt sequence.
    * Field10 selects the half destination, including odd half registers.
    * Fresh retained-source probes establish f27/f29 and f44/f46, while the
    * ordinary long f32mul family supplies the same six wait fields f80..f85. */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !len || (enc != AGX_ISA_ENCODING_AGX3B && !m5) || dest >= 256 ||
       f32 >= 256 || f16 >= 256 || (f32 & 1) ||
       (unsigned)hint32 > AGX3_ALU_HINT_CACHE ||
       (unsigned)hint16 > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {25, 2}, {26, 2}, {42, 2}, {43, 1}, {60, 2}, {74, 2},
   };
   struct agx3_field_kv fields[18] = {
      {2, dest},
      {18, f32},
      {35, f16},
      {11, dest_cache},
      {27, hint32 == AGX3_ALU_HINT_CACHE},
      {29, hint32 == AGX3_ALU_HINT_DISCARD},
      {44, hint16 == AGX3_ALU_HINT_CACHE},
      {46, hint16 == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[8 + i] = (struct agx3_field_kv){80 + i, !!(wait_mask & (1u << i))};
   if (m5) {
      fields[14] = (struct agx3_field_kv){78, 3};
      fields[15] = (struct agx3_field_kv){79, 62};
      fields[16] = (struct agx3_field_kv){86, !!(wait_mask & 64)};
      fields[17] = (struct agx3_field_kv){87, !!(wait_mask & 128)};
   }
   for (unsigned ep = 0; ep < (m5 ? 4 : 2); ++ep)
      if (agx3_bind_compose(enc, AGX3_MN_f32mul, ep, base, 8, fields, m5 ? 18 : 14,
                            out, len))
         return true;
   return false;
}

/* Ops that alu_compose can carry a wait on (fmin/fmax's tied-operand fsel form
 * is not modeled here; the backend keeps those on the compact packer). */
bool
agx3_alu_wait_supported(enum agx3_alu_op op)
{
   /* fmin/fmax use a tied-operand fsel form whose duplicated compare fields
    * this table does not carry; exclude them. */
   return op < AGX3_ALU_NUM && op != AGX3_ALU_FMIN && op != AGX3_ALU_FMAX;
}

/* One composition of `F`. `skip_pinned_waits` drops the baseline entries whose
 * field is also one of the form's wait fields, which is how the "what the
 * caller actually asked for" reference encoding is built below. The production
 * path passes false, so its bytes are unchanged. */
static bool
agx3_alu_compose_raw(enum agx_isa_encoding enc, const struct agx3_alu_form *F,
                     bool skip_pinned_waits, uint16_t dest, uint16_t src0,
                     uint16_t src1, uint16_t src2, uint32_t imm,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len)
{
   struct agx3_field_kv base[20];
   unsigned nbase = 0;
   for (unsigned i = 0; i < F->nbase; ++i) {
      if (skip_pinned_waits) {
         bool is_wait = false;
         for (unsigned j = 0; j < F->nwait; ++j)
            is_wait |= (F->base[i].f == F->wait[j]);
         if (is_wait)
            continue;
      }
      base[nbase].field = F->base[i].f;
      base[nbase].value = F->base[i].v;
      nbase++;
   }

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   if (F->f_dest >= 0) { ov[n].field = F->f_dest; ov[n].value = dest; n++; }
   if (F->f_src0 >= 0) { ov[n].field = F->f_src0; ov[n].value = src0; n++; }
   if (F->f_src1 >= 0) { ov[n].field = F->f_src1; ov[n].value = src1; n++; }
   if (F->f_src2 >= 0) { ov[n].field = F->f_src2; ov[n].value = src2; n++; }
   if (F->f_imm >= 0)  { ov[n].field = F->f_imm;  ov[n].value = imm;  n++; }
   for (unsigned i = 0; i < F->nwait; ++i)
      if (wait_mask & (1u << i)) { ov[n].field = F->wait[i]; ov[n].value = 1; n++; }

   return agx3_bind_compose(enc, F->mnemonic, agx3_bind_ep(enc, F->ep), base,
                            nbase, ov, n, out, len);
}

static bool
agx3_alu_form_pins_a_wait(const struct agx3_alu_form *F)
{
   for (unsigned i = 0; i < F->nbase; ++i)
      for (unsigned j = 0; j < F->nwait; ++j)
         if (F->base[i].f == F->wait[j])
            return true;
   return false;
}

bool
agx3_alu_compose(enum agx_isa_encoding enc, enum agx3_alu_op op, uint16_t dest,
                 uint16_t src0, uint16_t src1, uint16_t src2, uint32_t imm,
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                 uint8_t *len)
{
   if (op >= AGX3_ALU_NUM)
      return false;
   /*
    * G18G (agx3d) uses the agx3c column, but only for LDIMM so far: its row
    * pins no specimen lifetime or wait, and the applegpu_g18g 32-bit GPR
    * ldimm witnesses of the j813-linux gpu/m5-*-reference corpus carry
    * exactly its fields (f9=2 f10=2 f15=3 f16=105, f2 destination, f14
    * immediate: 41 03 00 08, 41 05 ff ff, 11 07 00 00 80 3f). The other rows
    * pin their specimen's source lifetimes, which agx3d instead takes from
    * the runtime encoders (agx3_falu_*_runtime). Its wait field is one slot
    * code; the additivity check below refuses anything else.
    */
   const bool agx3d = enc == AGX_ISA_ENCODING_AGX3D;
   if (agx3d && op != AGX3_ALU_LDIMM)
      return false;
   const unsigned ei = agx_encoding_is_g17_class(enc) ? 1
                       : (enc == AGX_ISA_ENCODING_AGX3B) ? 0
                                                         : 2;
   if (ei == 2)
      return false;
   const struct agx3_alu_form *F = &agx3_alu_forms[op][ei];
   if (F->mnemonic == 0)
      return false;
   if (wait_mask && !agx3_alu_wait_supported(op))
      return false;
   /* Only wait bits with a field in this form are deposited below. A bit past
    * the end of wait[] would be silently dropped, which is a missing
    * dependency rather than a refusal, so fail closed instead. Callers fall
    * back to a synchronous prelude that retires the slot (agx3_emit_alu). */
   if (F->nwait < 16 && (wait_mask >> F->nwait) != 0)
      return false;

   if (!agx3_alu_compose_raw(enc, F, agx3d, dest, src0, src1, src2, imm,
                             wait_mask, out, len))
      return false;

   /*
    * agx3_alu_table.h was extracted by decoding one witnessed Apple
    * instruction per form, so some baselines carry that specimen's own
    * scoreboard wait as if it were a form constant. Correcting that changes
    * the bytes of every wait-carrying ALU instruction and is held for
    * hardware. What cannot wait is the case where the pinned wait and the
    * requested wait are not independent bits.
    *
    * Two wait encodings exist. The ten-byte integer forms carry a six-bit
    * MASK, so a pinned slot plus a requested slot is simply "wait both" --
    * an over-wait, conservative and safe. The compact float forms instead
    * carry a three-bit CODE naming one slot, values 1..6, and the two
    * deposits merge arithmetically: f32add with f52 pinned (code 1) and slot
    * 2 (f53, code 2) requested emitted 09 01 04 05 00 e0, code 7 -- a value
    * Apple never emits and which names no slot. That is a wrong value in the
    * instruction stream, the one failure mode with no safe reading.
    *
    * Tell the two apart structurally rather than by hard-coding a form: an
    * additive (mask) deposit satisfies
    *    emitted == pinned_only | requested_only
    * byte for byte, because the two land in disjoint bits and nothing else
    * differs. A merged code does not. Refuse when it does not, and
    * agx3_emit_alu() retires the slot with a synchronous prelude instead.
    */
   if (wait_mask == 0)
      return true;

   uint8_t zero[AGX3_ALU_MAX_LENGTH], requested[AGX3_ALU_MAX_LENGTH];
   uint8_t one[AGX3_ALU_MAX_LENGTH], pinned[AGX3_ALU_MAX_LENGTH];
   uint8_t zero_len = 0, requested_len = 0, one_len = 0, pinned_len = 0;
   if (!agx3_alu_compose_raw(enc, F, true, dest, src0, src1, src2, imm, 0, zero,
                             &zero_len) ||
       !agx3_alu_compose_raw(enc, F, true, dest, src0, src1, src2, imm,
                             wait_mask, requested, &requested_len) ||
       zero_len != *len || requested_len != *len)
      return false;

   /* Requirement one: the multi-slot deposit is the union of the single-slot
    * deposits. On a mask form that is automatic. On a code form it is false --
    * f32add slots 1 and 2 deposit byte5 0x20 and 0x40, but together they
    * deposit 0xe0, which is code 7. */
   uint8_t all[AGX3_ALU_MAX_LENGTH];
   memcpy(all, zero, *len);
   for (unsigned i = 0; i < F->nwait; ++i) {
      if (!(wait_mask & (1u << i)))
         continue;
      if (!agx3_alu_compose_raw(enc, F, true, dest, src0, src1, src2, imm,
                                1u << i, one, &one_len) ||
          one_len != *len)
         return false;
      for (unsigned k = 0; k < *len; ++k)
         all[k] |= one[k];
   }
   if (memcmp(all, requested, *len) != 0)
      return false;

   /* Requirement two: the form's pinned specimen wait, if it has one, is
    * additive with what the caller asked for rather than merging into a third
    * value. agx3d never deposits the pinned wait. */
   if (agx3d || !agx3_alu_form_pins_a_wait(F))
      return true;
   if (!agx3_alu_compose_raw(enc, F, false, dest, src0, src1, src2, imm, 0,
                             pinned, &pinned_len) ||
       pinned_len != *len)
      return false;
   for (unsigned i = 0; i < *len; ++i)
      if (out[i] != (uint8_t)(pinned[i] | requested[i]))
         return false;

   return true;
}

/* Native and_live/or_live/xor_live retain their second input for a later
 * operation (f39), then discard it on its last use (f41). and_keep_both
 * clears both discard fields. The operation classes and register fields
 * are identical; only truth-table bits differ. */
static bool
agx3_alu_bit_width(enum agx_isa_encoding enc, enum agx3_alu_op op, bool narrow,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
      !src || !hints || !out || !len ||
      (op != AGX3_ALU_BAND && op != AGX3_ALU_BOR && op != AGX3_ALU_BXOR) ||
      dest >= 256 || (!narrow && (dest & 1)) || (wait_mask & ~(enc == AGX_ISA_ENCODING_AGX3C ? 0xffu : 0x3fu)))
      return false;
   for(unsigned i=0;i<2;i++)
      if(src[i]>=256 || (!narrow && (src[i]&1)) || (unsigned)hints[i]>AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_alu_form *f=&agx3_alu_forms[op][enc == AGX_ISA_ENCODING_AGX3C ? 1 : 0];
   struct agx3_field_kv base[20],ov[15];unsigned nb=0,n=0;
   for(unsigned i=0;i<f->nbase;i++){
      unsigned field=f->base[i].f;bool wait=false;
      for(unsigned j=0;j<f->nwait;j++)wait|=field==f->wait[j];
      if(field!=27 && field!=41 && !wait)
         base[nb++]=(struct agx3_field_kv){field,
            narrow && (field == 10 || field == 24 || field == 38) ? 1 : f->base[i].v};
   }
   ov[n++]=(struct agx3_field_kv){2,dest};
   const unsigned operand[]={16,30},cache[]={25,39},discard[]={27,41};
   for(unsigned i=0;i<2;i++){
      ov[n++]=(struct agx3_field_kv){operand[i],src[i]};
      ov[n++]=(struct agx3_field_kv){cache[i],hints[i]==AGX3_ALU_HINT_CACHE};
      ov[n++]=(struct agx3_field_kv){discard[i],hints[i]==AGX3_ALU_HINT_DISCARD};
   }
   for(unsigned i=0;i<f->nwait;i++)
      if(wait_mask&(1u<<i))ov[n++]=(struct agx3_field_kv){f->wait[i],1};
   if (enc == AGX_ISA_ENCODING_AGX3C &&
       agx3_bind_compose(enc,f->mnemonic,0,base,nb,ov,n,out,len))
      return true;
   return agx3_bind_compose(enc,f->mnemonic,f->ep,base,nb,ov,n,out,len);
}

/*
 * Each source's last-use field in the AGX3b composer forms, as the explicit-hint
 * encoders place it: integer f27/f42 (agx3_alu_integer_reg_shift), bitwise
 * f27/f41 (agx3_alu_bit_width), float f46/f29/f63 for operands f35/f18/f52
 * (agx3_alu_float_reg), and the single source of the float conversions and
 * unary float forms. A zero names no grounded field.
 */
static const uint8_t agx3_alu_source_last_use[AGX3_ALU_NUM][3] = {
   [AGX3_ALU_IADD] = {27, 42},
   [AGX3_ALU_ISUB] = {27, 42},
   [AGX3_ALU_FADD] = {46, 29},
   [AGX3_ALU_FMUL] = {46, 29},
   [AGX3_ALU_FMA] = {46, 29, 63},
   [AGX3_ALU_BAND] = {27, 41},
   [AGX3_ALU_BOR] = {27, 41},
   [AGX3_ALU_BXOR] = {27, 41},
   [AGX3_ALU_RCP] = {29},
   [AGX3_ALU_RSQRT] = {29},
   [AGX3_ALU_RINT] = {29},
   [AGX3_ALU_I2F] = {29},
   [AGX3_ALU_U2F] = {29},
   [AGX3_ALU_F2I] = {27},
};

bool
agx3_alu_form_ends_source(enum agx_isa_encoding enc, enum agx3_alu_op op,
                          unsigned i)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || op >= AGX3_ALU_NUM || i >= 3)
      return false;

   const uint8_t field = agx3_alu_source_last_use[op][i];
   const struct agx3_alu_form *F = &agx3_alu_forms[op][0];
   for (unsigned b = 0; field != 0 && b < F->nbase; ++b) {
      if (F->base[b].f == field && F->base[b].v != 0)
         return true;
   }
   return false;
}

/* G18G (M6), which compiles as agx3c: also reports the agx3c RCP, I2F, U2F
 * and F2I specimens that end a live source. agx3_alu_form_ends_source
 * keeps the G17G-family (agx3c) answer for every other GPU. */
bool
agx3_alu_form_ends_source_g18g(enum agx_isa_encoding enc, enum agx3_alu_op op,
                               unsigned i)
{
   if (op >= AGX3_ALU_NUM || i >= 3)
      return false;

   unsigned column;
   if (enc == AGX_ISA_ENCODING_AGX3B) {
      column = 0;
   } else if (enc == AGX_ISA_ENCODING_AGX3C &&
              (op == AGX3_ALU_RCP || op == AGX3_ALU_I2F ||
               op == AGX3_ALU_U2F || op == AGX3_ALU_F2I)) {
      /* G17G/G18G native live-input references establish the same unary
       * source-discard fields. A wait-carrying specimen must not bypass
       * the live-preserving compact path in agx3_select_core.c.
       */
      column = 1;
   } else {
      return false;
   }

   const uint8_t field = agx3_alu_source_last_use[op][i];
   const struct agx3_alu_form *F = &agx3_alu_forms[op][column];
   for (unsigned b = 0; field != 0 && b < F->nbase; ++b) {
      if (F->base[b].f == field && F->base[b].v != 0)
         return true;
   }
   return false;
}

bool
agx3_alu_bit_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_width(enc, op, false, dest, src, hints, wait_mask, out, len);
}

/* Native runtime fragment XORs at offsets108/152/198/222 use ep0,
 * f11 destination-cache and both source lifetimes. The same fields have
 * independent extended-form native witnesses. Let the composer reject an
 * unrepresentable compact operand/wait tuple before trying ep1: never clear
 * a dependency to make the compact form fit. Legacy callers stay unchanged. */
bool
agx3_alu_bit_reg_native(enum agx_isa_encoding enc, enum agx3_alu_op op,
                         uint16_t dest, const uint16_t src[2],
                         const enum agx3_alu_hint hints[2], bool dest_cache,
                         uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       (op != AGX3_ALU_BAND && op != AGX3_ALU_BOR && op != AGX3_ALU_BXOR) ||
       dest >= 256 || (dest & 1) || (wait_mask & ~UINT16_C(0x3f)))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (src[i] >= 256 || (src[i] & 1) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_field_kv base[] = {
      {9,2}, {10,2}, {23,2}, {24,2}, {37,2}, {38,2},
   };
   struct agx3_field_kv fields[] = {
      {2,dest}, {16,src[0]}, {30,src[1]}, {11,dest_cache},
      {25,hints[0] == AGX3_ALU_HINT_CACHE},
      {27,hints[0] == AGX3_ALU_HINT_DISCARD},
      {39,hints[1] == AGX3_ALU_HINT_CACHE},
      {41,hints[1] == AGX3_ALU_HINT_DISCARD},
      {44,!!(wait_mask & 1)}, {45,!!(wait_mask & 2)},
      {46,!!(wait_mask & 4)}, {47,!!(wait_mask & 8)},
      {48,!!(wait_mask & 16)}, {49,!!(wait_mask & 32)},
   };
   const enum agx3_mnemonic name = agx3_alu_forms[op][0].mnemonic;
   return agx3_bind_compose(enc,name,0,base,6,fields,14,out,len) ||
          agx3_bind_compose(enc,name,1,base,6,fields,14,out,len);
}

/*
 * The carrier must not end its source's lifetime. The composer table's
 * bit_and baseline is Apple's `bit_and r0, ^r0, r0` specimen, whose f27 marks
 * the first source's last use. Emitted as a no-op, that tells the G17P the
 * register is dead wherever it still holds a live value:
 * reconvergence.terminate_invocation.terminate_helpers keeps its
 * helper-invocation predicate in r0h across the retire in front of its first
 * branch, and fails in whole quads. G17P therefore takes the explicit-hint
 * form with no lifetime hint on either operand. The specimen's own slot-6
 * wait (f49) stays with the carrier, so the slots it retires are unchanged.
 */
bool
agx3_alu_band_identity_wait(enum agx_isa_encoding enc, uint16_t reg,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C)
      return false;

   const uint16_t src[] = {reg, reg};
   const enum agx3_alu_hint hints[] = {AGX3_ALU_HINT_NONE, AGX3_ALU_HINT_NONE};

   /* The witnessed carriers wait slot6 on G17P and slot8 on G17S. Neither
    * operand is dead: r0 may hold an independent live half-register value.
    * The raw form table includes its specimen's f27 discard and must not
    * be used for this identity operation. */
   const uint16_t extra = UINT16_C(1) <<
      (enc == AGX_ISA_ENCODING_AGX3C ? 7 : 5);
   return agx3_alu_bit_reg(enc, AGX3_ALU_BAND, reg, src, hints,
                           wait_mask | extra, out, len);
}

/*
 * The AGX3c path above composes that same `bit_and r0, ^r0, r0` specimen, so on
 * G17G, and on G18G through its re-encoder, the carrier ends the lifetime of a
 * register that may still be live. Apple's G18G compiler never emits a
 * register bitop without a lifetime hint: a live source is cached (f25, f39;
 * applegpu_g18g andc32_live `23048083c0200000`), a dead one discarded. This is
 * the live form on both sources. The specimen's own slot-8 wait (f51) stays
 * with the carrier, so the slots it retires are unchanged.
 */
bool
agx3_alu_band_identity_live_gpu1(uint16_t reg, uint16_t wait_mask,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return bit_reg_gpu1(AGX3_MN_bit_and, 1, 10, !(reg & 1), reg, false, reg,
                       AGX3_ALU_HINT_CACHE, reg, AGX3_ALU_HINT_CACHE,
                       wait_mask | (UINT16_C(1) << 7), out, len);
}

bool
agx3_alu_bit_imm(enum agx_isa_encoding enc, enum agx3_alu_op op, bool narrow,
                 uint16_t dest, uint16_t src, uint8_t immediate,
                 enum agx3_alu_hint hint, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* AGX3C uses its own table entry for the same fields. Native G17G
    * witnesses (the native G17G compiler, port/cwit k_or1/k_and255/
    * k_xor1/k_or1h) carry f37 = 1 and the constant in f29 on that point:
    *
    *   a | 1             2b0007803238a3024480   bit_or  f29=1   f43=16
    *   a & 0xff          2b0007803030a302c4bf   bit_and f29=255 f43=10
    *   a ^ 1             2b0006803238a3024480   bit_xor f29=1   f43=19
    *   (ushort) a | 1    3b0207003228a3024480   bit_or  f10=1 f24=1, f38=2
    *
    * each after a load whose slot-7 wait is f51. */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const uint16_t waits = m5 ? 0xff : 0x3f; /* eight slots on AGX3C, six on AGX3B */
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len ||
       (op != AGX3_ALU_BAND && op != AGX3_ALU_BOR && op != AGX3_ALU_BXOR) ||
       dest >= 256 || (!narrow && (dest & 1)) || src >= 256 ||
       (!narrow && (src & 1)) || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~waits))
      return false;

   const struct agx3_alu_form *f = &agx3_alu_forms[op][m5 ? 1 : 0];
   struct agx3_field_kv base[20], ov[12];
   unsigned nb = 0, n = 0;
   for (unsigned i = 0; i < f->nbase; ++i) {
      unsigned field = f->base[i].f;
      bool wait = false;
      for (unsigned j = 0; j < f->nwait; ++j)
         wait |= field == f->wait[j];
      /* f27 and f41 are the two sources' last-use hints and f37 the second
       * operand's kind: all three are deposited below. f38 is that operand's
       * width and stays 2 even for a narrow result, which is what Apple's own
       * `(ushort)a & 0xff` shows. */
      if (field != 27 && field != 41 && field != 37 && !wait)
         base[nb++] = (struct agx3_field_kv){
            field, narrow && (field == 10 || field == 24) ? 1 : f->base[i].v};
   }
   ov[n++] = (struct agx3_field_kv){2, dest};
   ov[n++] = (struct agx3_field_kv){16, src};
   ov[n++] = (struct agx3_field_kv){25, hint == AGX3_ALU_HINT_CACHE};
   ov[n++] = (struct agx3_field_kv){27, hint == AGX3_ALU_HINT_DISCARD};
   ov[n++] = (struct agx3_field_kv){37, 1};
   ov[n++] = (struct agx3_field_kv){29, immediate};
   for (unsigned i = 0; i < f->nwait; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){f->wait[i], 1};
   return agx3_bind_compose(enc, f->mnemonic, f->ep, base, nb, ov, n, out,
                            len);
}

bool
agx3_alu_bit_reg16(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_width(enc, op, true, dest, src, hints, wait_mask, out, len);
}

/* Native G17G multisample DEPTHF payloads select a low coverage bit with
 * AND and retain SR39's API byte with ANDC(0xff). Both are half operations;
 * their destination/source lifetime and eight wait fields are independent. */
bool
agx3_alu_and_imm16_m5(uint16_t dest, uint16_t src, uint8_t immediate,
                     bool complement, bool immediate_first,
                     enum agx3_alu_hint hint,
                     bool dest_cache, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 || wait_mask > 255 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (immediate_first && !complement))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, immediate_first ? 1 : 2},
      {24, immediate_first ? 2 : 1}, {37, immediate_first ? 2 : 1},
      {38, immediate_first ? 1 : 2},
      {42, 3}, {43, complement ? 11 : 10},
   };
   struct agx3_field_kv fields[15] = {
      {2, dest}, {11, dest_cache},
      {immediate_first ? 30 : 16, src},
      {immediate_first ? 15 : 29, immediate},
      {immediate_first ? 39 : 25, hint == AGX3_ALU_HINT_CACHE},
      {immediate_first ? 41 : 27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 6;
   for (unsigned slot = 0; slot < 8; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){44 + slot, 1};
   enum agx3_mnemonic name = complement ? AGX3_MN_bit_andc : AGX3_MN_bit_and;
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, name, 0, base, 8,
                             fields, n, out, len) ||
          agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, name, 1, base, 8,
                             fields, n, out, len);
}

bool
agx3_alu_complement_reg16_m5(uint16_t dest, const uint16_t src[2],
                            const enum agx3_alu_hint hints[2],
                            bool disjunction, bool dest_cache,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src || !hints || !out || !len || dest >= 256 || src[0] >= 256 ||
       src[1] >= 256 || wait_mask > 255 ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {37, 2}, {38, 1},
      {42, 3}, {43, disjunction ? 17 : 11},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {11, dest_cache}, {16, src[0]}, {30, src[1]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {39, hints[1] == AGX3_ALU_HINT_CACHE},
      {41, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 8; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){44 + slot, 1};
   enum agx3_mnemonic name = disjunction ? AGX3_MN_bit_orc : AGX3_MN_bit_andc;
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, name, 0, base, 8,
                             fields, n, out, len) ||
          agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, name, 1, base, 8,
                             fields, n, out, len);
}

bool
agx3_alu_bit_not_m5(uint16_t dest, uint16_t src, bool word,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 ||
       (word && ((dest | src) & 1)) ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_CACHE &&
        hint != AGX3_ALU_HINT_DISCARD) || (wait_mask & ~UINT16_C(0xff)))
      return false;

   /* Native G17G `~x` on ushort and uint (port/cwit3 k_not_u16/u32): the
    * eight-byte `inv` with C's form constants f28 = 3, f29 = 93, result and
    * source widths f10/f24, source cache/discard f25/f27 and the eight
    * scoreboard slots f30..f37 (bit-flip map of the witness):
    *   3b0226182108a002  h2 <- ~^h2, wait slot 8
    *   2b00261a2108a802  r0 <- ~^r0, wait slot 8 */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, word ? 2 : 1}, {23, 2}, {24, word ? 2 : 1},
      {28, 3}, {29, 93},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 8; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){30 + slot, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_inv, 0, base,
                            sizeof(base) / sizeof(base[0]), fields, n, out,
                            len);
}

bool
agx3_alu_bit_not_reg16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){30 + slot,
                                          !!(wait_mask & (1u << slot))};
   return agx3_bind_compose(enc, AGX3_MN_inv, 0, base,
                            sizeof(base) / sizeof(base[0]), fields, n, out,
                            len);
}

/* The two-register bitwise instructions beyond AND/OR/XOR share the
 * ordinary bitops' operand, hint and wait topology; only the mnemonic
 * carries the truth table. */
static bool
agx3_alu_bit_named_width(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
                   uint16_t dest, const uint16_t src[2],
                   const enum agx3_alu_hint hints[2], const bool half[2], bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !src || !hints || !out || !len ||
       dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       ((dest & 1) || (!half[0] && (src[0] & 1)) ||
        (!half[1] && (src[1] & 1))) || (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;

   const unsigned table = mnemonic == AGX3_MN_bit_andc ? 11 :
      mnemonic == AGX3_MN_bit_nor ? 14 : mnemonic == AGX3_MN_bit_nand ? 12 :
      mnemonic == AGX3_MN_bit_xnor ? 18 : mnemonic == AGX3_MN_bit_orc ? 17 : 0;
   if (m5 && !table)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, half[0] ? 1 : 2},
      {37, 2}, {38, half[1] ? 1 : 2}, {42, 3}, {43, table},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {16, src[0]}, {30, src[1]}, {11, dest_cache},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {39, hints[1] == AGX3_ALU_HINT_CACHE},
      {41, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      fields[n++] = (struct agx3_field_kv){44 + slot,
                                          !!(wait_mask & (1u << slot))};
   return agx3_bind_compose(enc, mnemonic, 0, base, m5 ? 8 : 6, fields, n, out, len) ||
          (m5 && agx3_bind_compose(enc, mnemonic, 1, base, 8, fields, n, out, len));
}

static bool
agx3_alu_bit_named(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
                   uint16_t dest, const uint16_t src[2],
                   const enum agx3_alu_hint hints[2], bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   const bool half[] = {false, false};
   return agx3_alu_bit_named_width(enc, mnemonic, dest, src, hints, half,
                                  dest_cache, wait_mask, out, len);
}

bool
agx3_alu_bit_andc_width_m5(uint16_t dest, const uint16_t src[2],
                           const bool half[2], const enum agx3_alu_hint hints[2],
                           bool dest_cache, uint16_t wait_mask,
                           uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_named_width(AGX_ISA_ENCODING_AGX3C, AGX3_MN_bit_andc, dest,
                                  src, hints, half, dest_cache, wait_mask,
                                  out, len);
}

bool
agx3_alu_bit_andc(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[2],
                  const enum agx3_alu_hint hints[2], bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   /* Apple `v.x & ~v.y`:
    *   0b010e03020000800000  bit_andc r0, ^r0, ^r1
    * Keeping the first input independently gives:
    *   0b031605020000800000  bit_andc r0, r1, ^r2 */
   return agx3_alu_bit_named(enc, AGX3_MN_bit_andc, dest, src, hints, dest_cache,
                             wait_mask, out, len);
}

bool
agx3_alu_bit_nor(enum agx_isa_encoding enc, uint16_t dest,
                 const uint16_t src[2], const enum agx3_alu_hint hints[2],
                 bool dest_cache, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Apple `~(x | y)` of two buffer-loaded uints, slot-6 wait:
    *   2b830681010000800000  bit_nor r2, $r1, $r0
    *   2b030e01010000800000  bit_nor r2, ^r1, r0
    *   2b011603010000800000  bit_nor r2, r0, ^r1 */
   return agx3_alu_bit_named(enc, AGX3_MN_bit_nor, dest, src, hints, dest_cache,
                             wait_mask, out, len);
}

/* Apple's two-register logic ops of uint attachments (blend/native/bitops.py),
 * r2 = a, r0 = b, both dead, slot-6 wait:
 *   ~(a & b)   0b031e01030800800000  bit_nand
 *   ~(a ^ b)   0b031f01010000800000  bit_xnor
 *   a | ~b     0b011f03030000800000  bit_orc r0, ^r0, ^r2 (a = r0, b = r2)
 *   ~a | b     0b031f01030000800000  bit_orc r0, ^r2, ^r0
 * bit_orc keeps its operand order, like bit_andc. */
bool
agx3_alu_bit_nand(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[2], const enum agx3_alu_hint hints[2],
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_named(enc, AGX3_MN_bit_nand, dest, src, hints, dest_cache,
                             wait_mask, out, len);
}

bool
agx3_alu_bit_xnor(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[2], const enum agx3_alu_hint hints[2],
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_named(enc, AGX3_MN_bit_xnor, dest, src, hints, dest_cache,
                             wait_mask, out, len);
}

bool
agx3_alu_bit_orc(enum agx_isa_encoding enc, uint16_t dest,
                 const uint16_t src[2], const enum agx3_alu_hint hints[2],
                 bool dest_cache, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_named(enc, AGX3_MN_bit_orc, dest, src, hints, dest_cache,
                             wait_mask, out, len);
}

/* Native iadd_live/isub_live retain both inputs, while their dead variants
 * set f27/f42. Keep the same canonical opcode/register fields. G17G spells
 * the same pair on iadd32/isub32 and the compose corpus holds Apple's own
 * instructions with each combination of the two, which is what admits the
 * G17G forms here rather than through the generic template whose baseline
 * pins f27. */
bool
agx3_alu_integer_reg(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_integer_reg_shift(enc, op, dest, src, hints, 0, wait_mask,
                                     out, len);
}

/*
 * The same two forms with slot B shifted left first (f54).
 *
 * `a - (b << N)` is how a multiply by a small negative constant is strength
 * reduced -- `x * -7` becomes `x - (x << 3)` -- so this is not a rare shape,
 * and without it every such multiply failed the whole shader closed. Note it
 * is NOT a degenerate self-subtraction: with the shift it computes -7x.
 *
 * Witnessed against Apple, one kernel per shift, everything else identical
 * (`o[i] = a[i] - (c[i] << N)`, `metal-nt -arch applegpu_g17p`):
 *
 *   N = 0   1f 01 56 00 02 00 10 a8 17 05   f54 absent
 *   N = 1   1f 01 56 00 02 00 10 e8 16 05   f54 = 1
 *   N = 2   1f 01 56 00 02 00 10 a8 16 15   f54 = 2
 *   N = 3   1f 01 56 00 02 00 10 e8 16 15   f54 = 3
 *   N = 4   1f 01 56 00 02 00 10 a8 16 05   f54 = 4
 *
 * Apple stops emitting this form at N = 5, so the shift is capped at the four
 * witnessed non-zero amounts rather than extrapolated.
 */
bool
agx3_alu_integer_reg_shift(enum agx_isa_encoding enc, enum agx3_alu_op op,
                 uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], uint8_t shift,
                 uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_integer_reg_shift_cache(enc, op, dest, src, hints, shift,
                                           false, wait_mask, out, len);
}

/* As above, optionally depositing f11 (destination register-cache hint).
 * Native G17P code sets f11 on every integer iadd/isub result, including
 * loop-carried ones (captured throughput fragment shader, 0xa8..0xe2).
 */
bool
agx3_alu_integer_reg_shift_cache(enum agx_isa_encoding enc,
                                 enum agx3_alu_op op, uint16_t dest,
                                 const uint16_t src[2],
                                 const enum agx3_alu_hint hints[2],
                                 uint8_t shift, bool dest_cache,
                                 uint16_t wait_mask,
                                 uint8_t out[AGX3_ALU_MAX_LENGTH],
                                 uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || !src || !hints || !out ||
      !len || (op != AGX3_ALU_IADD && op != AGX3_ALU_ISUB) ||
      dest >= 256 || (dest & 1) || (gpu1 && dest_cache) ||
      (wait_mask & ~(uint16_t)(gpu1 ? 0xff : 0x3f)))
      return false;
   /* Only the subtract's shift is witnessed; the add reaches its shifted
    * form through agx3_iadd_encode, which models it separately. G17G's
    * shifted subtract is not witnessed at all. */
   if (shift != 0 && (gpu1 || op != AGX3_ALU_ISUB || shift > 4))
      return false;
   for(unsigned i=0;i<2;i++)
      if(src[i]>=256 || (src[i]&1) || (unsigned)hints[i]>AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_alu_form *f=&agx3_alu_forms[op][gpu1];
   struct agx3_field_kv base[20],ov[13];unsigned nb=0,n=0;
   for(unsigned i=0;i<f->nbase;i++){
      unsigned field=f->base[i].f;bool wait=false;
      for(unsigned j=0;j<f->nwait;j++)wait|=field==f->wait[j];
      if(field!=27 && field!=42 && !wait)
         base[nb++]=(struct agx3_field_kv){field,f->base[i].v};
   }
   ov[n++]=(struct agx3_field_kv){2,dest};
   const unsigned operand[]={16,31},cache[]={25,40},discard[]={27,42};
   for(unsigned i=0;i<2;i++){
      ov[n++]=(struct agx3_field_kv){operand[i],src[i]};
      ov[n++]=(struct agx3_field_kv){cache[i],hints[i]==AGX3_ALU_HINT_CACHE};
      ov[n++]=(struct agx3_field_kv){discard[i],hints[i]==AGX3_ALU_HINT_DISCARD};
   }
   for(unsigned i=0;i<f->nwait;i++)
      if(wait_mask&(1u<<i))ov[n++]=(struct agx3_field_kv){f->wait[i],1};
   /* f54, slot B's shift. Zero is the form's own absence, so only deposit a
    * real shift; that keeps every previously encodable add byte-identical. */
   if(shift)ov[n++]=(struct agx3_field_kv){54,shift};
   if(dest_cache)ov[n++]=(struct agx3_field_kv){11,1};
   return agx3_bind_compose(enc,f->mnemonic,f->ep,base,nb,ov,n,out,len);
}

/* Native float comparison selects two integer immediates. Field 11 selects
 * the result width; a 16-bit result can target either register half. */
static bool
agx3_alu_fcmp_select_imm_width_internal(enum agx_isa_encoding enc,
                        uint16_t dest,
                        const uint16_t src[2], const bool src_16[2],
                        const enum agx3_alu_hint hints[2],
                        unsigned condition, uint8_t if_true, uint8_t if_false,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        bool src1_neg, const bool src_abs[2],
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !src_16 || !hints ||
       dest >= 256 || (!dest_16 && (dest & 1)) || (condition != 0 && condition != 1 && condition != 2 &&
       condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   /* f65 negates the SECOND comparator and nothing else: it is an ordinary
    * field in the same encoding point, orthogonal to the condition, to the
    * result width and to the two selected immediates.
    *
    * Grounded with metal-nt -arch applegpu_g17p, one kernel per cell, every
    * vector recovered by re-composing through this encoder. The reproducer's
    * own shape -- a 32-bit result selecting 1 and 0 -- across every relation:
    *
    *    a >  b      02011d03228102c02080   f35=0 f52=2 f110=2
    *   -a >  b      02031d01228103c82080   f35=2 f52=0 f110=1 f65
    *   -a <  b      02031d01228102c82080   f35=2 f52=0 f110=2 f65
    *   -a >= b      02031d01268103c82080   f35=2 f52=0 f110=6 f65
    *   -a <= b      02031d01268102c82080   f35=2 f52=0 f110=5 f65
    *   -a == b      02031d01268100c82080   f35=2 f52=0 f110=0 f65
    *    a < -b      02011d03228103c82080   f35=0 f52=2 f110=1 f65
    *
    * The first four are the SWAP: the two comparator registers exchange
    * places and the relation reverses, which is how a negated FIRST source
    * reaches an encoding that has no bit for it. The caller performs that
    * rewrite; this entry point only ever sees the negate on slot 1.
    *
    * Two controls make the rewrite a fact rather than a reading. `a < -b`
    * sets f65 with the operands in place, so f65 really is slot 1's. And
    * `-a < -b` comes back as 02011d03228102c02080 -- byte-identical to
    * `a > b`, with no f65 at all -- so negating both sides is the relation
    * flip alone, which is the same algebra the swap relies on.
    *
    * A fresh half/half comparison selecting a 32-bit integer is the same
    * endpoint with both comparator width fields set to 1:
    *
    *    a < b ? 2u : 0u   02001f00228213c02080   f43=1 f60=1
    *    a < b ? 7u : 3u   02001f00228713c02083   f43=1 f60=1
    *
    * so the two half widths compose independently of the 32-bit destination
    * and selected immediates. This is the exact shaderFloat16 comparison
    * shape used by the Vulkan precision tests.
    *
    * Mixing the negate with the absolute modifiers is still not witnessed:
    * ABS moves to a longer endpoint that has no f65, so that stays refused
    * in the caller rather than silently dropping a modifier. */
   for (unsigned i = 0; i < 2; ++i)
      if (src[i] >= 256 || (!src_16[i] && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, dest_16 ? 1 : 2}, {42, 2}, {43, src_16[0] ? 1 : 2},
      {59, 2}, {60, src_16[1] ? 1 : 2},
      {76, 1}, {77, dest_16 ? 1 : 2}, {93, 1}, {94, dest_16 ? 1 : 2},
   };
   const struct agx3_field_kv fields[] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {61, hints[1] == AGX3_ALU_HINT_CACHE},
      {63, hints[1] == AGX3_ALU_HINT_DISCARD}, {65, src1_neg},
      {47, src_abs[0]}, {64, src_abs[1]},
      {68, if_true}, {85, if_false}, {110, condition}, {107, wait_mask == 32},
   };
   /* An absolute source moves to the fourteen-byte ep28..35 endpoints; the
    * composer's predicates reject f47/f64 on the ten-byte ones. */
   for (unsigned ep = 20; ep < 36; ++ep)
      if (agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, 10, fields, 15, out, len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_select_imm_width(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2], const bool src_16[2],
                        const enum agx3_alu_hint hints[2],
                        unsigned condition, uint8_t if_true, uint8_t if_false,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_fcmp_select_imm_width_internal(
      enc, dest, src, src_16, hints, condition, if_true, if_false, dest_16,
      dest_cache, wait_mask, false, (const bool[2]){false, false}, out, len);
}

/* Reverse a comparison so its two operands can exchange places. */
static unsigned
agx3_fcond_reversed(unsigned condition)
{
   switch (condition) {
   case 1: return 2;  /* lt -> gt */
   case 2: return 1;  /* gt -> lt */
   case 5: return 6;  /* ge -> le */
   case 6: return 5;  /* le -> ge */
   default: return condition; /* eq is its own reverse */
   }
}

/*
 * The same compare-select with an ABSOLUTE VALUE on either comparison source.
 *
 * This is NOT a modifier bit on the ten-byte form: Apple moves to the
 * fourteen-byte ep28..35 endpoints, which carry f47 and f64, the two sources'
 * absolute-value bits. NEG stays on the ten-byte endpoint; ABS never does.
 *
 * Those endpoints place the destination block in one of two ways, and the
 * placement decides how an immediate select is read. With dest f3, width f11
 * and cache f12 the selects are integers; with f0 set and dest f17, width f25
 * and cache f26 they are minifloats. metal-nt -arch applegpu_g17p kernels
 * with the same float32 comparison, destination and hints:
 *
 *   x + ((fabs(a) < b) ? 3u : 1u)       22 05 3f 07 23 83 03 00 20 81 00 80 01 00
 *   x + ((fabs(a) < b) ? 1.0f : 0.0f)   22 05 39 07 23 b0 03 00 20 80 00 80 01 00
 *
 * (f68 = 3 against the minifloat 48 of 1.0.) FCMPSEL's selected immediates
 * are integer bit patterns, so they take the integer placement. The rest of
 * the integer-select ladder, one kernel per field, dest r2 with cache,
 * comparisons r2 and r3 with last use, wait slot 6:
 *
 *   ( a < fabs(b)) ? 3u : 1u        22 05 3f 07 23 83 03 00 20 81 00 80 02 00
 *   (fabs(a) < fabs(b)) ? 3u : 1u   22 05 3f 07 23 83 03 00 20 81 00 80 03 00
 *   (fabs(a) >  b) ? 3u : 1u        22 05 3f 07 23 83 02 00 20 81 00 80 01 00
 *   (fabs(a) == b) ? 3u : 1u        22 05 3f 07 27 83 00 00 20 81 00 80 01 00
 *   (fabs(a) >= b) ? 3u : 1u        22 05 3f 07 27 83 02 00 20 81 00 80 01 00
 *   (fabs(a) <= b) ? 3u : 1u        22 05 3f 07 27 83 03 00 20 81 00 80 01 00
 *   (fabs(a) <  b) ? 0u : 7u        22 05 3f 07 23 80 03 00 20 87 00 80 01 00
 *   16-bit result, 3 and 1          12 03 3d 05 23 83 03 00 20 81 00 80 01 00
 *   half/half, 16-bit result        1a 02 3d 04 23 83 0b 00 20 81 00 80 01 00
 *   half/half, fabs(b), 32-bit      22 04 3f 04 23 83 13 00 20 81 00 80 02 00
 *   float/half, 32-bit result       22 05 3f 06 23 83 03 00 20 81 00 80 01 00
 *
 * so f110 is the condition, f68 and f85 the two integer selects, f43 and f60
 * the comparators' widths and f44/f46 and f61/f63 their cache and last-use
 * hints, exactly as on the ten-byte endpoint.
 */
bool
agx3_alu_fcmp_select(enum agx_isa_encoding enc, const struct agx3_fsel_desc *in,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !in || !out || !len)
      return false;

   struct agx3_fsel_desc d = *in;

   if ((d.sel0_kind != AGX3_FSEL_SEL_IMMEDIATE &&
        d.sel0_kind != AGX3_FSEL_SEL_REGISTER &&
        d.sel0_kind != AGX3_FSEL_SEL_UNIFORM) ||
       (d.sel1_kind != AGX3_FSEL_SEL_IMMEDIATE &&
        d.sel1_kind != AGX3_FSEL_SEL_REGISTER &&
        d.sel1_kind != AGX3_FSEL_SEL_UNIFORM))
      return false;

   if (d.condition != 0 && d.condition != 1 && d.condition != 2 &&
       d.condition != 5 && d.condition != 6)
      return false;
   /* Native corpus rows 222/429 carry slot 5 on this long endpoint; the
    * existing slot 6 form remains unchanged. */
   if (d.wait_mask != 0 && d.wait_mask != 16 && d.wait_mask != 32)
      return false;
   if (d.dest >= 256 || (!d.dest_16 && (d.dest & 1)))
      return false;

   /*
    * Normalise the comparison modifiers.
    *
    * f65 is the only negate the encoding has and it names the SECOND
    * comparator. Every other placement reduces to it by algebra that Apple's
    * own compiler performs, so the shapes below are rewrites rather than
    * refusals. Witnessed with metal-nt -arch applegpu_g17p on a select whose
    * true value is a REGISTER -- the shape this entry point exists for --
    * one kernel per cell, each vector recovered by re-composing here:
    *
    *    a >= c      12051d07850302c0   f35=4 f52=6 f110=5
    *    a >= -c     12051d07850302c8   f35=4 f52=6 f110=5 f65
    *   -a >= c      12071d05850303c8   f35=6 f52=4 f110=6 f65
    *   -a >= -c     12051d07850303c0   f35=4 f52=6 f110=6
    *    a <= c      12051d07850303c0   byte-identical to the row above
    *
    * so a negated first comparator exchanges the operands and reverses the
    * relation, and negating both sides is the relation flip on its own. The
    * last two rows are the same instruction, which is what makes the
    * double-negation rewrite a measurement rather than an assumption.
    *
    * An immediate comparator carries its sign in bit 7 of the minifloat, so a
    * modifier on that slot is constant folding: Apple spells `-a >= 0.5f` as
    * `a <= -0.5f`, 12051da1850303c2 with f51 = 160 and f110 = 6 against
    * 12050da1850302c2 with f51 = 32 and f110 = 5 for `a >= 0.5f`.
    */
   /* A uniform comparand has no modifier or lifetime fields; normalisation
    * below never moves slot 0 when it carries no negate. */
   if (d.cmp0_uniform &&
       (d.cmp0_abs || d.cmp0_neg || d.cmp0_hint != AGX3_ALU_HINT_NONE))
      return false;
   /* Slot 1 takes a uniform the same way (Subnautica/Fallout NV native
    * corpus: 14,500 fsel with f59 = 3, f53 = the uniform half, f60 = its
    * width, e.g. b217677f822002020016 for r22 < u126 ? r32 : r22 and
    * c2190b4427800002802e00000100 for |r24| ? u68(half) with f47 = 1).
    * Nothing witnessed carries two uniform comparands or a modifier on
    * one. */
   if (d.cmp1_uniform &&
       (d.cmp1_is_immediate || d.cmp0_uniform || d.cmp1_abs || d.cmp1_neg ||
        d.cmp1_hint != AGX3_ALU_HINT_NONE))
      return false;

   if (d.cmp1_is_immediate) {
      if (d.cmp1 > 0xff)
         return false;
      if (d.cmp1_abs) {
         d.cmp1 &= 0x7f;
         d.cmp1_abs = false;
      }
      if (d.cmp1_neg) {
         d.cmp1 ^= 0x80;
         d.cmp1_neg = false;
      }
      if (d.cmp0_neg) {
         d.condition = agx3_fcond_reversed(d.condition);
         d.cmp1 ^= 0x80;
         d.cmp0_neg = false;
      }
   } else if (d.cmp0_neg && d.cmp1_neg) {
      d.condition = agx3_fcond_reversed(d.condition);
      d.cmp0_neg = d.cmp1_neg = false;
   } else if (d.cmp0_neg) {
      /* The exchange would move a slot-1 uniform into slot 0 beside a
       * negated register, a pairing with no witness; the packer keeps
       * such uniforms in a register. */
      if (d.cmp1_uniform)
         return false;
      const uint16_t reg = d.cmp0;
      const enum agx3_alu_hint hint = d.cmp0_hint;
      const bool abs = d.cmp0_abs, narrow = d.cmp0_16;
      d.cmp0 = d.cmp1;
      d.cmp0_hint = d.cmp1_hint;
      d.cmp0_abs = d.cmp1_abs;
      d.cmp0_16 = d.cmp1_16;
      d.cmp1 = reg;
      d.cmp1_hint = hint;
      d.cmp1_abs = abs;
      d.cmp1_16 = narrow;
      d.condition = agx3_fcond_reversed(d.condition);
      d.cmp0_neg = false;
      d.cmp1_neg = true;
   }

   /* Every path above clears it; nothing may reach the deposit with a negate
    * on the slot that has no field for one. */
   if (d.cmp0_neg)
      return false;

   /* An immediate compare slot still declares width 2 (see the base vector
    * below), so only a register slot carries a half width. */
   const bool cmp0_16 = d.cmp0_16;
   const bool cmp1_16 = d.cmp1_is_immediate ? false : d.cmp1_16;

   /* Both comparator widths remain independent when a selected value is a
    * register. Fresh applegpu_g17p kernels selecting between a 32-bit GPR and
    * immediate 32 use the same field map and set f43=f60=1:
    *
    *    a < b ? 32u : v   02041f0421a013c0   ep19
    *    a < b ? v : 32u   02041f0420a013c0   ep13
    *
    * This is the vector shaderFloat16 comparison shape emitted by NIR. */

   /* A 32-bit source is a register pair and must start on an even half; a
    * 16-bit source is a single half and may sit on an odd one.  Pinning both
    * to even is what refused every mediump comparator that reached here. */
   if (d.cmp0 >= 256 || (!cmp0_16 && (d.cmp0 & 1)))
      return false;
   /* A uniform half index is bounded by its field: the composer refuses an
    * over-wide f53 (the corpus witnesses up to 382). */
   if (!d.cmp1_is_immediate && !d.cmp1_uniform &&
       (d.cmp1 >= 256 || (!cmp1_16 && (d.cmp1 & 1))))
      return false;
   if (d.cmp1_uniform && !cmp1_16 && (d.cmp1 & 1))
      return false;
   if (d.cmp1_is_immediate && d.cmp1 > 0xff)
      return false;
   if (d.sel0 >= 256 || d.sel1 >= 256)
      return false;
   if (d.sel0_kind != AGX3_FSEL_SEL_IMMEDIATE && !d.sel_16 && (d.sel0 & 1))
      return false;
   if (d.sel1_kind != AGX3_FSEL_SEL_IMMEDIATE && !d.sel_16 && (d.sel1 & 1))
      return false;
   if ((unsigned)d.cmp0_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)d.cmp1_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)d.sel0_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)d.sel1_hint > AGX3_ALU_HINT_CACHE ||
       (d.sel0_kind != AGX3_FSEL_SEL_REGISTER && d.sel0_hint != AGX3_ALU_HINT_NONE) ||
       (d.sel1_kind != AGX3_FSEL_SEL_REGISTER && d.sel1_hint != AGX3_ALU_HINT_NONE))
      return false;

   /* f43 and f60 are the two comparators' width fields, exactly as in
    * agx3_alu_fcmp_select_imm_width_internal() above, which grounds them on
    * the same AGX3_MN_fsel endpoint family.  An immediate slot always declares
    * width 2. */
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, d.dest_16 ? 1 : 2},
      {42, d.cmp0_uniform ? AGX3_FSEL_CMP_UNIFORM : AGX3_FSEL_CMP_REGISTER},
      {43, cmp0_16 ? 1 : 2},
      {60, cmp1_16 ? 1 : 2},
      {77, d.sel_16 ? 1 : 2}, {94, d.sel_16 ? 1 : 2},
   };

   struct agx3_field_kv f[25];
   unsigned n = 0;
   f[n++] = (struct agx3_field_kv){3, d.dest};
   f[n++] = (struct agx3_field_kv){12, d.dest_cache};

   /* A uniform comparand, as Apple selects on a constant-program flag
    * (applegpu_g17p, `f ? x : a` with f in a constant buffer):
    *
    *    fsel d = (u17l == 0.0) ? r40 : r48   0222e78185508002   f36=34 f42=3 f43=1
    */
   if (d.cmp0_uniform) {
      f[n++] = (struct agx3_field_kv){36, d.cmp0};
   } else {
      f[n++] = (struct agx3_field_kv){35, d.cmp0};
      f[n++] = (struct agx3_field_kv){44,
                                      d.cmp0_hint == AGX3_ALU_HINT_CACHE};
      f[n++] = (struct agx3_field_kv){46,
                                      d.cmp0_hint == AGX3_ALU_HINT_DISCARD};
      f[n++] = (struct agx3_field_kv){47, d.cmp0_abs};
   }

   f[n++] = (struct agx3_field_kv){59, d.cmp1_is_immediate
                                          ? AGX3_FSEL_CMP_IMMEDIATE
                                          : d.cmp1_uniform
                                             ? AGX3_FSEL_CMP_UNIFORM
                                             : AGX3_FSEL_CMP_REGISTER};
   if (d.cmp1_uniform) {
      f[n++] = (struct agx3_field_kv){53, d.cmp1};
   } else {
      f[n++] = (struct agx3_field_kv){d.cmp1_is_immediate ? 51 : 52, d.cmp1};
      f[n++] = (struct agx3_field_kv){61,
                                      d.cmp1_hint == AGX3_ALU_HINT_CACHE};
      f[n++] = (struct agx3_field_kv){63,
                                      d.cmp1_hint == AGX3_ALU_HINT_DISCARD};
      f[n++] = (struct agx3_field_kv){64, d.cmp1_abs};
      f[n++] = (struct agx3_field_kv){65, d.cmp1_neg};
   }

   f[n++] = (struct agx3_field_kv){76, d.sel0_kind};
   f[n++] = (struct agx3_field_kv){
      d.sel0_kind == AGX3_FSEL_SEL_REGISTER ? 69 :
      d.sel0_kind == AGX3_FSEL_SEL_UNIFORM ? 70 : 68, d.sel0};
   f[n++] = (struct agx3_field_kv){93, d.sel1_kind};
   f[n++] = (struct agx3_field_kv){
      d.sel1_kind == AGX3_FSEL_SEL_REGISTER ? 86 :
      d.sel1_kind == AGX3_FSEL_SEL_UNIFORM ? 87 : 85, d.sel1};
   f[n++] = (struct agx3_field_kv){78, d.sel0_hint == AGX3_ALU_HINT_CACHE};
   f[n++] = (struct agx3_field_kv){80, d.sel0_hint == AGX3_ALU_HINT_DISCARD};
   f[n++] = (struct agx3_field_kv){95, d.sel1_hint == AGX3_ALU_HINT_CACHE};
   f[n++] = (struct agx3_field_kv){97, d.sel1_hint == AGX3_ALU_HINT_DISCARD};

   f[n++] = (struct agx3_field_kv){110, d.condition};
   f[n++] = (struct agx3_field_kv){106, d.wait_mask == 16};
   f[n++] = (struct agx3_field_kv){107, d.wait_mask == 32};

   /*
    * The destination block has TWO placements on this endpoint family: dest
    * f3 -> f17, its width f11 -> f25, its cache hint f12 -> f26, with f0 and
    * f24 fixed. Apple emits both --
    *
    *   (fabs(a) < 0.5f) ? 1.0h : 0.0h   120309a123b00302208000800100   ep28
    *   corpus vector 224                42d725d923810300208000000300   ep33
    *
    * -- but they are not interchangeable for a non-zero immediate select: the
    * f0 placement reads f68/f85 as minifloats and the f3 placement as
    * integers (see agx3_alu_fcmp_select_imm_width_abs() for the witnessed
    * pair). FCMPSEL selects integer bit patterns, so pick f3/f11 first and
    * fall back to the shifted block only when every immediate select is zero,
    * which is the same bits in either.
    *
    * What the first row does settle is that an IMMEDIATE comparator is not a
    * separate form: against (fabs(a) < c) ? 1.0h : 0.0h, which is
    * 1203190523b00300208000800100, it changes only f59 (2 -> 1) and the
    * operand field (f52 -> f51, holding the minifloat 32). Refusing every
    * non-zero immediate comparator recorded what had been measured, not what
    * the field can hold.
    */
   const bool integer_select =
      (d.sel0_kind == AGX3_FSEL_SEL_IMMEDIATE && d.sel0 != 0) ||
      (d.sel1_kind == AGX3_FSEL_SEL_IMMEDIATE && d.sel1 != 0);
   struct agx3_field_kv alt_base[9];
   unsigned nab = 0;
   alt_base[nab++] = (struct agx3_field_kv){0, 1};
   alt_base[nab++] = (struct agx3_field_kv){24, 2};
   alt_base[nab++] = (struct agx3_field_kv){25, d.dest_16 ? 1 : 2};
   alt_base[nab++] = (struct agx3_field_kv){
      42, d.cmp0_uniform ? AGX3_FSEL_CMP_UNIFORM : AGX3_FSEL_CMP_REGISTER};
   alt_base[nab++] = (struct agx3_field_kv){43, cmp0_16 ? 1 : 2};
   alt_base[nab++] = (struct agx3_field_kv){60, cmp1_16 ? 1 : 2};
   alt_base[nab++] = (struct agx3_field_kv){77, d.sel_16 ? 1 : 2};
   alt_base[nab++] = (struct agx3_field_kv){94, d.sel_16 ? 1 : 2};

   /* With a uniform comparand and a zero alternative Apple uses the shifted
    * block (form 130 / 124 in the native corpus); keep the f3 block first
    * everywhere else so existing encodings stay byte-identical. */
   const bool shifted_first =
      d.cmp0_uniform && !integer_select &&
      (d.sel0_kind == AGX3_FSEL_SEL_IMMEDIATE ||
       d.sel1_kind == AGX3_FSEL_SEL_IMMEDIATE);
   for (unsigned pass = 0; pass < (integer_select ? 1 : 2); ++pass) {
      const bool shifted = (pass == 1) != shifted_first;
      f[0] = (struct agx3_field_kv){shifted ? 17 : 3, d.dest};
      f[1] = (struct agx3_field_kv){shifted ? 26 : 12, d.dest_cache};
      /* Corpus rows 546/634 use the tied-result GPR forms 13/19. Let the
       * native predicates select them instead of growing an 8-byte select.
       * Apple's 16-bit select on a uniform comparand is endpoint 11. */
      for (unsigned ep = d.cmp0_uniform ? 10 : 13; ep < 36; ++ep) {
         if (agx3_bind_compose(enc, AGX3_MN_fsel, ep, shifted ? alt_base : base,
                               shifted ? nab : 7, f, n, out, len))
            return true;
      }
   }
   return false;
}

bool
agx3_alu_fcmp_select_imm_width_abs(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2], const bool src_16[2],
                        const enum agx3_alu_hint hints[2],
                        unsigned condition, uint8_t if_true, uint8_t if_false,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        const bool src_abs[2],
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* At least one absolute source: without one this is the shorter endpoint,
    * which agx3_alu_fcmp_select_imm_width() owns. */
   if (!src_abs || (!src_abs[0] && !src_abs[1]))
      return false;

   return agx3_alu_fcmp_select_imm_width_internal(
      enc, dest, src, src_16, hints, condition, if_true, if_false, dest_16,
      dest_cache, wait_mask, false, src_abs, out, len);
}

/* The same select with the negate on either comparator. A negate on the first
 * one has no bit: it is the operand swap with the relation reversed, which is
 * what `neg[0]` performs here so that no caller has to know the asymmetry.
 * Negating both sides is the relation flip on its own, with no f65 at all. */
bool
agx3_alu_fcmp_select_imm_width_neg(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   const bool neg[2], unsigned condition, uint8_t if_true, uint8_t if_false,
   bool dest_16, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src || !src_16 || !hints || !neg)
      return false;

   uint16_t s[2] = {src[0], src[1]};
   bool w[2] = {src_16[0], src_16[1]};
   enum agx3_alu_hint h[2] = {hints[0], hints[1]};
   bool negate_second = neg[1];

   /* Both negated, or only the first: the relation reverses. Only the
    * first-alone case also exchanges the operands, so that the remaining
    * negate lands on slot 1, the only slot f65 names. */
   if (neg[0]) {
      switch (condition) {
      case 1: condition = 2; break;   /* lt -> gt */
      case 2: condition = 1; break;   /* gt -> lt */
      case 5: condition = 6; break;   /* ge -> le */
      case 6: condition = 5; break;   /* le -> ge */
      case 0: break;                  /* eq is its own reverse */
      default: return false;
      }

      if (!neg[1]) {
         const uint16_t reg = s[0]; s[0] = s[1]; s[1] = reg;
         const bool width = w[0]; w[0] = w[1]; w[1] = width;
         const enum agx3_alu_hint hint = h[0]; h[0] = h[1]; h[1] = hint;
         negate_second = true;
      } else {
         negate_second = false;
      }
   }

   return agx3_alu_fcmp_select_imm_width_internal(
      enc, dest, s, w, h, condition, if_true, if_false, dest_16, dest_cache,
      wait_mask, negate_second, (const bool[2]){false, false}, out, len);
}

bool
agx3_alu_fcmp_minifloat_select_half(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, uint16_t if_true, uint8_t if_false,
   enum agx3_alu_hint compare_hint, enum agx3_alu_hint true_hint,
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       compare >= 256 || (compare & 1) || if_true >= 256 ||
       (unsigned)compare_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)true_hint > AGX3_ALU_HINT_CACHE ||
       condition > 2 ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   /* Native vector `select(half(0), h, x == -1.0f)` lanes:
    *   020119b1860a00c22080
    *   0a0319b1860b00022080
    * field51 is the minifloat compare operand, field69 the selected half
    * register and field85 the false immediate. Those take the f0
    * destination placement, which reads field85 as a minifloat; a non-zero
    * integer needs the f3 placement Apple's ushort kernel uses,
    * `x + ((a == -1.0f) ? h : 5)`: 12033db1860500222085. */
   const bool integer = if_false != 0;
   const struct agx3_field_kv float_base[] = {
      {0, 1}, {24, 2}, {25, 1}, {42, 2}, {43, 2}, {59, 1},
      {60, 2}, {76, 2}, {77, 1}, {93, 1}, {94, 1},
   };
   const struct agx3_field_kv integer_base[] = {
      {10, 2}, {11, 1}, {42, 2}, {43, 2}, {59, 1},
      {60, 2}, {76, 2}, {77, 1}, {93, 1}, {94, 1},
   };
   const struct agx3_field_kv fields[] = {
      {integer ? 3 : 17, dest}, {integer ? 12 : 26, dest_cache},
      {35, compare}, {51, minifloat}, {69, if_true}, {85, if_false},
      {44, compare_hint == AGX3_ALU_HINT_CACHE},
      {46, compare_hint == AGX3_ALU_HINT_DISCARD},
      {78, true_hint == AGX3_ALU_HINT_CACHE},
      {80, true_hint == AGX3_ALU_HINT_DISCARD},
      {107, wait_mask == 32}, {110, condition},
   };
   for (unsigned ep = 20; ep < 28; ++ep)
      if (agx3_bind_compose(enc, AGX3_MN_fsel, ep,
                            integer ? integer_base : float_base,
                            integer ? 10 : 11, fields, 12, out, len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_minifloat_select_mixed(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_16, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !selected ||
       !selected_immediate || !hints || !out || !len || dest >= 256 ||
       compare >= 256 || (compare & 1) || (!dest_16 && (dest & 1)) ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (dest_16 ? (wait_mask > UINT16_C(0x20) ||
                    (wait_mask && (wait_mask & (wait_mask - 1))))
                : (wait_mask != 0 && wait_mask != 32)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE)
      return false;

   /* The observed half result selects two byte constants; the full result
    * selects exactly one full register and one byte constant. */
   if (dest_16 ? (!selected_immediate[0] || !selected_immediate[1])
               : (selected_immediate[0] == selected_immediate[1]))
      return false;

   /* A full result's immediate alternative is floating zero. Fresh native
    * sweeps cover both placements and all five ordered conditions. */
   if (!dest_16 && selected[selected_immediate[0] ? 0 : 1] != 0)
      return false;

   for (unsigned i = 0; i < 2; ++i) {
      if (selected[i] >= 256 ||
          (selected_immediate[i]
              ? hints[i + 1] != AGX3_ALU_HINT_NONE
              : ((!dest_16 && (selected[i] & 1)) ||
                 (unsigned)hints[i + 1] > AGX3_ALU_HINT_CACHE)))
         return false;
   }

   struct agx3_field_kv base[11];
   unsigned nbase = 0;
   if (dest_16) {
      base[nbase++] = (struct agx3_field_kv){10, 2};
      base[nbase++] = (struct agx3_field_kv){11, 1};
   } else {
      base[nbase++] = (struct agx3_field_kv){0, 1};
      base[nbase++] = (struct agx3_field_kv){24, 2};
      base[nbase++] = (struct agx3_field_kv){25, 2};
   }
   base[nbase++] = (struct agx3_field_kv){42, 2};
   base[nbase++] = (struct agx3_field_kv){43, 2};
   base[nbase++] = (struct agx3_field_kv){59, 1};
   base[nbase++] = (struct agx3_field_kv){60, 2};
   base[nbase++] = (struct agx3_field_kv){76,
                                          selected_immediate[0] ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){77, dest_16 ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){93,
                                          selected_immediate[1] ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){94, dest_16 ? 1 : 2};
   struct agx3_field_kv fields[20] = {
      {dest_16 ? 3 : 17, dest},
      {dest_16 ? 12 : 26, dest_cache},
      {35, compare},
      {51, minifloat},
      {selected_immediate[0] ? 68 : 69, selected[0]},
      {selected_immediate[1] ? 85 : 86, selected[1]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {110, condition},
   };
   unsigned n = 9;
   const unsigned cache[] = {78, 95}, discard[] = {80, 97};
   for (unsigned i = 0; i < 2; ++i) {
      if (!selected_immediate[i]) {
         fields[n++] = (struct agx3_field_kv){cache[i],
                                              hints[i + 1] == AGX3_ALU_HINT_CACHE};
         fields[n++] = (struct agx3_field_kv){discard[i],
                                              hints[i + 1] == AGX3_ALU_HINT_DISCARD};
      }
   }
   for (unsigned slot = 0; slot < 6; ++slot)
      fields[n++] = (struct agx3_field_kv){102 + slot,
                                           !!(wait_mask & (1u << slot))};

   /* Apple's witnessed GE/minifloat vectors use the eight-byte ep10 form when
    * the physical destination permits it. The older EQ/LT corpus uses
    * ep22/25/26 according to result width. */
   const unsigned preferred = condition == 5
                                 ? 10
                                 : (dest_16 ? ((dest & 1) ? 26 : 25) : 22);
   if (agx3_bind_compose(enc, AGX3_MN_fsel, preferred, base, nbase, fields, n,
                         out, len))
      return true;
   for (unsigned ep = 0; ep < 28; ++ep) {
      if (ep != preferred &&
          agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, nbase, fields, n, out,
                            len))
         return true;
   }
   return false;
}

bool
agx3_alu_fcmp_half_minifloat_select_f32(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t compare,
   uint8_t minifloat, const uint16_t selected[2],
   const bool selected_immediate[2], const enum agx3_alu_hint hints[3],
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !selected ||
       !selected_immediate || !hints || !out || !len || dest >= 256 ||
       (dest & 1) || compare >= 256 ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   if (selected_immediate[0] && selected_immediate[1])
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      if (selected[i] >= 256 ||
          (!selected_immediate[i] && (selected[i] & 1)) ||
          (selected_immediate[i] && hints[i + 1] != AGX3_ALU_HINT_NONE))
         return false;
   }
   for (unsigned i = 0; i < 3; ++i) {
      if ((unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* Fresh Metal kernels for all five comparisons use fsel.ep27 with two
    * registers. Register/zero alternatives use ep22 and f68/f85 for the
    * immediate instead. In both, f35 is the scalar-half comparator and f51
    * the ordinary minifloat. These endpoints have only the slot-6 wait bit;
    * callers retire another producer slot before composing them.
    *
    * ep22's f0 destination placement reads the immediate as a minifloat. A
    * non-zero integer keeps the f3 placement, as Apple's
    * `x + ((a < 0.5h) ? v : 5u)` does in its tied-result form
    * 22062fa120850322, so it stays on ep27 with the immediate in f68/f85. */
   const bool mixed_selected =
      (selected_immediate[0] && selected[0] == 0) ||
      (selected_immediate[1] && selected[1] == 0);
   struct agx3_field_kv base[13];
   unsigned nbase = 0;
   if (mixed_selected) {
      base[nbase++] = (struct agx3_field_kv){0, 1};
      base[nbase++] = (struct agx3_field_kv){24, 2};
      base[nbase++] = (struct agx3_field_kv){25, 2};
   } else {
      base[nbase++] = (struct agx3_field_kv){10, 2};
      base[nbase++] = (struct agx3_field_kv){11, 2};
   }
   base[nbase++] = (struct agx3_field_kv){42, 2};
   base[nbase++] = (struct agx3_field_kv){43, 1};
   base[nbase++] = (struct agx3_field_kv){59, 1};
   base[nbase++] = (struct agx3_field_kv){60, 2};
   base[nbase++] = (struct agx3_field_kv){76,
                                          selected_immediate[0] ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){77, 2};
   base[nbase++] = (struct agx3_field_kv){93,
                                          selected_immediate[1] ? 1 : 2};
   base[nbase++] = (struct agx3_field_kv){94, 2};
   struct agx3_field_kv fields[14] = {
      {mixed_selected ? 17 : 3, dest},
      {mixed_selected ? 26 : 12, dest_cache},
      {35, compare},
      {51, minifloat},
      {selected_immediate[0] ? 68 : 69, selected[0]},
      {selected_immediate[1] ? 85 : 86, selected[1]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {78, hints[1] == AGX3_ALU_HINT_CACHE},
      {80, hints[1] == AGX3_ALU_HINT_DISCARD},
      {95, hints[2] == AGX3_ALU_HINT_CACHE},
      {97, hints[2] == AGX3_ALU_HINT_DISCARD},
      {107, wait_mask == 32},
      {110, condition},
   };
   return agx3_bind_compose(enc, AGX3_MN_fsel, mixed_selected ? 22 : 27, base,
                            nbase, fields, 14, out, len);
}

/* Compare a half/full float register with floating zero, selecting full float
 * registers. The three source lifetimes are independent (including aliases).
 * Native Metal uses these compact forms for mixed-precision shader selects. */
bool
agx3_alu_fcmp_select_zero(enum agx_isa_encoding enc, uint16_t dest,
                          const uint16_t src[3], bool compare_16,
                          const enum agx3_alu_hint hints[3], unsigned condition,
                          bool dest_cache, uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       dest >= 256 || (dest & 1) ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;
   for (unsigned i = 0; i < 3; ++i)
      if (src[i] >= 256 || ((i != 0 || !compare_16) && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 2}, {42, 2}, {43, compare_16 ? 1 : 2},
      {59, 1}, {60, 2}, {76, 2}, {77, 2}, {93, 2}, {94, 2},
   };
   const struct agx3_field_kv fields[] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {51, 0},
      {69, src[1]}, {86, src[2]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {78, hints[1] == AGX3_ALU_HINT_CACHE},
      {80, hints[1] == AGX3_ALU_HINT_DISCARD},
      {95, hints[2] == AGX3_ALU_HINT_CACHE},
      {97, hints[2] == AGX3_ALU_HINT_DISCARD},
      {110, condition}, {107, wait_mask == 32},
   };
   /* Native forms are ordered by size: 6, 8, then 10 bytes. */
   for (unsigned ep = 0; ep < 28; ++ep)
      if (agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, 10, fields, 14, out, len))
         return true;
   return false;
}

/* Integer source liveness remains relevant after conversion: IDs can still
 * address later stores. Keep the native instruction and actual RA hints. */
bool
agx3_alu_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t src, bool is_signed, enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len ||
       dest >= 256 || src >= 256 || ((dest | src) & 1) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {31, 3}, {32, 91},
   };
   struct agx3_field_kv fields[15] = {
      {2, dest}, {18, src}, {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD}, {30, is_signed},
      {11, m5 && dest_cache},
   };
   unsigned n = 6;
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      fields[n++] = (struct agx3_field_kv){33 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, AGX3_MN_int2flt, 0, base, m5 ? 6 : 4, fields, n, out, len);
}

bool
agx3_alu_narrow_int_to_float(enum agx_isa_encoding enc, uint16_t dest,
                             bool dest_16, uint16_t src, bool src_8,
                             bool is_signed, enum agx3_alu_hint hint,
                             bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) ||
       !out || !len || dest >= (m5 ? 256 : 192) ||
       (!dest_16 && (dest & 1)) || src >= (m5 ? 256 : 192) ||
       (src_8 && !is_signed && dest_16) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;

   /* applegpu_g17p `half(char)`, `float(char)`, `half(short)` and
    * `half(ushort)` share the int2flt point used for 32-bit inputs. f10 is
    * the result width; f15 selects a signed byte, even when the half holds
    * a wider sum (`half(char(a + b))`), and f26 = 1 a half input. Only the
    * float32 result has an unsigned byte form; `half(uchar)` masks with 0xff
    * and converts a ushort instead.
    *   a70756020204b070  h2 <- char(^h2), wait slot 6
    *   a707540202089270  h2 <- char($h4)
    *   a707540302049070  h3 <- char(h2)
    *   a70754030304b070  $h3 <- char(^h2)
    *   a7075402030e9670  $r1 <- char($h7)
    *   a72754020204b070  h2 <- char(^h2), wait slot 2
    *   a707560202068220  h2 <- ushort($h3), wait slot 6
    *   a707540202089430  r1 <- uchar(h4) */
   const struct agx3_field_kv base[] = {{9, 2}, {10, dest_16 ? 1 : 2}, {25, 2}, {31,3}, {32,91}};
   struct agx3_field_kv fields[16] = {
      {2, dest}, {11, dest_cache}, {18, src},
      {15, src_8}, {26, src_8 ? 0 : 1},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD}, {30, is_signed},
   };
   unsigned n = 8;
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){33 + i, 1};
   return agx3_bind_compose(enc, AGX3_MN_int2flt, 0, base, m5 ? 5 : 3, fields, n, out, len);
}

bool
agx3_alu_i32_to_f16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    bool is_signed, enum agx3_alu_hint hint,
                    bool dest_cache, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len || dest >= 256 ||
       src >= 256 || (src & 1) || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;

   /* applegpu-nt 32023.917 emits int2flt with f10=1 for a half result and
    * f25/f26=2 for a32-bit integer input. f30 selects signed conversion;
    * f27/f29 carry the post-RA source hint and f33..f38 are dependency waits.
    * Exact witnesses cover signed/unsigned, live/dead, no-wait and wait0. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {25, 2}, {26, 2}, {31, 3}, {32, 91},
   };
   /* f27 is the CACHE half of the post-RA source hint and f29 the DISCARD
    * half; the comment above named both but only f29 was ever set, so a
    * cached source had to be refused outright.  Established by round-trip
    * against Apple rather than by a byte diff, because applegpu_g17p
    * allocates different registers for the two kernels and a raw diff cannot
    * separate the hint from the allocation.  Composing with the registers
    * held equal, our existing field set reproduces Apple's dead-source
    * instruction exactly
    *
    *   a7 07 56 02 02 04 a8 60   dest=2 src=2 wait=0x20 f29=1
    *
    * and cannot reach the live-source one at any assignment; sweeping all 160
    * fields then gives exactly one explanation, with no second candidate:
    *
    *   a7 07 56 04 02 00 8a 60   dest=4 src=0 wait=0x20 f27=1
    *
    * The two are complementary -- dead sets f29 and clears f27, live the
    * reverse -- which is the one-hot {none, discard, cache} the enum already
    * describes. The typical user is an i32-to-f16 convert whose source stays
    * live. */
   struct agx3_field_kv fields[14] = {
      {2, dest}, {11, dest_cache}, {18, src},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD}, {30, is_signed},
   };
   unsigned n = 6;
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      fields[n++] = (struct agx3_field_kv){33 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, AGX3_MN_int2flt, 0, base, m5 ? 6 : 4, fields, n, out, len);
}

bool
agx3_alu_f16_to_int32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      bool is_signed, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len || dest >= (m5 ? 256 : 128) ||
       src >= (m5 ? 256 : 128) || (dest & 1) ||
       (hint != AGX3_ALU_HINT_NONE && hint != AGX3_ALU_HINT_DISCARD) ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /* Fresh applegpu_g17p kernels establish the flt2int width/sign axes:
    *
    *   270756000200ac480300  f16 -> i32, dead, wait5
    *   270756000200ac080200  f16 -> u32, dead, wait5
    * Matching f32 controls change only f24 from 1 to 2, proving that this is
    * the native half source selector rather than an unrelated instruction.
    *
    * Separate live-source, live-result, odd-half and texture-produced probes
    * correlate f16 with the source, f2 with the even i32 destination, f11
    * with destination cache, f27 with source discard and f33..f38 with the
    * six producer waits. The fresh retained-source probes use the zero hint;
    * no witness establishes a source-cache form, so that state is refused.
    * Apple materializes abs and neg as separate half operations, so modifiers
    * stay outside this helper.
   */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {24, 1}, {41, is_signed ? 5 : 4}, {42, 1}, {31,3}, {32,71},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {27, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 4;
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      fields[n++] = (struct agx3_field_kv){33 + slot,
                                           !!(wait_mask & (1u << slot))};

   return agx3_bind_compose(enc, AGX3_MN_flt2int, 0, base, m5 ? 8 : 6, fields, n, out, len);
}

static bool
agx3_alu_bfeil_width(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
               uint16_t shift, bool shift_is_register, uint8_t width,
               bool source_16, bool shift_16,
               const enum agx3_alu_hint hints[2], uint16_t wait_mask,
               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if (!out || !length || !hints || (!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) ||
       dest >= 256 || source >= 256 || ((dest & 1) || (!source_16 && (source & 1))) ||
       (shift_is_register ? (shift >= 256 || (!shift_16 && (shift & 1))) : shift >= 32) ||
       width < 1 || width > 31 || (wait_mask & ~(gpu1 ? 0xffu : 0x3fu)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE ||
       (!shift_is_register && hints[1] != AGX3_ALU_HINT_NONE))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,source_16 ? 1 : 2},{39,shift_16 ? 1 : 2},{52,1},{53,2},
      /* M5 SRL retains the extract width in f57 and uses its own dispatch. */
      {58,3},{59,183},
   };
   struct agx3_field_kv fields[19] = {
      {2,dest},{16,source},{25,hints[0] == AGX3_ALU_HINT_CACHE},
      {27,hints[0] == AGX3_ALU_HINT_DISCARD},
      {38,shift_is_register ? 2 : 1},{shift_is_register ? 31 : 30,shift},
      {44,0},{57,width},
   };
   unsigned count=8;
   if (shift_is_register) {
      fields[count++] = (struct agx3_field_kv){40,hints[1] == AGX3_ALU_HINT_CACHE};
      fields[count++] = (struct agx3_field_kv){42,hints[1] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i=0;i<(gpu1 ? 8 : 6);++i)
      if (wait_mask & (1u<<i))
         fields[count++] = (struct agx3_field_kv){60+i,1};
   return agx3_bind_compose(enc,AGX3_MN_srl,0,base,gpu1 ? 9 : 7,fields,count,out,length);
}

bool
agx3_alu_bfeil(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
               uint16_t shift, bool shift_is_register, uint8_t width,
               const enum agx3_alu_hint hints[2], uint16_t wait_mask,
               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   return agx3_alu_bfeil_width(enc,dest,source,shift,shift_is_register,width,
                              false,false,hints,wait_mask,out,length);
}

bool
agx3_alu_bfeil_m5(uint16_t dest, uint16_t source, bool source_16,
                  uint16_t shift, bool shift_is_register, bool shift_16,
                  uint8_t width, const enum agx3_alu_hint hints[2],
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *length)
{
   return agx3_alu_bfeil_width(AGX_ISA_ENCODING_AGX3C,dest,source,shift,
                              shift_is_register,width,source_16,shift_16,
                              hints,wait_mask,out,length);
}

bool
agx3_alu_shift_add16(enum agx_isa_encoding enc, uint16_t dest,
                     const uint16_t src[2], const enum agx3_alu_hint hints[2],
                     unsigned shift, bool dest_cache, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 192 || src[0] >= 192 || src[1] >= 192 || shift > 4 ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {38, 2}, {39, 1},
   };
   struct agx3_field_kv fields[15] = {
      {2, dest}, {11, dest_cache}, {16, src[0]}, {31, src[1]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {40, hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD}, {54, shift},
   };
   unsigned n = 9;
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){45 + i, 1};
   return agx3_bind_compose(enc, AGX3_MN_iadd, 0, base, 6, fields, n, out, len);
}

bool
agx3_alu_add_sat16(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[2], const enum agx3_alu_hint hints[2],
                   bool is_signed, bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 192 || src[0] >= 192 || src[1] >= 192 ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;

   /* applegpu_g17p `addsat(short, short)` and `addsat(ushort, ushort)` are
    * the iadd point above with f53 set; the signed form also marks both
    * half sources signed (f28/f43):
    *   9f01560202040c40170f  h2 <- ^h2 +sat ^h3, signed, wait slot 6
    *   9f01560002040e44110f  h0 <- $h2 +sat $h3, signed, wait slot 6
    *   9f01560402060840110f  h4 <- h3 +sat h2, signed, wait slot 6
    *   9f115401020208441305  h1 <- ^h1 +sat $h2, unsigned, wait slot 1
    *   9f01560303040c401705  $h3 <- ^h2 +sat ^h3, unsigned, wait slot 6 */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {38, 2}, {39, 1}, {53, 1},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {11, dest_cache}, {16, src[0]}, {31, src[1]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {40, hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD},
      {28, is_signed}, {43, is_signed},
   };
   unsigned n = 10;
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){45 + i, 1};
   return agx3_bind_compose(enc, AGX3_MN_iadd, 0, base, 7, fields, n, out, len);
}

bool
agx3_alu_addsub_sat16_m5(uint16_t dest, const uint16_t src[2],
                         const enum agx3_alu_hint hints[2], bool is_signed,
                         bool subtract, bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src || !hints || !out || !len || dest >= 256 || src[0] >= 256 ||
       src[1] >= 256 || (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask & ~0xffu))
      return false;

   /* Native G17G (port/cwit3) `addsat`/`subsat` on ushort/short: C's
    * iadd32 (f45 = 80) or isub32 (f45 = 100) register point with half
    * widths for the result and both operands (f10/f24/f39 = 1), saturation
    * f54, and the signed form marking both halves signed (f28/f43):
    *   3f0004382100a302b0002000  h2 <- ^h4 +sat ^h2, unsigned
    *   3f0004382100a302b000e000  h2 <- ^h4 +sat ^h2, signed
    *   3f0004382508a20290006000  h2 <- ^h4 -sat ^h2, signed
    * Waits are C's iadd32 f46..f53. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {38, 2}, {39, 1},
      {44, 3}, {45, subtract ? 100 : 80}, {54, 1},
   };
   struct agx3_field_kv fields[18] = {
      {2, dest}, {11, dest_cache}, {16, src[0]}, {31, src[1]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {40, hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD},
      {28, is_signed}, {43, is_signed},
   };
   unsigned n = 10;
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){46 + i, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,
                            subtract ? AGX3_MN_isub32 : AGX3_MN_iadd32, 0, base,
                            sizeof(base) / sizeof(base[0]), fields, n, out,
                            len);
}

/* The scalar-half isub point is the iadd layout above: slot A at f16 with
 * cache/discard f25/f27, slot B at f31 with f40/f42, destination cache f11
 * and waits f45..f50. Both register slots end below half 192. */
static bool
agx3_alu_isub16(enum agx_isa_encoding enc, uint16_t dest, bool zero_minuend,
                const uint16_t src[2], const enum agx3_alu_hint hints[2],
                bool dest_cache, uint16_t wait_mask,
                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned limit = m5 ? 256 : 192;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len || dest >= limit ||
       (!zero_minuend && src[0] >= limit) || src[1] >= limit ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask >> (m5 ? 8 : 6)))
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, zero_minuend ? 1 : 2}, {24, 1}, {38, 2}, {39, 1},
      {44, 3}, {45, 100},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {11, dest_cache}, {31, src[1]},
      {40, hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 5;
   if (!zero_minuend) {
      fields[n++] = (struct agx3_field_kv){16, src[0]};
      fields[n++] = (struct agx3_field_kv){25, hints[0] == AGX3_ALU_HINT_CACHE};
      fields[n++] =
         (struct agx3_field_kv){27, hints[0] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){(m5 ? 46 : 45) + i, 1};
   return agx3_bind_compose(enc, m5 ? AGX3_MN_isub32 : AGX3_MN_isub, 0, base,
                            m5 ? 8 : 6, fields, n, out, len);
}

bool
agx3_alu_sub16(enum agx_isa_encoding enc, uint16_t dest,
               const uint16_t src[2], const enum agx3_alu_hint hints[2],
               bool dest_cache, uint16_t wait_mask,
               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* applegpu_g17p `short(a - b)`:
    *   1f01560202040c001705  isub h2, ^h2, ^h3, wait slot 6
    *   1f01560002040e041105  isub h0, $h2, $h3, wait slot 6
    *   1f015604020608001105  isub h4, h3, h2, wait slot 6
    *   1f115401020208041305  isub h1, ^h1, $h2, wait slot 1
    *   1f01560303040c001705  isub $h3, ^h2, ^h3, wait slot 6 */
   return src && hints &&
          agx3_alu_isub16(enc, dest, false, src, hints, dest_cache, wait_mask,
                          out, len);
}

bool
agx3_alu_neg16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
               uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* applegpu_g17p `short(-a)` is isub with an immediate-zero slot A:
    *   1f015602020008001504  isub h2, #0, ^h2, wait slot 6
    *   1f01560202000c041104  isub h2, #0, $h3, wait slot 6
    *   1f015603020008001104  isub h3, #0, h2, wait slot 6
    *   1f215402020008001504  isub h2, #0, ^h2, wait slot 2
    *   1f015603030008001504  isub $h3, #0, ^h2, wait slot 6 */
   const uint16_t src_pair[2] = {0, src};
   const enum agx3_alu_hint hint_pair[2] = {AGX3_ALU_HINT_NONE, hint};
   return agx3_alu_isub16(enc, dest, true, src_pair, hint_pair, dest_cache,
                          wait_mask, out, len);
}

static bool
agx3_alu_shift_reg(enum agx3_mnemonic mnemonic, enum agx_isa_encoding enc,
                   uint16_t dest, bool dest_16, bool dest_cache,
                   const uint16_t src[2], const bool src_16[2],
                   const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) ||
       (m5 && mnemonic != AGX3_MN_sll && mnemonic != AGX3_MN_srl) || !src || !src_16 || !hints || !out ||
       !len || (dest_16 ? dest >= 256 : (dest >= 256 || (dest & 1))) ||
       (src_16[0] ? src[0] >= 256 : (src[0] >= 256 || (src[0] & 1))) ||
       (src_16[1] ? src[1] >= 256 : (src[1] >= 256 || (src[1] & 1))) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_DISCARD || (wait_mask >> (m5 ? 8 : 6)))
      return false;

   /* Fresh applegpu_g17p Metal vectors pin the independent result/source
    * widths, value cache/discard (f25/f27), shift discard (f42), and slot-2
    * wait. In particular, a uint shift count remains width 2 even when the
    * value is ushort. */
   struct agx3_field_kv base[10] = {
      {9, 2}, {10, dest_16 ? 1 : 2}, {23, 2}, {24, src_16[0] ? 1 : 2},
      {38, 2}, {39, src_16[1] ? 1 : 2}, {52, 1}, {53, 2},
   };
   unsigned nb = 8;
   if (m5) {
      base[nb++] = (struct agx3_field_kv){58, 3};
      base[nb++] = (struct agx3_field_kv){59, mnemonic == AGX3_MN_srl ? 183 : 162};
   }
   struct agx3_field_kv fields[16] = {
      {2, dest}, {11, dest_cache}, {16, src[0]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD}, {31, src[1]},
      {40, 0}, {42, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){60 + i, 1};
   return agx3_bind_compose(enc, mnemonic, 0, base, nb, fields, n, out, len);
}

/* G18G (M6), which compiles as agx3c: the all-half register shift below
 * half 128, with the shift count's cache hint (f40). Used by
 * agx3_alu_sll_reg_g18g and agx3_alu_srl_reg_g18g. */
static bool
agx3_alu_shift_reg_g18g(enum agx3_mnemonic mnemonic, enum agx_isa_encoding enc,
                        uint16_t dest, bool dest_16, bool dest_cache,
                        const uint16_t src[2], const bool src_16[2],
                        const enum agx3_alu_hint hints[2], uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned limit = gpu1 ? 128 : 256;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || !src || !src_16 ||
       !hints || !out || !len ||
       (gpu1 && (!dest_16 || !src_16[0] || !src_16[1])) ||
       (dest_16 ? dest >= limit : (dest >= limit || (dest & 1))) ||
       (src_16[0] ? src[0] >= limit : (src[0] >= limit || (src[0] & 1))) ||
       (src_16[1] ? src[1] >= limit : (src[1] >= limit || (src[1] & 1))) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > (gpu1 ? AGX3_ALU_HINT_CACHE : AGX3_ALU_HINT_DISCARD) ||
       (wait_mask & ~(gpu1 ? 0xffu : 0x3fu)))
      return false;

   /* Native905 pins the G17G/G18G all-half operand widths, both source
    * lifetimes and producer waits. Other GPU1 width combinations stay closed. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, dest_16 ? 1 : 2}, {23, 2}, {24, src_16[0] ? 1 : 2},
      {38, 2}, {39, src_16[1] ? 1 : 2}, {52, 1}, {53, 2},
      {58, 3}, {59, mnemonic == AGX3_MN_sll ? 162 : 183},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {11, dest_cache}, {16, src[0]},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD}, {31, src[1]},
      {40, gpu1 && hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned i = 0; i < (gpu1 ? 8 : 6); ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){60 + i, 1};
   return agx3_bind_compose(enc, mnemonic, 0, base, gpu1 ? 10 : 8,
                            fields, n, out, len);
}

bool
agx3_alu_sll_reg(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                 bool dest_cache, const uint16_t src[2],
                 const bool src_16[2], const enum agx3_alu_hint hints[2],
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                 uint8_t *len)
{
   return agx3_alu_shift_reg(AGX3_MN_sll, enc, dest, dest_16, dest_cache, src,
                             src_16, hints, wait_mask, out, len);
}

bool
agx3_alu_sll_reg_g18g(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                      bool dest_cache, const uint16_t src[2],
                      const bool src_16[2], const enum agx3_alu_hint hints[2],
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len)
{
   return agx3_alu_shift_reg_g18g(AGX3_MN_sll, enc, dest, dest_16, dest_cache, src,
                                  src_16, hints, wait_mask, out, len);
}

bool
agx3_alu_sll_one_reg32(enum agx_isa_encoding enc, uint16_t dest,
                       uint16_t shift, enum agx3_alu_hint shift_hint,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       shift >= 256 || (shift & 1) ||
       (unsigned)shift_hint > AGX3_ALU_HINT_DISCARD ||
       (wait_mask & ~0x3fu))
      return false;

   /* applegpu_g17p `ushort(1) << ushort(n)` is one sll with an immediate
    * one, a half destination and a half count (f39 = 1). The count here is
    * the low half of a uint register, which is exactly the half Apple reads
    * after its own `n & 15`:
    *   270054020302080080120400  $h2 <- 1 << ^h2
    *   270054040202000080100400  h4 <- 1 << h0 (count still live)
    *   270054010202040080120400  h1 <- 1 << ^h2
    * f42 is the count's last use and f11 the destination cache. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {15, 1}, {23, 1}, {24, 1},
      {38, 2}, {39, 1}, {52, 1}, {53, 2},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest},
      {11, dest_cache},
      {31, shift},
      {42, shift_hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned count = 4;
   for (unsigned slot = 0; slot < 6; ++slot) {
      if (wait_mask & (1u << slot))
         fields[count++] = (struct agx3_field_kv){60 + slot, 1};
   }

   return agx3_bind_compose(enc, AGX3_MN_sll, 0, base, 9, fields, count, out, len);
}

bool
agx3_alu_sll_one_m5(uint16_t dest, bool dest_16, uint16_t shift,
                    enum agx3_alu_hint shift_hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len)
{
   if (!out || !len || dest >= 256 || (!dest_16 && (dest & 1)) ||
       shift >= 256 || (shift & 1) ||
       (unsigned)shift_hint > AGX3_ALU_HINT_DISCARD || (wait_mask & ~0xffu))
      return false;

   /* Native G17G `ushort(1) << uint(n)` (port/cwit3 k_shl1_u16, also
    * k_shl1_u16b/k_shl1_u8) computes in 32 bits: one sll with the immediate
    * one in slot A (f15 = 1, f23 = 1, f24 = 1), the full uint count in slot B
    * (f31, f39 = 2, last use f42) and C's form constants f58 = 3, f59 = 162:
    *   2701201a2600a002008020340000  r0 <- 1 << ^r0
    * The result width f10 is the same independent field as in
    * agx3_alu_sll_reg(); a half destination sets f10 = 1. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, dest_16 ? 1 : 2}, {15, 1}, {23, 1}, {24, 1},
      {38, 2}, {39, 2}, {52, 1}, {53, 2}, {58, 3}, {59, 162},
   };
   struct agx3_field_kv fields[12] = {
      {2, dest},
      {11, dest_cache},
      {31, shift},
      {42, shift_hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned count = 4;
   for (unsigned slot = 0; slot < 8; ++slot) {
      if (wait_mask & (1u << slot))
         fields[count++] = (struct agx3_field_kv){60 + slot, 1};
   }

   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_sll, 0, base,
                            sizeof(base) / sizeof(base[0]), fields, count, out,
                            len);
}

bool
agx3_alu_srl_reg(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                 bool dest_cache, const uint16_t src[2],
                 const bool src_16[2], const enum agx3_alu_hint hints[2],
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                 uint8_t *len)
{
   return agx3_alu_shift_reg(AGX3_MN_srl, enc, dest, dest_16, dest_cache, src,
                             src_16, hints, wait_mask, out, len);
}

bool
agx3_alu_srl_reg_g18g(enum agx_isa_encoding enc, uint16_t dest, bool dest_16,
                      bool dest_cache, const uint16_t src[2],
                      const bool src_16[2], const enum agx3_alu_hint hints[2],
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len)
{
   return agx3_alu_shift_reg_g18g(AGX3_MN_srl, enc, dest, dest_16, dest_cache, src,
                                  src_16, hints, wait_mask, out, len);
}

bool
agx3_alu_asr_reg16(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                   uint16_t shift, const enum agx3_alu_hint hints[2],
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !hints || !out || !len ||
       dest >= 256 || source >= 256 || shift >= 256 || (shift & 1) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;

   /* applegpu_g17p `short(a) >> uint(n)`: the signed value and destination
    * remain scalar halves, while the register shift count retains its full
    * width. f28 selects signed fill and f45..f50 carry producer waits. The
    * value's and count's lifetimes are independent pairs, f25/f27 and
    * f40/f42, each witnessed dead, retained for a store and cached:
    *   a7115402020808e0e200  h2 <- ^h4 >> ^r1
    *   a7115402020808a0e200  h2 <- h4 >> ^r1
    *   a711540402080060e200  h4 <- ^h4 >> r0
    *   a7115402020a0aa0e200  h2 <- $h5 >> ^r1
    *   a7115403030a08e0e200  $h3 <- ^h5 >> ^r1 */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {28, 1}, {38, 2}, {39, 2},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest},
      {11, dest_cache},
      {16, source},
      {25, hints[0] == AGX3_ALU_HINT_CACHE},
      {27, hints[0] == AGX3_ALU_HINT_DISCARD},
      {31, shift},
      {40, hints[1] == AGX3_ALU_HINT_CACHE},
      {42, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned count = 8;
   for (unsigned slot = 0; slot < 6; ++slot) {
      if (wait_mask & (1u << slot))
         fields[count++] = (struct agx3_field_kv){45 + slot, 1};
   }

   return agx3_bind_compose(enc, AGX3_MN_sra, 0, base, 7, fields, count, out, len);
}

bool
agx3_alu_shift_imm32(enum agx_isa_encoding enc, bool left,
                      uint16_t dest, uint16_t src, unsigned shift,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !out || !len ||
       dest >= (m5 ? 256 : 192) || src >= (m5 ? 256 : 192) || ((dest | src) & 1) ||
       shift < 1 || shift > 31 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;

   /* Native runtime archives for uint shifts pin the literal count at f30,
    * independently of value width, lifetime hints, and scoreboard waits.
    * sll_13:2700560002003400f0110100
    * srl_17:a700560002004400f0110100
    * Both use a discarded r0 value and wait slot 6. No count register or
    * arithmetic boundary conversion is part of this immediate form. */
   /* G17G benchmark archive, the same immediate count at f30:
    * sll #13:0782001a0a00800a788300340000
    * srl #17:3782001a2a00a10a788400340000
    * M5 carries the implicit boundary at f58/f59; SLL additionally
    * selects the full-width source lanes at f71..f73. */
   struct agx3_field_kv base[13] = {
      {9, 2}, {10, 2}, {23, 2}, {24, 2},
      {38, 1}, {39, 2}, {52, 1}, {53, 2},
   };
   unsigned nb = 8;
   if (m5) {
      base[nb++] = (struct agx3_field_kv){58, 3};
      base[nb++] = (struct agx3_field_kv){59, left ? 162 : 183};
      if (left) {
         base[nb++] = (struct agx3_field_kv){71, 2};
         base[nb++] = (struct agx3_field_kv){72, 2};
         base[nb++] = (struct agx3_field_kv){73, 2};
      }
   }
   struct agx3_field_kv fields[14] = {
      {2, dest}, {11, dest_cache}, {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD}, {30, shift},
   };
   unsigned n = 6;
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){60 + i, 1};
   return agx3_bind_compose(enc, left ? AGX3_MN_sll : AGX3_MN_srl, 0,
                            base, nb, fields, n, out, len);
}

bool
agx3_alu_srl_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || out == NULL || len == NULL ||
       dest >= 256 || src >= 256 || shift < 1 || shift > 31 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~(m5 ? 0xffu : 0x3fu)))
      return false;

   /* applegpu_g17p emits this form when extracting [[sample_mask]] from
    * bits 4..7 of SR39. f30 is the immediate shift; f38/f39 select its
    * implicit full-width boundary operand rather than a second GPR.
    * applegpu_g17g and applegpu_g18g emit the same field vector for their
    * [[sample_mask]] (bits 8..15, srl #8), with the G17-class dispatch
    * f58 = 3 and f59 = 183 added (m6-sample-mask-in). */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1},
      {38, 1}, {39, 2}, {52, 1}, {53, 2}, {58, 3}, {59, 183},
   };
   struct agx3_field_kv fields[14] = {
      {2, dest},
      {11, dest_cache},
      {16, src},
      {25, hint == AGX3_ALU_HINT_CACHE},
      {27, hint == AGX3_ALU_HINT_DISCARD},
      {30, shift},
   };
   unsigned n = 6;
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i) {
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){60 + i, 1};
   }

   return agx3_bind_compose(enc, AGX3_MN_srl, 0, base, m5 ? 10 : 8, fields, n, out, len);
}

bool
agx3_alu_extract_imm32(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t low_high[2], unsigned shift,
                       const enum agx3_alu_hint hints[2], bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) || !out || !length || !low_high ||
       !hints || dest >= (gpu1 ? 256 : 192) ||
       low_high[0] >= (gpu1 ? 256 : 192) ||
       low_high[1] >= (gpu1 ? 256 : 192) ||
       ((dest | low_high[0] | low_high[1]) & 1) || shift<1 || shift>31 ||
       (unsigned)hints[0]>AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1]>AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(gpu1 ? 0xffu : 0x3fu)))
      return false;
   /* AGX3c retains the high/shift/low operand field mapping but requires
    * GPR dispatch f58=3, opcode f59=40 and has eight independent wait bits.
    * Descriptor address extraction consumes a load in slot7 on M5. */
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,2},{38,1},{39,2},{52,2},{53,2},
      {58,3},{59,40},
   };
   struct agx3_field_kv fields[19] = {
      {2,dest},{11,dest_cache},{16,low_high[1]},{30,shift},{45,low_high[0]},
      {25,hints[1]==AGX3_ALU_HINT_CACHE},{27,hints[1]==AGX3_ALU_HINT_DISCARD},
      {54,hints[0]==AGX3_ALU_HINT_CACHE},{56,hints[0]==AGX3_ALU_HINT_DISCARD},
      {57,0},
   };
   for(unsigned i=0;i<(gpu1 ? 8 : 6);i++)
      fields[10+i]=(struct agx3_field_kv){60+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,AGX3_MN_extract,0,base,gpu1 ? 10 : 8,
                           fields,gpu1 ? 18 : 16,out,length);
}

/* Native arithmetic right shift keeps signed fill without a second operation. */
static bool
agx3_alu_asr_imm32_source(enum agx_isa_encoding enc, uint16_t dest,
                   uint16_t source, bool source_half,
                   unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *length)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   /* AGX3B names word registers below 192. GPU1 (M5, and G18G, which packs
    * through this form) addresses the whole 256-half file, as
    * agx3_alu_asr_imm_m5() does with the same sra fields. */
   const unsigned word_regs = gpu1 ? 256 : 192;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || !out || !length ||
       dest >= word_regs || (dest & 1) ||
       source >= (source_half ? 256 : word_regs) ||
       (!source_half && (source & 1)) ||
       (source_half && !gpu1) ||
       shift < 1 || shift > (source_half ? 15 : 31) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,source_half ? 1 : 2},{38,1},{39,2},
      /* M5's native sra table requires the register-result selector and
       * GPU1 operation tag; reusing AGX3b's zero defaults is invalid. */
      {43,3},{44,167},
   };
   struct agx3_field_kv fields[13] = {
      {2,dest},{11,dest_cache},{16,source},{30,shift},
      {25,hint == AGX3_ALU_HINT_CACHE},{27,hint == AGX3_ALU_HINT_DISCARD},
      {28,1},
   };
   for (unsigned i=0;i<6;i++)
      fields[7+i]=(struct agx3_field_kv){45+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,AGX3_MN_sra,0,base,gpu1 ? 8 : 6,fields,13,out,length);
}

/* Metal532: runtime int and short shifts. The full-word value uses a
 * half count (only its low five bits matter); the short value uses a word
 * count. f28 selects sign fill and f45..52 carry independent waits. */
bool
agx3_alu_asr_reg_m5(uint16_t dest, uint16_t source, uint16_t shift,
                     bool dest_16, bool source_16,
                     const enum agx3_alu_hint hints[2], bool dest_cache,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *length)
{
   if (!out || !length || !hints || dest >= 256 || source >= 256 ||
       shift >= 256 || (shift & 1) || (dest_16 && !source_16) ||
       (!dest_16 && (dest & 1)) || (!source_16 && (source & 1)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE || (wait_mask & ~0xffu))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,dest_16 ? 1 : 2},{23,2},{24,source_16 ? 1 : 2},
      {28,1},{38,2},{39,dest_16 ? 2 : 1},{43,3},{44,167},
   };
   struct agx3_field_kv fields[16] = {
      {2,dest},{11,dest_cache},{16,source},{31,shift},
      {25,hints[0] == AGX3_ALU_HINT_CACHE},
      {27,hints[0] == AGX3_ALU_HINT_DISCARD},
      {40,hints[1] == AGX3_ALU_HINT_CACHE},
      {42,hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i=0;i<8;i++)
      fields[8+i]=(struct agx3_field_kv){45+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,AGX3_MN_sra,0,base,9,
                           fields,16,out,length);
}

bool
agx3_alu_asr_imm_m5(uint16_t dest, uint16_t source, bool dest_16,
                     bool source_16, unsigned shift, enum agx3_alu_hint hint,
                     bool dest_cache, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   if (!out || !length || dest >= 256 || source >= 256 ||
       (!dest_16 && (dest & 1)) || (!source_16 && (source & 1)) ||
       shift < 1 || shift > (source_16 ? 15 : 31) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0xffu))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,dest_16 ? 1 : 2},{23,2},{24,source_16 ? 1 : 2},
      {38,1},{39,2},{43,3},{44,167},
   };
   struct agx3_field_kv fields[15] = {
      {2,dest},{11,dest_cache},{16,source},{30,shift},
      {25,hint == AGX3_ALU_HINT_CACHE},{27,hint == AGX3_ALU_HINT_DISCARD},
      {28,1},
   };
   for (unsigned i=0;i<8;i++)
      fields[7+i]=(struct agx3_field_kv){45+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,AGX3_MN_sra,0,base,8,
                           fields,15,out,length);
}

bool
agx3_alu_asr_imm32(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                   unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *length)
{
   return agx3_alu_asr_imm32_source(enc, dest, source, false, shift, hint,
                                   dest_cache, wait_mask, out, length);
}

/* Native G17G/G18G short-to-int shifts (block46-native-447): f24=1 reads
 * only the named half while f10=2 writes the sign-extended word result. */
bool
agx3_alu_asr_imm16_to32(enum agx_isa_encoding enc, uint16_t dest,
                       uint16_t source, unsigned shift, enum agx3_alu_hint hint,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *length)
{
   return agx3_alu_asr_imm32_source(enc, dest, source, true, shift, hint,
                                   dest_cache, wait_mask, out, length);
}

/* Native short arithmetic shift: f10/f24 select half destination/source. */
bool
agx3_alu_asr_imm16(enum agx_isa_encoding enc, uint16_t dest, uint16_t source,
                   unsigned shift, enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *length)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !gpu1) || !out || !length ||
       dest >= 256 || source >= 256 || shift < 1 || shift > 15 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0x3fu))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{23,2},{24,1},{38,1},{39,2},
      {43,3},{44,167},
   };
   struct agx3_field_kv fields[13] = {
      {2,dest},{11,dest_cache},{16,source},{30,shift},
      {25,hint == AGX3_ALU_HINT_CACHE},{27,hint == AGX3_ALU_HINT_DISCARD},
      {28,1},
   };
   for (unsigned i=0;i<6;i++)
      fields[7+i]=(struct agx3_field_kv){45+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(enc,AGX3_MN_sra,0,base,gpu1 ? 8 : 6,fields,13,out,length);
}

bool agx3_alu_fcmp_select_reg_imm(enum agx_isa_encoding enc, uint16_t dest,
                                  const uint16_t src_in[4], bool compare_zero,
                                  const bool compare_abs_in[2],
                                  const bool compare_neg[2],
                                  const bool immediate[2],
                                  const enum agx3_alu_hint hints_in[4], unsigned condition,
                                  bool dest_16, bool dest_cache, uint16_t wait_mask,
                                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || !src_in || !compare_abs_in || !compare_neg ||
       !immediate || !hints_in)
      return false;

   /* Keep the independently qualified M5 form; G17P negate encodings
    * below have not been established for AGX3c. */
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      if (compare_neg[0] || compare_neg[1])
         return false;
      return agx3_alu_fcmp_select_gpu1(dest, src_in, compare_zero, compare_abs_in,
         immediate, hints_in, condition, dest_16, dest_cache, wait_mask, out, len);
   }

   /*
    * Normalise the comparison's negates before anything else looks at the
    * operands. f65 is the only negate field and it names the SECOND
    * comparator, so the other placements are rewrites, each one witnessed
    * with metal-nt -arch applegpu_g17p on a select whose true value is a
    * register -- the shape this encoder exists for -- and recovered field by
    * field through tools/decomp.c:
    *
    *    a >= c      12051d07850302c0   f35=4 f52=6 f110=5
    *    a >= -c     12051d07850302c8   f35=4 f52=6 f110=5 f65
    *   -a >= c      12071d05850303c8   f35=6 f52=4 f110=6 f65
    *   -a >= -c     12051d07850303c0   f35=4 f52=6 f110=6
    *    a <= c      12051d07850303c0   byte-identical to the row above
    *
    * The last two rows being one instruction is what makes the double-negate
    * rewrite a measurement rather than an assumption. f65 also composes with
    * an absolute comparator, which an earlier note recorded as unwitnessed:
    * `select(v.x, v.y, fabs(a) >= -c)` is 12051d0787030208800200800100, f47
    * and f65 together on the same fourteen-byte endpoint.
    */
   uint16_t src[4] = {src_in[0], src_in[1], src_in[2], src_in[3]};
   enum agx3_alu_hint hints[4] = {hints_in[0], hints_in[1], hints_in[2],
                                  hints_in[3]};
   bool compare_abs[2] = {compare_abs_in[0], compare_abs_in[1]};
   bool negate_second = compare_neg[1];
   if (compare_zero) {
      /* -0.0 and +0.0 compare equal, so a negate on the immediate slot is a
       * no-op and a negate on the other one is the relation reversal. */
      negate_second = false;
      if (compare_neg[0])
         condition = agx3_fcond_reversed(condition);
   } else if (compare_neg[0] && compare_neg[1]) {
      condition = agx3_fcond_reversed(condition);
      negate_second = false;
   } else if (compare_neg[0]) {
      const uint16_t reg = src[0];
      const enum agx3_alu_hint hint = hints[0];
      const bool abs = compare_abs[0];
      src[0] = src[1];
      hints[0] = hints[1];
      compare_abs[0] = compare_abs[1];
      src[1] = reg;
      hints[1] = hint;
      compare_abs[1] = abs;
      condition = agx3_fcond_reversed(condition);
      negate_second = true;
   }

   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest>=256 ||
       (!dest_16 && (dest&1)) ||
       (condition!=0 && condition!=1 && condition!=2 && condition!=5 && condition!=6) ||
       (wait_mask!=0 && wait_mask!=32) ||
       (compare_zero && (src[1]!=0 || compare_abs[1])))
      return false;
   for(unsigned i=0;i<4;i++) {
      const bool imm=(i==1 && compare_zero) || (i>=2 && immediate[i-2]);
      if(src[i]>=256 || (!imm && (i<2 || !dest_16) && (src[i]&1)) ||
         (unsigned)hints[i]>AGX3_ALU_HINT_CACHE || (imm && hints[i]!=AGX3_ALU_HINT_NONE))
         return false;
   }
   const bool abs_compare = compare_abs[0] || compare_abs[1];
   /* Apple's absolute comparisons selecting a zero for a float result take
    * the f0 destination placement, which reads f68/f85 as minifloats. A
    * non-zero selected immediate is an integer bit pattern and needs the f3
    * placement: `(fabs(a) < b) ? v : 5u` is 02011f0583080320208500000100.
    * Zero is the same bits in either. */
   const bool abs_immediate_select =
      abs_compare && (immediate[0] || immediate[1]) &&
      (!immediate[0] || src[2] == 0) && (!immediate[1] || src[3] == 0);
   /* Only a comparison against an immediate zero is refused here. The
    * destination width is an independent field on the long endpoint, exactly
    * as it is on the short one, and Apple's own output carries the
    * combination: eight of 84 Apple `fsel` instructions set both f11 = 1 and
    * f47/f64, on ep33 and ep34 --
    *
    *   42d725d923810300208000000300   ep33 f11=1 f47=1 f64=1 f76=1 f93=1
    *   3a8525db23820300800800000300   ep34 f11=1 f47=1 f64=1 f76=1 f93=2
    *
    * The absolute immediate-select branch below takes its f25 width from
    * dest_16 as well.
    *
    * `mediump` code reaches this every time it writes a comparison on an
    * absolute value -- the shape is fcmpsel [d16 s0=32:reg.abs s1=32:reg
    * s2=16:reg s3=16:imm0], which is what a half-precision `(abs(a) < b) ?
    * c : 0.0` compiles to. */
   if (abs_compare && compare_zero)
      return false;

   /* Ordered GE/LE with two absolute comparators needs the general fsel
    * endpoint selector. Preserve the already-normalized operands and the
    * native wait directly; no extra instructions or temporary registers. */
   if (compare_abs[0] && compare_abs[1] && condition > 2) {
      const struct agx3_fsel_desc d = {
         .dest = dest,
         .dest_16 = dest_16,
         .dest_cache = dest_cache,
         .cmp0 = src[0],
         .cmp0_hint = hints[0],
         .cmp0_abs = compare_abs[0],
         .cmp1 = src[1],
         .cmp1_hint = hints[1],
         .cmp1_abs = compare_abs[1],
         .cmp1_neg = negate_second,
         .sel0_kind = immediate[0] ? AGX3_FSEL_SEL_IMMEDIATE
                                  : AGX3_FSEL_SEL_REGISTER,
         .sel1_kind = immediate[1] ? AGX3_FSEL_SEL_IMMEDIATE
                                  : AGX3_FSEL_SEL_REGISTER,
         .sel0 = src[2],
         .sel1 = src[3],
         .sel0_hint = hints[2],
         .sel1_hint = hints[3],
         .sel_16 = dest_16,
         .condition = condition,
         .wait_mask = wait_mask,
      };
      return agx3_alu_fcmp_select(enc, &d, out, len);
   }

   /* f65 negates the SECOND compare source. Apple's applegpu_g17p output for
    * `select(x, y, a < -b)` differs from `select(x, y, a < b)` in exactly that
    * one field, in the same encoding point, for both result widths:
    *
    *   32-bit result  a <  b   02011f03820a03c08006   ... 63:1      69:10 ...
    *                  a < -b   02011f03820a03c88006   ... 63:1 65:1 69:10 ...
    *   16-bit result  a <  b   02011d03820b0320800a   ... 63:1      69:11 ...
    *                  a < -b   02011d03820b0328800a   ... 63:1 65:1 69:11 ...
    *
    * Three further refusals stood here, each recording what had been measured
    * rather than what the encoding rules out. All three are now witnessed, so
    * the normalisation at the top of this function performs them instead:
    *
    *  - A negated FIRST source. The note above already said `-a < b` comes
    *    back as 02031d01820b0228800a, "f65 again with the operands swapped and
    *    the relation flipped" -- which IS the rewrite, observed, and then
    *    declined. `select(v.x, v.y, -a >= c)` says the same on the
    *    register-select form (12071d05850303c8 against 12051d07850302c0).
    *
    *  - The negate mixed with an ABSOLUTE source, on either endpoint family.
    *    Immediate selects, where abs escalates to the fourteen-byte form:
    *
    *      (fabs(a) <  c) ? 1.0h : 0.0h   1203190523b00300208000800100
    *      (fabs(a) < -c) ? 1.0h : 0.0h   1203190523b00308208000800100   f65
    *      (fabs(a) < -c) ? 1.0f : 0.0f   02011b0523b00308208000800100   ep30
    *      (-a < fabs(c)) ? 1.0h : 0.0h   1205190323b00208208000800100
    *
    *    one field apart, and the last row is the swap again: f35 and f52
    *    exchange (2/4 -> 4/2), f110 flips lt -> gt, and the abs travels with
    *    the operand it belonged to. Register selects say it too:
    *    select(v.x, v.y, fabs(a) >= -c) is 12051d0787030208800200800100.
    *
    *  - The negate against an exact immediate ZERO. `-a < 0.0f` and `a > 0.0f`
    *    are one instruction, 1203098122b002c22080 both -- so the negate is the
    *    relation reversal and nothing else, which is what -0.0 comparing equal
    *    to +0.0 requires.
    *
    * What still fails closed is anything the composer itself refuses: the
    * field vector is deposited and agx3_bind_compose() either finds an
    * endpoint carrying every field or returns false. */

   struct agx3_field_kv base[12];
   unsigned nbase=0;
   base[nbase++]=(struct agx3_field_kv){42,2};
   base[nbase++]=(struct agx3_field_kv){43,2};
   base[nbase++]=(struct agx3_field_kv){59,compare_zero?1:2};
   base[nbase++]=(struct agx3_field_kv){60,2};
   base[nbase++]=(struct agx3_field_kv){76,immediate[0]?1:2};
   base[nbase++]=(struct agx3_field_kv){77,dest_16?1:2};
   base[nbase++]=(struct agx3_field_kv){93,immediate[1]?1:2};
   base[nbase++]=(struct agx3_field_kv){94,dest_16?1:2};
   if (abs_immediate_select) {
      base[nbase++]=(struct agx3_field_kv){0,1};
      base[nbase++]=(struct agx3_field_kv){24,2};
      base[nbase++]=(struct agx3_field_kv){25,dest_16?1:2};
   } else {
      base[nbase++]=(struct agx3_field_kv){10,2};
      base[nbase++]=(struct agx3_field_kv){11,dest_16?1:2};
   }
   struct agx3_field_kv fields[19]={
      {abs_immediate_select?17:3,dest},
      {abs_immediate_select?26:12,dest_cache},
      {35,src[0]},{compare_zero?51:52,src[1]},
      {immediate[0]?68:69,src[2]},{immediate[1]?85:86,src[3]},
      {110,condition},{107,wait_mask==32},{47,compare_abs[0]},
      {64,compare_abs[1]},{65,negate_second},
   };
   const unsigned cache[]={44,61,78,95},discard[]={46,63,80,97};
   unsigned n=11;
   for(unsigned i=0;i<4;i++) if((i<2 && !(i==1 && compare_zero)) || (i>=2 && !immediate[i-2])) {
      fields[n++]=(struct agx3_field_kv){cache[i],hints[i]==AGX3_ALU_HINT_CACHE};
      fields[n++]=(struct agx3_field_kv){discard[i],hints[i]==AGX3_ALU_HINT_DISCARD};
   }
   for(unsigned ep=0;ep<36;ep++)
      if(agx3_bind_compose(enc,AGX3_MN_fsel,ep,base,nbase,fields,n,out,len))
         return true;
   return false;
}

bool
agx3_alu_fcmp_select_f32_f16(enum agx_isa_encoding enc, uint16_t dest,
                             const uint16_t src[4],
                             const enum agx3_alu_hint hints[4],
                             unsigned condition, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       dest >= 256 || (dest & 1) ||
       /* f110 is enum agx_fcond (eq 0, lt 1, gt 2, ge 5, le 6), swept one
        * field at a time against Apple's own bytes for the same form; see
        * agx3_alu_fcmp_select_f32_imm. The NaN-ignoring ltn/gtn codes were
        * not witnessed here and stay out. */
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (i != 1 && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* Full/half comparison with two full-register alternatives. The operand
    * width and routing fields are shared with the native mixed comparison and
    * all-register FSEL witnesses already used by the adjacent encoders. */
   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 2}, {42, 2}, {43, 2}, {59, 2},
      {60, 1}, {76, 2}, {77, 2}, {93, 2}, {94, 2},
   };
   struct agx3_field_kv fields[16] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {69, src[2]}, {86, src[3]}, {110, condition}, {107, wait_mask == 32},
   };
   const unsigned cache[] = {44, 61, 78, 95};
   const unsigned discard[] = {46, 63, 80, 97};
   unsigned n = 8;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){cache[i],
                                           hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard[i],
                                           hints[i] == AGX3_ALU_HINT_DISCARD};
   }

   for (unsigned ep = 0; ep < 28; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, 10, fields, n, out, len))
         return true;
   }

   return false;
}

/* Compare two scalar half values and select between two full registers.
 *
 * Apple keeps the comparison narrow and the selected values full-width.  The
 * plain and absolute forms below are byte-exact applegpu_g17p witnesses from
 * `metal-nt` for `(a < b) ? t : f` and `(fabs(a) < b) ? t : f`:
 *
 *   plain  02 08 1f 08 80 04 13 c0
 *   abs(a) 02 08 1f 08 83 00 13 00 80 04 00 80 01 00
 *   abs(b) 02 08 1f 08 83 00 13 00 80 04 00 80 02 00
 *   both   02 08 1f 08 83 00 13 00 80 04 00 80 03 00
 *
 * Decomposition gives f43=f60=1 for the two half comparators, f77=f94=2 for
 * the two full selected values, and f47/f64 as their independent ABS bits.
 */
bool
agx3_alu_fcmp_select_f16_f16_f32(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[4],
   const enum agx3_alu_hint hints[4], const bool src_abs[2],
   unsigned condition, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       !src_abs || dest >= 256 || (dest & 1) ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask != 0 && wait_mask != 32))
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      const bool half = i < 2;
      if (src[i] >= 256 || (!half && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 2}, {42, 2}, {43, 1}, {59, 2},
      {60, 1}, {76, 2}, {77, 2}, {93, 2}, {94, 2},
   };
   struct agx3_field_kv fields[18] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {69, src[2]}, {86, src[3]}, {110, condition},
      {107, wait_mask == 32}, {47, src_abs[0]}, {64, src_abs[1]},
   };
   const unsigned cache[] = {44, 61, 78, 95};
   const unsigned discard[] = {46, 63, 80, 97};
   unsigned n = 10;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){cache[i],
                                           hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard[i],
                                           hints[i] == AGX3_ALU_HINT_DISCARD};
   }

   for (unsigned ep = 0; ep < 36; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, 10, fields, n, out, len))
         return true;
   }
   return false;
}

/* Native Metal keeps all four scalar half operands in their assigned halves:
 * src0/src1 are compared and src2/src3 are the selected alternatives.  The
 * instruction carries independent post-RA lifetime hints.  Only the no-wait
 * form is witnessed; callers retire scoreboard slots before composing it.
 */
/*
 * Compare a 32-bit float against a minifloat immediate and select either of
 * two 32-bit registers.
 *
 * This is what an ordinary `x < k ? a : b` becomes: NIR keeps the comparison
 * at 32 bits and Mesa carries the constant as a minifloat, so the packer sees
 * a mixed-width select with no register to widen.  Apple's own bytes, from
 * `metal-nt -arch applegpu_g17p` compiling `(a[i] OP 0.5f) ? c[i] : d[i]`:
 *
 *   <    12 03 0f a1 22 06 03 02 20 0a     f110 = 1
 *   >    12 03 0f a1 22 06 02 02 20 0a     f110 = 2
 *   >=   12 03 0f a1 26 06 02 02 20 0a     f110 = 5
 *   <=   12 03 0f a1 26 06 03 02 20 0a     f110 = 6
 *   ==   12 03 0f a1 26 06 00 02 20 0a     f110 = 0
 *   !=   12 03 0f a1 26 08 00 02 20 06     f110 = 0, selected operands swapped
 *
 * Six kernels differing only in the operator, and only f110 (and, for `!=`,
 * which operand lands in which select slot) changes: the condition selector is
 * literally `enum agx_fcond`, and it is orthogonal to the operand routing.
 * The last row is why an inverted condition needs no separate encoding --
 * Apple spells it by exchanging the two selected values.
 *
 * GTK's CICP conversion also reaches `0.5 >= fabs(x)`. Fresh G17P kernels
 * show Apple canonicalizing it to `fabs(x) <= 0.5` on long endpoint 35:
 *
 *   <=   02 03 07 a1 07 04 03 02 00 06 00 80 01 00
 *   >=   02 03 07 a1 07 04 02 02 00 06 00 80 01 00
 *   >    02 03 07 a1 03 04 02 02 00 06 00 80 01 00
 *   <    02 03 07 a1 03 04 03 02 00 06 00 80 01 00
 *   ==   02 03 07 a1 07 04 00 02 00 06 00 80 01 00
 *
 * Field47 is the only new semantic bit; operand fields, lifetime hints and
 * the condition retain their ordinary FSEL meanings.
 */
bool
agx3_alu_fcmp_select_f32_imm(enum agx_isa_encoding enc, uint16_t dest,
                             uint16_t src0, uint8_t minifloat, bool src0_abs,
                             uint16_t src2, uint16_t src3,
                             const enum agx3_alu_hint hints[4],
                             unsigned condition, bool dest_cache,
                             uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !hints ||
       dest >= 256 || src0 >= 256 || src2 >= 256 || src3 >= 256 ||
       (dest & 1) || (src0 & 1) || (src2 & 1) || (src3 & 1))
      return false;

   /* Only the five conditions Apple was witnessed emitting for this form:
    * eq = 0, lt = 1, gt = 2, ge = 5, le = 6, matching enum agx_fcond. */
   if (condition != 0 && condition != 1 && condition != 2 &&
       condition != 5 && condition != 6)
      return false;

   /* The compact endpoint exposes the full single-slot wait selector.  The
    * absolute endpoint is freshly witnessed with the native load slot (5);
    * callers retire any other slot before composing it. */
   if ((wait_mask & ~UINT16_C(0x3f)) || (wait_mask & (wait_mask - 1)))
      return false;
   if (src0_abs && wait_mask != 0 && wait_mask != 32)
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if ((unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* The compared immediate and the unused slot-1 register share a lifetime
    * that does not exist; refuse a hint on either. */
   if (hints[1] != AGX3_ALU_HINT_NONE)
      return false;

   struct agx3_field_kv fields[25];
   unsigned n = 0;
   fields[n++] = (struct agx3_field_kv){3, dest};
   fields[n++] = (struct agx3_field_kv){10, 2};
   fields[n++] = (struct agx3_field_kv){11, 2};
   fields[n++] = (struct agx3_field_kv){12, dest_cache};
   fields[n++] = (struct agx3_field_kv){35, src0};
   fields[n++] = (struct agx3_field_kv){42, 2};
   fields[n++] = (struct agx3_field_kv){43, 2};
   fields[n++] = (struct agx3_field_kv){44, hints[0] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){46, hints[0] == AGX3_ALU_HINT_DISCARD};
   if (src0_abs)
      fields[n++] = (struct agx3_field_kv){47, 1};
   fields[n++] = (struct agx3_field_kv){51, minifloat};
   fields[n++] = (struct agx3_field_kv){59, 1};
   fields[n++] = (struct agx3_field_kv){60, 2};
   fields[n++] = (struct agx3_field_kv){69, src2};
   fields[n++] = (struct agx3_field_kv){76, 2};
   fields[n++] = (struct agx3_field_kv){77, 2};
   fields[n++] = (struct agx3_field_kv){78, hints[2] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){80, hints[2] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){86, src3};
   fields[n++] = (struct agx3_field_kv){93, 2};
   fields[n++] = (struct agx3_field_kv){94, 2};
   fields[n++] = (struct agx3_field_kv){95, hints[3] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){97, hints[3] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){110, condition};
   for (unsigned i = 0; i < 6; ++i) {
      if (wait_mask & (1u << i)) {
         fields[n++] = (struct agx3_field_kv){(uint16_t)(102 + i), 1};
         break;
      }
   }

   return agx3_bind_compose(enc, AGX3_MN_fsel, src0_abs ? 35 : 27, NULL, 0,
                            fields, n, out, len);
}

bool
agx3_alu_fcmp_select_half(enum agx_isa_encoding enc, uint16_t dest,
                          const uint16_t src[4],
                          const enum agx3_alu_hint hints[4],
                          unsigned condition, bool dest_cache,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || !src || !hints ||
       dest >= 256 ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) || wait_mask != 0)
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   const struct agx3_field_kv base[] = {
      {10, 2}, {11, 1}, {42, 2}, {43, 1}, {59, 2}, {60, 1},
      {76, 2}, {77, 1}, {93, 2}, {94, 1},
   };
   const struct agx3_field_kv fields[] = {
      {3, dest}, {12, dest_cache}, {35, src[0]}, {52, src[1]},
      {69, src[2]}, {86, src[3]},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {61, hints[1] == AGX3_ALU_HINT_CACHE},
      {63, hints[1] == AGX3_ALU_HINT_DISCARD},
      {78, hints[2] == AGX3_ALU_HINT_CACHE},
      {80, hints[2] == AGX3_ALU_HINT_DISCARD},
      {95, hints[3] == AGX3_ALU_HINT_CACHE},
      {97, hints[3] == AGX3_ALU_HINT_DISCARD},
      {110, condition}, {107, 0},
   };

   for (unsigned ep = 0; ep < 28; ++ep) {
      uint8_t candidate[AGX3_ALU_MAX_LENGTH], candidate_len;
      if (!agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, 10, fields, 16,
                             candidate, &candidate_len))
         continue;

      /* Apple's G17P compiler canonicalizes scalar-half FSEL with bits62:63
       * set. Keep that exact choice. Destination cache is already field12;
       * byte0 bit3 is field3's low destination bit and must only come from
       * the requested half-register address.
       */
      if (candidate_len != 10)
         continue;
      candidate[7] |= 0xc0;
      for (unsigned i = 0; i < candidate_len; ++i)
         out[i] = candidate[i];
      *len = candidate_len;
      return true;
   }

   return false;
}

static bool
agx3_alu_m5_fsel_half(uint16_t dest, const uint16_t src[4],
                         const enum agx3_alu_hint hints[4], unsigned condition,
                         bool dest_cache, uint16_t wait_mask,
                         uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src || !hints || !out || !len || dest >= 256 ||
       (wait_mask & ~UINT16_C(0xff)) )
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* Native G17G min/max uses fseli16, with half compare operands and
    * half selected values. Conditions 3/7 preserve min/max NaN semantics;
    * ordered less/greater comparisons instead use conditions 1/2.
    * Live/dead half4 references pin the compact form and slot-8 endpoint. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {40, 2}, {41, 1},
      {57, 2}, {58, 1}, {71, 2}, {72, 1},
   };
   struct agx3_field_kv fields[25] = {
      {2, dest}, {11, dest_cache}, {16, src[0]}, {33, src[1]},
      {50, src[2]}, {64, src[3]}, {86, condition},
      {76, 3}, {77, 75},
   };
   const unsigned cache[] = {25, 42, 59, 73};
   const unsigned discard[] = {27, 44, 61, 75};
   unsigned n = 9;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){cache[i], hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard[i], hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < 8; ++i)
      fields[n++] = (struct agx3_field_kv){78 + i, !!(wait_mask & (1u << i))};
   for (unsigned ep = 0; ep <= 8; ++ep) {
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fseli16, ep,
                            base, 10, fields, n, out, len))
         return true;
   }
   return false;
}

static bool
agx3_alu_m5_fminmax_half(uint16_t dest, const uint16_t src[4],
                        const enum agx3_alu_hint hints[4], bool maximum,
                        bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src || src[0] != src[2] || src[1] != src[3])
      return false;
   return agx3_alu_m5_fsel_half(dest,src,hints,maximum ? 7 : 3,
                               dest_cache,wait_mask,out,len);
}

bool
agx3_alu_fcmp_select_half_gpu1(uint16_t dest, const uint16_t src[4],
                              const enum agx3_alu_hint hints[4],
                              unsigned condition, bool invert,
                              bool dest_cache, uint16_t wait_mask,
                              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src || !hints || (condition != 0 && condition != 1 && condition != 2 &&
                         condition != 5 && condition != 6))
      return false;
   uint16_t s[4] = {src[0],src[1],src[2],src[3]};
   enum agx3_alu_hint h[4] = {hints[0],hints[1],hints[2],hints[3]};
   if (invert) {
      s[2]=src[3];s[3]=src[2];h[2]=hints[3];h[3]=hints[2];
   }
   return agx3_alu_m5_fsel_half(dest,s,h,condition,dest_cache,wait_mask,out,len);
}

static bool
agx3_alu_fminmax_half(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[4],
                      const enum agx3_alu_hint hints[4], bool maximum,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C)
      return agx3_alu_m5_fminmax_half(dest, src, hints, maximum, dest_cache,
                                      wait_mask, out, len);

   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || (wait_mask & ~UINT16_C(0x3f)))
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   const struct agx3_field_kv base[] = {
      {10,2},{11,1},{42,2},{43,1},{59,2},{60,1},{76,2},{77,1},{93,2},{94,1},
   };
   struct agx3_field_kv fields[21] = {
      {3,dest},{12,dest_cache},{35,src[0]},{52,src[1]},
      {69,src[2]},{86,src[3]},{110,maximum ? 7 : 3},
   };
   const unsigned cache[] = {44,61,78,95};
   const unsigned discard[] = {46,63,80,97};
   unsigned n = 7;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){cache[i],
         hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard[i],
         hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < 6; ++i)
      fields[n++] =
         (struct agx3_field_kv){102 + i, !!(wait_mask & (1u << i))};
   for (unsigned ep = 0; ep < 28; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_fsel, ep, base, 10, fields, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_alu_fmin_half(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[4],
                   const enum agx3_alu_hint hints[4], bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   return agx3_alu_fminmax_half(enc, dest, src, hints, false, dest_cache,
                               wait_mask, out, len);
}

bool
agx3_alu_fmax_half(enum agx_isa_encoding enc, uint16_t dest,
                   const uint16_t src[4],
                   const enum agx3_alu_hint hints[4], bool dest_cache,
                   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                   uint8_t *len)
{
   return agx3_alu_fminmax_half(enc, dest, src, hints, true, dest_cache,
                               wait_mask, out, len);
}

/*
 * 32-bit tied fmin/fmax.
 *
 * The 32-bit destination form of a tied fcmpsel is the same Apple `fsel`
 * family the half encoder above uses; only the four width selectors change
 * (f11/f43/f60/f77/f94: 1 = one 16-bit half, 2 = a 32-bit pair).  Which
 * encoding profile Apple picks depends on the second operand:
 *
 *   fmax(a, b)     f32   12 03 1e 05 00 c0    fsel.ep7, src1 a register
 *   fmax(a, 0.0f)  f32   12 03 0a 81 00 c2    fsel.ep2, src1 a minifloat
 *
 * both witnessed from `metal-nt -arch applegpu_g17p`.  ep7 names the
 * destination in f3, ep2 in f17; every other field id is shared.  The
 * immediate is a plain AGX minifloat and appears twice, in f51 and f85,
 * because the compared value and the selected value are the same operand --
 * which is exactly what makes the instruction a minimum or a maximum rather
 * than a general select.
 *
 * The wait mask is the ordinary three-bit scoreboard selector at the top of
 * the last byte: slot n sets f(101 + n).  Only a single slot is encodable in
 * this form, so a multi-slot wait must be retired separately.
 */
static bool
agx3_alu_fminmax32_gpu1(uint16_t dest, uint16_t src0, uint16_t src2,
                       bool immediate, uint16_t src1, uint16_t src3,
                       const enum agx3_alu_hint hints[4], bool maximum,
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* M5 fseli32 compares floating values but selects 32-bit payloads. Zero
    * has the same representation in both immediate domains; other minifloat
    * constants need separately grounded result-immediate handling. */
   if ((immediate && (src1 || src3 || hints[1] || hints[3])) ||
       ((dest | src0 | src2 | (immediate ? 0 : src1 | src3)) & 1))
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{23,2},{24,2},{40,immediate ? 1 : 2},{41,2},
      {57,2},{58,2},{71,immediate ? 1 : 2},{72,2},{76,3},{77,76},
   };
   struct agx3_field_kv fields[23] = {
      {2,dest},{11,dest_cache},{16,src0},{immediate ? 32 : 33,src1},
      {50,src2},{immediate ? 63 : 64,src3},{86,maximum ? 7 : 3},
   };
   unsigned n=7;
   const uint16_t cache_fields[4] = {25,42,59,73};
   const uint16_t discard_fields[4] = {27,44,61,75};
   for (unsigned i=0;i<4;++i) {
      fields[n++] = (struct agx3_field_kv){cache_fields[i],hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){discard_fields[i],hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i=0;i<8;++i)
      fields[n++] = (struct agx3_field_kv){78+i,!!(wait_mask & (1u<<i))};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,AGX3_MN_fseli32,4,
                            base,12,fields,n,out,len);
}

static bool
agx3_alu_fminmax32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                   uint16_t src2, bool src1_immediate, uint16_t src1,
                   uint16_t src3, const enum agx3_alu_hint hints[4],
                   bool maximum, bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!gpu1 && enc != AGX_ISA_ENCODING_AGX3B) || !hints || !out || !len || dest >= 256 ||
       src0 >= 256 || src2 >= 256 || src1 >= 256 || src3 >= 256)
      return false;
   if (wait_mask & ~(gpu1 ? UINT16_C(0xff) : UINT16_C(0x3f)))
      return false;
   /* One scoreboard slot at a time: the selector is a three-bit code. */
   if (!gpu1 && (wait_mask & (wait_mask - 1)))
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if ((unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }
   if (src1_immediate && (src1 > 255 || src3 > 255))
      return false;
   if (gpu1)
      return agx3_alu_fminmax32_gpu1(dest,src0,src2,src1_immediate,src1,src3,
                                    hints,maximum,dest_cache,wait_mask,out,len);

   struct agx3_field_kv fields[24];
   unsigned n = 0;
   const uint16_t dest_field = src1_immediate ? 17 : 3;
   const uint16_t dest_kind = src1_immediate ? 24 : 10;
   const uint16_t dest_size = src1_immediate ? 25 : 11;
   const uint16_t dest_cache_field = src1_immediate ? 26 : 12;
   if (src1_immediate)
      fields[n++] = (struct agx3_field_kv){0, 1};
   fields[n++] = (struct agx3_field_kv){dest_field, dest};
   fields[n++] = (struct agx3_field_kv){dest_kind, 2};
   fields[n++] = (struct agx3_field_kv){dest_size, 2};
   fields[n++] = (struct agx3_field_kv){dest_cache_field, dest_cache};

   /* src0 and src2 are the tied register operand. */
   fields[n++] = (struct agx3_field_kv){35, src0};
   fields[n++] = (struct agx3_field_kv){42, 2};
   fields[n++] = (struct agx3_field_kv){43, 2};
   fields[n++] = (struct agx3_field_kv){44, hints[0] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){46, hints[0] == AGX3_ALU_HINT_DISCARD};
   fields[n++] = (struct agx3_field_kv){69, src2};
   fields[n++] = (struct agx3_field_kv){76, 2};
   fields[n++] = (struct agx3_field_kv){77, 2};
   fields[n++] = (struct agx3_field_kv){78, hints[2] == AGX3_ALU_HINT_CACHE};
   fields[n++] = (struct agx3_field_kv){80, hints[2] == AGX3_ALU_HINT_DISCARD};

   /* src1 and src3 are the tied second operand: a register pair or the same
    * minifloat deposited twice. */
   fields[n++] = (struct agx3_field_kv){src1_immediate ? 51 : 52, src1};
   fields[n++] = (struct agx3_field_kv){59, src1_immediate ? 1 : 2};
   fields[n++] = (struct agx3_field_kv){60, 2};
   fields[n++] = (struct agx3_field_kv){src1_immediate ? 85 : 86, src3};
   fields[n++] = (struct agx3_field_kv){93, src1_immediate ? 1 : 2};
   fields[n++] = (struct agx3_field_kv){94, 2};
   if (!src1_immediate) {
      fields[n++] = (struct agx3_field_kv){61, hints[1] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){63, hints[1] == AGX3_ALU_HINT_DISCARD};
      fields[n++] = (struct agx3_field_kv){95, hints[3] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){97, hints[3] == AGX3_ALU_HINT_DISCARD};
   } else if (hints[1] != AGX3_ALU_HINT_NONE ||
              hints[3] != AGX3_ALU_HINT_NONE) {
      return false;             /* an immediate has no lifetime */
   }

   struct agx3_field_kv all[32];
   unsigned m = 0;
   for (unsigned i = 0; i < n; ++i)
      all[m++] = fields[i];
   for (unsigned i = 0; i < 6; ++i) {
      if (wait_mask & (1u << i))
         all[m++] = (struct agx3_field_kv){(uint16_t)(102 + i), 1};
   }
   all[m++] = (struct agx3_field_kv){110, maximum ? 7 : 3};

   return agx3_bind_compose(enc, AGX3_MN_fsel, src1_immediate ? 2 : 7, NULL, 0, all,
                            m, out, len);
}

bool
agx3_alu_fmin_f32(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[4], const enum agx3_alu_hint hints[4],
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src)
      return false;
   return agx3_alu_fminmax32(enc, dest, src[0], src[2], false, src[1], src[3],
                             hints, false, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmax_f32(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[4], const enum agx3_alu_hint hints[4],
                  bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!src)
      return false;
   return agx3_alu_fminmax32(enc, dest, src[0], src[2], false, src[1], src[3],
                             hints, true, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmax_abs_f32(enum agx_isa_encoding enc, uint16_t dest,
                     const uint16_t src[2],
                     const enum agx3_alu_hint hints[2], bool dest_cache,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !hints || !out || !len ||
       dest >= 256 || (dest & 1))
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      if (src[i] >= 256 || (src[i] & 1) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE)
         return false;
   }

   /* Matched native Minecraft terrain: 020b3b01830a0000880000000200.
    * f64 and f98 apply ABS to both halves of the second tied operand.
    * The f0 destination placement selects floating values, unlike the
    * general integer-valued select. Keep this separate from that encoder.
    * No negate, half-width, immediate, or producer-wait form is admitted.
    */
   const struct agx3_field_kv f[] = {
      {0, 1}, {17, dest}, {24, 2}, {25, 2}, {26, dest_cache},
      {35, src[0]}, {42, 2}, {43, 2},
      {44, hints[0] == AGX3_ALU_HINT_CACHE},
      {46, hints[0] == AGX3_ALU_HINT_DISCARD},
      {52, src[1]}, {59, 2}, {60, 2},
      {61, hints[1] == AGX3_ALU_HINT_CACHE},
      {63, hints[1] == AGX3_ALU_HINT_DISCARD}, {64, 1},
      {69, src[0]}, {76, 2}, {77, 2},
      {78, hints[0] == AGX3_ALU_HINT_CACHE},
      {80, hints[0] == AGX3_ALU_HINT_DISCARD},
      {86, src[1]}, {93, 2}, {94, 2},
      {95, hints[1] == AGX3_ALU_HINT_CACHE},
      {97, hints[1] == AGX3_ALU_HINT_DISCARD}, {98, 1}, {110, 7},
   };
   return agx3_bind_compose(enc, AGX3_MN_fsel, 30, NULL, 0, f,
                           sizeof(f) / sizeof(f[0]), out, len);
}

bool
agx3_alu_fmin_f32_imm(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      uint8_t minifloat, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const enum agx3_alu_hint hints[4] = {hint, AGX3_ALU_HINT_NONE, hint,
                                        AGX3_ALU_HINT_NONE};
   return agx3_alu_fminmax32(enc, dest, src, src, true, minifloat, minifloat,
                             hints, false, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmax_f32_imm(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      uint8_t minifloat, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const enum agx3_alu_hint hints[4] = {hint, AGX3_ALU_HINT_NONE, hint,
                                        AGX3_ALU_HINT_NONE};
   return agx3_alu_fminmax32(enc, dest, src, src, true, minifloat, minifloat,
                             hints, true, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
             bool dest_half, bool src_half, bool src_abs, bool src_neg,
             enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* 192, not 128. The register fields of this form are range-limited by
    * the encoder tables -- f2 and f18 both accept [0, 191], 192
    * distinct encodings with no aliasing, and refuse 192 -- and the composer
    * enforces that, so the gate here only has to agree with it. At 128 it did
    * not: AGX_NUM_REGS is 256 half-registers, so a shader under any real
    * register pressure gets a reciprocal whose destination is above 128, and
    * hk_shader.c turns a packer refusal into VK_ERROR_FEATURE_NOT_PRESENT for
    * the whole pipeline. Three of the ten distinct programs three.js compiles
    * for its first few examples died on `rcp [d32:reg140 s0=32:reg86.abs]`. */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned regs = m5 ? 256 : 192;
   const unsigned slots = m5 ? 8 : 6;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !out || !len ||
       dest >= regs || src >= regs ||
       (!dest_half && (dest & 1)) || (!src_half && (src & 1)) ||
       (wait_mask & ~((1u << slots) - 1)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   /* Fresh applegpu_g17p kernels cover all four result/source width pairs.
    * f10 and f26 independently select a scalar half (1) or float (2), while
    * f30/f31 fold ABS/NEG into the source. Lifetimes, result caching, and all
    * six producer slots are orthogonal to those arithmetic fields. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, dest_half ? 1 : 2}, {25, 2},
      {26, src_half ? 1 : 2}, {43, 2},
   };
   struct agx3_field_kv fields[17] = {
      {2, dest},
      {11, dest_cache},
      {18, src},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
      {30, src_abs},
      {31, src_neg},
   };
   unsigned n = 7;
   if (m5) {
      fields[n++] = (struct agx3_field_kv){47, 3};
      fields[n++] = (struct agx3_field_kv){48, 64};
   }
   for (unsigned slot = 0; slot < slots; ++slot)
      fields[n++] = (struct agx3_field_kv){49 + slot,
                                               !!(wait_mask & (1u << slot))};

   return agx3_bind_compose(enc, AGX3_MN_f32rcp, 0, base, 5, fields, n, out, len);
}

bool
agx3_alu_f32rcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_rcp(enc, dest, src, false, false, false, false, hint,
                       dest_cache, wait_mask, out, len);
}

static bool agx3_alu_unop(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
                          uint16_t dest, uint16_t src, bool dest_half,
                          bool src_half, enum agx3_alu_hint hint,
                          bool dest_cache, uint16_t wait_mask, bool src_neg,
                          bool src_abs, uint8_t out[AGX3_ALU_MAX_LENGTH],
                          uint8_t *len);

bool
agx3_alu_f32rsqrt_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, bool src_neg, bool src_abs,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* f10 and f26 are the destination's and source's widths; the half form
    * sets both to 1, the 32-bit form to 2. f30 and f31 fold the source's
    * absolute value and negation in, exactly as on f32log2/f32exp2/f32rcp;
    * see agx3_alu_core.h for the four witnessed bytes. */
   if (enc == AGX_ISA_ENCODING_AGX3C)
      return agx3_alu_unop(enc, AGX3_MN_f32rsqrt, dest, src, false, false, hint,
                           dest_cache, wait_mask, src_neg, src_abs, out, len);
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
      {30, src_abs}, {31, src_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[7 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, AGX3_MN_f32rsqrt, 0, base, 4, fields, 13, out, len);
}

bool
agx3_alu_unary_m5(enum agx3_funop op, uint16_t dest, uint16_t src,
                   bool dest_half, bool src_half, bool src_abs,
                   bool src_neg, bool saturate, enum agx3_alu_hint hint,
                   bool dest_cache, uint16_t wait_mask,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || (unsigned)op > AGX3_FUNOP_EXP2 || dest >= 256 || src >= 256 ||
       (!dest_half && (dest & 1)) || (!src_half && (src & 1)) ||
       /* No direct narrowing witness; captured probes convert separately. */
       (dest_half && !src_half) || (wait_mask & ~0xffu) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const bool rounding = op <= AGX3_FUNOP_RINT;
   enum agx3_mnemonic mnemonic = rounding ? AGX3_MN_f32rint :
                          op == AGX3_FUNOP_LOG2 ? AGX3_MN_f32log2 : AGX3_MN_f32exp2;
   const unsigned opcode = rounding ? 65 : op == AGX3_FUNOP_LOG2 ? 59 : 52;
   /* Fresh G17G profiles prove odd half registers, half-to-full inputs,
    * f14 saturation, f30/f31 modifiers, actual lifetimes and zero/slot8
    * waits on this same native instruction family. */
   struct agx3_field_kv base[7] = {
      {9, 2}, {10, dest_half ? 1 : 2}, {25, 2},
      {26, src_half ? 1 : 2}, {33, 3}, {34, opcode},
   };
   unsigned nbase = 6;
   if (rounding)
      base[nbase++] = (struct agx3_field_kv){43, 3 - op};
   struct agx3_field_kv fields[16] = {
      {2, dest}, {18, src}, {11, dest_cache}, {14, saturate},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD}, {30, src_abs}, {31, src_neg},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 8; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){35 + slot, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, mnemonic, 0, base, nbase,
                            fields, n, out, len);
}

/*
 * The gradients are the same f33=3 class as rint/log2/exp2 above (f34 49 and
 * 50): the destination and source registers, widths, modifiers, saturation
 * and wait slots sit in the same fields. Native G17G fragment programs use
 * 16-bit gradients with odd half registers and sources above h127, and wait
 * on slots 0, 1, 2 and 7 directly. The destination and source share one width
 * there; no mixed-width gradient was observed, so none is encoded.
 *
 * The cache hints keep the canonical gradient values (f11 set, and f27 set
 * on dfdx only), as the 32-bit low-register encoding always has.
 */
bool
agx3_alu_derivative_m5(bool dfdy, uint16_t dest, uint16_t src, bool half,
                       bool src_abs, bool src_neg, bool saturate,
                       uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                       uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 ||
       (!half && ((dest & 1) || (src & 1))) || (wait_mask & ~0xffu))
      return false;

   struct agx3_field_kv base[8] = {
      {9, 2}, {10, half ? 1 : 2}, {11, 1}, {25, 2}, {26, half ? 1 : 2},
      {33, 3}, {34, dfdy ? 50 : 49},
   };
   unsigned nbase = 7;
   if (!dfdy)
      base[nbase++] = (struct agx3_field_kv){27, 1};

   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {14, saturate}, {30, src_abs}, {31, src_neg},
   };
   unsigned n = 5;
   for (unsigned slot = 0; slot < 8; ++slot)
      if (wait_mask & (1u << slot))
         fields[n++] = (struct agx3_field_kv){35 + slot, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,
                            dfdy ? AGX3_MN_f32dfdy : AGX3_MN_f32dfdx, 0, base, nbase,
                            fields, n, out, len);
}

/* log2/exp2 share independently selected f10 destination and f26 source
 * widths. Full-width forms have all six recovered wait bits; any half-width
 * log2 is currently grounded for no wait and slot 6. */
static bool
agx3_alu_unop(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
              uint16_t dest, uint16_t src, bool dest_half, bool src_half,
              enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
              bool src_neg, bool src_abs,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      if (mnemonic == AGX3_MN_f32rsqrt) {
         if (!out || !len || dest >= 256 || src >= 256 ||
             (!dest_half && (dest & 1)) || (!src_half && (src & 1)) ||
             (wait_mask & ~0xffu) || (unsigned)hint > AGX3_ALU_HINT_CACHE)
            return false;
         const struct agx3_field_kv base[] = {
            {9, 2},  {10, dest_half ? 1 : 2},
            {25, 2}, {26, src_half ? 1 : 2},
            {33, 3}, {34, 66},
         };
         struct agx3_field_kv f[15] = {
            {2, dest},
            {18, src},
            {11, dest_cache},
            {27, hint == AGX3_ALU_HINT_CACHE},
            {29, hint == AGX3_ALU_HINT_DISCARD},
            {30, src_abs},
            {31, src_neg},
         };
         for (unsigned i = 0; i < 8; i++)
            f[7 + i] =
               (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
         return agx3_bind_compose(enc, mnemonic, 0, base, 6, f, 15, out, len);
      }
      if (mnemonic != AGX3_MN_f32log2 && mnemonic != AGX3_MN_f32exp2)
         return false;
      return agx3_alu_unary_m5(mnemonic == AGX3_MN_f32log2
                                 ? AGX3_FUNOP_LOG2 : AGX3_FUNOP_EXP2,
                                dest, src, dest_half, src_half, src_abs,
                                src_neg, false, hint, dest_cache, wait_mask,
                                out, len);
   }

   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (!dest_half && (dest & 1)) ||
       (!src_half && (src & 1)) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, dest_half ? 1 : 2}, {25, 2},
      {26, src_half ? 1 : 2},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
      {30, src_abs}, {31, src_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[7 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, mnemonic, 0, base, 4, fields, 13, out, len);
}

bool
agx3_alu_log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
              bool dest_half, bool src_half, bool src_abs, bool src_neg,
              enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* applegpu_g17p's f32log2 operand grammar names halves 0..191. Full
    * scalars additionally require their ordinary even-pair alignment. */
   const unsigned registers = enc == AGX_ISA_ENCODING_AGX3C ? 256 : 192;
   if (dest >= registers || src >= registers)
      return false;

   return agx3_alu_unop(enc, AGX3_MN_f32log2, dest, src, dest_half, src_half, hint,
                        dest_cache, wait_mask, src_neg, src_abs, out, len);
}

bool
agx3_alu_f32exp2_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, bool src_neg, bool src_abs,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_unop(enc, AGX3_MN_f32exp2, dest, src, false, false, hint,
                        dest_cache, wait_mask, src_neg, src_abs, out, len);
}

/*
 * The 32-bit rounding modes. Field-for-field the half forms below with f10
 * and f26 raised from 1 to 2, exactly as the 32-bit logarithm relates to the
 * half one, plus the source modifiers f30 (absolute value) and f31 (negate)
 * that every member of this form family carries. f43 is the mode.
 */
static bool
agx3_alu_f32_rint(enum agx_isa_encoding enc, uint64_t mode, uint16_t dest,
                  uint16_t src, enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, bool src_neg, bool src_abs,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      if (mode > 3)
         return false;
      return agx3_alu_unary_m5(AGX3_FUNOP_RINT - mode, dest, src, false,
                                false, src_abs, src_neg, false, hint,
                                dest_cache, wait_mask, out, len);
   }

   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || (dest & 1) || (src & 1) ||
       (wait_mask & ~UINT16_C(0x3f)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 2}, {26, 2}, {43, mode},
   };
   struct agx3_field_kv fields[13] = {
      {2, dest}, {18, src}, {11, dest_cache},
      {27, hint == AGX3_ALU_HINT_CACHE},
      {29, hint == AGX3_ALU_HINT_DISCARD},
      {30, src_abs}, {31, src_neg},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[7 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, AGX3_MN_f32rint, 0, base, 5, fields, 13, out, len);
}

bool
agx3_alu_f32floor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                  bool src_neg, bool src_abs,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_rint(enc, 3, dest, src, hint, dest_cache, wait_mask,
                            src_neg, src_abs, out, len);
}

bool
agx3_alu_f32ceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                 bool src_neg, bool src_abs,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_rint(enc, 2, dest, src, hint, dest_cache, wait_mask,
                            src_neg, src_abs, out, len);
}

bool
agx3_alu_f32roundeven(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask, bool src_neg, bool src_abs,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_f32_rint(enc, 0, dest, src, hint, dest_cache, wait_mask,
                            src_neg, src_abs, out, len);
}

bool
agx3_alu_hrsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache,
                uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                uint8_t *len)
{
   return agx3_alu_unop(enc, AGX3_MN_f32rsqrt, dest, src, true, true, hint,
                        dest_cache, wait_mask, false, false, out, len);
}

bool
agx3_alu_hexp2_neg(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                   enum agx3_alu_hint hint, bool dest_cache,
                   uint16_t wait_mask, bool src_neg,
                   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_unop(enc, AGX3_MN_f32exp2, dest, src, true, true, hint,
                        dest_cache, wait_mask, src_neg, false, out, len);
}

static bool
agx3_alu_hrint(enum agx_isa_encoding enc, uint64_t mode, uint16_t dest,
               uint16_t src, enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !out || !len || dest >= 256 ||
       src >= 256 || mode > 3 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;
   /* f32rint modes 0..3 are round-even, trunc, ceil and floor. Six-load
    * applegpu_g17p kernels witness f35..f40 independently of the mode. */
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,1},{43,mode},
   };
   struct agx3_field_kv fields[11] = {
      {2,dest},{18,src},{11,dest_cache},
      {27,hint == AGX3_ALU_HINT_CACHE},
      {29,hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[5 + i] =
         (struct agx3_field_kv){35 + i, !!(wait_mask & (1u << i))};
   return agx3_bind_compose(enc, AGX3_MN_f32rint, 0, base, 5, fields, 11,
                            out, len);
}

bool
agx3_alu_hroundeven(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_hrint(enc, 0, dest, src, hint, dest_cache, wait_mask,
                         out, len);
}

bool
agx3_alu_htrunc(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache,
                uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                uint8_t *len)
{
   return agx3_alu_hrint(enc, 1, dest, src, hint, dest_cache, wait_mask,
                         out, len);
}

bool
agx3_alu_hceil(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   return agx3_alu_hrint(enc, 2, dest, src, hint, dest_cache, wait_mask,
                         out, len);
}

bool
agx3_alu_hfloor(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                enum agx3_alu_hint hint, bool dest_cache,
                uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                uint8_t *len)
{
   return agx3_alu_hrint(enc, 3, dest, src, hint, dest_cache, wait_mask,
                         out, len);
}

/* Native float32 multiplication rounds directly to the half destination. */
bool
agx3_alu_fmul_narrow(enum agx_isa_encoding enc, uint16_t dest,
                     const uint16_t src[2], const enum agx3_alu_hint hints[2],
                     bool dest_cache, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !src || !hints || !out ||
       !len || dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       ((src[0] | src[1]) & 1) || (wait_mask & ~(m5 ? 0xffu : 0x3fu)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{25,2},{26,2},{42,2},{43,2},{60,2},{74,2},
   };
   struct agx3_field_kv fields[18] = {
      {2, dest},
      {18, src[0]},
      {35, src[1]},
      {11, dest_cache},
      {27, hints[0] == AGX3_ALU_HINT_CACHE},
      {29, hints[0] == AGX3_ALU_HINT_DISCARD},
      {44, hints[1] == AGX3_ALU_HINT_CACHE},
      {46, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i=0;i<6;i++)
      fields[8+i] = (struct agx3_field_kv){80+i,!!(wait_mask & (1u<<i))};
   if (m5) {
      fields[14] = (struct agx3_field_kv){78, 3};
      fields[15] = (struct agx3_field_kv){79, 62};
      fields[16] = (struct agx3_field_kv){86, !!(wait_mask & 64)};
      fields[17] = (struct agx3_field_kv){87, !!(wait_mask & 128)};
   }
   for (unsigned ep = 0; ep < (m5 ? 4 : 2); ep++)
      if (agx3_bind_compose(enc, AGX3_MN_f32mul, ep, base, 8, fields, m5 ? 18 : 14,
                            out, len))
         return true;
   return false;
}

/*
 * Shared body for the scalar-half register/register multiply and add.
 *
 * G17P carries the float saturate modifier in f14. The compact ep0/ep1
 * encoding points have no such field, so a saturating half operation lands on
 * the two-byte-longer ep2 form; the composer fails closed on f14 below that,
 * which is what selects the point. Every other field -- destination, both
 * source registers, the destination cache bit, the two source lifetime bits
 * and all six scoreboard waits -- is the same map the unsaturated form uses,
 * so one body serves both. Passing f14 = 0 is byte-identical to omitting it at
 * every encoding point, so the unsaturated callers are unchanged.
 *
 * Apple witnesses, metal-nt -arch applegpu_g17p, macOS 26 Metal toolchain:
 *    a[i] * b[i]                     10 03 1d 02 00 c0            f16mul.ep1
 *    saturate(a[i] * b[i])           10 03 1d 02 01 00 00 82      f16mul.ep2
 *    a[i] + b[i]                     10 03 1c 02 00 c0            f16add.ep1
 *    saturate(a[i] + b[i])           10 03 1c 02 01 00 00 82      f16add.ep2
 *    saturate(x * y), src1 live      18 03 0d 02 01 00 00 82      f45 = 0
 *    saturate(x * y), src0 live      18 02 15 03 01 00 00 82      f28 = 0
 * Each saturating vector differs from its unsaturated partner in f14 alone.
 */
static bool
agx3_alu_h_binop(enum agx_isa_encoding enc, enum agx3_mnemonic mnemonic,
                 unsigned max_ep, uint16_t dest, const uint16_t src[2],
                 const enum agx3_alu_hint hints[2], bool dest_cache,
                 bool saturate, const bool src_abs[2], const bool src_neg[2], uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) ||
       !src || !hints || !src_abs || !out || !len || dest >= 256 ||
       src[0] >= 256 || src[1] >= 256 ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1}, {41, 2}, {42, 1},
   };
   struct agx3_field_kv fields[24] = {
      {2, dest},
      {17, src[0]},
      {34, src[1]},
      {11, dest_cache},
      {14, saturate},
      {26, hints[0] == AGX3_ALU_HINT_CACHE},
      {28, hints[0] == AGX3_ALU_HINT_DISCARD},
      {43, hints[1] == AGX3_ALU_HINT_CACHE},
      {45, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 9;
   if (src_abs[0])
      fields[n++] = (struct agx3_field_kv){29, 1};
   if (src_abs[1])
      fields[n++] = (struct agx3_field_kv){46, 1};
   if (src_neg && src_neg[0])
      fields[n++] = (struct agx3_field_kv){30, 1};
   if (src_neg && src_neg[1])
      fields[n++] = (struct agx3_field_kv){47, 1};
   if (m5) {
      fields[n++] = (struct agx3_field_kv){49, 3};
      fields[n++] = (struct agx3_field_kv){50, mnemonic == AGX3_MN_f16mul ? 47 : 41};
   }
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      fields[n++] =
         (struct agx3_field_kv){51 + i, !!(wait_mask & (1u << i))};

   for (unsigned ep = 0; ep < (m5 ? 4 : max_ep); ++ep) {
      if (agx3_bind_compose(enc, mnemonic, ep, base, 6, fields, n, out, len))
         return true;
   }

   return false;
}

bool
agx3_alu_hmul(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[2], const enum agx3_alu_hint hints[2],
              bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool src_abs[2] = {false, false};
   return agx3_alu_h_binop(enc, AGX3_MN_f16mul, 2, dest, src, hints, dest_cache,
                           false, src_abs, NULL, wait_mask, out, len);
}

bool
agx3_alu_hmul_abs(enum agx_isa_encoding enc, uint16_t dest,
                  const uint16_t src[2],
                  const enum agx3_alu_hint hints[2],
                  const bool src_abs[2], bool dest_cache,
                  uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_h_binop(enc, AGX3_MN_f16mul, 4, dest, src, hints, dest_cache,
                           false, src_abs, NULL, wait_mask, out, len);
}

bool
agx3_alu_hmul_saturate(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2], const enum agx3_alu_hint hints[2],
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool src_abs[2] = {false, false};
   return agx3_alu_h_binop(enc, AGX3_MN_f16mul, 3, dest, src, hints, dest_cache,
                           true, src_abs, NULL, wait_mask, out, len);
}

bool
agx3_alu_hmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, uint16_t immediate,
                        enum agx3_alu_hint hint, bool dest_cache,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !out || !len || dest >= 256 ||
       src >= 256 || immediate > UINT8_MAX ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /* Fresh G17P Metal witnesses place the minifloat in field16, retain the
    * ordinary half register/lifetime fields, and carry all six wait bits.
    * G17G uses the same compact operands and its f16mul ep3 selectors;
    * native constant sweeps and live/dead half4 vectors pin slot 8. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 1}, {25, 1}, {41, 2}, {42, 1},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest},
      {16, immediate},
      {34, src},
      {11, dest_cache},
      {43, hint == AGX3_ALU_HINT_CACHE},
      {45, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 6;
   if (m5) {
      fields[n++] = (struct agx3_field_kv){49, 3};
      fields[n++] = (struct agx3_field_kv){50, 47};
   }
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      fields[n++] = (struct agx3_field_kv){51 + i,
                                            !!(wait_mask & (1u << i))};

   for (unsigned ep = 0; ep < (m5 ? 4 : 2); ++ep)
      if (agx3_bind_compose(enc, AGX3_MN_f16mul, ep, base, 6, fields, n, out,
                            len))
         return true;

   return false;
}

/* Scalar half add uses the register, lifetime and six-slot wait fields pinned
 * by fresh G17P add_dead/add_live Metal witnesses. */
bool
agx3_alu_hadd(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[2], const enum agx3_alu_hint hints[2],
              bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool src_abs[2] = {false, false};
   return agx3_alu_h_binop(enc, AGX3_MN_f16add, 4, dest, src, hints, dest_cache,
                           false, src_abs, NULL, wait_mask, out, len);
}

bool
agx3_alu_hadd_mod_m5(uint16_t dest, const uint16_t src[2],
                     const enum agx3_alu_hint hints[2], const bool src_abs[2],
                     const bool src_neg[2], bool dest_cache, bool saturate,
                     uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                     uint8_t *len)
{
   return agx3_alu_h_binop(AGX_ISA_ENCODING_AGX3C, AGX3_MN_f16add, 4, dest, src,
                           hints, dest_cache, saturate, src_abs, src_neg,
                           wait_mask, out, len);
}

bool
agx3_alu_half_add_imm_to_f32_m5(uint16_t dest, uint16_t src, uint8_t immediate,
                                bool src_abs, bool src_neg,
                                enum agx3_alu_hint hint, bool dest_cache,
                                uint16_t wait_mask,
                                uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 || (dest & 1) ||
       (wait_mask & ~0xffu) || (unsigned)hint > AGX3_ALU_HINT_CACHE)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {25, 1}, {26, 2}, {42, 2}, {43, 1}, {50, 3}, {51, 48},
   };
   struct agx3_field_kv f[16] = {
      {2, dest},
      {11, dest_cache},
      {17, immediate},
      {35, src},
      {44, hint == AGX3_ALU_HINT_CACHE},
      {46, hint == AGX3_ALU_HINT_DISCARD},
      {47, src_abs},
      {48, src_neg},
   };
   for (unsigned i = 0; i < 8; i++)
      f[8 + i] = (struct agx3_field_kv){52 + i, !!(wait_mask & (1u << i))};
   for (unsigned ep = 0; ep < 4; ep++)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_f32add, ep, base, 8, f,
                            16, out, len))
         return true;
   return false;
}

bool
agx3_alu_hadd_signed(enum agx_isa_encoding enc, uint16_t dest,
                      const uint16_t src[2], const enum agx3_alu_hint hints[2],
                      const bool src_neg[2], bool dest_cache,
                      uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                      uint8_t *len)
{
   const bool src_abs[] = {false, false};
   return agx3_alu_h_binop(enc, AGX3_MN_f16add, 4, dest, src, hints, dest_cache,
                           false, src_abs, src_neg, wait_mask, out, len);
}

bool
agx3_alu_hadd_saturate(enum agx_isa_encoding enc, uint16_t dest,
                       const uint16_t src[2], const enum agx3_alu_hint hints[2],
                       bool dest_cache, uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool src_abs[2] = {false, false};
   return agx3_alu_h_binop(enc, AGX3_MN_f16add, 4, dest, src, hints, dest_cache,
                           true, src_abs, NULL, wait_mask, out, len);
}

bool
agx3_alu_hadd_neg(enum agx_isa_encoding enc, uint16_t dest,
                 const uint16_t src[2], const enum agx3_alu_hint hints[2],
                 unsigned neg_mask, bool saturate, bool dest_cache,
                 uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3C || !src || !hints || neg_mask > 3)
      return false;

   /* Native G17G x-y, -x-y and saturate(x-y) vectors pin source negate
    * fields 47/30, saturation and slot-8 dependencies. Canonicalize a
    * single negation onto source 1, retaining each operand's lifetime. */
   const bool swap = neg_mask == 1;
   const uint16_t ordered_src[] = {src[swap], src[!swap]};
   const enum agx3_alu_hint ordered_hints[] = {hints[swap], hints[!swap]};
   const unsigned negate = swap ? 2 : neg_mask;
   const bool src_abs[2] = {false, false};
   const bool src_neg[2] = {negate & 1, negate & 2};
   return agx3_alu_h_binop(enc, AGX3_MN_f16add, 4, dest, ordered_src, ordered_hints,
                          dest_cache, saturate, src_abs, src_neg, wait_mask,
                          out, len);
}

bool
agx3_alu_hadd_abs_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool src_neg, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3C || !out || !len || dest >= 128 ||
       src >= 128 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~UINT16_C(0xff)))
      return false;

   /* Native910 abs/negabs keeps the half input and half result in f16add,
    * with -0 in operand 1. Both lifetimes and slot0/slot7 waits are witnessed. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1}, {41, 1}, {42, 1},
      {49, 3}, {50, 41},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {17, src}, {33, UINT8_C(0x80)}, {29, 1},
      {30, src_neg}, {11, dest_cache},
      {26, hint == AGX3_ALU_HINT_CACHE},
      {28, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned slot = 0; slot < 8; ++slot)
      fields[n++] = (struct agx3_field_kv){51 + slot,
                                           !!(wait_mask & (1u << slot))};
   for (unsigned ep = 2; ep <= 3; ++ep)
      if (agx3_bind_compose(enc,AGX3_MN_f16add,ep,base,8,fields,n,out,len))
         return true;
   return false;
}

bool
agx3_alu_half_add_negzero_m5(uint16_t dest, uint16_t src, bool src_abs,
                             bool src_neg, bool saturate, enum agx3_alu_hint hint,
                             bool dest_cache, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || (wait_mask & ~0xffu))
      return false;

   /* G17G device-half witnesses: abs, negate and negative-abs keep the
    * source in a scalar half and use the native modifier fields.
    * saturate(half), with or without either modifier, is this same form
    * with f14 set (port/hsatwit k_hsat, k_hsat_neg, k_hsat_abs). */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1}, {41, 1}, {42, 1}, {49, 3}, {50, 41},
   };
   struct agx3_field_kv fields[17] = {
      {2, dest},
      {17, src},
      {33, 0x80},
      {11, dest_cache},
      {26, hint == AGX3_ALU_HINT_CACHE},
      {28, hint == AGX3_ALU_HINT_DISCARD},
      {29, src_abs},
      {30, src_neg},
   };
   for (unsigned i = 0; i < 8; ++i)
      fields[8 + i] = (struct agx3_field_kv){51 + i, !!(wait_mask & (1u << i))};
   fields[16] = (struct agx3_field_kv){14, saturate};
   for (unsigned ep = 0; ep < 4; ++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_f16add, ep, base, 8,
                            fields, 17, out, len))
         return true;
   return false;
}

bool
agx3_alu_hadd_neg_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool g17g = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!g17g && enc != AGX_ISA_ENCODING_AGX3B) || !out || !len ||
       dest >= (g17g ? 128 : 256) || src >= (g17g ? 128 : 256) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(g17g ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1}, {41, 1}, {42, 1},
      {49, 3}, {50, 41},
   };
   struct agx3_field_kv fields[16] = {
      {2, dest}, {17, src}, {33, UINT8_C(0x80)}, {30, 1},
      {11, dest_cache},
      {26, hint == AGX3_ALU_HINT_CACHE},
      {28, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 7;
   for (unsigned slot = 0; slot < (g17g ? 8 : 6); ++slot)
      fields[n++] = (struct agx3_field_kv){51 + slot,
                                           !!(wait_mask & (1u << slot))};

   /* Native969: plain negation keeps the half source and -0 operand,
    * clearing only the abs modifier relative to Native910. */
   if (g17g) {
      for (unsigned ep = 2; ep <= 3; ++ep)
         if (agx3_bind_compose(enc, AGX3_MN_f16add, ep, base, 8, fields, n, out, len))
            return true;
      return false;
   }
   return agx3_bind_compose(enc, AGX3_MN_f16add, 2, base, 6, fields, n, out, len);
}

bool
agx3_alu_hadd_saturate_negzero(
   enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
   enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !out || !len || dest >= 256 ||
       src >= 256 || (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /* applegpu_g17p lowers saturate(half) to f16add.ep2 with an exact
    * negative-zero minifloat in field33. The ordinary half-add witnesses
    * establish the shared register, lifetime, destination-retention and
    * six-slot wait axes; four texture-dependent saturation witnesses pin
    * this endpoint, the saturation bit and slot-0 dependency exactly.
    * applegpu_g17g preserves the short form and adds ep3 for eight-slot
    * dependencies. Native live/dead half4 kernels pin its long selectors
    * and slot-8 wait independently of the register lifetime hint. */
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1}, {41, 1}, {42, 1},
   };
   struct agx3_field_kv fields[17] = {
      {2, dest},
      {17, src},
      {33, UINT8_C(0x80)},
      {11, dest_cache},
      {14, 1},
      {26, hint == AGX3_ALU_HINT_CACHE},
      {28, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 7;
   if (m5) {
      fields[n++] = (struct agx3_field_kv){49, 3};
      fields[n++] = (struct agx3_field_kv){50, 41};
   }
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      fields[n++] = (struct agx3_field_kv){51 + slot,
                                           !!(wait_mask & (1u << slot))};

   for (unsigned ep = 2; ep <= (m5 ? 3 : 2); ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f16add, ep, base, 6, fields, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_alu_hadd_one_neg(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t src, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_hadd_minifloat(enc, dest, src, UINT8_C(0x30), true, false,
                                  hint, dest_cache, wait_mask, out, len);
}

/* Native scalar half FMA preserves the single-rounding f16 semantics. */
static bool
hfma(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[3], const enum agx3_alu_hint hints[3],
              bool product_neg, bool src2_neg, bool dest_cache, bool saturate,
              uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !src || !hints || !out || !len ||
       (saturate && !m5) || dest >= 256 || src[0] >= 256 || src[1] >= 256 || src[2] >= 256 ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[2] > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1},
      {41, 2}, {42, 1}, {58, 2}, {59, 1},
   };
   struct agx3_field_kv fields[24] = {
      {2, dest},
      {17, src[0]},
      {34, src[1]},
      {51, src[2]},
      {11, dest_cache},
      {26, hints[0] == AGX3_ALU_HINT_CACHE},
      {28, hints[0] == AGX3_ALU_HINT_DISCARD},
      {43, hints[1] == AGX3_ALU_HINT_CACHE},
      {45, hints[1] == AGX3_ALU_HINT_DISCARD},
      {60, hints[2] == AGX3_ALU_HINT_CACHE},
      {62, hints[2] == AGX3_ALU_HINT_DISCARD},
      {47, product_neg},
      {64, src2_neg},
      {14, saturate},
   };
   unsigned n = 14;
   if (m5) {
      fields[n++] = (struct agx3_field_kv){66, 3};
      fields[n++] = (struct agx3_field_kv){67, 42};
   }
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      fields[n++] =
         (struct agx3_field_kv){68 + i, !!(wait_mask & (1u << i))};

   /* G17G has native ep2 witnesses and an ep5 slot-8 dependency witness.
    * G17P's compact ep3 tail is a one-of-six wait selector. ep4 and ep5 carry
    * the six fields independently, so dependency sets with several producers
    * must begin there instead of aliasing to a different compact selector. */
   const unsigned first_ep =
      m5 ? 2 : (wait_mask && (wait_mask & (wait_mask - 1)) ? 4 : 3);
   for (unsigned ep = first_ep; ep < 6; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f16fma, ep, base, 8, fields, n, out,
                            len))
         return true;
   }

   return false;
}

bool
agx3_alu_hfma(enum agx_isa_encoding enc, uint16_t dest,
              const uint16_t src[3], const enum agx3_alu_hint hints[3],
              bool product_neg, bool src2_neg, bool dest_cache,
              uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return hfma(enc, dest, src, hints, product_neg, src2_neg, dest_cache,
               false, wait_mask, out, len);
}

bool
agx3_alu_hfma_saturate_m5(uint16_t dest, const uint16_t src[3],
              const enum agx3_alu_hint hints[3], bool product_neg,
              bool src2_neg, bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return hfma(AGX_ISA_ENCODING_AGX3C, dest, src, hints, product_neg,
               src2_neg, dest_cache, true, wait_mask, out, len);
}

/* Native scalar half FMA with an 8-bit minifloat addend. Fresh G17P Metal
 * witnesses place the addend in field50 and select the immediate third-source
 * form with fields58..59 = 1,1. This is distinct from both the three-register
 * form above and f16mul's field16 immediate form. */
bool
agx3_alu_hfma_addend_minifloat(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   uint16_t immediate, const enum agx3_alu_hint hints[2], bool dest_cache,
   uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !src || !hints || !out ||
       !len || dest >= 256 || src[0] >= 256 || src[1] >= 256 ||
       immediate > UINT8_MAX || (wait_mask & ~(m5 ? 0xffu : 0x3fu)) ||
       (unsigned)hints[0] > AGX3_ALU_HINT_CACHE ||
       (unsigned)hints[1] > AGX3_ALU_HINT_CACHE)
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 2}, {25, 1},
      {41, 2}, {42, 1}, {58, 1}, {59, 1},
   };
   struct agx3_field_kv fields[19] = {
      {2, dest},
      {17, src[0]},
      {34, src[1]},
      {50, immediate},
      {11, dest_cache},
      {26, hints[0] == AGX3_ALU_HINT_CACHE},
      {28, hints[0] == AGX3_ALU_HINT_DISCARD},
      {43, hints[1] == AGX3_ALU_HINT_CACHE},
      {45, hints[1] == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i = 0; i < 6; ++i)
      fields[9 + i] =
         (struct agx3_field_kv){68 + i, !!(wait_mask & (1u << i))};

   if (m5) {
      fields[15] = (struct agx3_field_kv){66, 3};
      fields[16] = (struct agx3_field_kv){67, 42};
      fields[17] = (struct agx3_field_kv){74, !!(wait_mask & 64)};
      fields[18] = (struct agx3_field_kv){75, !!(wait_mask & 128)};
   }

   /* ep3 carries one selected wait; the extended endpoints carry a mask. */
   const unsigned first_ep =
      wait_mask && (wait_mask & (wait_mask - 1)) ? 4 : 3;
   for (unsigned ep = first_ep; ep < 6; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f16fma, ep, base, 8, fields, m5 ? 19 : 15,
                            out, len))
         return true;
   }

   return false;
}

/* Exact half-to-float widening uses native mixed-width ADD with negative zero. */
bool
agx3_alu_widen_half(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool src_abs, bool src_neg,
                    bool dest_cache, uint16_t wait_mask,
                    uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !out || !len ||
       dest >= 256 || src >= 256 || (dest & 1) ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (!m5 && src_abs && src_neg))
      return false;
   /* G17G Metal half4 -> float4 witnesses retain the half operand and add
    * negative zero. M5 also requires its floating-point pipeline selectors;
    * these are not the G17P long-form tail bits. Native modifier and slot-8
    * wait variants round-trip through the AGX3C composer without patching.
    */
   const struct agx3_field_kv base[] = {
      {9,2},{10,2},{25,2},{26,1},{42,1},{43,2},
      {50,3},{51,48},
   };
   struct agx3_field_kv fields[16] = {
      {2,dest},{18,src},{34,0x80},{11,dest_cache},
      {27,hint == AGX3_ALU_HINT_CACHE},{29,hint == AGX3_ALU_HINT_DISCARD},
      {30,src_abs},{31,src_neg},
   };
   const unsigned slots = m5 ? 8 : 6;
   for (unsigned i=0;i<slots;i++)
      fields[8+i] = (struct agx3_field_kv){52+i,!!(wait_mask & (1u<<i))};
   for (unsigned ep=0;ep<4;ep++)
      if (agx3_bind_compose(enc,AGX3_MN_f32add,ep,base,m5 ? 8 : 6,
                           fields,8+slots,out,len)) {
         /* metal-nt canonicalizes modifier-specific tail bits. */
         if (!m5 && src_abs && *len == 10)
            out[7] |= 0x80;
         if (!m5 && src_neg && *len == 8)
            out[7] |= 0x80;
         return true;
      }
   return false;
}

bool
agx3_alu_fcmp_select_imm(enum agx_isa_encoding enc, uint16_t dest,
                        const uint16_t src[2], const enum agx3_alu_hint hints[2],
                        unsigned condition, uint8_t if_true, uint8_t if_false,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool src_16[2] = {false, false};
   return agx3_alu_fcmp_select_imm_width(enc, dest, src, src_16, hints, condition,
      if_true, if_false, dest_16, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fcmp_select_imm_width_neg_second(
   enum agx_isa_encoding enc, uint16_t dest, const uint16_t src[2],
   const bool src_16[2], const enum agx3_alu_hint hints[2],
   unsigned condition, uint8_t if_true, uint8_t if_false, bool dest_16,
   bool dest_cache, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_fcmp_select_imm_width_internal(
      enc, dest, src, src_16, hints, condition, if_true, if_false, dest_16,
      dest_cache, wait_mask, true, (const bool[2]){false, false}, out, len);
}

bool
agx3_alu_fma_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                       uint16_t src, uint8_t immediate, uint16_t addend,
                       enum agx3_alu_hint src_hint,
                       enum agx3_alu_hint addend_hint, bool dest_cache,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_fma_minifloat_mod(enc, dest, src, immediate, addend, false, false,
                                     src_hint, addend_hint, dest_cache,
                                     wait_mask, out, len);
}

bool
agx3_alu_f32_to_s16(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                    enum agx3_alu_hint hint, bool dest_cache,
                    uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                    uint8_t *len)
{
   if (wait_mask != 0 && wait_mask != 32)
      return false;
   return agx3_alu_float_to_int16(enc, dest, src, false, true, hint,
                                  dest_cache, wait_mask, out, len);
}

bool
agx3_alu_fmul_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                        uint16_t src, uint16_t immediate,
                        enum agx3_alu_hint hint, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_fmul_minifloat_mod(
      enc, dest, src, immediate, false, false, hint, wait_mask, out, len);
}

bool
agx3_alu_hrcp(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
              enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
              uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_rcp(enc, dest, src, true, true, false, false, hint,
                       dest_cache, wait_mask, out, len);
}

bool
agx3_alu_f32rsqrt(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  enum agx3_alu_hint hint, bool dest_cache,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   return agx3_alu_f32rsqrt_mod(enc, dest, src, hint, dest_cache, wait_mask,
                                false, false, out, len);
}

bool
agx3_alu_f32log2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_log2(enc, dest, src, false, false, false, false, hint,
                        dest_cache, wait_mask, out, len);
}

bool
agx3_alu_f32exp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                 enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                 uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_unop(enc, AGX3_MN_f32exp2, dest, src, false, false, hint,
                        dest_cache, wait_mask, false, false, out, len);
}

bool
agx3_alu_hexp2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   return agx3_alu_hexp2_neg(enc, dest, src, hint, dest_cache, wait_mask,
                             false, out, len);
}

bool
agx3_alu_hlog2(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
               enum agx3_alu_hint hint, bool dest_cache,
               uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
               uint8_t *len)
{
   return agx3_alu_log2(enc, dest, src, true, true, false, false, hint,
                        dest_cache, wait_mask, out, len);
}

bool
agx3_alu_sqrt_step_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                       enum agx3_alu_hint hint, bool dest_cache,
                       uint16_t wait_mask, bool src_abs,
                       uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_sqrt_step_source_modifiers(enc, dest, src, hint, dest_cache,
                                             wait_mask, src_abs, false, out, len);
}

bool
agx3_alu_f32log2_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     enum agx3_alu_hint hint, bool dest_cache,
                     uint16_t wait_mask, bool src_neg, bool src_abs,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_log2(enc, dest, src, false, false, src_abs, src_neg, hint,
                        dest_cache, wait_mask, out, len);
}

bool
agx3_alu_sqrt_step(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                  enum agx3_alu_hint hint, bool dest_cache, uint16_t wait_mask,
                  uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_sqrt_step_mod(enc, dest, src, hint, dest_cache, wait_mask,
                                 false, out, len);
}

/* G17G no-fast-math fmin/fmax emit condition 3/7 with selected values
 * tied to their comparands. Do not admit those codes for an arbitrary
 * compare/select: only this native minNum/maxNum domain is witnessed. */
static bool
agx3_gpu1_tied_minmax(const struct agx3_fsel32_gpu1_desc *d, bool half)
{
   return (d->condition == 3 || d->condition == 7) &&
          !d->cmp1_immediate && !d->sel_immediate[0] &&
          !d->sel_immediate[1] && d->cmp0_16 == half && d->cmp1_16 == half &&
          d->cmp0 == d->sel[0] && d->cmp1 == d->sel[1];
}

bool
agx3_alu_fcmp_select32_gpu1(const struct agx3_fsel32_gpu1_desc *d,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!d || !out || !len || d->dest >= 256 || (d->dest & 1) ||
       (d->wait_mask & ~0xffu) || (d->cmp1_16 && d->cmp1_immediate) ||
       (d->condition != 0 && d->condition != 1 && d->condition != 2 &&
        d->condition != 5 && d->condition != 6 &&
        !agx3_gpu1_tied_minmax(d, false)))
      return false;

   /* Two float constants select a half result on GPU1 (`fself16` in
    * gpu/m5-xfb-alu10-reference, kernel fself_one_zero, followed by a widen);
    * no 32-bit form carries both minifloats.
    *
    * Integer constants are another matter: `fseli32` takes an 8-bit integer
    * in either selected slot, and both at once. applegpu_g18g and
    * applegpu_g17g alike (g18ref-out/probe/m6-alu-sel, -sel2, -hints):
    *
    *   (a <  b)    ? 2u : 1u   f49 = 2, f57 = 1; f63 = 1, f71 = 1; f86 = 1
    *   (a <  b)    ? 5u : 0u   f49 = 5 (f63 empty)       -- likewise 255u : 0u
    *   (a <  b)    ? 0u : 2u   f63 = 2 (f49 empty)
    *   (a == 0.0f) ? 2u : 1u   comparand minifloat 0: f40 = 1, f86 = 0
    *   (a <  b)    ? c  : 1u   f50 = c, f57 = 2; f63 = 1, f71 = 1
    *   (a <  b)    ? 1u : c    f49 = 1, f57 = 1; f64 = c, f71 = 2
    *
    * the immediate in f49/f63 with kind 1, a register in f50/f64 with kind
    * 2, both widths f58/f72 = 2 -- exactly this function's `fseli32` layout
    * with the selected kinds opened up. Comparand cache hints f25/f42 are
    * witnessed on the same form (h_f_u32_lt21_live). */
   if (d->sel_immediate[0] && d->sel_immediate[1] && !d->sel_integer)
      return false;

   /* Native G17G fragments compare a uniform against a register
    * and select a minifloat or a register: fself32 with f19 = the uniform
    * half and f25 = 3, conditions 5 and 6, no lifetime hints on the
    * uniform (16 Minecraft terrain/entity witnesses, asserted in the M5
    * instruction test). Only that shape is admitted. */
   if (d->cmp0_uniform &&
       (d->cmp0_16 || (d->cmp0 & 1) || d->cmp0 >= 256 ||
        d->cmp0_hint != AGX3_ALU_HINT_NONE || d->cmp1_immediate ||
        d->cmp1_16 || d->sel_integer || !d->sel_immediate[0] ||
        d->sel_immediate[1] || (d->condition != 5 && d->condition != 6)))
      return false;

   const uint16_t src[4] = {d->cmp0, d->cmp1, d->sel[0], d->sel[1]};
   const bool imm[4] = {false, d->cmp1_immediate, d->sel_immediate[0],
                        d->sel_immediate[1]};
   const enum agx3_alu_hint hints[4] = {d->cmp0_hint, d->cmp1_hint,
                                        d->sel_hint[0], d->sel_hint[1]};
   const bool half[4] = {d->cmp0_16, d->cmp1_16, false, false};
   for (unsigned i = 0; i < 4; ++i) {
      if (src[i] >= 256 || (!imm[i] && !half[i] && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE ||
          (imm[i] && hints[i] != AGX3_ALU_HINT_NONE))
         return false;
   }


   const bool fself = !d->sel_integer && (imm[2] || imm[3]);
   const unsigned o2 = fself ? 2 : 0;
   const unsigned o5 = fself ? 5 : 0;
   const unsigned o8 = fself ? 8 : 0;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23 + o2, d->cmp0_uniform ? 3 : 2},
      {24 + o2, d->cmp0_16 ? 1 : 2},
      {40 + o2, imm[1] ? 1 : 2}, {41 + o2, d->cmp1_16 ? 1 : 2},
      {57 + o2, imm[2] ? 1 : 2}, {58 + o2, 2},
      {71 + o5, imm[3] ? 1 : 2}, {72 + o5, 2},
      {76 + o8, 3}, {77 + o8, fself ? 74 : 76},
   };
   struct agx3_field_kv fields[23] = {
      {2, d->dest}, {11, d->dest_cache},
      {(d->cmp0_uniform ? 17 : 16) + o2, src[0]},
      {(imm[1] ? 32 : 33) + o2, src[1]},
      {(imm[2] ? 49 : 50) + o2, src[2]},
      {(imm[3] ? 63 : 64) + o5, src[3]},
      {86 + o8, d->condition},
   };
   const uint16_t cache[] = {25 + o2, 42 + o2, 59 + o2, 73 + o5};
   const uint16_t discard[] = {27 + o2, 44 + o2, 61 + o2, 75 + o5};
   unsigned n = 7;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){
         cache[i], hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){
         discard[i], hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < 8; ++i) {
      fields[n++] = (struct agx3_field_kv){
         78 + o8 + i, !!(d->wait_mask & (1u << i))};
   }
   /* G17G's second NClamp operation is the compact six-byte tied fmin,
    * with the slot-one wait in ep0. Other general selects retain ep4. */
   if (agx3_gpu1_tied_minmax(d, false) &&
       agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fseli32, 0, base, 12,
                        fields, n, out, len))
      return true;
   /* The uniform-comparand witnesses use the compact fself32 endpoints
    * (8 and 10 bytes): take the shortest that holds the fields. */
   if (d->cmp0_uniform) {
      uint8_t best[AGX3_ALU_MAX_LENGTH], best_len = 0;
      for (unsigned ep = 0; ep < 8; ++ep) {
         uint8_t bytes[AGX3_ALU_MAX_LENGTH], size;
         if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fself32, ep, base,
                               12, fields, n, bytes, &size) &&
             (!best_len || size < best_len)) {
            memcpy(best, bytes, size);
            best_len = size;
         }
      }
      if (!best_len)
         return false;
      memcpy(out, best, best_len);
      *len = best_len;
      return true;
   }
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,
                            fself ? AGX3_MN_fself32 : AGX3_MN_fseli32, 4, base, 12,
                            fields, n, out, len);
}

/* The general select's GPU1 branch, over the descriptor above. Mesa's
 * fcmpsel carries integer immediates in its selected slots, so zero is the
 * only constant it can hand to the minifloat-selecting `fself32`. */
static bool
agx3_alu_fcmp_select_gpu1(uint16_t dest, const uint16_t src[4],
                        bool compare_zero, const bool compare_abs[2],
                        const bool immediate[2],
                        const enum agx3_alu_hint hints[4], unsigned condition,
                        bool dest_16, bool dest_cache, uint16_t wait_mask,
                        uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* A 16-bit result selecting a small constant against a register is its
    * own form; the 32-bit one below selects two registers only. */
   if (dest_16 && compare_zero && src[1] == 0 && !compare_abs[0] &&
       !compare_abs[1] && immediate[0] && !immediate[1] && src[2] <= 7 &&
       hints[2] == AGX3_ALU_HINT_NONE &&
       agx3_alu_fcmp_select16_const_reg_gpu1(
          dest, src[0], hints[0], 0, (uint8_t)src[2], src[3], condition,
          wait_mask, out, len))
      return true;

   /* Every other 16-bit result is the general `fseli16`: registers or
    * integer bytes on either side, a register or minifloat comparand. */
   if (dest_16 && !compare_abs[0] && !compare_abs[1]) {
      const struct agx3_fsel32_gpu1_desc d = {
         .dest = dest,
         .dest_cache = dest_cache,
         .cmp0 = src[0],
         .cmp0_hint = hints[0],
         .cmp1_immediate = compare_zero,
         .cmp1 = src[1],
         .cmp1_hint = hints[1],
         .sel_immediate = {immediate[0], immediate[1]},
         .sel = {src[2], src[3]},
         .sel_hint = {hints[2], hints[3]},
         .condition = condition,
         .wait_mask = wait_mask,
      };
      return agx3_alu_fcmp_select16_gpu1(&d, out, len);
   }

   if (dest_16 || compare_abs[0] || compare_abs[1] ||
       (compare_zero && src[1] != 0) || (immediate[0] && src[2] != 0) ||
       (immediate[1] && src[3] != 0))
      return false;
   const struct agx3_fsel32_gpu1_desc d = {
      .dest = dest,
      .dest_cache = dest_cache,
      .cmp0 = src[0],
      .cmp0_hint = hints[0],
      .cmp1_immediate = compare_zero,
      .cmp1 = src[1],
      .cmp1_hint = hints[1],
      .sel_immediate = {immediate[0], immediate[1]},
      .sel = {src[2], src[3]},
      .sel_hint = {hints[2], hints[3]},
      .condition = condition,
      .wait_mask = wait_mask,
   };
   return agx3_alu_fcmp_select32_gpu1(&d, out, len);
}

/*
 * A 32-bit float comparison whose 16-bit result selects between a small
 * constant and a register. Apple's fseli16 corpus above only ever selects
 * two constants and the 32-bit fsel form only ever selects two registers, so
 * `cond ? 1 : x` -- which is what Gallium hands the packer once it has
 * lowered a boolean blend -- had no encoding at all and refused the whole
 * shader.
 *
 * From applegpu_g17g compute kernels written with vector select() so the
 * compiler cannot turn them into the conditional store it emits for the
 * scalar form (gpu/m5-mixed-select3-reference.metal), comparing against an
 * immediate zero with the register in half 10 and the constant 1:
 *
 *   select(c, 1, a <  0.0f)   2200050a2300a00214804e6021048000   f86 = 1
 *   select(c, 1, a >= 0.0f)   2200050a2300a00214804e2421048000   f86 = 5
 *
 * f49 carries the selected constant, f64 the selected register and f71 = 2
 * says the constant is the true value; f32 is the comparison's minifloat.
 * The register slot has no lifetime field on this form, so a last-use hint
 * on it is dropped rather than encoded -- conservative, and what the witness
 * spells. Selecting the constant on the FALSE side, and comparing against a
 * register rather than an immediate, move other fields and stay refused
 * until each is witnessed on its own.
 */
bool
agx3_alu_fcmp_select16_const_reg_gpu1(
   uint16_t dest, uint16_t src0, enum agx3_alu_hint src0_hint,
   uint8_t compare_minifloat, uint8_t if_true_constant,
   uint16_t if_false_register, unsigned condition, uint16_t wait_mask,
   uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src0 >= 256 || (src0 & 1) ||
       (if_false_register & 1) || if_false_register >= 256 ||
       if_true_constant > 7 ||
       (unsigned)src0_hint > AGX3_ALU_HINT_CACHE ||
       (condition != 0 && condition != 1 && condition != 2 &&
        condition != 5 && condition != 6) ||
       (wait_mask & ~128u))
      return false;

   const struct agx3_field_kv base[] = {
      {9, 2},  {10, 1}, {23, 2}, {24, 2}, {40, 1}, {41, 2},
      {57, 1}, {58, 1}, {71, 2}, {72, 1}, {75, 1}, {76, 3}, {77, 75},
   };
   const struct agx3_field_kv fields[] = {
      {2, dest},
      {16, src0},
      {25, src0_hint == AGX3_ALU_HINT_CACHE},
      {27, src0_hint == AGX3_ALU_HINT_DISCARD},
      {32, compare_minifloat},
      {49, if_true_constant},
      {64, if_false_register},
      {85, !!(wait_mask & 128u)},
      {86, condition},
   };
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fseli16, 8, base,
                            sizeof(base) / sizeof(base[0]), fields,
                            sizeof(fields) / sizeof(fields[0]), out, len);
}


bool
agx3_alu_hadd_minifloat(enum agx_isa_encoding enc, uint16_t dest,
                      uint16_t src, uint16_t immediate, bool src_neg,
                      bool saturate, enum agx3_alu_hint hint,
                      bool dest_cache, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !out || !len || dest >= 256 ||
       src >= 256 || immediate > UINT8_MAX ||
       (!m5 && (immediate != UINT8_C(0x30) || !src_neg)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;


   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 1}, {25, 1}, {41, 2}, {42, 1},
   };
   struct agx3_field_kv fields[18] = {
      {2, dest},
      {16, immediate},
      {34, src},
      {47, src_neg},
      {14, saturate},
      {11, dest_cache},
      {43, hint == AGX3_ALU_HINT_CACHE},
      {45, hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   if (m5) {
      fields[n++] = (struct agx3_field_kv){49, 3};
      fields[n++] = (struct agx3_field_kv){50, 41};
   }
   for (unsigned slot = 0; slot < (m5 ? 8 : 6); ++slot)
      fields[n++] = (struct agx3_field_kv){51 + slot,
                                           !!(wait_mask & (1u << slot))};

   for (unsigned ep = 1; ep <= (m5 ? 3 : 1); ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f16add, ep, base, 6, fields, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_alu_m5_ldimm(uint16_t dest, bool half, bool dest_cache, uint32_t bits,
                  uint16_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
                  uint8_t *len)
{
   if (!out || !len || dest >= 256 || (!half && (dest & 1)) ||
       (half && bits > UINT16_MAX) || wait_mask > 0xff)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, half ? 1 : 2}, {15, 3}, {16, 105},
   };
   struct agx3_field_kv fields[11] = {
      {2, dest}, {11, dest_cache}, {14, bits},
   };
   for (unsigned slot = 0; slot < 8; ++slot)
      fields[3 + slot] = (struct agx3_field_kv){17 + slot,
                                               !!(wait_mask & (1u << slot))};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_ldimm, 2, base, 4,
                            fields, 11, out, len);
}

bool
agx3_alu_fcmp_select16_gpu1(const struct agx3_fsel32_gpu1_desc *d,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!d || !out || !len || d->dest >= 256 || (d->wait_mask & ~0xffu) ||
       d->cmp0_uniform ||
       (d->condition != 0 && d->condition != 1 && d->condition != 2 &&
        d->condition != 5 && d->condition != 6 &&
        !agx3_gpu1_tied_minmax(d, true)))
      return false;

   const uint16_t src[4] = {d->cmp0, d->cmp1, d->sel[0], d->sel[1]};
   const bool imm[4] = {false, d->cmp1_immediate, d->sel_immediate[0],
                        d->sel_immediate[1]};
   const enum agx3_alu_hint hints[4] = {d->cmp0_hint, d->cmp1_hint,
                                        d->sel_hint[0], d->sel_hint[1]};

   /*
    * The second comparand may also be a 16-bit register, which the
    * comparison widens: f41, its width, drops to 1. Apple puts the float
    * first and the half second, reversing the relation when the source had
    * them the other way (applegpu_g18g/g17g, g18ref-out/probe/m6-alu-sel and
    * m6-alu-hints):
    *
    *   (a != float(h)) ? 1 : 0   fseli16 h2 = ^r1 == h4 ? #0 : #1   f41 = 1
    *   (float(h) != b) ? 1 : 0   the same instruction, operands exchanged
    *   uint(float(h) >= b)       fseli16 h2 = ^r1 <= h4 ? #1 : #0; clr h3
    *   ... with h reused         f42 = 1, h's cache hint
    *
    * Native float/half selects with a live half comparand also use odd
    * halves (f33 = 5, f41 = 1, f42 = 1). Only full-width comparands need
    * even half-register indices; selected values may likewise use either
    * half independently.
    */
   if (d->cmp1_16 && d->cmp1_immediate)
      return false;
   const bool half[4] = {d->cmp0_16, d->cmp1_16, true, true};
   for (unsigned i = 0; i < 4; ++i) {

      if (src[i] >= 256 || (!imm[i] && !half[i] && (src[i] & 1)) ||
          (unsigned)hints[i] > AGX3_ALU_HINT_CACHE ||
          (imm[i] && hints[i] != AGX3_ALU_HINT_NONE))
         return false;
   }

   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, d->cmp0_16 ? 1 : 2},
      {40, imm[1] ? 1 : 2}, {41, d->cmp1_16 ? 1 : 2},
      {57, imm[2] ? 1 : 2}, {58, 1},
      {71, imm[3] ? 1 : 2}, {72, 1},
      {76, 3}, {77, 75},
   };
   struct agx3_field_kv fields[23] = {
      {2, d->dest}, {11, d->dest_cache}, {16, src[0]},
      {imm[1] ? 32 : 33, src[1]},
      {imm[2] ? 49 : 50, src[2]},
      {imm[3] ? 63 : 64, src[3]},
      {86, d->condition},
   };
   static const uint16_t cache[] = {25, 42, 59, 73};
   static const uint16_t discard[] = {27, 44, 61, 75};
   unsigned n = 7;
   for (unsigned i = 0; i < 4; ++i) {
      fields[n++] = (struct agx3_field_kv){
         cache[i], hints[i] == AGX3_ALU_HINT_CACHE};
      fields[n++] = (struct agx3_field_kv){
         discard[i], hints[i] == AGX3_ALU_HINT_DISCARD};
   }
   for (unsigned i = 0; i < 8; ++i)
      fields[n++] = (struct agx3_field_kv){78 + i, !!(d->wait_mask & (1u << i))};
   for (unsigned ep = 0; ep < 16; ++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fseli16, ep, base,
                             sizeof(base) / sizeof(base[0]), fields, n, out, len))
         return true;
   return false;
}

bool
agx3_alu_iadd16_gpu1(uint16_t dest, bool dest_cache, bool src0_immediate,
                     uint16_t src0, enum agx3_alu_hint src0_hint,
                     uint16_t src1, enum agx3_alu_hint src1_hint,
                     unsigned shift, uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{

   if (!out || !len || dest >= 256 || src1 >= 256 || shift > 4 ||
       (src0_immediate ? src0 > 255 : src0 >= 256) ||
       (unsigned)src0_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)src1_hint > AGX3_ALU_HINT_CACHE || wait_mask > 0xff)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {24, 1}, {38, 2}, {39, 1}, {44, 3}, {45, 80},
   };
   struct agx3_field_kv ov[20];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv){2, dest};
   ov[n++] = (struct agx3_field_kv){11, dest_cache ? 1 : 0};
   if (src0_immediate) {
      ov[n++] = (struct agx3_field_kv){23, 1};
      ov[n++] = (struct agx3_field_kv){15, src0};
   } else {
      ov[n++] = (struct agx3_field_kv){23, 2};
      ov[n++] = (struct agx3_field_kv){16, src0};
      ov[n++] = (struct agx3_field_kv){25, src0_hint == AGX3_ALU_HINT_CACHE};
      ov[n++] = (struct agx3_field_kv){27, src0_hint == AGX3_ALU_HINT_DISCARD};
   }
   ov[n++] = (struct agx3_field_kv){31, src1};
   ov[n++] = (struct agx3_field_kv){40, src1_hint == AGX3_ALU_HINT_CACHE};
   ov[n++] = (struct agx3_field_kv){42, src1_hint == AGX3_ALU_HINT_DISCARD};
   ov[n++] = (struct agx3_field_kv){55, shift};
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){46 + i, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_iadd32, 0, base, 7, ov,
                            n, out, len);
}

bool
agx3_alu_bit_imm_gpu1(enum agx3_alu_op op, uint16_t dest, bool dest_cache,
                      uint16_t src, uint8_t immediate,
                      enum agx3_alu_hint hint, uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* Apple's varying-stencil fragment masks the reference with
    * `bit_and r0h = r0l & 255`: 23 00 07 01 38 20 a3 02 c4 bf. As on G17P,
    * the immediate operand's width field f38 stays 2 for a narrow result. */
   if (!out || !len || op != AGX3_ALU_BAND || dest >= 256 || src >= 256 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || wait_mask > 0xff)
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 1}, {23, 2}, {24, 1}, {38, 2}, {42, 3}, {43, 10},
   };
   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv){2, dest};
   ov[n++] = (struct agx3_field_kv){11, dest_cache ? 1 : 0};
   ov[n++] = (struct agx3_field_kv){16, src};
   ov[n++] = (struct agx3_field_kv){25, hint == AGX3_ALU_HINT_CACHE};
   ov[n++] = (struct agx3_field_kv){27, hint == AGX3_ALU_HINT_DISCARD};
   ov[n++] = (struct agx3_field_kv){37, 1};
   ov[n++] = (struct agx3_field_kv){29, immediate};
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){44 + i, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_bit_and, 1, base, 7, ov,
                            n, out, len);
}

bool
agx3_alu_iadd32_reg_imm_gpu1(uint16_t dest, bool dest_cache, uint16_t src,
                             bool src_16, bool sign_extend,
                             enum agx3_alu_hint hint, uint8_t immediate,
                             bool saturate, uint16_t wait_mask,
                             uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   /* `iadd32` with the register in slot A (f16, kind f23 = 2, width f24)
    * and an immediate in slot B (f30, kind f38 = 1, width f39 = 1), the
    * order Apple keeps for these two shapes; applegpu_g18g and
    * applegpu_g17g give the same vector but for the wait slot
    * (g18ref-out/probe/m6-alu-misc, m6-alu-hints):
    *
    *   addsat(x, 1u), x dead    f16 0, f24 2, f27, f30 = 1, f54 = 1
    *   int(short x),  x dead    f16 0, f24 1, f27, f28 = 1, f30 = 0
    *   int(short x),  to r1     f2 2, f11 (the result is reused)
    *
    * f54 is the saturate bit (G17G's iadd32 keeps its shift in f55 and
    * saturation beside it) and f28 slot A's sign extension. Only these two
    * combinations are witnessed: an unsigned 32-bit saturating add, and a
    * plain sign-extending add of a half. The eight waits are f46..f53, as
    * on every GPU1 iadd32. */
   if (!out || !len || (dest & 1) || dest >= 256 ||
       (src_16 ? src >= 256 : ((src & 1) || src >= 256)) ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || wait_mask > 0xff ||
       !(saturate ? !src_16 && !sign_extend
                  : src_16 && sign_extend && immediate == 0))
      return false;
   const struct agx3_field_kv base[] = {
      {9, 2}, {10, 2}, {23, 2}, {38, 1}, {39, 1}, {44, 3}, {45, 80},
   };
   struct agx3_field_kv ov[20];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv){2, dest};
   ov[n++] = (struct agx3_field_kv){11, dest_cache ? 1 : 0};
   ov[n++] = (struct agx3_field_kv){16, src};
   ov[n++] = (struct agx3_field_kv){24, src_16 ? 1 : 2};
   ov[n++] = (struct agx3_field_kv){25, hint == AGX3_ALU_HINT_CACHE};
   ov[n++] = (struct agx3_field_kv){27, hint == AGX3_ALU_HINT_DISCARD};
   ov[n++] = (struct agx3_field_kv){28, sign_extend ? 1 : 0};
   ov[n++] = (struct agx3_field_kv){30, immediate};
   ov[n++] = (struct agx3_field_kv){54, saturate ? 1 : 0};
   for (unsigned i = 0; i < 8; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv){46 + i, 1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_iadd32,
                            AGX3_BIND_APPLE_ORDER, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len);
}


bool
agx3_alu_isub_imm16_m5(uint16_t dest, uint16_t src, uint16_t immediate,
                      enum agx3_alu_hint hint, bool dest_cache,
                      uint16_t wait_mask,
                      uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len || dest >= 256 || src >= 256 || immediate > 255 ||
       (unsigned)hint > AGX3_ALU_HINT_CACHE || wait_mask > 255)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2}, {10,1}, {23,2}, {24,1}, {38,1}, {39,1}, {44,3}, {45,100},
   };
   struct agx3_field_kv fields[14] = {
      {2,dest}, {11,dest_cache}, {16,src}, {30,immediate},
      {25,hint == AGX3_ALU_HINT_CACHE}, {27,hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 6;
   for (unsigned i=0; i<8; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){46+i,1};
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_isub32, 0,
                            base, 8, fields, n, out, len);
}

static bool
agx3_alu_bit_widen_m5(enum agx3_alu_op op, bool first_half, uint16_t dest, uint16_t word, uint16_t half,
                     enum agx3_alu_hint word_hint,
                     enum agx3_alu_hint half_hint, bool dest_cache,
                     uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   if ((op != AGX3_ALU_BAND && op != AGX3_ALU_BOR && op != AGX3_ALU_BXOR) || !out || !len || dest >= 256 || word >= 256 || half >= 256 ||
       ((dest & 1) || (!first_half && (word & 1))) || (unsigned)word_hint > AGX3_ALU_HINT_CACHE ||
       (unsigned)half_hint > AGX3_ALU_HINT_CACHE || wait_mask > 255)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2}, {10,2}, {23,2}, {24,first_half ? 1 : 2}, {37,2}, {38,1}, {42,3}, {43,op == AGX3_ALU_BAND ? 10 : op == AGX3_ALU_BOR ? 16 : 19},
   };
   struct agx3_field_kv fields[16] = {
      {2,dest}, {11,dest_cache}, {16,word}, {30,half},
      {25,word_hint == AGX3_ALU_HINT_CACHE}, {27,word_hint == AGX3_ALU_HINT_DISCARD},
      {39,half_hint == AGX3_ALU_HINT_CACHE}, {41,half_hint == AGX3_ALU_HINT_DISCARD},
   };
   unsigned n = 8;
   for (unsigned i=0; i<8; ++i)
      if (wait_mask & (1u << i))
         fields[n++] = (struct agx3_field_kv){44+i,1};
   for (unsigned ep=0; ep<2; ++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, op == AGX3_ALU_BAND ? AGX3_MN_bit_and : op == AGX3_ALU_BOR ? AGX3_MN_bit_or : AGX3_MN_bit_xor, ep,
                             base, 8, fields, n, out, len))
         return true;
   return false;
}

/* Metal's ushort2 -> uint3 AND/OR/XOR witness uses a word destination
 * (field 10 = 2) and two half sources (fields 24 and 38 = 1). */
bool
agx3_alu_bit_half_to_word_m5(enum agx3_alu_op op, uint16_t dest,
                            const uint16_t src[2],
                            const enum agx3_alu_hint hints[2], bool dest_cache,
                            uint16_t wait_mask,
                            uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return src && hints && agx3_alu_bit_widen_m5(
      op, true, dest, src[0], src[1], hints[0], hints[1], dest_cache,
      wait_mask, out, len);
}

bool
agx3_alu_bit_mixed_m5(enum agx3_alu_op op, uint16_t dest, uint16_t word,
                     uint16_t half, enum agx3_alu_hint word_hint,
                     enum agx3_alu_hint half_hint, bool dest_cache,
                     uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_widen_m5(op, false, dest, word, half, word_hint,
                               half_hint, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_and_mixed_m5(uint16_t dest, uint16_t word, uint16_t half,
                     enum agx3_alu_hint word_hint,
                     enum agx3_alu_hint half_hint, bool dest_cache,
                     uint16_t wait_mask,
                     uint8_t out[AGX3_ALU_MAX_LENGTH], uint8_t *len)
{
   return agx3_alu_bit_mixed_m5(AGX3_ALU_BAND, dest, word, half, word_hint,
                               half_hint, dest_cache, wait_mask, out, len);
}

bool
agx3_alu_hfma_immediates_m5(uint16_t dest, uint16_t src,
   uint8_t multiplier, uint8_t addend, enum agx3_alu_hint hint,
   bool dest_cache, uint8_t wait_mask, uint8_t out[AGX3_ALU_MAX_LENGTH],
   uint8_t *len)
{
   if (dest >= 256 || src >= 256 || hint > AGX3_ALU_HINT_CACHE || !out || !len)
      return false;
   const struct agx3_field_kv base[] = {
      {9,2},{10,1},{24,2},{25,1},{41,1},{42,1},{58,1},{59,1},{66,3},{67,42},
   };
   struct agx3_field_kv fields[15] = {
      {2,dest},{17,src},{33,multiplier},{50,addend},{11,dest_cache},
      {26,hint == AGX3_ALU_HINT_CACHE},{28,hint == AGX3_ALU_HINT_DISCARD},
   };
   for (unsigned i=0;i<8;++i)
      fields[7+i]=(struct agx3_field_kv){68+i,!!(wait_mask & (1u<<i))};
   for (unsigned ep=2;ep<6;++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,AGX3_MN_f16fma,ep,
                           base,10,fields,15,out,len))
         return true;
   return false;
}
