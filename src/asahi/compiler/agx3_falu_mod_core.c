/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_falu_mod_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

/* Long-form baselines: the encoding point that carries every modifier, per
 * mnemonic and encoding. Decoded from Apple's witnessed instructions. */
static const struct agx3_field_kv add_b[] = {KV(9, 2),  KV(10, 2), KV(25, 2),
                                             KV(26, 2), KV(29, 1), KV(42, 2),
                                             KV(43, 2), KV(46, 1), KV(57, 1)};
static const struct agx3_field_kv add_c[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1), KV(42, 2),
   KV(43, 2), KV(46, 1), KV(50, 3), KV(51, 48), KV(59, 1)};
static const struct agx3_field_kv mul_b[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1), KV(42, 2),
   KV(43, 2), KV(46, 1), KV(60, 2), KV(74, 2), KV(85, 1)};
static const struct agx3_field_kv mul_c[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),  KV(42, 2), KV(43, 2),
   KV(46, 1), KV(60, 2), KV(74, 2), KV(78, 3), KV(79, 62), KV(87, 1)};
static const struct agx3_field_kv fma_b[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),  KV(42, 2), KV(43, 2),
   KV(46, 1), KV(59, 2), KV(60, 2), KV(63, 1), KV(77, 2), KV(91, 2), KV(102, 1)};
static const struct agx3_field_kv fma_c[] = {
   KV(9, 2),  KV(10, 2), KV(25, 2), KV(26, 2), KV(29, 1),  KV(42, 2),
   KV(43, 2), KV(46, 1), KV(59, 2), KV(60, 2), KV(63, 1),  KV(77, 2),
   KV(91, 2), KV(95, 3), KV(96, 55), KV(104, 1)};

static bool
b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

/*
 * The runtime-hint float forms below also encode G18G (agx3d), from the
 * agx3c field vector: f32add, f32mul and f32fma differ between the aligned
 * applegpu_g17g/applegpu_g18g pairs of the j813-linux gpu/m5-*-reference
 * corpus (58, 371 and 79 of them) only in register, cache/discard hint,
 * output-register and wait choices. agx3d picks its own encoding point,
 * smallest first, and its wait field is one slot code rather than agx3c's
 * mask, so at most one wait slot is admitted.
 */
static bool
runtime_enc_ok(enum agx_isa_encoding e, uint16_t wait_mask)
{
   if (e == AGX_ISA_ENCODING_AGX3D)
      return (wait_mask & (wait_mask - 1)) == 0 && wait_mask <= 0xff;
   return e == AGX_ISA_ENCODING_AGX3B || e == AGX_ISA_ENCODING_AGX3C;
}

static unsigned
common_mods(uint16_t dest, uint16_t src0, uint16_t src1,
            const struct agx3_falu_mods *m, struct agx3_field_kv *ov)
{
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(35, src0);
   ov[n++] = (struct agx3_field_kv)KV(18, src1);
   ov[n++] = (struct agx3_field_kv)KV(47, m->src0_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(48, m->src0_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(43, m->src0_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(30, m->src1_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(31, m->src1_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(26, m->src1_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(14, m->saturate ? 1 : 0);
   return n;
}

/* A 32-bit operand is a register pair and starts on an even half; a 16-bit
 * one is a single half and may sit on an odd index. */
static bool
slot_ok(uint16_t r, bool narrow)
{
   return r < 256 && (narrow || (r & 1) == 0);
}

bool
agx3_falu_add_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                  uint16_t src1, const struct agx3_falu_mods *m,
                  uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (m == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       m->addend_abs || m->addend_neg)
      return false;
   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   return agx3_bind_compose(enc, AGX3_MN_f32add, 3, b(enc) ? add_b : add_c,
                            b(enc) ? 9 : 11, ov, n, out, len);
}

/* Keep subtraction and other modified adds fused while preserving both live
 * operands and their producer slots. The native forms grow from4 to10bytes
 * only when modifiers or wait combinations require it. */
static bool
add_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src0, uint16_t src1,
                         bool dest_half, const struct agx3_falu_mods *m,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = agx_encoding_is_g17_class(enc);
   if (!runtime || !runtime_enc_ok(enc, runtime->wait_mask) || !m || !out || !len ||
       (!dest_half && (dest & 1)) || (dest_half && !m5) || dest >= 256 || !slot_ok(src0, m->src0_16) ||
       !slot_ok(src1, m->src1_16) || m->addend_abs || m->addend_neg ||
       runtime->wait_mask > (m5 ? 0xff : 0x3f) ||
       (enc == AGX_ISA_ENCODING_AGX3D && (m->src0_16 || m->src1_16)))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   uint8_t best[AGX3_FALU_MOD_MAX_LENGTH], best_size = 0;
   /* Addition is commutative, but the compact form gives the first source
    * the cheap negate modifier. Native subtraction places its negated input
    * there. Swap both modifiers and lifetime hints when this saves bytes. */
   for (unsigned swap = 0; swap < 2; ++swap) {
      struct agx3_falu_mods mods = *m;
      if (swap) {
         mods.src0_abs = m->src1_abs;
         mods.src0_neg = m->src1_neg;
         mods.src0_16 = m->src1_16;
         mods.src1_abs = m->src0_abs;
         mods.src1_neg = m->src0_neg;
         mods.src1_16 = m->src0_16;
      }
      struct agx3_field_kv ov[32];
      unsigned n = common_mods(dest, swap ? src1 : src0, swap ? src0 : src1,
                               &mods, ov);
      ov[n++] = (struct agx3_field_kv)KV(10, dest_half ? 1 : 2);
      ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
      ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[swap]);
      ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[swap]);
      ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1 - swap]);
      ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1 - swap]);
      for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
         ov[n++] = (struct agx3_field_kv)KV(52 + i,
                                          (runtime->wait_mask >> i) & 1);
      for (unsigned ep = 0; ep <= 3; ++ep) {
         uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
         if (agx3_bind_compose(enc, AGX3_MN_f32add, agx3_bind_ep(enc, ep), m5 ? add_c : add_b, m5 ? 11 : 9, ov, n, bytes, &size)) {
            if (!best_size || size < best_size) {
               for (unsigned i = 0; i < size; ++i)
                  best[i] = bytes[i];
               best_size = size;
            }
            break;
         }
      }
   }
   if (!best_size)
      return false;
   for (unsigned i = 0; i < best_size; ++i)
      out[i] = best[i];
   *len = best_size;
   return true;
}

bool
agx3_falu_add_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return add_mod_runtime(enc, dest, src0, src1, false, m, runtime, out, len);
}

bool
agx3_falu_add_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
                         const struct agx3_falu_mods *m,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return add_mod_runtime(AGX_ISA_ENCODING_AGX3C, dest, src0, src1, true,
                           m, runtime, out, len);
}

bool
agx3_falu_add_imm_runtime(uint16_t dest, uint16_t src0, uint8_t minifloat,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (!m || !runtime || !out || !len || (dest & 1) || dest >= 256 ||
       (src0 & 1) || src0 >= 256 || m->src0_16 ||
       m->src1_abs || m->src1_neg || m->src1_16 || m->addend_abs ||
       m->addend_neg || m->saturate || (runtime->cache[0] && runtime->discard[0]))
      return false;

   static const struct agx3_field_kv add_imm_c[] = {
      KV(9, 2),  KV(10, 2), KV(25, 1), KV(26, 2), KV(42, 2),
      KV(43, 2), KV(50, 3), KV(51, 48)};
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(35, src0);
   ov[n++] = (struct agx3_field_kv)KV(17, minifloat);
   ov[n++] = (struct agx3_field_kv)KV(47, m->src0_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(48, m->src0_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(14, 0);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[0]);
   for (unsigned i = 0; i < 8; ++i)
      ov[n++] = (struct agx3_field_kv)KV(52 + i, (runtime->wait_mask >> i) & 1);
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_f32add, 3, add_imm_c,
                            sizeof(add_imm_c) / sizeof(add_imm_c[0]), ov, n,
                            out, len);
}

bool
agx3_falu_mul_imm_runtime(uint16_t dest, uint16_t src0, uint8_t minifloat,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (!m || !runtime || !out || !len || (dest & 1) || dest >= 256 ||
       (src0 & 1) || src0 >= 256 || m->src0_16 || m->src0_neg ||
       m->src1_abs || m->src1_neg || m->src1_16 || m->addend_abs ||
       m->addend_neg || m->saturate || (runtime->cache[0] && runtime->discard[0]))
      return false;

   static const struct agx3_field_kv mul_imm_c[] = {
      KV(9, 2),  KV(10, 2), KV(25, 1), KV(26, 2), KV(42, 2), KV(43, 2),
      KV(60, 2), KV(74, 2), KV(78, 3), KV(79, 62)};
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(35, src0);
   ov[n++] = (struct agx3_field_kv)KV(17, minifloat);
   ov[n++] = (struct agx3_field_kv)KV(47, m->src0_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(48, 0);
   ov[n++] = (struct agx3_field_kv)KV(14, 0);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[0]);
   for (unsigned i = 0; i < 8; ++i)
      ov[n++] = (struct agx3_field_kv)KV(80 + i, (runtime->wait_mask >> i) & 1);
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_f32mul, 3, mul_imm_c,
                            sizeof(mul_imm_c) / sizeof(mul_imm_c[0]), ov, n,
                            out, len);
}

/* Apple lowers float-to-half conversion to f32add with a16-bit destination
 * and the exact negative-zero minifloat in the other input. This preserves
 * signed zero and rounds only once; half4 witnesses include odd halves. */
static bool
add_immediate(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     bool half, uint8_t immediate, bool src_abs, bool src_neg,
                     bool saturate, const struct agx3_falu_runtime *runtime,
                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool agx3c = enc == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = agx3c ? 8 : 6;
   if ((!b(enc) && !agx3c) || !runtime || !out || !len ||
       dest >= 256 || (!half && (dest & 1)) || src >= 256 || (src & 1) ||
       runtime->wait_mask >= (1u << slots) ||
       (runtime->cache[0] && runtime->discard[0]))
      return false;
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, half ? 1 : 2), KV(25, 2), KV(26, 2), KV(42, 1), KV(43, 2),
   };
   /* G17S cvt_f2h, r_mrt and tx_cube_sample witness this narrowing
    * operation, with f50=3/f51=48 and an eight-slot wait bank. The native
    * r_fragcoord variants set f15=1; its meaning is not established here
    * and that variant is not modeled by this API.
    * Saturating narrowing remains separate from the full-width add. */
   const struct agx3_field_kv base_c[] = {
      KV(9, 2), KV(10, half ? 1 : 2), KV(25, 2), KV(26, 2), KV(42, 1), KV(43, 2),
      KV(50, 3), KV(51, 48),
   };
   struct agx3_field_kv ov[32] = {
      KV(2, dest), KV(18, src), KV(34, immediate), KV(11, runtime->dest_cache),
      KV(27, runtime->cache[0]), KV(29, runtime->discard[0]), KV(14, saturate),
      KV(30, src_abs), KV(31, src_neg),
   };
   unsigned n = 9;
   for (unsigned i = 0; i < slots; ++i)
      ov[n++] = (struct agx3_field_kv)KV(52 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 0; ep <= 3; ++ep) {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      if (agx3_bind_compose(enc, AGX3_MN_f32add, ep, agx3c ? base_c : base,
                           agx3c ? 8 : 6, ov, n, bytes, &size)) {
         for (unsigned i = 0; i < size; ++i)
            out[i] = bytes[i];
         *len = size;
         return true;
      }
   }
   return false;
}

bool
agx3_falu_narrow_f32(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     bool saturate, const struct agx3_falu_runtime *runtime,
                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return add_immediate(enc, dest, src, true, 0x80, false, false, saturate,
                         runtime, out, len);
}

bool
agx3_falu_add_imm_m5(uint16_t dest, uint16_t src, uint8_t immediate,
                     bool src_abs, bool src_neg, bool saturate,
                     const struct agx3_falu_runtime *runtime,
                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return add_immediate(AGX_ISA_ENCODING_AGX3C, dest, src, false, immediate,
                         src_abs, src_neg, saturate, runtime, out, len);
}

/* Half of a minifloat minus a register: f32add with a 16-bit destination,
 * the register negated in the first operand and the minifloat in the other.
 * Apple's half4(1.0 - x) and half4(-x - 1.0) (blend/native/negpack3.py
 * fs_h_onem, fs_h_negm1) differ only in the immediate; with exact negative
 * zero this is the negated float-to-half conversion. */
bool
agx3_falu_narrow_imm_sub_f32(enum agx_isa_encoding enc, uint16_t dest,
                             uint8_t minifloat, uint16_t src,
                             const struct agx3_falu_runtime *runtime,
                             uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                             uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !runtime || !out || !len ||
       dest >= 256 || src >= 256 || (src & 1) || runtime->wait_mask > 0x3f ||
       (runtime->cache[0] && runtime->discard[0]))
      return false;
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 1), KV(25, 1), KV(26, 2), KV(42, 2), KV(43, 2),
      KV(48, 1),
   };
   struct agx3_field_kv ov[16] = {
      KV(2, dest), KV(35, src), KV(17, minifloat),
      KV(11, runtime->dest_cache), KV(44, runtime->cache[0]),
      KV(46, runtime->discard[0]),
   };
   unsigned n = 6;
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(52 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 0; ep <= 3; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32add, ep, base, 7, ov, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_falu_mul_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                  uint16_t src1, const struct agx3_falu_mods *m,
                  uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (m == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       m->addend_abs || m->addend_neg)
      return false;
   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   return agx3_bind_compose(enc, AGX3_MN_f32mul, 3, b(enc) ? mul_b : mul_c,
                            b(enc) ? 11 : 13, ov, n, out, len);
}

/* Saturation and source modifiers stay fused with the multiply. Native
 * live/dead witnesses carry independent hints and the same six dependency
 * slots as plain FMUL; select the shortest encoding that fits them. */
static bool
mul_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          bool dest_half, const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = agx_encoding_is_g17_class(enc);
   if (!runtime || !runtime_enc_ok(enc, runtime->wait_mask) || !m || !out || !len ||
       (!dest_half && (dest & 1)) || (dest_half && !m5) || dest >= 256 || !slot_ok(src0, m->src0_16) ||
       !slot_ok(src1, m->src1_16) || m->addend_abs || m->addend_neg ||
       runtime->wait_mask > (m5 ? 0xff : 0x3f) ||
       (enc == AGX_ISA_ENCODING_AGX3D && (m->src0_16 || m->src1_16)))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   ov[n++] = (struct agx3_field_kv)KV(10, dest_half ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[0]);
   ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1]);
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      ov[n++] = (struct agx3_field_kv)KV(80 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = 0; ep <= 3; ++ep) {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      if (agx3_bind_compose(enc, AGX3_MN_f32mul, agx3_bind_ep(enc, ep), m5 ? mul_c : mul_b, m5 ? 13 : 11, ov, n, bytes, &size)) {
         for (unsigned i = 0; i < size; ++i)
            out[i] = bytes[i];
         *len = size;
         return true;
      }
   }
   return false;
}

bool
agx3_falu_mul_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return mul_mod_runtime(enc, dest, src0, src1, false, m, runtime, out, len);
}

bool
agx3_falu_mul_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
                         const struct agx3_falu_mods *m,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return mul_mod_runtime(AGX_ISA_ENCODING_AGX3C, dest, src0, src1, true,
                           m, runtime, out, len);
}

bool
agx3_falu_fma_mod(enum agx_isa_encoding enc, uint16_t dest, uint16_t src0,
                  uint16_t src1, uint16_t addend,
                  const struct agx3_falu_mods *m,
                  uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   if (m == NULL || (dest & 1) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       !slot_ok(addend, m->addend_16))
      return false;
   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   ov[n++] = (struct agx3_field_kv)KV(52, addend);
   ov[n++] = (struct agx3_field_kv)KV(64, m->addend_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(65, m->addend_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(60, m->addend_16 ? 1 : 2);
   return agx3_bind_compose(enc, AGX3_MN_f32fma, b(enc) ? 5 : 7, b(enc) ? fma_b : fma_c,
                            b(enc) ? 14 : 16, ov, n, out, len);
}

/* Native dead/live modifier witnesses use the same source cache/discard
 * fields as ordinary FMA. Encoding points 2..5 carry progressively wider
 * modifiers and dependency waits; choose the shortest exact representation.
 * This keeps fog subtraction fused while waiting only on its dependencies.
 */
static bool
fma_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1, uint16_t addend, bool dest_half,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = agx_encoding_is_g17_class(enc);
   if (runtime == NULL || !runtime_enc_ok(enc, runtime->wait_mask) ||
       m == NULL || out == NULL || len == NULL || (!dest_half && (dest & 1)) ||
       (dest_half && !m5) || dest >= 256 ||
       !slot_ok(src0, m->src0_16) || !slot_ok(src1, m->src1_16) ||
       !slot_ok(addend, m->addend_16) || runtime->wait_mask > (m5 ? 0xff : 0x3f) ||
       (enc == AGX_ISA_ENCODING_AGX3D &&
        (m->src0_16 || m->src1_16 || m->addend_16)))
      return false;
   for (unsigned i = 0; i < 3; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   struct agx3_field_kv ov[32];
   unsigned n = common_mods(dest, src0, src1, m, ov);
   ov[n++] = (struct agx3_field_kv)KV(10, dest_half ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(52, addend);
   ov[n++] = (struct agx3_field_kv)KV(64, m->addend_abs);
   ov[n++] = (struct agx3_field_kv)KV(65, m->addend_neg);
   ov[n++] = (struct agx3_field_kv)KV(60, m->addend_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(44, runtime->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, runtime->discard[0]);
   ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1]);
   ov[n++] = (struct agx3_field_kv)KV(61, runtime->cache[2]);
   ov[n++] = (struct agx3_field_kv)KV(63, runtime->discard[2]);
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      ov[n++] = (struct agx3_field_kv)KV(97 + i,
                                       (runtime->wait_mask >> i) & 1);
   for (unsigned ep = m5 ? 0 : 2; ep <= (m5 ? 7 : 5); ++ep) {
      uint8_t bytes[AGX3_FALU_MOD_MAX_LENGTH], size;
      if (agx3_bind_compose(enc, AGX3_MN_f32fma, agx3_bind_ep(enc, ep),
                           m5 ? fma_c : fma_b, m5 ? 16 : 14, ov, n,
                           bytes, &size)) {
         for (unsigned i = 0; i < size; ++i)
            out[i] = bytes[i];
         *len = size;
         return true;
      }
   }
   return false;
}

bool
agx3_falu_fma_mod_runtime(enum agx_isa_encoding enc, uint16_t dest,
                          uint16_t src0, uint16_t src1, uint16_t addend,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_mod_runtime(enc, dest, src0, src1, addend, false, m, runtime,
                          out, len);
}

bool
agx3_falu_fma_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
                        uint16_t addend, const struct agx3_falu_mods *m,
                        const struct agx3_falu_runtime *runtime,
                        uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_mod_runtime(AGX_ISA_ENCODING_AGX3C, dest, src0, src1, addend,
                          true, m, runtime, out, len);
}

/* Native FMA has a minifloat addend form: keep a*b+c fused without a
 * temporary constant register, including absolute/negated multipliers, and
 * preserve each multiplier's lifetime. */
static bool
fma_addend_imm_width_mod(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src0, uint16_t src1, uint8_t immediate,
                            bool dest_half, const struct agx3_falu_mods *m,
                            const struct agx3_falu_runtime *runtime,
                            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !m || !runtime || !out || !len ||
       (!dest_half && (dest & 1)) || (dest_half && !m5) || dest >= 256 || !slot_ok(src0, m->src0_16) ||
       !slot_ok(src1, m->src1_16) || m->addend_abs || m->addend_neg ||
       m->addend_16 || (!m5 && (m->src0_16 || m->src1_16 || m->saturate)) ||
       runtime->wait_mask > (m5 ? 0xff : 0x3f))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,2), KV(25,2), KV(26,2), KV(42,2), KV(43,2),
      KV(59,1), KV(60,2), KV(77,2), KV(91,2), KV(102,1),
      KV(95,3), KV(96,55),
   };
   struct agx3_field_kv ov[32] = {
      KV(10,dest_half ? 1 : 2), KV(2,dest), KV(35,src0), KV(18,src1), KV(51,immediate),
      KV(11,runtime->dest_cache), KV(44,runtime->cache[0]),
      KV(46,runtime->discard[0]), KV(27,runtime->cache[1]),
      KV(29,runtime->discard[1]), KV(47,m->src0_abs), KV(48,m->src0_neg),
      KV(30,m->src1_abs), KV(31,m->src1_neg),
      KV(43,m->src0_16 ? 1 : 2), KV(26,m->src1_16 ? 1 : 2),
      KV(14,m->saturate),
   };
   unsigned n = 17;
   /* G17G preserves the operand/modifier fields, but its long endpoint
    * carries eight dependency slots. Override the G17P template's f102
    * too, so a specimen's slot-6 wait cannot leak into the instruction. */
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      if (m5 || (runtime->wait_mask & (1u << i)))
         ov[n++] = (struct agx3_field_kv)KV(97+i, !!(runtime->wait_mask & (1u << i)));
   for (unsigned ep = 0; ep <= (m5 ? 7 : 5); ++ep) {
      if (agx3_bind_compose(enc,AGX3_MN_f32fma,ep,base,m5 ? 13 : 11,ov,n,out,len))
         return true;
   }
   return false;
}

bool
agx3_falu_fma_addend_imm_width_mod(enum agx_isa_encoding enc, uint16_t dest,
   uint16_t src0, uint16_t src1, uint8_t immediate,
   const struct agx3_falu_mods *m, const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_addend_imm_width_mod(enc, dest, src0, src1, immediate, false,
                                  m, runtime, out, len);
}

bool
agx3_falu_fma_addend_imm_narrow_m5(uint16_t dest, uint16_t src0, uint16_t src1,
   uint8_t immediate, const struct agx3_falu_mods *m,
   const struct agx3_falu_runtime *runtime,
   uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_addend_imm_width_mod(AGX_ISA_ENCODING_AGX3C, dest, src0, src1,
                                  immediate, true, m, runtime, out, len);
}

bool
agx3_falu_fma_addend_imm_mod(enum agx_isa_encoding enc, uint16_t dest,
                            uint16_t src0, uint16_t src1, uint8_t immediate,
                            bool src0_abs, bool src0_neg, bool src1_abs,
                            bool src1_neg,
                            const struct agx3_falu_runtime *runtime,
                            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const struct agx3_falu_mods m = {
      .src0_abs = src0_abs, .src0_neg = src0_neg,
      .src1_abs = src1_abs, .src1_neg = src1_neg,
   };
   return agx3_falu_fma_addend_imm_width_mod(
      enc, dest, src0, src1, immediate, &m, runtime, out, len);
}

bool
agx3_falu_fma_addend_imm(enum agx_isa_encoding enc, uint16_t dest,
                         uint16_t src0, uint16_t src1, uint8_t immediate,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return agx3_falu_fma_addend_imm_mod(enc, dest, src0, src1, immediate,
                                       false, false, false, false, runtime,
                                       out, len);
}


static bool
fma_two_imm(enum agx_isa_encoding enc, uint16_t dest,
            uint16_t src, uint8_t multiplier, uint8_t addend,
            bool dest_half, bool src_abs, bool saturate,
            const struct agx3_falu_runtime *runtime,
            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((enc != AGX_ISA_ENCODING_AGX3B && !m5) || !runtime || !out || !len ||
       dest >= 256 || src >= 256 || (src & 1) || (!dest_half && (dest & 1)) ||
       (!m5 && (dest_half || src_abs || saturate)) ||
       runtime->wait_mask > (m5 ? 0xff : 0x3f) ||
       (runtime->cache[0] && runtime->discard[0]))
      return false;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,dest_half ? 1 : 2), KV(25,2), KV(26,2), KV(42,1), KV(43,2),
      KV(59,1), KV(60,2), KV(77,2), KV(91,2), KV(95,3), KV(96,55), KV(104,1),
   };
   struct agx3_field_kv ov[32] = {
      KV(2,dest), KV(18,src), KV(34,multiplier), KV(51,addend),
      KV(11,runtime->dest_cache), KV(27,runtime->cache[0]),
      KV(29,runtime->discard[0]), KV(30,src_abs), KV(14,saturate),
   };
   unsigned n = 9;
   /* The specimen's f104 is a slot-8 dependency, not an opcode bit.
    * Override every M5 wait field, including zero, before selecting an
    * encoding point. Otherwise every full-width FMA waits on slot 8 and
    * expands to the 16-byte endpoint even without a dependency. */
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      if (m5 || (runtime->wait_mask & (1u << i)))
         ov[n++] = (struct agx3_field_kv)KV(97+i,!!(runtime->wait_mask & (1u << i)));
   for (unsigned ep = 0; ep <= (m5 ? 7 : 5); ++ep)
      if (agx3_bind_compose(enc,AGX3_MN_f32fma,ep,base,m5 ? 13 : 10,ov,n,out,len))
         return true;
   return false;
}

bool
agx3_falu_fma_two_imm(enum agx_isa_encoding enc, uint16_t dest,
                     uint16_t src, uint8_t multiplier, uint8_t addend,
                     const struct agx3_falu_runtime *runtime,
                     uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_two_imm(enc, dest, src, multiplier, addend, false, false, false,
                       runtime, out, len);
}

/* G17G narrows a full-input FMA directly through f32fma's result width f10.
 * A forced device-half store proves this form, distinct from f16fma's half
 * arithmetic and operand widths. Keep its single F32 rounding to F16. */
bool
agx3_falu_fma_two_imm_narrow_m5(uint16_t dest, uint16_t src,
                                uint8_t multiplier, uint8_t addend,
                                bool src_abs, bool saturate,
                                const struct agx3_falu_runtime *runtime,
                                uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_two_imm(AGX_ISA_ENCODING_AGX3C, dest, src, multiplier, addend,
                       true, src_abs, saturate, runtime, out, len);
}

/* Immediate multiplier (f34, selected by f42=1) with a modifier-bearing
 * register multiplicand (f18 + f26/f30/f31) and addend (f52 + f60/f64/f65).
 * Baseline is the constant part of Apple's witnessed fma(0.5, b, c); every
 * operand, modifier, lifetime hint and wait bit is an override, so a field
 * the selected encoding point does not consume makes the composer refuse
 * rather than silently drop it. */
static bool
fma_mul_imm(enum agx_isa_encoding enc, uint16_t dest, bool dest_half,
            uint8_t multiplier, uint16_t src1, uint16_t addend,
            const struct agx3_falu_mods *m,
            const struct agx3_falu_runtime *runtime,
            uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || m == NULL || runtime == NULL ||
       out == NULL || len == NULL || (dest_half && !m5) ||
       (!dest_half && (dest & 1)) || dest >= 256 ||
       !slot_ok(src1, m->src1_16) || !slot_ok(addend, m->addend_16) ||
       runtime->wait_mask > (m5 ? 0xff : 0x3f))
      return false;

   /* There is no f35 register slot in this form, so nothing can carry a
    * multiplier-side modifier or lifetime hint; a caller that has one has not
    * folded it into the immediate and must not be silently obeyed.
    *
    * f31 -- the multiplicand's negate -- is refused for a different reason:
    * it is not witnessed here. Apple never emits it beside an immediate
    * multiplier, folding the sign into the minifloat instead
    * (fma(0.5,-fabs(b),c) is f34=160 with f30, no f31), which is exact
    * because k * -b == -k * b. The caller folds it the same way. */
   if (m->src0_abs || m->src0_neg || m->src0_16 || m->src1_neg ||
       runtime->cache[0] || runtime->discard[0])
      return false;

   for (unsigned i = 1; i < 3; ++i)
      if (runtime->cache[i] && runtime->discard[i])
         return false;

   /* f10 is the destination width: G17G narrows the fused result straight
    * into a half (port/hfmawit, half(fma(a, 0.5f, b))). */
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, dest_half ? 1 : 2), KV(25, 2), KV(26, 2), KV(42, 1),
      KV(43, 2), KV(59, 2), KV(60, 2), KV(77, 2), KV(91, 2),
      /* Native G17G retains operand fields and its own long endpoint. */
      KV(95, 3), KV(96, 55),
   };
   /* Appended rather than counted: a designated-initialiser list with a
    * hand-written length silently grows a zeroed {field 0, value 0} entry the
    * moment an override is removed. */
   struct agx3_field_kv ov[32];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(34, multiplier);
   ov[n++] = (struct agx3_field_kv)KV(18, src1);
   ov[n++] = (struct agx3_field_kv)KV(52, addend);
   ov[n++] = (struct agx3_field_kv)KV(26, m->src1_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(30, m->src1_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(60, m->addend_16 ? 1 : 2);
   ov[n++] = (struct agx3_field_kv)KV(64, m->addend_abs ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(65, m->addend_neg ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(14, m->saturate ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(11, runtime->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, runtime->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(29, runtime->discard[1]);
   ov[n++] = (struct agx3_field_kv)KV(61, runtime->cache[2]);
   ov[n++] = (struct agx3_field_kv)KV(63, runtime->discard[2]);
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i)
      ov[n++] = (struct agx3_field_kv)KV(97 + i, (runtime->wait_mask >> i) & 1);

   for (unsigned ep = 0; ep <= (m5 ? 7 : 5); ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_f32fma, ep, base, m5 ? 12 : 10, ov, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_falu_fma_mul_imm_mod(enum agx_isa_encoding enc, uint16_t dest,
                          uint8_t multiplier, uint16_t src1, uint16_t addend,
                          const struct agx3_falu_mods *m,
                          const struct agx3_falu_runtime *runtime,
                          uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return fma_mul_imm(enc, dest, false, multiplier, src1, addend, m, runtime,
                      out, len);
}

bool
agx3_falu_fma_mul_imm_narrow_m5(uint16_t dest, uint8_t multiplier,
                                uint16_t src1, uint16_t addend,
                                const struct agx3_falu_mods *m,
                                const struct agx3_falu_runtime *runtime,
                                uint8_t out[AGX3_FALU_MOD_MAX_LENGTH],
                                uint8_t *len)
{
   return fma_mul_imm(AGX_ISA_ENCODING_AGX3C, dest, true, multiplier, src1,
                      addend, m, runtime, out, len);
}

bool
agx3_falu_add_imm_narrow_m5(uint16_t dest, uint16_t src, uint8_t immediate,
                         bool src_abs, bool src_neg, bool saturate,
                         const struct agx3_falu_runtime *runtime,
                         uint8_t out[AGX3_FALU_MOD_MAX_LENGTH], uint8_t *len)
{
   return add_immediate(AGX_ISA_ENCODING_AGX3C, dest, src, true, immediate,
                        src_abs, src_neg, saturate, runtime, out, len);
}
