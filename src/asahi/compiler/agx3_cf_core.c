/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_cf_core.h"

#include "agx3_binder.h"

#include <string.h>

/* Baselines recovered by decoding Apple's witnessed templates; only the
 * non-zero constant fields need setting (the composer defaults the rest to 0).
 * AGX3c compares, cnd ops and branches carry extra selector constants. */
#define KV(f, v) {(f), (v)}

static bool
agx3b(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B;
}

static bool
slot_ok(unsigned slot)
{
   return slot <= AGX3_CF_MAX_FLAG_SLOT;
}

static bool
icond_ok(enum agx3_cf_icond cond)
{
   return cond == AGX3_CF_ICOND_ULT || cond == AGX3_CF_ICOND_UGT ||
          cond == AGX3_CF_ICOND_EQ || cond == AGX3_CF_ICOND_SLT ||
          cond == AGX3_CF_ICOND_SGT;
}

static bool
fcond_ok(enum agx3_cf_fcond cond)
{
   return cond == AGX3_CF_FCOND_EQ || cond == AGX3_CF_FCOND_LT ||
          cond == AGX3_CF_FCOND_GT;
}

/* Native half comparisons use the half indices directly. Signed relations
 * sign-extend both inputs; cache/discard belongs to each source independently.
 * The compact unsigned form is six bytes, signed forms are ten bytes. */
bool
agx3_cf_icmp_reg16(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                   unsigned slot, const uint16_t src[2],
                   const struct agx3_cf_source_hints hints[2], uint16_t wait_mask,
                   uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   /* agx3c (G17G and G18G) is the same compare with its class constants f32/f33 = 3/82,
    * waiting at f33 + slot (AGX3b's slot-6 f39 is the same rule): metal-nt
    * applegpu_g17g breaks a loop on ushort a == b with icmp f41 after
    * slot-8 loads, applegpu_g18g tests x < y with f34 after slot-1 loads
    * (g18ref-out/probe/m6-libagx-forms3 lk3_break_eq16, lk3_if_lt16). */
   const bool c = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!agx3b(enc) && !c) || !icond_ok(cond) || !slot_ok(slot) || !src ||
       !hints || !out || !len ||
       (c ? (wait_mask >> 8) != 0 : (wait_mask != 0 && wait_mask != 32)))
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (src[i] >= 256 || (hints[i].cache && hints[i].discard))
         return false;
   const bool sign_extend = cond == AGX3_CF_ICOND_SLT || cond == AGX3_CF_ICOND_SGT;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,1), KV(24,2), KV(25,1), KV(32,3), KV(33,82),
   };
   struct agx3_field_kv fields[19] = {
      KV(2,src[0]), KV(17,src[1]), KV(31,slot), KV(42,cond),
      KV(11,hints[0].cache), KV(13,hints[0].discard), KV(14,sign_extend),
      KV(26,hints[1].cache), KV(28,hints[1].discard), KV(29,sign_extend),
   };
   unsigned n = 10;
   if (c) {
      for (unsigned i = 0; i < 8; ++i)
         if (wait_mask & (1u << i))
            fields[n++] = (struct agx3_field_kv)KV(34 + i, 1);
   } else {
      fields[n++] = (struct agx3_field_kv)KV(39, wait_mask == 32);
   }
   for (unsigned ep = 0; ep < 2; ++ep)
      if (agx3_bind_compose(enc, AGX3_MN_icmp, ep, base, c ? 6 : 4, fields, n, out,
                            len))
         return true;
   return false;
}

bool
agx3_cf_icmp_width_m5(enum agx3_cf_icond cond, unsigned slot,
                      const uint16_t src[2], const bool half[2],
                      const struct agx3_cf_source_hints hints[2],
                      uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!icond_ok(cond) || !slot_ok(slot) || !src || !half || !hints || !out || !len)
      return false;
   for (unsigned i = 0; i < 2; ++i)
      if (src[i] >= 256 || (!half[i] && (src[i] & 1)) ||
          (hints[i].cache && hints[i].discard))
         return false;
   const bool sign = cond == AGX3_CF_ICOND_SLT || cond == AGX3_CF_ICOND_SGT;
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,half[0]?1:2), KV(24,2), KV(25,half[1]?1:2),
      KV(32,3), KV(33,82),
   };
   const struct agx3_field_kv fields[] = {
      KV(2,src[0]), KV(17,src[1]), KV(31,slot), KV(42,cond),
      KV(11,hints[0].cache), KV(13,hints[0].discard), KV(14,sign && half[0]),
      KV(26,hints[1].cache), KV(28,hints[1].discard), KV(29,sign && half[1]),
   };
   for (unsigned ep = 0; ep < 3; ++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,AGX3_MN_icmp,ep,base,6,fields,10,out,len))
         return true;
   return false;
}

/*
 * The source fields of every comparison form are HALF-register indices with
 * eight bits, so halves 0..255 -- all 128 architectural 32-bit registers --
 * are addressable.  These entry points used to stop at half 128, which cut
 * the usable 32-bit register file in half for any shader that branches, and
 * nothing about the encoding asked for that.
 *
 * Apple's own applegpu_g17p output, from kernels with ninety simultaneously
 * live loaded values, compares high halves in every shape:
 *
 *   if (x88 >  x89)          0a9722950485                    icmp ep0  f2=150 f17=148
 *   if (x88 <  17u)          0a9922901583                    icmp ep0  f2=152 f16=17
 *   if (x88 != 0u)           0a992380060007830000            icmp ep1  f2=152
 *   if (fx88 <  fx89)        0a9722950385                    fcmp ep0  f2=150 f19=148
 *   if (fx88 < -fx89)        0a972295038d                    fcmp ep0  f32=1 (negate)
 *   if (|fx88| < |fx89|)     0a97239503000305000000200300    fcmp ep2  f14=1 f31=1
 *   if (fx88 == fx89)        0a972395060000850000            fcmp ep1
 *   if (fx88 <  0.0f)        0a9922810383                    fcmp ep0  f2=152
 *   if (fx88 <  1.0f)        0a9922b10383                    fcmp ep0  f18=48
 *
 * and the surrounding loads and multiply-adds in the same kernel write the
 * same field just as high (`lddev` f2=178, `imadd` f2=182), so the index is
 * plainly a half number and not a narrower operand class.  The generated
 * composer -- which comes from the encoder tables -- accepts f2/f17/f19
 * up to 255 and refuses 256, and every value round-trips back through the
 * structural decomposer unchanged.
 */
bool
agx3_cf_icmp_reg(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                 unsigned slot, uint16_t src0, uint16_t src1,
                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!icond_ok(cond) || !slot_ok(slot) || src0 >= 256 || src1 >= 256)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(24, 2),
                                          KV(25, 2)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(24, 2),
                                          KV(25, 2), KV(32, 3), KV(33, 82)};
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(17, src1), KV(42, cond),
                                      KV(31, slot)};
   return agx3_bind_compose(enc, AGX3_MN_icmp, 1, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 6, ov, 4, out, len);
}

/* Apple's G17P loop tests compare the counter register with the trip count
 * straight from the uniform file: f18 is the uniform half index, f24 = 3 its
 * operand class and f11 the register's cache hint, on the same ten-byte ep1
 * point as the register/register compare, e.g. `i == u6` with the counter
 * cached, or `i == u10` without the hint, both in flag slot 0.
 * f13 is the register's discard hint, as on the register forms. G17G
 * (AGX3C) uses this shape with the additional f32=3/f33=82 selectors. */
bool
agx3_cf_icmp_reg_uniform32(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                           unsigned slot, uint16_t src0, uint16_t uniform,
                           struct agx3_cf_source_hints hint,
                           uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       !icond_ok(cond) || !slot_ok(slot) || src0 >= 256 ||
       (src0 & 1) || uniform >= 256 || (uniform & 1) ||
       (hint.cache && hint.discard) || !out || !len)
      return false;
   const struct agx3_field_kv base[] = {KV(9, 2), KV(10, 2), KV(24, 3),
                                        KV(25, 2), KV(32, 3), KV(33, 82)};
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(18, uniform),
                                      KV(31, slot), KV(42, cond),
                                      KV(11, hint.cache),
                                      KV(13, hint.discard)};
   return agx3_bind_compose(enc, AGX3_MN_icmp, 1, base, agx3b(enc) ? 4 : 6, ov, 6, out, len);
}

/* An aligned 32-bit uniform against an 8-bit immediate: Apple's G17P loop
 * guard reads the trip count straight from the uniform file instead of
 * copying it into a GPR first. Same ten-byte ep1 point as agx3_cf_icmp_imm();
 * the first operand's class f9 = 3 selects the uniform, whose half index is
 * f3 (the register form's f2 is absent), e.g. `u6 == 0` in flag slot 0.
 * G17G (AGX3C) encodes the same operands with its M5 f32=3/f33=82
 * selectors. */
bool
agx3_cf_icmp_uniform_imm(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                         unsigned slot, uint16_t uniform, uint8_t imm,
                         uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       !icond_ok(cond) || !slot_ok(slot) || uniform >= 256 ||
       (uniform & 1) || !out || !len)
      return false;
   const struct agx3_field_kv base[] = {KV(9, 3), KV(10, 2), KV(24, 1),
                                        KV(25, 1), KV(32, 3), KV(33, 82)};
   const struct agx3_field_kv ov[] = {KV(3, uniform), KV(16, imm),
                                      KV(42, cond), KV(31, slot)};
   return agx3_bind_compose(enc, AGX3_MN_icmp, 1, base, agx3b(enc) ? 4 : 6, ov, 4, out, len);
}

bool
agx3_cf_icmp_imm(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                 unsigned slot, uint16_t src0, uint8_t imm,
                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!icond_ok(cond) || !slot_ok(slot) || src0 >= 256)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(24, 1),
                                          KV(25, 1)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(24, 1),
                                          KV(25, 1), KV(32, 3), KV(33, 82)};
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(16, imm), KV(42, cond),
                                      KV(31, slot)};
   return agx3_bind_compose(enc, AGX3_MN_icmp, 1, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 6, ov, 4, out, len);
}

bool
agx3_cf_icmp_imm16_runtime(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                           unsigned slot, uint16_t src, uint8_t imm,
                           struct agx3_cf_source_hints hint, uint16_t wait_mask,
                           uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!agx3b(enc) && !gpu1) || !icond_ok(cond) || !slot_ok(slot) ||
       src >= 256 ||
       !out || !len || (hint.cache && hint.discard) ||
       (gpu1 ? wait_mask > 255 : (wait_mask != 0 && wait_mask != 32)))
      return false;
   if (gpu1) {
      const bool sign = cond == AGX3_CF_ICOND_SLT || cond == AGX3_CF_ICOND_SGT;
      const struct agx3_field_kv base[] = {
         KV(9,2), KV(10,1), KV(14,sign), KV(24,1), KV(25,1), KV(32,3), KV(33,82),
      };
      struct agx3_field_kv fields[14] = {
         KV(2,src), KV(16,imm), KV(42,cond), KV(31,slot),
         KV(11,hint.cache), KV(13,hint.discard),
      };
      for (unsigned i = 0; i < 8; ++i)
         fields[6+i] = (struct agx3_field_kv)KV(34+i, !!(wait_mask & (1u << i)));
      for (unsigned ep = 0; ep < 3; ++ep)
         if (agx3_bind_compose(enc,AGX3_MN_icmp,ep,base,7,fields,14,out,len))
            return true;
      return false;
   }
   if (imm != 0 &&
       (cond == AGX3_CF_ICOND_ULT || cond == AGX3_CF_ICOND_UGT)) {
      const struct agx3_field_kv base[] = {
         KV(9, 2), KV(10, 1), KV(24, 1), KV(25, 1),
      };
      const struct agx3_field_kv fields[] = {
         KV(2, src), KV(16, imm), KV(42, cond), KV(31, slot),
         KV(39, wait_mask == 32),
      };
      return agx3_bind_compose(enc, AGX3_MN_icmp, 0, base, 4, fields, 5, out,
                               len);
   }
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,1), KV(14,1), KV(24,1), KV(25,1), KV(32,3), KV(33,82),
   };
   const struct agx3_field_kv fields[] = {
      KV(2,src), KV(16,imm), KV(42,cond), KV(31,slot), KV(11,hint.cache),
      KV(13,hint.discard), KV(39,wait_mask == 32),
   };
   return agx3_bind_compose(enc,AGX3_MN_icmp,1,base,gpu1 ? 7 : 5,fields,7,out,len);
}

/* G18G only: agx3_cf_icmp_imm16_runtime() as the G18G layer emits it on
 * AGX_ISA_ENCODING_AGX3C (re-encoded as agx3d afterwards). */
bool
agx3_cf_icmp_imm16_runtime_g18g(enum agx_isa_encoding enc,
                                enum agx3_cf_icond cond, unsigned slot,
                                uint16_t src, uint8_t imm,
                                struct agx3_cf_source_hints hint,
                                uint16_t wait_mask,
                                uint8_t out[AGX3_CF_MAX_LENGTH],
                                uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3C)
      return agx3_cf_icmp_imm16_runtime(enc, cond, slot, src, imm, hint,
                                        wait_mask, out, len);

   /* The source is the icmp's eight-bit half index on both encodings:
    * Apple's own applegpu_g17g/g18g compares name halves up to 204 in f2
    * (g18ref-out/probe/m6-high-cf16), and the 16-bit form round-trips
    * 0..254 through the same field. */
   if (!icond_ok(cond) || !slot_ok(slot) || src >= 256 || !out || !len ||
       (hint.cache && hint.discard))
      return false;
   /* Control flow arrives with a retired scoreboard. Ground the signed
    * half/zero predicate needed by sampler-border selection independently
    * of the wider immediate and compact-wait domains. */
   if (imm != 0 || wait_mask != 0 ||
       (cond != AGX3_CF_ICOND_EQ && cond != AGX3_CF_ICOND_SLT &&
        cond != AGX3_CF_ICOND_SGT))
      return false;
   /* Native signed and unsigned zero tests both sign-extend the half.
    * This preserves equality and every relation to zero. Live witnesses
    * retain the source for the taken branch instead of discarding it. */
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,1), KV(14,1), KV(24,1), KV(25,1), KV(32,3), KV(33,82),
   };
   const struct agx3_field_kv fields[] = {
      KV(2,src), KV(16,imm), KV(42,cond), KV(31,slot), KV(11,hint.cache),
      KV(13,hint.discard), KV(39,wait_mask == 32),
   };
   return agx3_bind_compose(enc,AGX3_MN_icmp,1,base,7,fields,7,out,len);
}

/*
 * Compare a 16-bit register against the immediate zero -- the shape every
 * branch on a boolean takes, since Mesa keeps booleans in 16-bit halves.
 *
 * The compare width lives in f10 (1 = 16-bit, 2 = 32-bit); Apple's own
 * `if (ushort != 0)` sets it to 1 and adds f14, where the 32-bit form has
 * neither. Probe-witnessed with `metal-nt -arch applegpu_g16g`; the non-zero
 * 16-bit immediate forms are not covered here. Register/register compares
 * use agx3_cf_icmp_reg16(), including its compact unsigned encoding point.
 */
bool
agx3_cf_icmp_zero16_runtime(enum agx_isa_encoding enc, enum agx3_cf_icond cond,
                            unsigned slot, uint16_t src,
                            struct agx3_cf_source_hints hint, uint16_t wait_mask,
                            uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   return agx3_cf_icmp_imm16_runtime(enc, cond, slot, src, 0, hint, wait_mask,
                                     out, len);
}

bool
agx3_cf_fcmp_reg(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                 bool or_nan, unsigned slot, uint16_t src0, uint16_t src1,
                 uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!fcond_ok(cond) || !slot_ok(slot) || src0 >= 256 || src1 >= 256)
      return false;
   const struct agx3_field_kv base_b[] = {KV(9, 2), KV(10, 2), KV(26, 2),
                                          KV(27, 2)};
   const struct agx3_field_kv base_c[] = {KV(9, 2),  KV(10, 2), KV(26, 2),
                                          KV(27, 2), KV(36, 3), KV(37, 69)};
   /* or_nan folds into the condition: lt->5, gt->6 (eq stays 0). */
   uint64_t fcond = cond;
   if (or_nan && cond == AGX3_CF_FCOND_LT)
      fcond = 5;
   else if (or_nan && cond == AGX3_CF_FCOND_GT)
      fcond = 6;
   else if (or_nan)
      return false;
   const struct agx3_field_kv ov[] = {KV(2, src0), KV(19, src1), KV(46, fcond),
                                      KV(34, or_nan ? 1 : 0), KV(35, slot)};
   return agx3_bind_compose(enc, AGX3_MN_fcmp, 1, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 6, ov, 5, out, len);
}

bool
agx3_cf_fcmp_reg_neg(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                     bool or_nan, unsigned slot, const uint16_t src_in[2],
                     const bool neg_in[2],
                     const struct agx3_cf_source_hints hints_in[2],
                     uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if ((enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) || !src_in || !neg_in || !hints_in ||
       !out || !len || !slot_ok(slot) ||
       (or_nan && enc != AGX_ISA_ENCODING_AGX3C) ||
       (cond != AGX3_CF_FCOND_LT && cond != AGX3_CF_FCOND_GT &&
        !(enc == AGX_ISA_ENCODING_AGX3C && cond == AGX3_CF_FCOND_EQ)) ||
       (neg_in[0] == neg_in[1]) || src_in[0] >= 256 || src_in[1] >= 256 ||
       (src_in[0] & 1) || (src_in[1] & 1) ||
       (hints_in[0].cache && hints_in[0].discard) ||
       (hints_in[1].cache && hints_in[1].discard))
      return false;

   uint16_t src[2] = {src_in[0], src_in[1]};
   struct agx3_cf_source_hints hints[2] = {hints_in[0], hints_in[1]};
   if (neg_in[0]) {
      const uint16_t tmp = src[0];
      src[0] = src[1];
      src[1] = tmp;
      const struct agx3_cf_source_hints hint = hints[0];
      hints[0] = hints[1];
      hints[1] = hint;
      if (cond != AGX3_CF_FCOND_EQ)
         cond = cond == AGX3_CF_FCOND_LT ? AGX3_CF_FCOND_GT
                                       : AGX3_CF_FCOND_LT;
   }

   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 2), KV(26, 2), KV(27, 2), KV(38, 1)};
   struct agx3_field_kv ov[] = {
      KV(2, src[0]), KV(11, hints[0].cache), KV(13, hints[0].discard),
      KV(19, src[1]), KV(28, hints[1].cache), KV(30, hints[1].discard),
      KV(32, 1), KV(35, slot), KV(46, cond), KV(34, 0)};
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      /* Metal537/540: ordered and unordered comparisons with a negated
       * f32 source use the M5 selector plus independent negate
       * and NaN fields. CF boundaries have already retired producer waits. */
      ov[8].value = (unsigned)cond + (or_nan && cond != AGX3_CF_FCOND_EQ ? 4 : 0);
      ov[9].value = or_nan;
      const struct agx3_field_kv m5_base[] = {
         KV(9,2), KV(10,2), KV(26,2), KV(27,2), KV(36,3), KV(37,69),
      };
      for (unsigned ep=0; ep<4; ep++)
         if (agx3_bind_compose(enc, AGX3_MN_fcmp, ep, m5_base, 6, ov, 10, out, len))
            return true;
      return false;
   }
   return agx3_bind_compose(enc, AGX3_MN_fcmp, 0, base, 5, ov, 9, out, len);
}

/*
 * Two half-float registers, compared natively rather than widened: Apple's
 * G17P compiler branches on `if (h0 < h1)` with the f32 form's fields and both
 * source widths set to half (f10 = f27 = 1). As on every float compare, each
 * source's cache and discard sit one and three fields after its width. Pinned
 * by 24 metal-nt -arch applegpu_g17p kernels over lt, gt, eq, ge, le and ne
 * with every combination of the two operands read again after the branch.
 */
bool
agx3_cf_fcmp_reg16(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                   bool or_nan, unsigned slot, const uint16_t src[2],
                   const struct agx3_cf_source_hints hints[2],
                   uint16_t wait_mask, uint8_t out[AGX3_CF_MAX_LENGTH],
                   uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !src || !hints || !out || !len ||
       !slot_ok(slot) || !fcond_ok(cond) || src[0] >= 256 || src[1] >= 256 ||
       (hints[0].cache && hints[0].discard) ||
       (hints[1].cache && hints[1].discard) ||
       (wait_mask & ~(m5 ? UINT16_C(0xff) : UINT16_C(0x3f))))
      return false;

   /* Same condition folding as the f32 form: lt/gt or unordered are codes
    * 5/6 with f34, and a greater-or-equal is its inverse. */
   uint64_t fcond = cond;
   if (or_nan && cond == AGX3_CF_FCOND_LT)
      fcond = 5;
   else if (or_nan && cond == AGX3_CF_FCOND_GT)
      fcond = 6;
   else if (or_nan)
      return false;

   const struct agx3_field_kv base[] = {KV(9, 2), KV(10, 1), KV(26, 2),
                                        KV(27, 1)};
   struct agx3_field_kv ov[19] = {
      KV(2, src[0]), KV(11, hints[0].cache), KV(13, hints[0].discard),
      KV(19, src[1]), KV(28, hints[1].cache), KV(30, hints[1].discard),
      KV(34, or_nan), KV(35, slot), KV(46, fcond),
   };
   unsigned n = 9;
   if (m5) {
      ov[n++] = (struct agx3_field_kv){36, 3};
      ov[n++] = (struct agx3_field_kv){37, 69};
   }
   for (unsigned wait_slot = 0; wait_slot < (m5 ? 8 : 6); ++wait_slot)
      ov[n++] = (struct agx3_field_kv){38 + wait_slot,
                                       !!(wait_mask & (1u << wait_slot))};

   for (unsigned ep = 0; ep < 4; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_fcmp, ep, base, 4, ov, n, out, len))
         return true;
   }
   return false;
}

/*
 * The same register comparison with an ABSOLUTE VALUE on either source.
 *
 * As on `fsel`, ABS is not a bit on the six-byte form: Apple moves to a
 * longer encoding point. Six bytes become fourteen, ep0 becomes ep2, and f14
 * and f31 are the two sources' bits. Grounded with metal-nt -arch
 * applegpu_g17p, one kernel per field. The branch body is a device atomic so
 * that Metal cannot if-convert the compare into an `fsel` and answer a
 * different question:
 *
 *   if (a < b)                  0a013a0303c0                    ep0,  6 B
 *   if (fabs(a) < b)            0a013b0303000300000000800100    ep2, 14 B  f14
 *   if (a < fabs(b))            0a013b0303000300000000800200    ep2, 14 B  f31
 *   if (fabs(a) < fabs(b))      0a013b0303000300000000800300    ep2, 14 B  both
 *   if (fabs(a) > fabs(b))      0a013b0303000200000000800300    f46 = 2
 *   if (fabs(a) == fabs(b))     0a013b0307000000000000800300    f46 absent
 *
 * Every other field keeps its number, so the long form differs from the short
 * one by the endpoint and the two bits alone. Qt Quick's conicalgradient.frag
 * branches on `abs(coord.y) == abs(coord.x)` and failed closed on this.
 *
 * f38..f43 are the six producer wait slots on this form; Apple's kernels
 * above wait on slot 6 because their operands come from a load.
 */
bool
agx3_cf_fcmp_width_m5(enum agx3_cf_fcond cond, bool or_nan, unsigned slot,
                     const uint16_t src[2], const bool half[2],
                     const bool absolute[2],
                     const struct agx3_cf_source_hints hints[2],
                     uint16_t wait_mask,
                     uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!src || !half || !absolute || !hints || !out || !len ||
       !slot_ok(slot) || !fcond_ok(cond) || wait_mask > 0xff)
      return false;
   for (unsigned i = 0; i < 2; ++i) {
      if (src[i] >= 256 || (!half[i] && (src[i] & 1)) ||
          (hints[i].cache && hints[i].discard))
         return false;
   }
   unsigned condition = cond;
   if (or_nan && cond == AGX3_CF_FCOND_LT)
      condition = 5;
   else if (or_nan && cond == AGX3_CF_FCOND_GT)
      condition = 6;
   else if (or_nan)
      return false;
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(26, 2), KV(36, 3), KV(37, 69),
   };
   struct agx3_field_kv fields[22] = {
      KV(2, src[0]), KV(10, half[0] ? 1 : 2),
      KV(11, hints[0].cache), KV(13, hints[0].discard), KV(14, absolute[0]),
      KV(19, src[1]), KV(27, half[1] ? 1 : 2),
      KV(28, hints[1].cache), KV(30, hints[1].discard), KV(31, absolute[1]),
      KV(34, or_nan), KV(35, slot), KV(46, condition),
   };
   unsigned count = 13;
   for (unsigned i = 0; i < 8; ++i)
      fields[count++] = (struct agx3_field_kv)KV(38 + i, (wait_mask >> i) & 1);
   for (unsigned ep = 0; ep < 4; ++ep) {
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, AGX3_MN_fcmp, ep, base, 4,
                           fields, count, out, len))
         return true;
   }
   return false;
}

bool
agx3_cf_fcmp_reg_abs(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                     bool or_nan, unsigned slot, const uint16_t src[2],
                     const bool abs[2],
                     const struct agx3_cf_source_hints hints[2],
                     uint16_t wait_mask,
                     uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (enc != AGX_ISA_ENCODING_AGX3B || !src || !abs || !hints || !out ||
       !len || !slot_ok(slot) || or_nan || !fcond_ok(cond) ||
       src[0] >= 256 || src[1] >= 256 || (src[0] & 1) || (src[1] & 1) ||
       (hints[0].cache && hints[0].discard) ||
       (hints[1].cache && hints[1].discard) ||
       (wait_mask & ~UINT16_C(0x3f)))
      return false;

   /* Without an absolute source this is the short endpoint, and emitting the
    * long form for it would change bytes that already work. */
   if (!abs[0] && !abs[1])
      return false;

   const struct agx3_field_kv base[] = {KV(9, 2), KV(10, 2), KV(26, 2),
                                        KV(27, 2)};
   struct agx3_field_kv ov[16] = {
      KV(2, src[0]), KV(11, hints[0].cache), KV(13, hints[0].discard),
      KV(14, abs[0]),
      KV(19, src[1]), KV(28, hints[1].cache), KV(30, hints[1].discard),
      KV(31, abs[1]),
      KV(35, slot), KV(46, cond),
   };
   unsigned n = 10;
   for (unsigned wait_slot = 0; wait_slot < 6; ++wait_slot)
      ov[n++] = (struct agx3_field_kv){38 + wait_slot,
                                       !!(wait_mask & (1u << wait_slot))};

   return agx3_bind_compose(enc, AGX3_MN_fcmp, 2, base, 4, ov, n, out, len);
}

bool
agx3_cf_cndst(enum agx_isa_encoding enc, unsigned slot, bool invert,
              unsigned push_count, uint8_t out[AGX3_CF_MAX_LENGTH],
              uint8_t *len)
{
   if ((slot != 6 && !slot_ok(slot)) || (push_count != 1 && push_count != 2))
      return false;
   const struct agx3_field_kv base_c[] = {KV(2, 3), KV(3, 33)};
   const struct agx3_field_kv ov[] = {KV(0, invert ? 1 : 0), KV(1, slot),
                                      KV(12, push_count)};
   return agx3_bind_compose(enc, AGX3_MN_cndst, 0, base_c, agx3b(enc) ? 0 : 2, ov, 3,
                            out, len);
}

bool
agx3_cf_cndlt(enum agx_isa_encoding enc, unsigned slot, bool invert,
              uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (!slot_ok(slot))
      return false;
   const struct agx3_field_kv base_b[] = {KV(12, 2)};
   const struct agx3_field_kv base_c[] = {KV(2, 3), KV(3, 31), KV(12, 2)};
   const struct agx3_field_kv ov[] = {KV(0, invert ? 1 : 0), KV(1, slot)};
   return agx3_bind_compose(enc, AGX3_MN_cndlt, 0, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 1 : 3, ov, 2, out, len);
}

bool
agx3_cf_cndcf(enum agx_isa_encoding enc, unsigned slot, bool invert,
              unsigned level_count, uint8_t out[AGX3_CF_MAX_LENGTH],
              uint8_t *len)
{
   if (!slot_ok(slot) || level_count < 2 || level_count > 0xffff)
      return false;

   const struct agx3_field_kv base_c[] = {KV(2, 3), KV(3, 28)};
   const struct agx3_field_kv ov[] = {KV(0, invert ? 1 : 0), KV(1, slot),
                                      KV(12, level_count)};

   /* AGX3b is one form; AGX3c splits the same operation into a compact ep0
    * and a wide ep1, and only ep1 carries the larger level counts. Try the
    * compact point first so a shallow break keeps the short encoding. */
   if (agx3b(enc))
      return agx3_bind_compose(enc, AGX3_MN_cndcf, 0, NULL, 0, ov, 3, out, len);

   return agx3_bind_compose(enc, AGX3_MN_cndcf, 0, base_c, 2, ov, 3, out, len) ||
          agx3_bind_compose(enc, AGX3_MN_cndcf, 1, base_c, 2, ov, 3, out, len);
}

bool
agx3_cf_cndef(enum agx_isa_encoding enc, uint8_t out[AGX3_CF_MAX_LENGTH],
              uint8_t *len)
{
   const struct agx3_field_kv base_b[] = {KV(1, 6), KV(12, 1), KV(13, 2),
                                          KV(14, 1)};
   const struct agx3_field_kv base_c[] = {KV(1, 6),  KV(2, 3),  KV(3, 29),
                                          KV(12, 1), KV(16, 2), KV(17, 1),
                                          KV(18, 2)};
   return agx3_bind_compose(enc, AGX3_MN_cndef, 0, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 4 : 7, NULL, 0, out, len);
}

bool
agx3_cf_cndend(enum agx_isa_encoding enc, unsigned pop_count,
               uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   if (pop_count != 1 && pop_count != 2)
      return false;
   const struct agx3_field_kv base_b[] = {KV(11, 2), KV(12, 2)};
   const struct agx3_field_kv base_c[] = {KV(0, 3), KV(1, 30), KV(14, 2),
                                          KV(15, 2), KV(16, 2)};
   const struct agx3_field_kv ov[] = {KV(10, pop_count)};
   return agx3_bind_compose(enc, AGX3_MN_cndend, 0, agx3b(enc) ? base_b : base_c,
                            agx3b(enc) ? 2 : 5, ov, 1, out, len);
}

bool
agx3_cf_branch(enum agx_isa_encoding enc, bool is_bn,
               uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   enum agx3_mnemonic mnem = is_bn ? AGX3_MN_bn : AGX3_MN_b;
   const int ep = agx3b(enc) ? 0 : 1;
   const struct agx3_field_kv base_c[] = {KV(1, 3), KV(2, is_bn ? 21 : 8)};
   return agx3_bind_compose(enc, mnem, ep, base_c, agx3b(enc) ? 0 : 2, NULL, 0,
                            out, len);
}

/* OR `width` bits of `value` into `buf` starting at instruction bit
 * `instr_bit` (little-endian byte order). The reserved template has zeros in
 * every displacement slice, so a pure OR is a faithful deposit. */
static void
deposit(uint8_t *buf, unsigned instr_bit, unsigned width, uint64_t value)
{
   for (unsigned i = 0; i < width; ++i) {
      if ((value >> i) & 1u) {
         const unsigned bit = instr_bit + i;
         buf[bit / 8] |= (uint8_t)(1u << (bit % 8));
      }
   }
}

bool
agx3_cf_patch_branch(enum agx_isa_encoding enc, bool is_bn, uint8_t *branch,
                     int64_t displacement)
{
   (void)is_bn; /* b and bn share the f0 scatter within an encoding. */
   if (branch == NULL || (displacement & 1) != 0)
      return false;

   /* 48-bit two's-complement range. */
   if (displacement < -(INT64_C(1) << 47) || displacement >= (INT64_C(1) << 47))
      return false;

   const uint64_t d = (uint64_t)displacement & ((INT64_C(1) << 48) - 1);

   if (enc == AGX_ISA_ENCODING_AGX3B) {
      deposit(branch, 1 * 16 + 9, 7, d >> 1);   /* disp[1:7]   */
      deposit(branch, 2 * 16 + 0, 16, d >> 8);  /* disp[8:23]  */
      deposit(branch, 3 * 16 + 0, 16, d >> 24); /* disp[24:39] */
      deposit(branch, 4 * 16 + 0, 8, d >> 40);  /* disp[40:47] */
      return true;
   }

   if (enc == AGX_ISA_ENCODING_AGX3C) {
      deposit(branch, 0 * 16 + 7, 1, d >> 1);   /* disp[1]     */
      deposit(branch, 0 * 16 + 12, 3, d >> 2);  /* disp[2:4]   */
      deposit(branch, 1 * 16 + 1, 1, d >> 5);   /* disp[5]     */
      deposit(branch, 1 * 16 + 3, 1, d >> 6);   /* disp[6]     */
      deposit(branch, 1 * 16 + 5, 3, d >> 7);   /* disp[7:9]   */
      deposit(branch, 0 * 16 + 8, 4, d >> 10);  /* disp[10:13] */
      deposit(branch, 1 * 16 + 12, 1, d >> 14); /* disp[14]    */
      deposit(branch, 2 * 16 + 0, 5, d >> 15);  /* disp[15:19] */
      deposit(branch, 3 * 16 + 0, 11, d >> 20); /* disp[20:30] */
      deposit(branch, 3 * 16 + 14, 2, d >> 31); /* disp[31:32] */
      deposit(branch, 4 * 16 + 0, 15, d >> 33); /* disp[33:47] */
      return true;
   }

   return false;
}

/*
 * G17G (AGX3c) spells the same compare with its selector constants f36/f37
 * and eight wait slots (f38..f45). Apple's applegpu_g17g branches in
 * test/native/g17g/m5-compare-select-reference.metal, each a register that is
 * not read again (f13), slot 0, chosen smallest first as Apple does:
 *
 *   if (x <  0.5f)    0a012aa10302             f18 = 32, f46 = 1
 *   if (x >  1.0f)    0a012ab10202             f18 = 48, f46 = 2
 *   if (x < -1.0f)    0a013ab10302             f18 = 176
 *   half x < 1.0h     0a002ab10302             f10 = 1
 *   if (x >= 0.0f)    1a012b81060002020000     lt, unordered: f34, f46 = 5
 *   if (x <= 2.0f)    1a012bc1060003020000     gt, unordered: f46 = 6
 *   if (x == 0.0f)    0a012b81060000020000     f46 = 0
 */
bool agx3_cf_fcmp_zero_imm(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                        bool or_nan, unsigned slot, uint16_t source, bool source_16,
                        uint8_t minifloat, struct agx3_cf_source_hints hint,
                        uint16_t wait_mask,
                        uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!agx3b(enc) && !m5) || !out || !len || !fcond_ok(cond) || !slot_ok(slot) ||
       source >= 256 || (!source_16 && (source & 1)) || (hint.cache && hint.discard) ||
       (wait_mask >> (m5 ? 8 : 6)) || (or_nan && cond == AGX3_CF_FCOND_EQ))
      return false;
   const struct agx3_field_kv base[]={KV(9,2),KV(10,source_16?1:2),KV(26,1),KV(27,2),KV(36,3),KV(37,69)};
   struct agx3_field_kv fields[15]={
      KV(2,source),KV(18,minifloat),KV(11,hint.cache),KV(13,hint.discard),
      KV(34,or_nan),KV(35,slot),KV(46,(unsigned)cond+(or_nan?4:0)),
   };
   for(unsigned i=0;i<(m5?8:6);i++)fields[7+i]=(struct agx3_field_kv)KV(38+i,!!(wait_mask&(1u<<i)));
   for(unsigned ep=0;ep<=(m5?3:1);ep++)
      if(agx3_bind_compose(enc,AGX3_MN_fcmp,ep,base,m5?6:4,fields,m5?15:13,out,len))return true;
   return false;
}

/*
 * The same encoding point comparing against a NON-ZERO constant.
 *
 * f18 is the second operand and it was pinned to zero, so `if (x < 1.0)` --
 * ordinary GLSL, and what dEQP's workgroup-memory tests branch on -- had no
 * encoding. This is the same defect the integer side already carried and
 * fixed: agx3_cf_icmp_zero16_runtime() never set f16, which is what made it a
 * zero compare.
 *
 * Apple's bytes, one kernel per variable, the branch body made a loop so it
 * cannot be if-converted into a select (a trivial `if`/`else` compiles to
 * fsel and emits no fcmp at all, which is why this needs a real branch):
 *
 *   if (v < 0.0f)   0a 03 2a 81 03 c2      f18 absent
 *   if (v < 0.5f)   0a 03 2a a1 03 c2      f18 = 32
 *   if (v < 1.0f)   0a 03 2a b1 03 c2      f18 = 48
 *   if (v < 2.0f)   0a 03 2a c1 03 c2      f18 = 64
 *   if (v > 1.0f)   0a 03 2a b1 02 c2      f18 = 48, relation in f46
 *   half v < 1.0h   0a 02 2a b1 03 c2      f18 = 48, f10 = 1
 *
 * so the constant is the ordinary minifloat this ISA uses everywhere else.
 */
bool agx3_cf_fcmp_zero(enum agx_isa_encoding enc, enum agx3_cf_fcond cond,
                       bool or_nan, unsigned slot, uint16_t source,
                       bool source_16, struct agx3_cf_source_hints hint,
                       uint16_t wait_mask,
                       uint8_t out[AGX3_CF_MAX_LENGTH], uint8_t *len)
{
   return agx3_cf_fcmp_zero_imm(enc, cond, or_nan, slot, source, source_16, 0,
                                hint, wait_mask, out, len);
}
