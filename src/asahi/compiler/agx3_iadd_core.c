/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_iadd_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

bool
agx3_iadd_encode(enum agx_isa_encoding encoding,
                 const struct agx3_iadd_desc *d,
                 uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   const bool agx3b = encoding == AGX_ISA_ENCODING_AGX3B;
   if (!agx3b && encoding != AGX_ISA_ENCODING_AGX3C)
      return false;
   /* The 32-bit signedness bits describe the G18G saturating forms
    * (agx3_iadd_encode_g18g); this encoder has no such form. */
   if (d->src_a_sign_extend_32 || d->sign_extend_32)
      return false;

   /* Six scoreboard slots on AGX3b, eight on AGX3c. */
   const unsigned slots = agx3b ? 6 : 8;

   /* A 32-bit destination spans two halves. The source is one half when
    * narrow, an aligned pair otherwise. */
   if ((d->dest & 1) || d->dest + 2 > 256)
      return false;
   if (d->src_16 ? (d->src >= 256) : ((d->src & 1) || d->src + 2 > 256))
      return false;

   /* Only the witnessed shift range, and only real scoreboard slots. Apple's
    * shifted adds witness 1..4; shift 0 is the unshifted form. */
   if (d->shift > 4 || d->wait_mask >= (1u << slots))
      return false;
   const bool m5_imm_sub = !agx3b && d->subtract &&
      !d->src_a_is_register && !d->saturate && !d->shift;
   const bool m5_reg_sub = !agx3b && d->subtract &&
      d->src_a_is_register && !d->saturate;
   /* Native G17G (port/cwit3) addsat/subsat on uint and int: the iadd32 or
    * isub32 register point with f54; the signed form also sets f28/f43:
    *   2f00041a2100a30238812000  r0 <- ^r0 +sat ^r2, unsigned
    *   2f00041a2100a3023881e000  r0 <- ^r0 +sat ^r2, signed
    *   2f00041a2500a20218812000  r0 <- ^r0 -sat ^r2, unsigned */
   const bool m5_sat = !agx3b && d->src_a_is_register &&
      !d->src_a_16 && !d->src_16 && !d->shift && d->saturate &&
      d->src_a_sign_extend == d->sign_extend;
   if (!agx3b && (d->subtract || d->saturate) &&
       !m5_imm_sub && !m5_reg_sub && !m5_sat)
      return false;
   if ((d->saturate || d->src_a_cache) && !d->src_a_is_register)
      return false;
   if (d->src_cache && !d->src_a_is_register && agx3b)
      return false;
   if ((d->src_a_cache && d->src_a_kill) || (d->src_cache && d->src_kill))
      return false;
   /* Saturation is witnessed at full width with matching signedness only. */
   if (d->saturate &&
       (d->src_a_16 || d->src_16 || d->shift ||
        d->src_a_sign_extend != d->sign_extend))
      return false;
   if (d->src_a_is_register) {
      if (d->immediate != 0)
         return false;
      if (d->src_a_16 ? (d->src_a_reg >= 256)
                      : ((d->src_a_reg & 1) || d->src_a_reg + 2 > 256))
         return false;
   } else if (d->src_a_16 || d->src_a_sign_extend) {
      /* Slot A width and sign-extension only exist for a register slot A. */
      return false;
   }

   const struct agx3_field_kv baseline_b[] = {
      KV(9, 2), KV(10, 2), KV(23, 1), KV(24, 1), KV(38, 2), KV(42, 1),
   };
   /*
    * Register slot A: f23 = 2 with the value in f16 instead of f15. f24 and
    * f39 are the two operands' WIDTHS (1 = one 16-bit half, 2 = a 32-bit
    * pair), not part of the operand kind -- the same role f24 already plays
    * in agx3_isub_immediate_encode(). Pinning both to 2 here is what made
    * `iadd d32, h_a, h_b` inexpressible.
    */
   const struct agx3_field_kv baseline_b_reg[] = {
      KV(9, 2),  KV(10, 2), KV(23, 2), KV(24, d->src_a_16 ? 1 : 2),
      KV(38, 2), KV(39, d->src_16 ? 1 : 2),
   };
   /* AGX3c's iadd32 uses the identical operand fields on top of its own two
    * form constants. Witnessed with metal-nt -arch applegpu_g17s on the same
    * lid.x+lid.y / +lid.z pair:
    *   16 + 16   27 04 04 1a 29 00 a3 02 60 00 21 00  f24=1 f39=1 f11=1
    *   32 + 16   27 08 04 1a 21 00 a3 02 a8 00 21 00  f24=2 f39=1
    */
   const struct agx3_field_kv baseline_c_reg[] = {
      KV(9, 2),  KV(10, 2), KV(23, 2), KV(24, d->src_a_16 ? 1 : 2),
      KV(38, 2), KV(39, d->src_16 ? 1 : 2), KV(44, 3), KV(45, d->subtract ? 100 : 80),
   };
   const struct agx3_field_kv baseline_c[] = {
      KV(9, 2),  KV(10, 2), KV(23, 1), KV(24, 1),
      KV(38, 2), KV(42, 1), KV(44, 3),  KV(45, d->subtract ? 100 : 80),
   };
   const struct agx3_field_kv *baseline =
      d->src_a_is_register ? (agx3b ? baseline_b_reg : baseline_c_reg)
                           : (agx3b ? baseline_b : baseline_c);
   const unsigned nbaseline =
      d->src_a_is_register
         ? (agx3b ? sizeof(baseline_b_reg) / sizeof(baseline_b_reg[0])
                  : sizeof(baseline_c_reg) / sizeof(baseline_c_reg[0]))
      : agx3b ? sizeof(baseline_b) / sizeof(baseline_b[0])
              : sizeof(baseline_c) / sizeof(baseline_c[0]);

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_cache ? 1 : 0);
   if (d->src_a_is_register) {
      ov[n++] = (struct agx3_field_kv)KV(16, d->src_a_reg);
      /* Liveness, not baseline: see the note in the header. */
      ov[n++] = (struct agx3_field_kv)KV(27, d->src_a_kill ? 1 : 0);
      ov[n++] = (struct agx3_field_kv)KV(42, d->src_kill ? 1 : 0);
      if (d->src_a_sign_extend)
         ov[n++] = (struct agx3_field_kv)KV(28, 1);
      if (d->src_a_cache)
         ov[n++] = (struct agx3_field_kv)KV(25, 1);
      if (d->src_cache)
         ov[n++] = (struct agx3_field_kv)KV(40, 1);
      if (d->saturate && agx3b)
         ov[n++] = (struct agx3_field_kv)KV(53, 1);
   }
   else
      ov[n++] = (struct agx3_field_kv)KV(15, d->immediate);
   /* The immediate form must preserve live inputs, like the register form. */
   if (!d->src_a_is_register)
      ov[n++] = (struct agx3_field_kv)KV(42, d->src_kill ? 1 : 0);
   if (!d->src_a_is_register && d->src_cache)
      ov[n++] = (struct agx3_field_kv)KV(40, 1);
   ov[n++] = (struct agx3_field_kv)KV(31, d->src);
   if (!d->src_a_is_register)
      ov[n++] = (struct agx3_field_kv)KV(39, d->src_16 ? 1 : 2);
   if (d->sign_extend)
      ov[n++] = (struct agx3_field_kv)KV(43, 1);
   if (d->shift)
      ov[n++] = (struct agx3_field_kv)KV(agx3b ? 54 : 55, d->shift);
   if (m5_sat)
      ov[n++] = (struct agx3_field_kv)KV(54, 1);
   for (unsigned i = 0; i < slots; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV((agx3b ? 45 : 46) + i, 1);
   }

   return agx3_bind_compose(encoding,
                            agx3b ? (d->subtract ? AGX3_MN_isub : AGX3_MN_iadd)
                                  : (d->subtract ? AGX3_MN_isub32 : AGX3_MN_iadd32),
                            0, baseline, nbaseline, ov, n, out, len);
}

static bool
agx3_add_sub_immediate_encode(enum agx_isa_encoding encoding, bool subtract,
                            const struct agx3_isub_imm_desc *d,
                            uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   const bool gpu1 = encoding == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = gpu1 ? 8 : 6;
   if (!d || !out || !len ||
       (encoding != AGX_ISA_ENCODING_AGX3B && !gpu1) || (!subtract && !gpu1) ||
       d->dest > 254 || (d->dest & 1) || d->src >= 256 ||
       (!d->src_16 && ((d->src & 1) || d->sign_extend)) ||
       d->immediate > 255 || (d->src_cache && d->src_kill) ||
       (d->wait_mask & ~((1u << slots) - 1)))
      return false;

   /* Native ushort/short/uint minus1/7/28/255, with paired live/dead
    * sources. f24 selects source width, f28 sign extension, and f25/f27
    * are independent cache/discard hints. Wait fields45..50 retain all
    * producer dependencies; no additional retirement instruction is needed.
    */
   const struct agx3_field_kv base[] = {
      KV(9,2), KV(10,2), KV(23,2), KV(24,d->src_16 ? 1 : 2),
      KV(38,1), KV(39,1), KV(44,3), KV(45,subtract ? 100 : 80),
   };
   struct agx3_field_kv fields[15] = {
      KV(2,d->dest), KV(16,d->src), KV(30,d->immediate),
      KV(25,d->src_cache), KV(27,d->src_kill), KV(28,d->sign_extend),
      KV(11,d->dest_cache),
   };
   unsigned n=7;
   /* HAL300 AGX3c isub32 retains the operand fields, with explicit GPR
    * dispatch/opcode and eight dependencies in fields 46..53. */
   for (unsigned slot=0;slot<slots;slot++)
      if(d->wait_mask & (1u<<slot))
         fields[n++]=(struct agx3_field_kv)KV((gpu1 ? 46 : 45)+slot,1);
   return agx3_bind_compose(encoding,gpu1 ? (subtract ? AGX3_MN_isub32 : AGX3_MN_iadd32) : AGX3_MN_isub,0,
                            base,gpu1 ? 8 : 6,fields,n,out,len);
}

bool
agx3_isub_immediate_encode(enum agx_isa_encoding encoding,
                           const struct agx3_isub_imm_desc *d,
                           uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   return agx3_add_sub_immediate_encode(encoding, true, d, out, len);
}

bool
agx3_iadd_immediate_encode(enum agx_isa_encoding encoding,
                           const struct agx3_isub_imm_desc *d,
                           uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   return agx3_add_sub_immediate_encode(encoding, false, d, out, len);
}

bool
agx3_imad_encode(enum agx_isa_encoding encoding,
                 const struct agx3_imad_desc *d,
                 uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   const bool m5 = encoding == AGX_ISA_ENCODING_AGX3C;
   if (encoding != AGX_ISA_ENCODING_AGX3B && !m5)
      return false;

   if (m5 && ((d->src_c_16 && d->dest_64) || (d->src_c_64 && !d->dest_64) ||
              (d->dest_64 && d->no_addend) ||
              (!d->dest_64 && d->src_a_sign_extend && !d->src_a_16) ||
              d->src_c_sign_extend))
      return false;

   if (d->dest_16 && d->dest_64)
      return false;
   /* A half result occupies one half; wider products remain 32-bit aligned. */
   if (d->dest_16 ? d->dest >= 256
                  : ((d->dest & 1) ||
                     d->dest + (d->dest_64 ? 4u : 2u) > 256))
      return false;
   if ((d->src_a_kill && d->src_a_cache) ||
       (d->multiplier_kill && d->multiplier_cache) ||
       (d->src_c_kill && d->src_c_cache))
      return false;

   if (d->src_a_16 ? (d->src_a >= 256) : ((d->src_a & 1) || d->src_a + 2 > 256))
      return false;
   if (d->addend_is_immediate &&
       (d->no_addend || d->addend_immediate > 255 ||
        d->src_c || d->src_c_16 || d->src_c_64 || d->src_c_sign_extend ||
        d->src_c_cache || d->src_c_kill))
      return false;
   if (d->src_c_16 && d->src_c_64)
      return false;
   if (!d->no_addend && !d->addend_is_immediate &&
       (d->src_c_16 ? (d->src_c >= 256)
                    : ((d->src_c & 1) ||
                       d->src_c + (d->src_c_64 ? 4u : 2u) > 256)))
      return false;
   /* A pure multiply has no addend to widen or sign-extend. */
   if (d->no_addend &&
       (d->src_c || d->src_c_16 || d->src_c_64 || d->src_c_sign_extend))
      return false;
   if (d->dest_64) {
      /* A 32->64 product widens operands that are already 32-bit, so a narrow
       * operand carrying the bit is unwitnessed here. */
      if (!m5 && ((d->src_a_sign_extend && d->src_a_16) ||
                  (d->multiplier_sign_extend && d->multiplier_16)))
         return false;
   } else if (!d->dest_16) {
      /* A 32-bit product widens each 16-bit operand by exactly one step, by
       * that operand's own bit: f28 for slot A, f43 for a slot-B register.
       * A 32-bit operand widens nothing and carries neither.
       *
       *   int(short  a) * 129 + int(short  c)
       *      9f 00 56 00 02 02 04 02 10 2a 66 00     f24 = 1, f28 = 1
       *   int(ushort a) * 129 + int(ushort c)
       *      9f 00 56 00 02 02 04 02 10 2a 22 00     f24 = 1, f28 = 0
       *   int(short) * int(ushort), unsigned slot A
       *      9f 00 56 00 02 00 04 00 10 26 1a 00     f43 = 1 alone
       *   int(ushort) * int(short), unsigned slot B
       *      9f 00 56 00 02 00 04 00 10 26 0e 00     f28 = 1 alone
       *   int a * int(short b)
       *      9f 00 56 00 02 00 10 00 50 26 1a 00     f24 = 2, f39 = 1, f43 = 1
       *   int(short a) * int(short b) + int c
       *      9f 00 56 00 02 08 14 00 10 2f 3e 00     f54 = 2
       *   int(short a) * 129 + int c
       *      9f 00 56 00 02 08 04 02 10 2b 26 00     f54 = 2
       *
       * A half multiplier appears only with no addend or a 32-bit register
       * addend, and no narrow operand appears with an immediate addend. */
      if ((d->src_a_sign_extend && !d->src_a_16) ||
          (d->multiplier_sign_extend && !d->multiplier_16))
         return false;
      if ((d->src_a_16 || d->multiplier_16) && d->addend_is_immediate)
         return false;
      if (!m5 && d->multiplier_16 && !d->no_addend &&
          (d->src_c_16 || d->src_c_64))
         return false;
   }
   /* The 8-bit immediate multiplier has no sign-extend bit of its own. */
   if (!d->multiplier_is_register && d->multiplier_sign_extend)
      return false;
   if (!d->multiplier_is_register && d->multiplier_16)
      return false;
   /*
    * Fresh G17P ushort witnesses ground both narrow source-B classes.  The
    * register form is a low-half multiply, so signed and unsigned MSL produce
    * the same instruction. Current G17P Metal also witnesses a plain half
    * register C for ushort a*b+c. Keep wider C and every widening bit closed:
    * none has a meaning for a 16-bit result in the witnessed domain.
    */
   if (d->dest_16 &&
       (!d->src_a_16 ||
        (d->multiplier_is_register ? !d->multiplier_16 : d->no_addend) ||
        (!d->addend_is_immediate && !d->no_addend &&
         !d->src_c_16) ||
        d->src_a_sign_extend || d->multiplier_sign_extend ||
        d->src_c_sign_extend))
      return false;
   if (d->wait_mask >= (1u << (m5 ? 8 : 6)))
      return false;

   if (d->multiplier_is_register &&
       (d->multiplier_16 ? (d->multiplier_reg >= 256)
                         : ((d->multiplier_reg & 1) ||
                            d->multiplier_reg + 2 > 256)))
      return false;

   /*
    * Slot B holds the multiplier. Immediate is f38/f39 = 1 with the value in
    * f30; a register is f38/f39 = 2 with the value in f31, and carries f42,
    * exactly as Apple emits it and as every register-slot-B `imadd` in the
    * command-stream corpus does.
    */
   const uint64_t dest_width = d->dest_16 ? 1 : (d->dest_64 ? 3 : 2);
   const uint64_t src_a_width = d->src_a_16 ? 1 : 2;
   const uint64_t mult_width = d->multiplier_16 ? 1 : 2;
   const struct agx3_field_kv baseline_imm[] = {
      KV(9, 2),  KV(10, dest_width), KV(23, 2), KV(24, src_a_width),
      KV(38, 1), KV(39, 1), KV(53, 2),
   };
   const struct agx3_field_kv baseline_reg[] = {
      KV(9, 2),  KV(10, dest_width), KV(23, 2), KV(24, src_a_width),
      KV(38, 2), KV(39, mult_width), KV(53, 2),
   };
   /* An absent or immediate addend selects the immediate-C endpoint and f53
    * drops to 1. Fresh G17P code for a*b+1 proves this independently for a
    * register multiplier as well as the existing immediate multiplier. */
   const struct agx3_field_kv baseline_reg_nc[] = {
      KV(9, 2),  KV(10, dest_width), KV(23, 2),
      KV(24, src_a_width),  KV(38, 2), KV(39, mult_width), KV(53, 1),
   };
   /* Native mul37/mul255 use an immediate multiplier and absent addend. */
   const struct agx3_field_kv baseline_imm_nc[] = {
      KV(9, 2), KV(10, dest_width), KV(23, 2), KV(24, src_a_width),
      KV(38, 1), KV(39, 1), KV(53, 1),
   };
   const bool immediate_c = d->no_addend || d->addend_is_immediate;
   const struct agx3_field_kv *baseline =
      immediate_c ? (d->multiplier_is_register ? baseline_reg_nc : baseline_imm_nc)
                   : (d->multiplier_is_register ? baseline_reg : baseline_imm);
   const unsigned nbaseline =
      immediate_c ? (d->multiplier_is_register
         ? sizeof(baseline_reg_nc) / sizeof(baseline_reg_nc[0])
         : sizeof(baseline_imm_nc) / sizeof(baseline_imm_nc[0]))
      : d->multiplier_is_register
         ? (sizeof(baseline_reg) / sizeof(baseline_reg[0]))
         : (sizeof(baseline_imm) / sizeof(baseline_imm[0]));

   struct agx3_field_kv ov[28];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(16, d->src_a);
   ov[n++] = (struct agx3_field_kv)KV(25, d->src_a_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, d->src_a_kill);
   if (d->src_a_sign_extend)
      ov[n++] = (struct agx3_field_kv)KV(28, 1);
   if (d->multiplier_is_register) {
      ov[n++] = (struct agx3_field_kv)KV(31, d->multiplier_reg);
      /* f42 is the multiplier's last-use hint, not a baseline bit. */
      ov[n++] = (struct agx3_field_kv)KV(42, d->multiplier_kill ? 1 : 0);
      ov[n++] = (struct agx3_field_kv)KV(40, d->multiplier_cache);
      if (d->multiplier_sign_extend)
         ov[n++] = (struct agx3_field_kv)KV(43, 1);
   }
   else
      ov[n++] = (struct agx3_field_kv)KV(30, d->multiplier);
   if (immediate_c) {
      ov[n++] = (struct agx3_field_kv)KV(54, 1);
      if (d->addend_is_immediate)
         ov[n++] = (struct agx3_field_kv)KV(45, d->addend_immediate);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(46, d->src_c);
      ov[n++] = (struct agx3_field_kv)KV(55, d->src_c_cache);
      ov[n++] = (struct agx3_field_kv)KV(57, d->src_c_kill);
      ov[n++] = (struct agx3_field_kv)KV(
         54, d->src_c_64 ? 3 : (d->src_c_16 ? 1 : 2));
   }
   if (d->src_c_sign_extend)
      ov[n++] = (struct agx3_field_kv)KV(58, 1);
   if (m5) {
      ov[n++] = (struct agx3_field_kv)KV(59, 3);
      ov[n++] = (struct agx3_field_kv)KV(60, d->dest_64 ? 85 : 84);
   }
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV((m5 ? 61 : 60) + i, 1);
   }

   /* A negated addend is Apple's `imsub`: same fields, one opcode bit. A
    * pure multiply has no addend to negate, so refuse that rather than emit a
    * form nothing witnesses. */
   if (d->negate_addend && (d->no_addend || m5))
      return false;

   return agx3_bind_compose(encoding,
                            m5 ? (d->dest_64 ? AGX3_MN_imadd64 : AGX3_MN_imadd32)
                               : (d->negate_addend ? AGX3_MN_imsub : AGX3_MN_imadd), 0,
                            baseline, nbaseline, ov, n, out, len);
}

/*
 * G18G (M6) compiles as agx3c. These are the G17G/G18G iadd32/isub32 and
 * imadd32/imadd64 encoders as qualified on the M6: their admission (the
 * immediate-minus-register, wide-pair, shifted-subtract and 32-bit signedness
 * forms; the 16x16 and narrow products) differs from agx3_iadd_encode and
 * agx3_imad_encode above, which keep the G17G-family (agx3c) admission for
 * every other GPU. Where both admit a descriptor the field vectors agree.
 * Candidates for all chips once qualified there. Declared in agx3_iadd_core.h.
 */
bool
agx3_iadd_encode_g18g(enum agx_isa_encoding encoding,
                      const struct agx3_iadd_desc *d,
                      uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   /*
    * G18G (agx3d) spells iadd32/isub32 with the agx3c field vector: across
    * the 178 aligned applegpu_g17g/applegpu_g18g iadd32 pairs of the j813-linux
    * gpu/m5-*-reference corpus only the destination, source, cache hint and
    * wait-slot choices differ (f2, f11, f16, f31, f40, f46..f53), and its
    * forms are taken in Apple's order below.
    */
   const bool agx3b = encoding == AGX_ISA_ENCODING_AGX3B;
   if (!agx3b && !agx_encoding_is_g17_class(encoding))
      return false;

   /* Six scoreboard slots on AGX3b, eight on AGX3c and agx3d. */
   const unsigned slots = agx3b ? 6 : 8;

   /* A 32-bit destination spans two halves. The source is one half when
    * narrow, an aligned pair otherwise. */
   if ((d->dest & 1) || d->dest + 2 > 256)
      return false;
   if (d->src_16 ? (d->src >= 256) : ((d->src & 1) || d->src + 2 > 256))
      return false;

   /* Only the witnessed shift range, and only real scoreboard slots. Apple's
    * shifted adds witness 1..4; shift 0 is the unshifted form. */
   if (d->shift > 4 || d->wait_mask >= (1u << slots))
      return false;

   /* G17G's immediate minus a register: the immediate in slot A, a 32-bit or
    * a sign-/zero-extended 16-bit register in slot B, unshifted, with the
    * source's cache (f40) or discard (f42) hint. Apple Metal for
    * applegpu_g17g (test/native/g17g/m5-libagx-reference.metal):
    *
    *   0u - x, x dead          27 00 24 1a 25 00 a2 02 00 80 21 00
    *   31u - x, x live (to r3) 37 03 24 fa 25 00 a2 0a 00 80 11 00   f40
    *   16 - int(short x)       27 00 24 9a 25 08 a2 02 00 00 21 00   f39, f43
    *   16u - uint(ushort x)    27 00 24 9a 25 00 a2 02 00 00 21 00   f39
    */
   const bool g17g_imm_minus_reg = !agx3b && d->subtract &&
                                   !d->src_a_is_register && !d->saturate &&
                                   d->shift == 0;
   if (d->src_a_cache && !agx3b)
      return false;
   if (d->src_cache && !agx3b && !g17g_imm_minus_reg)
      return false;
   if ((d->saturate || d->src_a_cache) && !d->src_a_is_register)
      return false;
   if (d->src_cache && !d->src_a_is_register && !g17g_imm_minus_reg)
      return false;
   if ((d->src_a_cache && d->src_a_kill) || (d->src_cache && d->src_kill))
      return false;
   /* Saturation is witnessed at full width with matching signedness only. */
   if (d->saturate &&
       (d->src_a_16 || d->src_16 || d->shift ||
        d->src_a_sign_extend != d->sign_extend))
      return false;
   if (d->src_a_is_register) {
      if (d->immediate != 0)
         return false;
      if (d->src_a_16 ? (d->src_a_reg >= 256)
                      : ((d->src_a_reg & 1) || d->src_a_reg + 2 > 256))
         return false;
   } else if (d->src_a_16 || d->src_a_sign_extend) {
      /* Slot A width and sign-extension only exist for a register slot A. */
      return false;
   }

   /* The G17G subtract and saturate forms are witnessed only with two
    * 32-bit register operands and no shift; the 32-bit signedness bits only
    * alongside saturation. */
   const bool g17g_wide_pair = !agx3b && d->src_a_is_register &&
                               !d->src_a_16 && !d->src_16 && d->shift == 0;
   /* a - (b << n) is isub32 with the shift in f55, n = 1..4, exactly as
    * iadd32 carries a shifted addend: metal-nt applegpu_g17g/g18g
    * (g18ref-out/probe/m6-isub-shift, is_sub_shl1..4). */
   const bool g17g_shifted_sub = !agx3b && d->subtract && !d->saturate &&
                                 d->src_a_is_register && !d->src_a_16 &&
                                 !d->src_16 && d->shift >= 1;
   if (!agx3b && (d->subtract || d->saturate) && !g17g_wide_pair &&
       !g17g_imm_minus_reg && !g17g_shifted_sub)
      return false;
   if ((d->src_a_sign_extend_32 || d->sign_extend_32) && !d->saturate)
      return false;

   const struct agx3_field_kv baseline_b[] = {
      KV(9, 2), KV(10, 2), KV(23, 1), KV(24, 1), KV(38, 2), KV(42, 1),
   };
   /*
    * Register slot A: f23 = 2 with the value in f16 instead of f15. f24 and
    * f39 are the two operands' WIDTHS (1 = one 16-bit half, 2 = a 32-bit
    * pair), not part of the operand kind -- the same role f24 already plays
    * in agx3_isub_immediate_encode(). Pinning both to 2 here is what made
    * `iadd d32, h_a, h_b` inexpressible.
    */
   const struct agx3_field_kv baseline_b_reg[] = {
      KV(9, 2),  KV(10, 2), KV(23, 2), KV(24, d->src_a_16 ? 1 : 2),
      KV(38, 2), KV(39, d->src_16 ? 1 : 2),
   };
   /* AGX3c's iadd32 uses the identical operand fields on top of its own two
    * form constants. Witnessed with metal-nt -arch applegpu_g17s on the same
    * lid.x+lid.y / +lid.z pair:
    *   16 + 16   27 04 04 1a 29 00 a3 02 60 00 21 00  f24=1 f39=1 f11=1
    *   32 + 16   27 08 04 1a 21 00 a3 02 a8 00 21 00  f24=2 f39=1
    */
   const struct agx3_field_kv baseline_c_reg[] = {
      KV(9, 2),  KV(10, 2), KV(23, 2), KV(24, d->src_a_16 ? 1 : 2),
      KV(38, 2), KV(39, d->src_16 ? 1 : 2), KV(44, 3),
      KV(45, d->subtract ? 100 : 80),
   };
   const struct agx3_field_kv baseline_c[] = {
      KV(9, 2),  KV(10, 2), KV(23, 1), KV(24, 1),
      KV(38, 2), KV(42, 1), KV(44, 3),  KV(45, d->subtract ? 100 : 80),
   };
   const struct agx3_field_kv *baseline =
      d->src_a_is_register ? (agx3b ? baseline_b_reg : baseline_c_reg)
                           : (agx3b ? baseline_b : baseline_c);
   const unsigned nbaseline =
      d->src_a_is_register
         ? (agx3b ? sizeof(baseline_b_reg) / sizeof(baseline_b_reg[0])
                  : sizeof(baseline_c_reg) / sizeof(baseline_c_reg[0]))
      : agx3b ? sizeof(baseline_b) / sizeof(baseline_b[0])
              : sizeof(baseline_c) / sizeof(baseline_c[0]);

   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_cache ? 1 : 0);
   if (d->src_a_is_register) {
      ov[n++] = (struct agx3_field_kv)KV(16, d->src_a_reg);
      /* Liveness, not baseline: see the note in the header. */
      ov[n++] = (struct agx3_field_kv)KV(27, d->src_a_kill ? 1 : 0);
      ov[n++] = (struct agx3_field_kv)KV(42, d->src_kill ? 1 : 0);
      if (d->src_a_sign_extend || d->src_a_sign_extend_32)
         ov[n++] = (struct agx3_field_kv)KV(28, 1);
      if (d->src_a_cache)
         ov[n++] = (struct agx3_field_kv)KV(25, 1);
      if (d->src_cache)
         ov[n++] = (struct agx3_field_kv)KV(40, 1);
      if (d->saturate && agx3b)
         ov[n++] = (struct agx3_field_kv)KV(53, 1);
   }
   else
      ov[n++] = (struct agx3_field_kv)KV(15, d->immediate);
   /* f42 is source liveness for both immediate and register slot A, on both
    * encodings. In particular, an address add must preserve an ID used by
    * later stores. The AGX3c baseline above pins it the way its witness
    * happened to be, and a 32-bit register move -- which is this add with a
    * zero addend -- then killed a register the shader still reads: a
    * `vec4(texture(s, uv).r)` broadcast is two moves off one sampled
    * register, and every reader after the first saw a dead one. Apple's own
    * G17S corpus carries iadd32 with f42 set and clear. */
   if (!d->src_a_is_register)
      ov[n++] = (struct agx3_field_kv)KV(42, d->src_kill ? 1 : 0);
   if (!d->src_a_is_register && d->src_cache)
      ov[n++] = (struct agx3_field_kv)KV(40, 1);
   ov[n++] = (struct agx3_field_kv)KV(31, d->src);
   if (!d->src_a_is_register)
      ov[n++] = (struct agx3_field_kv)KV(39, d->src_16 ? 1 : 2);
   if (d->sign_extend || d->sign_extend_32)
      ov[n++] = (struct agx3_field_kv)KV(43, 1);
   if (d->shift)
      ov[n++] = (struct agx3_field_kv)KV(agx3b ? 54 : 55, d->shift);
   else if (d->saturate && !agx3b)
      ov[n++] = (struct agx3_field_kv)KV(54, 1);
   for (unsigned i = 0; i < slots; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV((agx3b ? 45 : 46) + i, 1);
   }

   return agx3_bind_compose(encoding,
                            agx3b ? (d->subtract ? AGX3_MN_isub : AGX3_MN_iadd) : (d->subtract ? AGX3_MN_isub32 : AGX3_MN_iadd32),
                            agx3_bind_ep(encoding, 0), baseline, nbaseline,
                            ov, n, out, len);
}

bool
agx3_imad_encode_g18g(enum agx_isa_encoding encoding,
                      const struct agx3_imad_desc *d,
                      uint8_t out[AGX3_IADD_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   if (encoding != AGX_ISA_ENCODING_AGX3B &&
       !agx_encoding_is_g17_class(encoding))
      return false;

   /* G18G (agx3d) takes the same imadd32/imadd64 field vector: the
    * applegpu_g18g builds of the m5 reference corpus and of the glxgears vertex probe
    * (2b 80 85 8c 12 04 01 90 13 00 00 00, the vertex_id * 12 + base address)
    * differ from G17G only in the wait slot (f61..f68). */
   const bool m5 = agx_encoding_is_g17_class(encoding);
   /* G17G and G18G's 16x16->32 products use the same width/sign fields.
    * test/native/g18g/narrow-imad.metal witnesses unsigned, signed, mixed
    * signedness, live inputs, and a 32-bit register addend. For example:
    *
    *   G18G unsigned  eb 01 00 00 12 80 41 40 00 00
    *   G18G signed    eb 01 00 00 12 80 41 a0 00 00 20 00
    *   G17G unsigned  2f 00 04 1a 21 00 a1 02 60 00 20 00 01 02
    *   G17G signed    2f 00 04 1a 21 00 a1 02 60 00 20 c0 01 02
    *
    * Pure 32x16->32 products are independently witnessed by mul_u32_u16
    * and mul_i32_i16; these set f24=2 while f39 remains 1. A register
    * addend is witnessed only with two half inputs. Keep immediate-addend,
    * narrow-result and 64-bit forms closed. G18G's packer uses the G17G form
    * before transcoding, preserving the same operand field vector. */
   /* Native902 G17G/G18G ushort/short a*b and a*b+c use this same
    * width-one field vector. A narrow result sign-extends no operand.
    * Keep immediate and negated addends outside the witnessed domain. */
   const bool g17_narrow_product = m5 && d->dest_16 && !d->dest_64 &&
      d->src_a_16 && d->multiplier_is_register && d->multiplier_16 &&
      !d->src_a_sign_extend && !d->multiplier_sign_extend &&
      !d->src_c_64 && !d->src_c_sign_extend &&
      !d->addend_is_immediate && !d->negate_addend &&
      (d->no_addend || d->src_c_16);
   const bool g17_half_product = m5 &&
      !d->dest_16 && !d->dest_64 && (d->src_a_16 || d->no_addend) &&
      d->multiplier_is_register && d->multiplier_16 &&
      !d->src_c_16 && !d->src_c_64 && !d->src_c_sign_extend &&
      !d->addend_is_immediate;
   /* M5's 32x32->64 form is `imadd64` (f10=3, f60=85): a register
    * multiplier with an immediate or a 64-bit register addend, both operands
    * zero-extended. Witnessed byte-exactly against applegpu_g17g
    * (gpu/m5-xfb-alu-reference.metal):
    *
    *   ulong(a)*ulong(b) + 12    2f 00 04 03 21 00 a9 02 a8 80 21 30
    *   ulong(a)*ulong(b) + c     2f 02 04 22 21 00 a9 02 a8 81 20 48
    *   ulong(a)*ulong(b)         2f 00 04 02 21 00 a9 02 a8 80 20 30
    *   long(a)*long(b)           2f 00 04 02 21 08 a9 02 a8 80 20 b0   f28, f43
    *
    * The addend-free product is the immediate form adding #0 (identical
    * bytes); a product signed on one side only has no witness. */
   if (m5 && !g17_half_product && !g17_narrow_product &&
       (d->dest_16 || d->src_a_16 ||
              d->multiplier_16 || d->src_c_16 ||
              (d->src_c_64 && !d->dest_64) ||
              (d->dest_64 && (!d->multiplier_is_register || d->no_addend)) ||
              (d->dest_64 ? d->src_a_sign_extend != d->multiplier_sign_extend
                          : (d->src_a_sign_extend || d->multiplier_sign_extend)) ||
              d->src_c_sign_extend))
      return false;

   if (d->dest_16 && d->dest_64)
      return false;
   /* A half result occupies one half; wider products remain 32-bit aligned. */
   if (d->dest_16 ? d->dest >= 256
                  : ((d->dest & 1) ||
                     d->dest + (d->dest_64 ? 4u : 2u) > 256))
      return false;
   if ((d->src_a_kill && d->src_a_cache) ||
       (d->multiplier_kill && d->multiplier_cache) ||
       (d->src_c_kill && d->src_c_cache))
      return false;

   if (d->src_a_16 ? (d->src_a >= 256) : ((d->src_a & 1) || d->src_a + 2 > 256))
      return false;
   if (d->addend_is_immediate &&
       (d->no_addend || d->addend_immediate > 255 ||
        d->src_c || d->src_c_16 || d->src_c_64 || d->src_c_sign_extend ||
        d->src_c_cache || d->src_c_kill))
      return false;
   if (d->src_c_16 && d->src_c_64)
      return false;
   if (!d->no_addend && !d->addend_is_immediate &&
       (d->src_c_16 ? (d->src_c >= 256)
                    : ((d->src_c & 1) ||
                       d->src_c + (d->src_c_64 ? 4u : 2u) > 256)))
      return false;
   /* A pure multiply has no addend to widen or sign-extend. */
   if (d->no_addend &&
       (d->src_c || d->src_c_16 || d->src_c_64 || d->src_c_sign_extend))
      return false;
   if (d->dest_64) {
      /* A 32->64 product widens operands that are already 32-bit, so a narrow
       * operand carrying the bit is unwitnessed here. */
      if ((d->src_a_sign_extend && d->src_a_16) ||
          (d->multiplier_sign_extend && d->multiplier_16))
         return false;
   } else if (!d->dest_16) {
      /* A 32-bit product widens each 16-bit operand by exactly one step, by
       * that operand's own bit: f28 for slot A, f43 for a slot-B register.
       * A 32-bit operand widens nothing and carries neither.
       *
       *   int(short  a) * 129 + int(short  c)
       *      9f 00 56 00 02 02 04 02 10 2a 66 00     f24 = 1, f28 = 1
       *   int(ushort a) * 129 + int(ushort c)
       *      9f 00 56 00 02 02 04 02 10 2a 22 00     f24 = 1, f28 = 0
       *   int(short) * int(ushort), unsigned slot A
       *      9f 00 56 00 02 00 04 00 10 26 1a 00     f43 = 1 alone
       *   int(ushort) * int(short), unsigned slot B
       *      9f 00 56 00 02 00 04 00 10 26 0e 00     f28 = 1 alone
       *   int a * int(short b)
       *      9f 00 56 00 02 00 10 00 50 26 1a 00     f24 = 2, f39 = 1, f43 = 1
       *   int(short a) * int(short b) + int c
       *      9f 00 56 00 02 08 14 00 10 2f 3e 00     f54 = 2
       *   int(short a) * 129 + int c
       *      9f 00 56 00 02 08 04 02 10 2b 26 00     f54 = 2
       *
       * A half multiplier appears only with no addend or a 32-bit register
       * addend, and no narrow operand appears with an immediate addend. */
      if ((d->src_a_sign_extend && !d->src_a_16) ||
          (d->multiplier_sign_extend && !d->multiplier_16))
         return false;
      if ((d->src_a_16 || d->multiplier_16) && d->addend_is_immediate)
         return false;
      if (d->multiplier_16 && !d->no_addend &&
          (d->src_c_16 || d->src_c_64))
         return false;
   }
   /* The 8-bit immediate multiplier has no sign-extend bit of its own. */
   if (!d->multiplier_is_register && d->multiplier_sign_extend)
      return false;
   if (!d->multiplier_is_register && d->multiplier_16)
      return false;
   /*
    * Fresh G17P ushort witnesses ground both narrow source-B classes.  The
    * register form is a low-half multiply, so signed and unsigned MSL produce
    * the same instruction. Current G17P Metal also witnesses a plain half
    * register C for ushort a*b+c. Keep wider C and every widening bit closed:
    * none has a meaning for a 16-bit result in the witnessed domain.
    */
   if (d->dest_16 &&
       (!d->src_a_16 ||
        (d->multiplier_is_register ? !d->multiplier_16 : d->no_addend) ||
        (!d->addend_is_immediate && !d->no_addend &&
         !d->src_c_16) ||
        d->src_a_sign_extend || d->multiplier_sign_extend ||
        d->src_c_sign_extend))
      return false;
   if (d->wait_mask >= (1u << (m5 ? 8 : 6)))
      return false;

   if (d->multiplier_is_register &&
       (d->multiplier_16 ? (d->multiplier_reg >= 256)
                         : ((d->multiplier_reg & 1) ||
                            d->multiplier_reg + 2 > 256)))
      return false;

   /*
    * Slot B holds the multiplier. Immediate is f38/f39 = 1 with the value in
    * f30; a register is f38/f39 = 2 with the value in f31, and carries f42,
    * exactly as Apple emits it and as every register-slot-B `imadd` in the
    * command-stream corpus does.
    */
   const uint64_t dest_width = d->dest_16 ? 1 : (d->dest_64 ? 3 : 2);
   const uint64_t src_a_width = d->src_a_16 ? 1 : 2;
   const uint64_t mult_width = d->multiplier_16 ? 1 : 2;
   const struct agx3_field_kv baseline_imm[] = {
      KV(9, 2),  KV(10, dest_width), KV(23, 2), KV(24, src_a_width),
      KV(38, 1), KV(39, 1), KV(53, 2),
   };
   const struct agx3_field_kv baseline_reg[] = {
      KV(9, 2),  KV(10, dest_width), KV(23, 2), KV(24, src_a_width),
      KV(38, 2), KV(39, mult_width), KV(53, 2),
   };
   /* An absent or immediate addend selects the immediate-C endpoint and f53
    * drops to 1. Fresh G17P code for a*b+1 proves this independently for a
    * register multiplier as well as the existing immediate multiplier. */
   const struct agx3_field_kv baseline_reg_nc[] = {
      KV(9, 2),  KV(10, dest_width), KV(23, 2),
      KV(24, src_a_width),  KV(38, 2), KV(39, mult_width), KV(53, 1),
   };
   /* Native mul37/mul255 use an immediate multiplier and absent addend. */
   const struct agx3_field_kv baseline_imm_nc[] = {
      KV(9, 2), KV(10, dest_width), KV(23, 2), KV(24, src_a_width),
      KV(38, 1), KV(39, 1), KV(53, 1),
   };
   const bool immediate_c = d->no_addend || d->addend_is_immediate;
   const struct agx3_field_kv *baseline =
      immediate_c ? (d->multiplier_is_register ? baseline_reg_nc : baseline_imm_nc)
                   : (d->multiplier_is_register ? baseline_reg : baseline_imm);
   const unsigned nbaseline =
      immediate_c ? (d->multiplier_is_register
         ? sizeof(baseline_reg_nc) / sizeof(baseline_reg_nc[0])
         : sizeof(baseline_imm_nc) / sizeof(baseline_imm_nc[0]))
      : d->multiplier_is_register
         ? (sizeof(baseline_reg) / sizeof(baseline_reg[0]))
         : (sizeof(baseline_imm) / sizeof(baseline_imm[0]));

   struct agx3_field_kv ov[28];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_cache);
   ov[n++] = (struct agx3_field_kv)KV(16, d->src_a);
   ov[n++] = (struct agx3_field_kv)KV(25, d->src_a_cache);
   ov[n++] = (struct agx3_field_kv)KV(27, d->src_a_kill);
   if (d->src_a_sign_extend)
      ov[n++] = (struct agx3_field_kv)KV(28, 1);
   if (d->multiplier_is_register) {
      ov[n++] = (struct agx3_field_kv)KV(31, d->multiplier_reg);
      /* f42 is the multiplier's last-use hint, not a baseline bit. */
      ov[n++] = (struct agx3_field_kv)KV(42, d->multiplier_kill ? 1 : 0);
      ov[n++] = (struct agx3_field_kv)KV(40, d->multiplier_cache);
      if (d->multiplier_sign_extend)
         ov[n++] = (struct agx3_field_kv)KV(43, 1);
   }
   else
      ov[n++] = (struct agx3_field_kv)KV(30, d->multiplier);
   if (immediate_c) {
      ov[n++] = (struct agx3_field_kv)KV(54, 1);
      if (d->addend_is_immediate)
         ov[n++] = (struct agx3_field_kv)KV(45, d->addend_immediate);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(46, d->src_c);
      ov[n++] = (struct agx3_field_kv)KV(55, d->src_c_cache);
      ov[n++] = (struct agx3_field_kv)KV(57, d->src_c_kill);
      ov[n++] = (struct agx3_field_kv)KV(
         54, d->src_c_64 ? 3 : (d->src_c_16 ? 1 : 2));
   }
   if (d->src_c_sign_extend)
      ov[n++] = (struct agx3_field_kv)KV(58, 1);
   if (m5) {
      ov[n++] = (struct agx3_field_kv)KV(59, 3);
      ov[n++] = (struct agx3_field_kv)KV(60, d->dest_64 ? 85 : 84);
   }
   for (unsigned i = 0; i < (m5 ? 8 : 6); ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV((m5 ? 61 : 60) + i, 1);
   }

   /* A negated addend is Apple's `imsub`: same fields, one opcode bit. A
    * pure multiply has no addend to negate, so refuse that rather than emit a
    * form nothing witnesses. */
   if (d->negate_addend && (d->no_addend || m5))
      return false;

   return agx3_bind_compose(
      encoding,
      m5 ? (d->dest_64 ? AGX3_MN_imadd64 : AGX3_MN_imadd32)
         : (d->negate_addend ? AGX3_MN_imsub : AGX3_MN_imadd),
      agx3_bind_ep(encoding, 0), baseline, nbaseline, ov, n, out, len);
}
