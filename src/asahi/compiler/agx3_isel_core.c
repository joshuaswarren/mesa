/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_isel_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

bool
agx3_bittest_encode(enum agx_isa_encoding enc,
                   const struct agx3_bittest_desc *d,
                   uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   if (!out || !len)
      return false;
   memset(out, 0, AGX3_ISEL_MAX_LENGTH);
   *len = 0;
   if (!d || enc != AGX_ISA_ENCODING_AGX3B ||
       d->dest >= 128 || (d->dest & 1) ||
       (d->mask != 1 && d->mask != 2 && d->mask != 4) ||
       (d->negate[0] && d->negate[1]) || d->wait_mask >= 64)
      return false;
   for (unsigned i = 0; i < 3; ++i) {
      if (d->src[i] >= 128 || (d->src[i] & 1) ||
          (d->cache[i] && d->discard[i]))
         return false;
   }

   const bool negate = d->negate[0] || d->negate[1];
   const struct agx3_field_kv base[] = {
      KV(42, 2), KV(43, 2), KV(57, 1), KV(58, 1),
      KV(72, 2), KV(73, 2), KV(89, 2), KV(90, 2), KV(106, 3),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   if (negate) {
      ov[n++] = (struct agx3_field_kv)KV(0, 1);
      ov[n++] = (struct agx3_field_kv)KV(17, d->dest);
      ov[n++] = (struct agx3_field_kv)KV(24, 2);
      ov[n++] = (struct agx3_field_kv)KV(25, 2);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(3, d->dest);
      ov[n++] = (struct agx3_field_kv)KV(10, 2);
      ov[n++] = (struct agx3_field_kv)KV(11, 2);
   }
   ov[n++] = (struct agx3_field_kv)KV(35, d->src[0]);
   ov[n++] = (struct agx3_field_kv)KV(49, d->mask);
   ov[n++] = (struct agx3_field_kv)KV(65, d->src[1]);
   ov[n++] = (struct agx3_field_kv)KV(82, d->src[2]);
   ov[n++] = (struct agx3_field_kv)KV(44, d->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(46, d->discard[0]);
   ov[n++] = (struct agx3_field_kv)KV(74, d->cache[1]);
   ov[n++] = (struct agx3_field_kv)KV(76, d->discard[1]);
   ov[n++] = (struct agx3_field_kv)KV(91, d->cache[2]);
   ov[n++] = (struct agx3_field_kv)KV(93, d->discard[2]);
   ov[n++] = (struct agx3_field_kv)KV(78, d->negate[0]);
   ov[n++] = (struct agx3_field_kv)KV(95, d->negate[1]);
   for (unsigned slot = 0; slot < 6; ++slot) {
      if (d->wait_mask & (1u << slot))
         ov[n++] = (struct agx3_field_kv)KV(98 + slot, 1);
   }
   return agx3_bind_compose(enc, AGX3_MN_isel, negate ? 22 : 27, base,
                           sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

static bool
cond_ok(enum agx3_isel_icond cond)
{
   return cond == AGX3_ISEL_ICOND_ULT || cond == AGX3_ISEL_ICOND_UGT ||
          cond == AGX3_ISEL_ICOND_EQ || cond == AGX3_ISEL_ICOND_SLT ||
          cond == AGX3_ISEL_ICOND_SGT;
}

/* A 32-bit value spans two halves, so its index is 2-aligned and the pair
 * must fit the 256-half register file. A 16-bit one is a single half. */
static bool
reg32_ok(uint16_t half)
{
   return (half & 1) == 0 && half + 2 <= 256;
}

static bool
reg16_ok(uint16_t half)
{
   return half < 256;
}

/*
 * One entry per witnessed operand shape. `base` is the field vector of an
 * Apple-compiled instruction with the operands removed; `true_field` and
 * `false_field` name where this shape puts its two selected operands, which
 * differs by operand kind: an immediate goes to f64/f81, a kind-2 register to
 * f65/f82, and the 32-bit both-registers form's kind-3 registers to f66/f83.
 */
struct agx3_isel_form {
   uint8_t ep;
   uint8_t true_field;
   uint8_t false_field;
   uint8_t nbase;
   struct agx3_field_kv base[16];
};

/* 32-bit result: ep27. */
static const struct agx3_isel_form form_32_reg_reg = {
   27, 66, 83, 12,
   {KV(10, 2), KV(11, 2), KV(42, 2), KV(43, 2), KV(44, 1), KV(57, 2),
    KV(58, 2), KV(59, 1), KV(72, 3), KV(73, 2), KV(89, 3), KV(90, 2)}};
static const struct agx3_isel_form form_32_imm_imm = {
   27, 64, 81, 12,
   {KV(10, 2), KV(11, 2), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 2), KV(89, 1), KV(90, 2)}};
/* These legacy vectors predate explicit lifetime arguments. Both happened to
 * keep their results live and therefore carry destination-cache f12. New
 * production callers use encode_explicit(), which derives f12 from post-RA
 * state for every operand shape and result width. */
static const struct agx3_isel_form form_32_reg_imm = {
   27, 65, 81, 14,
   {KV(10, 2), KV(11, 2), KV(12, 1), KV(42, 2), KV(43, 2), KV(46, 1),
    KV(57, 2), KV(58, 2), KV(61, 1), KV(72, 2), KV(73, 2), KV(76, 1),
    KV(89, 1), KV(90, 2)}};
static const struct agx3_isel_form form_32_imm_reg = {
   27, 64, 82, 14,
   {KV(10, 2), KV(11, 2), KV(12, 1), KV(42, 2), KV(43, 2), KV(46, 1),
    KV(57, 2), KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 2), KV(89, 2),
    KV(90, 2), KV(93, 1)}};

/* 16-bit result: ep25. */
static const struct agx3_isel_form form_16_imm_imm = {
   25, 64, 81, 12,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 1), KV(89, 1), KV(90, 1)}};
static const struct agx3_isel_form form_16_reg_imm = {
   25, 65, 81, 13,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 2), KV(73, 1), KV(76, 1), KV(89, 1),
    KV(90, 1)}};
static const struct agx3_isel_form form_16_imm_reg = {
   25, 64, 82, 13,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 1), KV(73, 1), KV(89, 2), KV(90, 1),
    KV(93, 1)}};
static const struct agx3_isel_form form_16_reg_reg = {
   25, 65, 82, 14,
   {KV(10, 2), KV(11, 1), KV(42, 2), KV(43, 2), KV(46, 1), KV(57, 2),
    KV(58, 2), KV(61, 1), KV(72, 2), KV(73, 1), KV(76, 1), KV(89, 2),
    KV(90, 1), KV(93, 1)}};

/*
 * A 16-bit result in an odd half is a different encoding point, not a field:
 * Apple emits ep25 for an even half and ep26 for an odd one, with an
 * otherwise identical field layout. Witnessed by ushort2/ushort4 kernels,
 * whose components must land in adjacent halves, so the compiler had to
 * produce both. (Scalar code never does -- Apple's allocator always picks an
 * even half, which is why the first pass here saw only ep25.)
 */
static const struct agx3_isel_form *
pick_form(bool dest_16, enum agx3_isel_operand t, enum agx3_isel_operand f)
{
   const bool ti = t == AGX3_ISEL_IMMEDIATE, fi = f == AGX3_ISEL_IMMEDIATE;

   if (dest_16) {
      if (ti && fi)
         return &form_16_imm_imm;
      if (!ti && fi)
         return &form_16_reg_imm;
      if (ti && !fi)
         return &form_16_imm_reg;
      return &form_16_reg_reg;
   }

   if (ti && fi)
      return &form_32_imm_imm;
   if (!ti && fi)
      return &form_32_reg_imm;
   if (ti && !fi)
      return &form_32_imm_reg;
   return &form_32_reg_reg;
}

static bool
operand_ok(enum agx3_isel_operand kind, uint16_t value, bool narrow)
{
   if (kind == AGX3_ISEL_IMMEDIATE)
      return value <= 0xff;
   if (kind == AGX3_ISEL_REGISTER)
      return narrow ? reg16_ok(value) : reg32_ok(value);
   return false;
}

/* Fresh G17P eq0/eq0_live/lt2/gt255 native witnesses. This form uses kind-2
 * selected registers even with two register selections; it cannot reuse the
 * older register-comparand form's kind-3 fields. */
/* Uniform operands: aligned 32-bit words in the low uniform file. */
static bool
uniform32_ok(uint16_t half)
{
   return (half & 1) == 0 && half + 2 <= 256;
}

static bool
encode_explicit(const struct agx3_isel_desc *d,
                 uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   const bool tu = d->true_kind == AGX3_ISEL_UNIFORM;
   const bool fu = d->false_kind == AGX3_ISEL_UNIFORM;
   const bool any_uniform = d->cmp0_uniform || d->cmp1_uniform || tu || fu;

   /* Native G17P (metal-nt -arch applegpu_g17p) uniform isel witnesses:
    *   (u < r) ? r : r      02091701020285c00004   cmp0 f36, f42 = 3
    *   (r < u) ? r : r      02010f09020205c20004   cmp1 f51, f57 = 3
    *   (r == 7) ? u : r     02010f86260a17c20002   true f66, f72 = 3
    *   (r == 7) ? r : u     02010f86060217c2200c   false f83, f89 = 3
    *   (r == 7) ? u : u     02010f86260a17c2200c
    *   (r < r) ? u : 3      02010f03220a05c02083
    * Keep to those shapes: one uniform comparand beside a 32-bit register,
    * or uniform selects beside register/immediate comparands, 32-bit result. */
   if (any_uniform) {
      if (d->dest_16 || d->cmp_16 ||
          (d->cmp0_uniform && d->cmp1_uniform) ||
          ((d->cmp0_uniform || d->cmp1_uniform) &&
           (d->cmp1_immediate || tu || fu)) ||
          (d->cmp0_uniform && !uniform32_ok(d->cmp0)) ||
          (d->cmp1_uniform && !uniform32_ok(d->cmp1)) ||
          (tu && !uniform32_ok(d->sel_true)) ||
          (fu && !uniform32_ok(d->sel_false)) ||
          (d->cmp0_uniform && (d->cache[0] || d->discard[0])) ||
          (d->cmp1_uniform && (d->cache[1] || d->discard[1])) ||
          (tu && (d->cache[2] || d->discard[2])) ||
          (fu && (d->cache[3] || d->discard[3])) ||
          (!tu && !operand_ok(d->true_kind, d->sel_true, false)) ||
          (!fu && !operand_ok(d->false_kind, d->sel_false, false)))
         return false;
   }

   if ((!d->cmp0_uniform &&
        !(d->cmp_16 ? reg16_ok(d->cmp0) : reg32_ok(d->cmp0))) ||
       (!d->cmp1_uniform &&
        (d->cmp1_immediate ? d->cmp1 > 255 :
         !(d->cmp_16 ? reg16_ok(d->cmp1) : reg32_ok(d->cmp1)))))
      return false;

   for (unsigned i = 0; i < 4; ++i) {
      if (d->cache[i] && d->discard[i])
         return false;
   }

   const bool ti = d->true_kind == AGX3_ISEL_IMMEDIATE;
   const bool fi = d->false_kind == AGX3_ISEL_IMMEDIATE;
   if ((d->cmp1_immediate && (d->cache[1] || d->discard[1])) ||
       (ti && (d->cache[2] || d->discard[2])) ||
       (fi && (d->cache[3] || d->discard[3])))
      return false;

   const unsigned width = d->dest_16 ? 1 : 2;
   const struct agx3_field_kv base[] = {
      KV(10, 2), KV(11, width), KV(12, d->dest_cache),
      KV(42, d->cmp0_uniform ? 3 : 2), KV(43, d->cmp_16 ? 1 : 2),
      KV(57, d->cmp1_immediate ? 1 : d->cmp1_uniform ? 3 : 2),
      KV(58, (d->cmp1_immediate || d->cmp_16) ? 1 : 2),
      KV(72, ti ? 1 : tu ? 3 : 2), KV(73, width),
      KV(89, fi ? 1 : fu ? 3 : 2), KV(90, width),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(3, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(d->cmp0_uniform ? 36 : 35, d->cmp0);
   ov[n++] = (struct agx3_field_kv)KV(d->cmp1_immediate ? 49 :
                                      d->cmp1_uniform ? 51 : 50, d->cmp1);
   ov[n++] = (struct agx3_field_kv)KV(106, d->cond);
   ov[n++] = (struct agx3_field_kv)KV(ti ? 64 : tu ? 66 : 65, d->sel_true);
   ov[n++] = (struct agx3_field_kv)KV(fi ? 81 : fu ? 83 : 82, d->sel_false);
   if (!d->cmp0_uniform) {
      ov[n++] = (struct agx3_field_kv)KV(44, d->cache[0]);
      ov[n++] = (struct agx3_field_kv)KV(46, d->discard[0]);
   }
   if (d->cmp_16 && d->cond >= AGX3_ISEL_ICOND_EQ)
      ov[n++] = (struct agx3_field_kv)KV(47, 1);
   if (!d->cmp1_immediate && !d->cmp1_uniform) {
      ov[n++] = (struct agx3_field_kv)KV(59, d->cache[1]);
      ov[n++] = (struct agx3_field_kv)KV(61, d->discard[1]);
      if (d->cmp_16 && d->cond >= AGX3_ISEL_ICOND_EQ)
         ov[n++] = (struct agx3_field_kv)KV(62, 1);
   }
   if (!ti && !tu) {
      ov[n++] = (struct agx3_field_kv)KV(74, d->cache[2]);
      ov[n++] = (struct agx3_field_kv)KV(76, d->discard[2]);
   }
   if (!fi && !fu) {
      ov[n++] = (struct agx3_field_kv)KV(91, d->cache[3]);
      ov[n++] = (struct agx3_field_kv)KV(93, d->discard[3]);
   }
   for (unsigned i = 0; i < 6; ++i)
      ov[n++] = (struct agx3_field_kv)KV(98 + i, (d->wait_mask >> i) & 1);

   const unsigned ep = d->dest_16 ? 25 + (d->dest & 1) : 27;
   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3B, AGX3_MN_isel, ep,
                           base, sizeof(base) / sizeof(base[0]), ov, n, out, len);
}

/* GPU1 separates integer and floating-point selects. These field locations
 * come from its own encoder table, not the GPU0 isel layout. Native M5
 * ushort kernels witness odd/even half results and half comparisons to 0/1,
 * and gpu/m5-xfb-alu8-reference.metal a half comparison to 2 with a word
 * result.
 */
static bool
encode_gpu1(const struct agx3_isel_desc *d,
              uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   /* A half comparand is witnessed only against an equality immediate: 0
    * and 1 in both result widths, and 2 with a word result (applegpu_g17g,
    * `select(0u, v, k == 2)` and `select(w, v, k == 2)` for ushort k). f30
    * is the same 8-bit deposit in every form, and
    * test/native/g17g/m5-libagx-reference.metal carries it at 3 with a half
    * result and at 4 and 255 with a word result, the last with the
    * immediate on the false side:
    *
    *   (k == 4) ? 0u : w      02 08 0f 84 26 80 07 02 82 00
    *   (k == 255) ? w : 7u    02 08 1f fe 86 00 17 02 22 87
    *   (k == 3) ? 0 : j       02 00 0d 82 26 80 17 02 82 01   half result
    *
    * Two half comparands (f24 = 1, the second a register: f38 = 2, f39 = 1,
    * f31) are witnessed with a half result, for equality and unsigned
    * less-than, on applegpu_g18g and applegpu_g17g alike
    * (g18ref-out/probe/m6-alu-sel i_u16_*, i_s16_*, m6-alu-hints):
    *
    *   (a == b) ? 1 : 0     iseli16 h2 = ^h2 == ^h3 ? #1 : #0   f28, f43
    *   (a != b) ? 1 : 0     iseli16 h2 = ^h2 == ^h3 ? #0 : #1   f28, f43
    *   short a == short b   the same bytes as ushort
    *   (a <  b) ? 1 : 0     iseli16 h2 = ^h2 <  ^h3 ? #1 : #0   f82 = 1
    *   ... a, b reused      f25/f40, the comparands' cache hints
    *   odd halves           h3 == h4, h2 == h3: any half index
    *
    * f28 and f43 widen the two halves by sign extension; Apple sets both for
    * equality, where the extension cannot change the outcome, and neither
    * for the unsigned relation.
    *
    * Native mixed-select probes also witness two half comparands selecting
    * full-width registers on G17G and G18G: eq/ne, unsigned lt/gt and signed
    * lt/gt, with live and discarded operands. This is iseli32, retaining
    * f24=f39=1 for the comparands while f10=f54=f68=2 select the full word.
    * Signed relations set both extension fields, as equality does. Equality
    * additionally witnesses one selected u8 immediate, on either side. */
   const bool half_regs = d->cmp_16 && !d->cmp1_immediate;
   const bool ti = d->true_kind == AGX3_ISEL_IMMEDIATE;
   const bool fi = d->false_kind == AGX3_ISEL_IMMEDIATE;
   if (half_regs && d->dest_16 && d->cond != AGX3_ISEL_ICOND_EQ &&
       d->cond != AGX3_ISEL_ICOND_ULT)
      return false;
   if (half_regs && !d->dest_16 && (ti || fi) &&
       (d->cond != AGX3_ISEL_ICOND_EQ || (ti && fi)))
      return false;
   if (d->cmp_16 && !half_regs &&
       (d->cmp1 > 255 || d->cond != AGX3_ISEL_ICOND_EQ))
      return false;
   /* A narrow second comparand widens into a full-width comparison only.
    * Apple's kernels load each ushort lane into its own register, so every
    * witnessed narrow comparand is a low half; odd halves stay refused. */
   if (d->cmp1_16 && (d->cmp_16 || d->cmp1_immediate || (d->cmp1 & 1) ||
                      !reg16_ok(d->cmp1)))
      return false;
   if (d->cmp1_sign_extend && !d->cmp1_16)
      return false;
   /* A signed relation with a 16-bit result is the same instruction with
    * f82 = 5/6, on a full-width comparand: applegpu_g17g kernels in
    * test/native/g17g/m5-compare-select-reference.metal, which is also the
    * shape glamor's gradient stop search reaches --
    *
    *   (i > 1) ? 0 : 1       02010d82228107022080   slt 2, constants exchanged
    *   (i < 1) ? 0 : 1       02010d80228106022080   sgt 0
    *   (i < j) ? 0 : 1       02011d03228007002081   slt, register comparand
    *
    * A narrow comparand with a signed relation stays unwitnessed. */
   if (d->dest_16 &&
       (d->cond != AGX3_ISEL_ICOND_ULT && d->cond != AGX3_ISEL_ICOND_UGT &&
        d->cond != AGX3_ISEL_ICOND_EQ) &&
       (d->cmp_16 || d->cmp1_16))
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if (d->cache[i] && d->discard[i])
         return false;
   }
   if ((d->cmp1_immediate && (d->cache[1] || d->discard[1])) ||
       (ti && (d->cache[2] || d->discard[2])) ||
       (fi && (d->cache[3] || d->discard[3])))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, d->dest_16 ? 1 : 2),
      /* f11 is the destination-cache hint, as in every other GPU1 form
       * sharing this f9/f10/f2/f11 header (iadd32's `16 + 16` witness
       * carries f11=1 with a live result; cbs, copies and funops write it
       * from dest_cache too). The three m5-xfb-alu8 selects store their
       * result directly and carry f11=0 whatever the selected-operand mix,
       * including the word-comparand register/zero shape the earlier
       * "mixed selection" reading of this field set to 1 -- the same
       * correction GPU0's f12 received. */
      KV(11, d->dest_cache),
      KV(23, 2), KV(24, d->cmp_16 ? 1 : 2),
      /* Native ushort equality sets f28 with its half comparator. */
      KV(28, half_regs ? d->cond >= AGX3_ISEL_ICOND_EQ : d->cmp_16),
      KV(38, d->cmp1_immediate ? 1 : 2),
      KV(39, (d->cmp1_immediate || d->cmp1_16 || half_regs) ? 1 : 2),
      KV(43, half_regs ? d->cond >= AGX3_ISEL_ICOND_EQ
                       : d->cmp1_sign_extend),
      KV(53, ti ? 1 : 2), KV(54, d->dest_16 ? 1 : 2),
      KV(67, fi ? 1 : 2), KV(68, d->dest_16 ? 1 : 2),
      KV(72, 3), KV(73, d->dest_16 ? 96 : 97),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   /* A GPU1 GPR destination is f2/kind2; f3/kind3 names a uniform. */
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(16, d->cmp0);
   ov[n++] = (struct agx3_field_kv)KV(d->cmp1_immediate ? 30 : 31, d->cmp1);
   ov[n++] = (struct agx3_field_kv)KV(ti ? 45 : 46, d->sel_true);
   ov[n++] = (struct agx3_field_kv)KV(fi ? 59 : 60, d->sel_false);
   ov[n++] = (struct agx3_field_kv)KV(82, d->cond);
   static const unsigned cache_fields[] = {25, 40, 55, 69};
   static const unsigned discard_fields[] = {27, 42, 57, 71};
   for (unsigned i = 0; i < 4; ++i) {
      ov[n++] = (struct agx3_field_kv)KV(cache_fields[i], d->cache[i]);
      ov[n++] = (struct agx3_field_kv)KV(discard_fields[i], d->discard[i]);
   }
   for (unsigned i = 0; i < 8; ++i)
      ov[n++] = (struct agx3_field_kv)KV(74 + i, (d->wait_mask >> i) & 1);
   const unsigned first = d->dest_16 ? 6 + (d->dest & 1) : 3;
   const unsigned last = d->dest_16 ? 8 : 4;
   for (unsigned ep = first; ep <= last; ++ep)
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C,
                            d->dest_16 ? AGX3_MN_iseli16 : AGX3_MN_iseli32, ep,
                            base, sizeof(base) / sizeof(base[0]), ov, n, out, len))
         return true;
   return false;
}

static bool
encode_agx3c(const struct agx3_isel_desc *d,
             uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   const bool cmp1_16 = d->cmp_16 != d->cmp_mixed_width;
   if (!(d->cmp_16 ? reg16_ok(d->cmp0) : reg32_ok(d->cmp0)) ||
       (d->cmp1_uniform ? ((d->cmp1 & 1) || d->cmp1 >= 512) :
        d->cmp1_immediate ? d->cmp1 > 255 :
        !(cmp1_16 ? reg16_ok(d->cmp1) : reg32_ok(d->cmp1))))
      return false;
   if (d->wait_mask >= (1u << 8))
      return false;
   for (unsigned i = 0; i < 4; ++i) {
      if (d->cache[i] && d->discard[i])
         return false;
   }

   const bool ti = d->true_kind == AGX3_ISEL_IMMEDIATE;
   const bool fi = d->false_kind == AGX3_ISEL_IMMEDIATE;
   if ((d->cmp1_immediate && (d->cache[1] || d->discard[1])) ||
       (ti && (d->cache[2] || d->discard[2])) ||
       (fi && (d->cache[3] || d->discard[3])))
      return false;

   const unsigned width = d->dest_16 ? 1 : 2;
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, width), KV(11, d->dest_cache),
      KV(23, 2), KV(24, d->cmp_16 ? 1 : 2),
      KV(38, d->cmp1_uniform ? 3 : d->cmp1_immediate ? 1 : 2),
      KV(39, (d->cmp1_immediate || cmp1_16) ? 1 : 2),
      KV(53, d->true_kind == AGX3_ISEL_UNIFORM ? 3 : ti ? 1 : 2), KV(54, width),
      KV(67, d->false_kind == AGX3_ISEL_UNIFORM ? 3 : fi ? 1 : 2), KV(68, width),
      KV(72, 3), KV(73, d->dest_16 ? 96 : 97),
   };
   struct agx3_field_kv ov[24];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(16, d->cmp0);
   ov[n++] = (struct agx3_field_kv)KV(d->cmp1_uniform ? 32 : d->cmp1_immediate ? 30 : 31, d->cmp1);
   ov[n++] = (struct agx3_field_kv)KV(82, d->cond);
   ov[n++] = (struct agx3_field_kv)KV(d->true_kind == AGX3_ISEL_UNIFORM ? 47 : ti ? 45 : 46, d->sel_true);
   ov[n++] = (struct agx3_field_kv)KV(d->false_kind == AGX3_ISEL_UNIFORM ? 61 : fi ? 59 : 60, d->sel_false);
   ov[n++] = (struct agx3_field_kv)KV(25, d->cache[0]);
   ov[n++] = (struct agx3_field_kv)KV(27, d->discard[0]);
   if (d->cmp_16 && d->cond >= AGX3_ISEL_ICOND_EQ)
      ov[n++] = (struct agx3_field_kv)KV(28, 1);
   if (!d->cmp1_immediate && !d->cmp1_uniform) {
      ov[n++] = (struct agx3_field_kv)KV(40, d->cache[1]);
      ov[n++] = (struct agx3_field_kv)KV(42, d->discard[1]);
      if (cmp1_16 && d->cond >= AGX3_ISEL_ICOND_EQ)
         ov[n++] = (struct agx3_field_kv)KV(43, 1);
   }
   if (!ti && d->true_kind != AGX3_ISEL_UNIFORM) {
      ov[n++] = (struct agx3_field_kv)KV(55, d->cache[2]);
      ov[n++] = (struct agx3_field_kv)KV(57, d->discard[2]);
   }
   if (!fi && d->false_kind != AGX3_ISEL_UNIFORM) {
      ov[n++] = (struct agx3_field_kv)KV(69, d->cache[3]);
      ov[n++] = (struct agx3_field_kv)KV(71, d->discard[3]);
   }
   for (unsigned i = 0; i < 8; ++i)
      ov[n++] = (struct agx3_field_kv)KV(74 + i, (d->wait_mask >> i) & 1);

   enum agx3_mnemonic mnemonic = d->dest_16 ? AGX3_MN_iseli16 : AGX3_MN_iseli32;
   for (unsigned ep = 0; ep < 16; ++ep) {
      if (agx3_bind_compose(AGX_ISA_ENCODING_AGX3C, mnemonic, ep, base,
                            sizeof(base) / sizeof(base[0]), ov, n, out, len))
         return true;
   }
   return false;
}
bool
agx3_isel_encode(enum agx_isa_encoding encoding,
                 const struct agx3_isel_desc *d,
                 uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   /* cmp1_16 and cmp1_sign_extend describe the G18G encoder
    * (agx3_isel_encode_g18g); this one has no such form. */
   if (d->cmp1_16 || d->cmp1_sign_extend)
      return false;

   if (encoding == AGX_ISA_ENCODING_AGX3C) {
      const bool tu = d->true_kind == AGX3_ISEL_UNIFORM;
      const bool fu = d->false_kind == AGX3_ISEL_UNIFORM;
      const bool any_uniform = d->cmp1_uniform || tu || fu;
      /* Native M5 ray/SVGF shaders use an aligned full32 uniform as the
       * right comparand and/or either selected value. A left uniform and
       * narrow/mixed-width uniform forms are not part of this admission. */
      if (!cond_ok(d->cond) || d->cmp0_uniform ||
          (any_uniform && (d->dest_16 || d->cmp_16 || d->cmp_mixed_width)) ||
          (d->cmp1_uniform && (d->cmp1_immediate || (d->cmp1 & 1) ||
                              d->cmp1 >= 512 || d->cache[1] || d->discard[1])) ||
          !(d->dest_16 ? reg16_ok(d->dest) : reg32_ok(d->dest)) ||
          (tu ? ((d->sel_true & 1) || d->sel_true >= 512 ||
                 d->cache[2] || d->discard[2]) :
                !operand_ok(d->true_kind, d->sel_true, d->dest_16)) ||
          (fu ? ((d->sel_false & 1) || d->sel_false >= 512 ||
                 d->cache[3] || d->discard[3]) :
                !operand_ok(d->false_kind, d->sel_false, d->dest_16)))
         return false;
      return encode_agx3c(d, out, len);
   }
   if (encoding != AGX_ISA_ENCODING_AGX3B || d->cmp_mixed_width)
      return false;

   /* The register comparands share a width; immediates carry no register. */
   if (!cond_ok(d->cond) ||
       (!d->cmp0_uniform &&
        !(d->cmp_16 ? reg16_ok(d->cmp0) : reg32_ok(d->cmp0))) ||
       (!d->cmp1_immediate && !d->cmp1_uniform &&
        !(d->cmp_16 ? reg16_ok(d->cmp1) : reg32_ok(d->cmp1))))
      return false;
   if (d->wait_mask >= (1u << 6))
      return false;

   if (d->dest_16 ? !reg16_ok(d->dest) : !reg32_ok(d->dest))
      return false;
   const bool uniform = d->cmp0_uniform || d->cmp1_uniform ||
                        d->true_kind == AGX3_ISEL_UNIFORM ||
                        d->false_kind == AGX3_ISEL_UNIFORM;
   if (uniform)
      return encode_explicit(d, out, len);

   if (!operand_ok(d->true_kind, d->sel_true, d->dest_16) ||
       !operand_ok(d->false_kind, d->sel_false, d->dest_16))
      return false;

   if (d->cmp1_immediate || d->cmp_16 || d->explicit_hints ||
       d->dest_cache)
      return encode_explicit(d, out, len);

   const struct agx3_isel_form *form =
      pick_form(d->dest_16, d->true_kind, d->false_kind);
   const uint8_t ep =
      (d->dest_16 && (d->dest & 1)) ? (uint8_t)(form->ep + 1) : form->ep;

   struct agx3_field_kv ov[12];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(3, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(35, d->cmp0);
   ov[n++] = (struct agx3_field_kv)KV(50, d->cmp1);
   ov[n++] = (struct agx3_field_kv)KV(106, d->cond);
   ov[n++] = (struct agx3_field_kv)KV(form->true_field, d->sel_true);
   ov[n++] = (struct agx3_field_kv)KV(form->false_field, d->sel_false);

   /* Waits ride on f98..f103 for slots 1..6. */
   for (unsigned i = 0; i < 6; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(98 + i, 1);
   }

   return agx3_bind_compose(encoding, AGX3_MN_isel, ep, form->base, form->nbase, ov,
                            n, out, len);
}

/* G18G (M6) compiles as agx3c and keeps the GPU1 integer select it was
 * qualified with (encode_gpu1): the widened narrow comparand (cmp1_16,
 * cmp1_sign_extend) and its own field and form choices. The uniform and cmp_mixed_width forms, which
 * that encoder never had, take agx3_isel_encode's agx3c forms. */
bool
agx3_isel_encode_g18g(enum agx_isa_encoding encoding,
                      const struct agx3_isel_desc *d,
                      uint8_t out[AGX3_ISEL_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || out == NULL || len == NULL)
      return false;

   const bool agx3c_form = d->cmp_mixed_width || d->cmp0_uniform ||
                           d->cmp1_uniform ||
                           d->true_kind == AGX3_ISEL_UNIFORM ||
                           d->false_kind == AGX3_ISEL_UNIFORM;
   const bool gpu1_form = d->cmp1_16 || d->cmp1_sign_extend;
   if (encoding != AGX_ISA_ENCODING_AGX3C || (agx3c_form && !gpu1_form))
      return agx3_isel_encode(encoding, d, out, len);
   if (agx3c_form)
      return false;

   /* The register comparands share a width; immediates carry no register. */
   if (!cond_ok(d->cond) ||
       !(d->cmp_16 ? reg16_ok(d->cmp0) : reg32_ok(d->cmp0)) ||
       (!d->cmp1_immediate &&
        !(d->cmp_16 ? reg16_ok(d->cmp1) : reg32_ok(d->cmp1))))
      return false;
   if (d->wait_mask >= (1u << 8))
      return false;

   if (d->dest_16 ? !reg16_ok(d->dest) : !reg32_ok(d->dest))
      return false;
   if (!operand_ok(d->true_kind, d->sel_true, d->dest_16) ||
       !operand_ok(d->false_kind, d->sel_false, d->dest_16))
      return false;

   return encode_gpu1(d, out, len);
}
