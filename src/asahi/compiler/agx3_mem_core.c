/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_mem_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

/* The register operands name halves 0..191 on AGX3b and the whole 256-half
 * file on AGX3c/AGX3d: metal-nt applegpu_g17g and applegpu_g18g, under
 * register pressure, store a vec3 through a base in half 214 (stdev f31=214),
 * index stores with f16=216 and load into half 194 (lddev f2=194)
 * (g18ref-out/probe/m6-high-base3 hb3_96, m6-high-base4 hb4_88_10). */
static bool
common_ok(enum agx_isa_encoding enc, const struct agx3_mem_desc *d)
{
   const bool wide = agx_encoding_is_g17_class(enc);
   const unsigned base_max = wide ? 252 : 188, idx_max = wide ? 254 : 191;

   if (d == NULL || d->data >= 256 || d->mask < 1 || d->mask > 15 ||
       d->shift > 4)
      return false;
   if (d->width != AGX3_MEM_W8 && d->width != AGX3_MEM_W16 &&
       d->width != AGX3_MEM_W32)
      return false;
   if (d->base_is_uniform) {
      if ((d->base & 3) != 0 || d->base > 508)
         return false;
   } else if ((d->base & 1) != 0 || d->base > base_max) {
      return false;
   }
   if (d->idx_kind == AGX3_MEM_IDX_REG &&
       ((d->idx_reg & 1) != 0 || d->idx_reg > idx_max))
      return false;
   if (d->idx_kind == AGX3_MEM_IDX_IMM && d->idx_scaled) {
      if (d->elem_idx > 0xffff)
         return false;
   } else if (d->byte_off < -32768 || d->byte_off > 32767) {
      return false;
   }
   return true;
}

/* Build the operand overrides shared by loads and stores. `off_field`,
 * `shift_field` and `mask_field` differ between the two. Returns the number of
 * overrides written. */
static unsigned
common_overrides(const struct agx3_mem_desc *d, uint16_t off_field,
                 uint16_t shift_field, uint16_t mask_field,
                 struct agx3_field_kv *ov)
{
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->data);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(mask_field, d->mask);

   /* Index. */
   if (d->idx_kind == AGX3_MEM_IDX_IMM) {
      ov[n++] = (struct agx3_field_kv)KV(23, 1);
      ov[n++] = (struct agx3_field_kv)KV(24, 1);
      if (d->idx_scaled) {
         ov[n++] = (struct agx3_field_kv)KV(15, d->elem_idx);
         ov[n++] = (struct agx3_field_kv)KV(shift_field, d->shift);
      } else {
         ov[n++] =
            (struct agx3_field_kv)KV(off_field, (uint16_t)d->byte_off);
         ov[n++] = (struct agx3_field_kv)KV(shift_field, 0);
      }
   } else {
      ov[n++] = (struct agx3_field_kv)KV(23, 2);
      ov[n++] = (struct agx3_field_kv)KV(24, 2);
      ov[n++] = (struct agx3_field_kv)KV(16, d->idx_reg);
      ov[n++] = (struct agx3_field_kv)KV(27, d->idx_kill ? 1 : 0);
      ov[n++] = (struct agx3_field_kv)KV(28, d->idx_signed ? 1 : 0);
      ov[n++] = (struct agx3_field_kv)KV(shift_field, d->shift);
      /* The byte offset is added after the index is scaled. */
      ov[n++] = (struct agx3_field_kv)KV(off_field, (uint16_t)d->byte_off);
   }

   /* Base. */
   if (d->base_is_uniform) {
      ov[n++] = (struct agx3_field_kv)KV(38, 3);
      ov[n++] = (struct agx3_field_kv)KV(32, d->base);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(38, 2);
      ov[n++] = (struct agx3_field_kv)KV(31, d->base);
      ov[n++] = (struct agx3_field_kv)KV(42, d->base_kill ? 1 : 0);
   }

   /* Wait mask -> f60+i. */
   for (unsigned i = 0; i < 8; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(60 + i, 1);
   }
   return n;
}

static void
mem_baseline(enum agx_isa_encoding enc, bool is_load,
             struct agx3_field_kv *base, unsigned *n)
{
   unsigned i = 0;
   base[i++] = (struct agx3_field_kv)KV(9, 2);
   base[i++] = (struct agx3_field_kv)KV(39, 3);
   base[i++] = (struct agx3_field_kv)KV(54, 1);
   if (agx_encoding_is_g17_class(enc)) {
      base[i++] = (struct agx3_field_kv)KV(45, 15);
      base[i++] = (struct agx3_field_kv)KV(53, 1);
      base[i++] = (struct agx3_field_kv)KV(59, is_load ? 104 : 187);
   }
   *n = i;
}

/*
 * The field vector of one LDDEV. `ep` is an agx3b/agx3c encoding point; it
 * is only range-checked here. Returns false for a descriptor it cannot hold.
 */
static bool
mem_load_fields(enum agx_isa_encoding enc, const struct agx3_mem_desc *d,
                uint8_t ep, struct agx3_field_kv *base, unsigned *nbase,
                struct agx3_field_kv *ov, unsigned *n)
{
   if (!common_ok(enc, d) || d->slot < 1 ||
       d->slot > (agx_encoding_is_g17_class(enc) ? 8 : 6) ||
       (enc == AGX_ISA_ENCODING_AGX3B ? ep != 0 : ep > 5))
      return false;

   mem_baseline(enc, true, base, nbase);
   if (d->agx3b_last_load && enc != AGX_ISA_ENCODING_AGX3B)
      return false;
   if ((d->last_load || d->agx3b_last_load) && enc == AGX_ISA_ENCODING_AGX3B)
      base[(*nbase)++] = (struct agx3_field_kv)KV(79, 2);

   *n = common_overrides(d, /*off*/ 76, /*shift*/ 75, /*mask*/ 73, ov);
   ov[(*n)++] = (struct agx3_field_kv)KV(12, d->slot);
   /* Only ep2/ep5 carry f82; the other forms fail the compose and the
    * caller's form order lands on ep2 like Apple's emitter. */
   if (d->last_load && agx_encoding_is_g17_class(enc))
      ov[(*n)++] = (struct agx3_field_kv)KV(82, 2);
   ov[(*n)++] = (struct agx3_field_kv)KV(77, d->coherent ? 1 : 0);
   return true;
}

bool
agx3_mem_encode_load_form(enum agx_isa_encoding enc,
                          const struct agx3_mem_desc *d, uint8_t ep,
                          uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len)
{
   /* agx3b/agx3c encoding points do not name agx3d forms. */
   if (enc == AGX_ISA_ENCODING_AGX3D)
      return false;

   struct agx3_field_kv base[7];
   struct agx3_field_kv ov[32];
   unsigned nbase, n;
   if (!mem_load_fields(enc, d, ep, base, &nbase, ov, &n))
      return false;

   return agx3_bind_compose(enc, AGX3_MN_lddev, ep, base, nbase, ov, n, out, len);
}

/*
 * G18G (agx3d) LDDEV: the agx3c field vector plus f87 = 2, in Apple's form
 * order. Grounded on the applegpu_g18g builds of the j813-linux
 * gpu/m5-*-reference corpus: the 436 LDDEVs aligned with a G17G one differ
 * only in slot, wait and register choices, except that the 409 main-program
 * data loads (uniform or GPR base, immediate or register index) all carry
 * f87 = 2, which no G17G LDDEV has; the glxgears vertex fetch
 * (49 07 00 80 08 10 80 60 08 00) is one of them. f87 is 0 on the loads of
 * argument-buffer and descriptor words in constant programs; what it selects
 * is not decoded, so this encoder is for data loads only.
 *
 * agx3d's LDDEV wait field is one slot code, not a mask (composing slots 1
 * and 3 gives the code of neither), so at most one wait slot is admitted.
 * Coherent (f77) loads have no G18G witness and are refused.
 */
static bool
agx3d_mem_encode_load(const struct agx3_mem_desc *d,
                      uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || d->coherent || (d->wait_mask & (d->wait_mask - 1)))
      return false;

   struct agx3_field_kv base[7];
   struct agx3_field_kv ov[33];
   unsigned nbase, n;
   if (!mem_load_fields(AGX_ISA_ENCODING_AGX3D, d, 0, base, &nbase, ov, &n))
      return false;
   ov[n++] = (struct agx3_field_kv)KV(87, 2);

   return agx3_bind_compose(AGX_ISA_ENCODING_AGX3D, AGX3_MN_lddev,
                            AGX3_BIND_APPLE_ORDER, base, nbase, ov, n, out,
                            len);
}

bool
agx3_mem_encode_load(enum agx_isa_encoding enc, const struct agx3_mem_desc *d,
                     uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len)
{
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      static const uint8_t forms[] = {0, 1, 3, 4, 2, 5};
      for (unsigned i = 0; i < sizeof(forms); ++i) {
         if (agx3_mem_encode_load_form(enc, d, forms[i], out, len))
            return true;
      }
      return false;
   }
   if (enc == AGX_ISA_ENCODING_AGX3D)
      return agx3d_mem_encode_load(d, out, len);
   return agx3_mem_encode_load_form(enc, d, 0, out, len);
}

bool
agx3_mem_encode_store(enum agx_isa_encoding enc, const struct agx3_mem_desc *d,
                      uint8_t out[AGX3_MEM_MAX_LENGTH], uint8_t *len)
{
   if (!common_ok(enc, d) || d->last_load || d->agx3b_last_load)
      return false;

   /* G18G (agx3d) STDEV takes the agx3c field vector: the 248 aligned
    * applegpu_g17g/applegpu_g18g stores of the m5 reference corpus differ
    * only in their wait slot. Its wait field is one slot code, and no G18G
    * store witnesses the coherent bit. */
   if (enc == AGX_ISA_ENCODING_AGX3D &&
       (d->coherent || (d->wait_mask & (d->wait_mask - 1))))
      return false;

   struct agx3_field_kv base[6];
   unsigned nbase;
   mem_baseline(enc, false, base, &nbase);

   struct agx3_field_kv ov[32];
   unsigned n = common_overrides(d, /*off*/ 74, /*shift*/ 73, /*mask*/ 71, ov);
   ov[n++] = (struct agx3_field_kv)KV(75, d->coherent ? 1 : 0);

   if (enc == AGX_ISA_ENCODING_AGX3C) {
      static const uint8_t forms[] = {0, 1, 3, 4, 2, 5};
      for (unsigned i = 0; i < sizeof(forms); ++i) {
         if (agx3_bind_compose(enc, AGX3_MN_stdev, forms[i], base, nbase,
                               ov, n, out, len))
            return true;
      }
      return false;
   }
   return agx3_bind_compose(enc, AGX3_MN_stdev, agx3_bind_ep(enc, 0), base, nbase,
                            ov, n, out, len);
}
