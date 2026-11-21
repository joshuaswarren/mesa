/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_zs_core.h"

#include "agx3_binder.h"

#include <string.h>

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}
#define NELEM(a) (sizeof(a) / sizeof((a)[0]))

static bool
is_b(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B;
}

static bool
enc_ok(enum agx_isa_encoding e)
{
   return e == AGX_ISA_ENCODING_AGX3B || e == AGX_ISA_ENCODING_AGX3C;
}

static unsigned
nslots(enum agx_isa_encoding e)
{
   return is_b(e) ? 6 : 8;
}

static bool
desc_ok(enum agx_isa_encoding e, const struct agx3_zs_desc *d)
{
   if (d == NULL || !enc_ok(e))
      return false;

   /* The staged word is a 32-bit register pair in every witness. */
   if ((d->value & 1) || d->value >= 256)
      return false;

   const unsigned slots = nslots(e);
   return d->slot >= 1 && d->slot <= slots &&
          (d->wait_mask & ~(uint16_t)((1u << slots) - 1)) == 0;
}

/* The wait bits are consecutive fields from f17, one per slot. */
static unsigned
append_waits(enum agx_isa_encoding e, uint16_t wait_mask,
             struct agx3_field_kv *ov, unsigned n)
{
   for (unsigned i = 0; i < nslots(e); ++i) {
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(17 + i, 1);
   }
   return n;
}

static unsigned
append_common(enum agx_isa_encoding e, const struct agx3_zs_desc *d,
              struct agx3_field_kv *ov, unsigned n)
{
   ov[n++] = (struct agx3_field_kv)KV(5, d->value);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   ov[n++] = (struct agx3_field_kv)KV(13, d->value_kill ? 1 : 0);
   return append_waits(e, d->wait_mask, ov, n);
}

static bool
finish(enum agx_isa_encoding e, enum agx3_mnemonic mnemonic,
       const struct agx3_field_kv *base_c, unsigned nbase_c,
       const struct agx3_field_kv *ov, unsigned n, uint8_t *out, uint8_t *len)
{
   /* f9=5 selects the 13-bit operand field on both encodings; AGX3c adds the
    * width, form and sub-opcode constants its longer form carries. */
   const struct agx3_field_kv base_b[] = {KV(9, 5)};

   if (!agx3_bind_compose(e, mnemonic, 0, is_b(e) ? base_b : base_c,
                          is_b(e) ? NELEM(base_b) : nbase_c, ov, n, out,
                          len)) {
      memset(out, 0, AGX3_ZS_MAX_LENGTH);
      *len = 0;
      return false;
   }
   return true;
}

bool
agx3_zs_encode_sample_mask(enum agx_isa_encoding enc,
                           const struct agx3_zs_desc *d,
                           uint8_t out[AGX3_ZS_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_ZS_MAX_LENGTH);
   *len = 0;

   if (!desc_ok(enc, d))
      return false;

   const struct agx3_field_kv base_c[] = {KV(9, 5), KV(10, 2), KV(14, 2),
                                          KV(16, 112)};

   struct agx3_field_kv ov[12];
   const unsigned n = append_common(enc, d, ov, 0);

   return finish(enc, AGX3_MN_movmsk, base_c, NELEM(base_c), ov, n, out, len);
}

bool
agx3_zs_encode_depth_stencil(enum agx_isa_encoding enc,
                             const struct agx3_zs_desc *d, bool depth,
                             bool stencil, uint8_t out[AGX3_ZS_MAX_LENGTH],
                             uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_ZS_MAX_LENGTH);
   *len = 0;

   if (!desc_ok(enc, d) || (!depth && !stencil))
      return false;

   const struct agx3_field_kv base_c[] = {KV(9, 5), KV(10, 2), KV(14, 2),
                                          KV(16, 37)};

   struct agx3_field_kv ov[15];
   unsigned n = append_common(enc, d, ov, 0);
   ov[n++] = (struct agx3_field_kv)KV(25, depth ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(26, stencil ? 1 : 0);

   /* AGX3c repeats the enable pair as a mode selector; AGX3b's form has no
    * such field and the composer rejects it there. */
   if (!is_b(enc)) {
      ov[n++] = (struct agx3_field_kv)KV(
         35, (uint64_t)((depth ? 1u : 0u) | (stencil ? 2u : 0u)));
   }

   return finish(enc, AGX3_MN_depthf, base_c, NELEM(base_c), ov, n, out, len);
}
