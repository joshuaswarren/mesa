/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_writeout_core.h"

#include <string.h>

#include "agx3_binder.h"
#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

bool
agx3_shift_add16_encode(enum agx_isa_encoding enc,
                        const struct agx3_shift_add_desc *d,
                        uint8_t out[AGX3_WRITEOUT_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL || d == NULL)
      return false;
   memset(out, 0, AGX3_WRITEOUT_MAX_LENGTH);
   *len = 0;

   /* AGX3c has no `iadd` form at all; the encoding is AGX3b-only. */
   if (enc != AGX_ISA_ENCODING_AGX3B)
      return false;
   if (d->dest >= 256 || d->src1 >= 256 || d->shift > 4)
      return false;
   if (d->src0_immediate ? d->src0 > 0xff : d->src0 >= 256)
      return false;
   if ((d->wait_mask & ~(uint16_t)0x3f) != 0)
      return false;

   /* 16-bit everywhere. Fresh live/dead shift witnesses pin f42 as the
    * shifted operand's last-read hint, independently of its width selector. */
   const struct agx3_field_kv base[] = {
      KV(9, 2), KV(10, 1), KV(24, 1), KV(38, 2), KV(39, 1),
   };

   struct agx3_field_kv ov[16];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(11, d->dest_hint ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(31, d->src1);
   ov[n++] = (struct agx3_field_kv)KV(42, d->src1_kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(54, d->shift);
   if (d->src0_immediate) {
      ov[n++] = (struct agx3_field_kv)KV(23, 1);
      ov[n++] = (struct agx3_field_kv)KV(15, d->src0);
   } else {
      ov[n++] = (struct agx3_field_kv)KV(23, 2);
      ov[n++] = (struct agx3_field_kv)KV(16, d->src0);
      ov[n++] = (struct agx3_field_kv)KV(27, 1);
      if (d->src0_kill)
         ov[n++] = (struct agx3_field_kv)KV(25, 1);
   }

   for (unsigned i = 0; i < 6; ++i)
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(45 + i, 1);

   if (agx3_bind_compose(enc, AGX3_MN_iadd, 0, base, 5, ov, n, out, len))
      return true;

   memset(out, 0, AGX3_WRITEOUT_MAX_LENGTH);
   *len = 0;
   return false;
}
