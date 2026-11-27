/* SPDX-License-Identifier: MIT */
#include "agx3_rt_m5_core.h"
#include "agx3_binder.h"
#include <string.h>

#define KV(f, v) {(f), (v)}

bool
agx3_rt_m5_encode_enter_query(enum agx_isa_encoding encoding,
                              uint16_t destination, uint16_t allocation_index,
                              uint8_t out[16], uint8_t *length)
{
   if (!out || !length)
      return false;
   memset(out, 0, 16);
   *length = 0;
   if (encoding != AGX_ISA_ENCODING_AGX3C || destination > 252 ||
       (destination & 1) || allocation_index < 4 ||
       allocation_index >= 4 + AGX3_RT_M5_MAX_LIVE_QUERIES)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(2, destination), KV(9, 2),   KV(10, 2),
      KV(14, 3),          KV(15, 90), KV(24, allocation_index),
   };
   return agx3_bind_compose(encoding, AGX3_MN_incStk, 0, fields,
                            sizeof(fields) / sizeof(fields[0]), NULL, 0, out,
                            length);
}

bool
agx3_rt_m5_encode_intersect(enum agx_isa_encoding encoding,
                           const struct agx3_rt_m5_intersect_desc *d,
                           uint8_t out[16], uint8_t *length)
{
   if (!out || !length)
      return false;
   memset(out, 0, 16);
   *length = 0;
   if (encoding != AGX_ISA_ENCODING_AGX3C || !d ||
       d->address_register > 252 || (d->address_register & 1) ||
       d->bvh_base > (d->bvh_uniform ? 508 : 252) ||
       (d->bvh_base & 1) || (d->bvh_uniform && d->bvh_kill))
      return false;

   const struct agx3_field_kv base[] = {
      KV(9, 5), KV(10, 2), KV(12, 8), KV(13, 1), KV(14, 2),
      KV(24, 2), KV(25, 3), KV(28, 1), KV(29, 92), KV(30, 1),
      KV(49, 32), KV(55, 3), KV(56, 3), KV(70, 2), KV(85, 1), KV(99, 1),
   };
   const struct agx3_field_kv operands[] = {
      KV(17, d->address_register),
      KV(d->bvh_uniform ? 63 : 62, d->bvh_base),
      KV(69, d->bvh_uniform ? 3 : 2), KV(73, d->bvh_kill),
   };
   return agx3_bind_compose(encoding, AGX3_MN_intersect_ray, 0, base,
                           sizeof(base) / sizeof(base[0]), operands,
                           sizeof(operands) / sizeof(operands[0]), out, length);
}

bool
agx3_rt_m5_encode_continue(enum agx_isa_encoding encoding,
                          uint8_t out[16], uint8_t *length)
{
   if (!out || !length)
      return false;
   memset(out, 0, 16);
   *length = 0;
   if (encoding != AGX_ISA_ENCODING_AGX3C)
      return false;
   const struct agx3_field_kv fields[] = {
      KV(5, 8), KV(9, 5), KV(10, 2), KV(12, 8), KV(13, 1),
      KV(14, 2), KV(15, 34), KV(16, 1),
   };
   return agx3_bind_compose(encoding, AGX3_MN_continue_ray, 0, fields,
                           sizeof(fields) / sizeof(fields[0]), NULL, 0,
                           out, length);
}
