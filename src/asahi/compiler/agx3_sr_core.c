/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_sr_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

/*
 * Supported AGX3B/AGX3C (SR, width) pairs:
 *  - 16-bit compute/subgroup: 6 simdgroup_in_tg, 7 lane_in_simd,
 *    9..11 thread_in_tg.xyz, 13 linear_thread_in_tg, 17..19 threads_per_tg.xyz,
 *    20..22 dispatch_threads_per_tg.xyz, 34 active_thread_index (fragment
 *    colour-only reads also set f48 via fragment_active_index), 36 (seen in
 *    device-atomic kernels; attribution unknown).
 *  - 16-bit fragment: 2 target-sample mask (inferred from 1x coverage-word
 *    arithmetic), 39 incoming coverage (bits 4..7), 41 back-facing predicate
 *    (0 = front-facing).
 *  - 32-bit compute/vertex/tess: 23..25 thread_in_grid.xyz, 26..28 tg_in_grid.xyz,
 *    46 instance_id (0xd8), 47 patch_id (0xd9), 48..49 tess_coord.uv (0xda/0xdb),
 *    51 vertex_id (0xdd).
 */
static bool
agx3_sr_witnessed(enum agx_isa_encoding enc, uint8_t sr, enum agx3_sr_width width)
{
   if (width == AGX3_SR_W16) {
      switch (sr) {
      case 6: case 7: case 9: case 10: case 11: case 13:
      case 17: case 18: case 19: case 20: case 21: case 22:
      case 34: case 36: case 59:
      case 2: case 39: case 41: case 12:
         return true;
      case 63:
         return enc == AGX_ISA_ENCODING_AGX3C;
      case 65:
         return agx_isa_is_agx3(enc);
      default:
         return false;
      }
   }
   if (width == AGX3_SR_W32 && sr == 64)
      return agx_isa_is_agx3(enc);
   if (width == AGX3_SR_W32 && sr == 62)
      return enc == AGX_ISA_ENCODING_AGX3C;
   if (width == AGX3_SR_W32)
      return (sr >= 23 && sr <= 28) || (sr >= 46 && sr <= 49) || sr == 51;
   return false;
}

bool
agx3_sr_encode(enum agx_isa_encoding enc, const struct agx3_sr_desc *d,
               uint8_t out[AGX3_SR_MAX_LENGTH], uint8_t *len)
{
   if (d == NULL || d->dest >= 256)
      return false;
   if (!agx3_sr_witnessed(enc, d->sr, d->width))
      return false;
   /* 32-bit reads write a register pair; every witness is 2-aligned. */
   if (d->width == AGX3_SR_W32 && (d->dest & 1) != 0)
      return false;

   /*
    * G18G (agx3d) reads special registers with the agx3c `rld` field vector:
    * the f28/f29 pair and every operand field are the same in all 374
    * aligned applegpu_g17g/applegpu_g18g rld pairs of the j813-linux
    * gpu/m5-*-reference corpus (only the destination, slot and wait choices
    * differ), and the G18G builds read SR 2 (W16), 23/24/25 and 51 (W32)
    * for the same builtins, and SR 39 (W16) in the native msaa4 fragment
    * (21 0a 42 00 00 00). Only those numbers are admitted for agx3d.
    */
   const bool g17 = agx_encoding_is_g17_class(enc);
   if (enc != AGX_ISA_ENCODING_AGX3B && !g17)
      return false;
   if (enc == AGX_ISA_ENCODING_AGX3D &&
       !(d->width == AGX3_SR_W16 ? d->sr == 2 || d->sr == 39
                                 : (d->sr >= 23 && d->sr <= 25) || d->sr == 51))
      return false;

   const unsigned nslots = g17 ? 8 : 6;
   if (d->slot < 1 || d->slot > nslots || (d->wait_mask >> nslots) != 0)
      return false;

   /* The entry marker is witnessed on G17P/G17G's W16 SR39 coverage read:
    * `[[sample_mask]]` and centroid inputs, and a multisample fragment with no
    * other input, each open with `rld ep1 ... f48=1`. */
   /* Apple's throughput fragment (metal-nt applegpu_g17p, fragment.fields
    * offset 64) also opens with a marked 32-bit SR24 read,
    * 1ca1110601000000, so the pixel-coordinate pair qualifies too. */
   /* G17G and G18G open a [[sample_mask]] fragment the same way, with the
    * marked W16 SR39 read (m6-sample-mask-in: 248201002000b002 on
    * applegpu_g17g, 2100420000080100 on applegpu_g18g). */
   const bool coverage_entry = d->sr == 39 && d->width == AGX3_SR_W16 &&
      agx_isa_is_agx3(enc);
   const bool pixel_entry = enc == AGX_ISA_ENCODING_AGX3B &&
      (d->sr == 23 || d->sr == 24) && d->width == AGX3_SR_W32;
   if (d->first && !coverage_entry && !pixel_entry)
      return false;

   struct agx3_field_kv base[5];
   unsigned nbase = 0;
   base[nbase++] = (struct agx3_field_kv)KV(9, 2);
   base[nbase++] = (struct agx3_field_kv)KV(38, 1);
   base[nbase++] = (struct agx3_field_kv)KV(23, 7);
   if (g17) {
      base[nbase++] = (struct agx3_field_kv)KV(28, 4);
      base[nbase++] = (struct agx3_field_kv)KV(29, 144);
   }

   struct agx3_field_kv ov[14];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(2, d->dest);
   ov[n++] = (struct agx3_field_kv)KV(10, d->width);
   ov[n++] = (struct agx3_field_kv)KV(24, d->width);
   ov[n++] = (struct agx3_field_kv)KV(21, d->sr);
   ov[n++] = (struct agx3_field_kv)KV(12, d->slot);
   if (d->fragment_active_index) {
      if (d->sr != 34 || d->width != AGX3_SR_W16 ||
          enc == AGX_ISA_ENCODING_AGX3D)
         return false;
      ov[n++] = (struct agx3_field_kv)KV(48, 1);
   }

   /* Wait mask -> f30+i. */
   for (unsigned i = 0; i < 8; ++i) {
      if (d->wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }

   if (d->first)
      ov[n++] = (struct agx3_field_kv)KV(48, 1);

   /* agx3d picks its own rld form, smallest first, as Apple's encoder does:
    * 21 05 5d 00 00 00 is the G18G `rld r2, SR 51` of every vertex shader
    * (applegpu_g18g builds of the oracle and glxgears probes). */
   if (enc == AGX_ISA_ENCODING_AGX3D)
      return !d->first &&
             agx3_bind_compose(enc, AGX3_MN_rld, AGX3_BIND_APPLE_ORDER, base, nbase,
                               ov, n, out, len);

   /* The compact form's selector requires every wait bit zero, a narrow
    * destination field and no entry marker; the composer refuses it otherwise
    * and the long form carries the same operands. */
   if (!d->first &&
       agx3_bind_compose(enc, AGX3_MN_rld, 0, base, nbase, ov, n, out, len))
      return true;
   return agx3_bind_compose(enc, AGX3_MN_rld, 1, base, nbase, ov, n, out, len);
}
