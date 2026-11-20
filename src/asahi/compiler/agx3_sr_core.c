/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_sr_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

/*
 * Witnessed (SR number, width) pairs: one Apple-compiled kernel per MSL
 * builtin, identical numbers on agx3b and
 * agx3c. 16-bit: 6 simdgroup_index_in_threadgroup, 7 thread_index_in_simdgroup,
 * 9/10/11 thread_position_in_threadgroup.xyz, 13 thread_index_in_threadgroup,
 * 17/18/19 threads_per_threadgroup.xyz, 20/21/22
 * dispatch_threads_per_threadgroup.xyz, 34 active thread index in the
 * subgroup, 36 (witnessed in Apple device-atomic kernels, attribution
 * UNKNOWN). 32-bit: 23/24/25 thread_position_in_grid.xyz,
 * 26/27/28 threadgroup_position_in_grid.xyz.
 *
 * Two 32-bit numbers are hardware-vertex invocation inputs. The composer's
 * anonymous f21 values 51 and 46 encode hardware SR bytes 0xdd vertex ID and
 * 0xd8 instance ID respectively. They are isolated by otherwise identical
 * G17P vertex shaders and compile to `0cdd1006` / `0cd81006` for destination
 * r0.
 *
 * Three more feed the post-tessellation vertex program of a Metal
 * tessellation pipeline, between those two: 47 (byte 0xd9) is [[patch_id]],
 * and 48 and 49 (0xda, 0xdb) are the two [[position_in_patch]] coordinates as
 * 32-bit floats, for triangle and quad patches alike (a triangle's third
 * barycentric is computed as 1 - u - v). metal-nt -arch applegpu_g17p reads
 * them as `0cd91006`, `2cda1046` and `1cdb1066`.
 *
 * Three more 16-bit numbers are fragment-only (`metal-nt -arch
 * applegpu_g17p`):
 *
 *  - 39 is the *incoming* coverage word. PROVEN: a fragment shader that reads
 *    `uint m [[sample_mask]]` compiles to exactly `rld h0 <- SR 39 (W16)`,
 *    `h0 >>= 4`, `int2flt`, so the incoming mask sits in bits 4..7.
 *  - 2 is read by every 1x fragment shader that writes a sample mask or a
 *    depth and by no colour-only shader; its value is combined into the staged
 *    coverage word as N | (T << 4) | (T << 8). Reproducing Apple's own
 *    kill-all constant from that formula requires T = 0xF0, i.e. a four-bit
 *    mask in bits 4..7 exactly like SR 39. Presence and use are PROVEN;
 *    "the per-invocation target-sample mask" is INFERRED from that arithmetic.
 *  - 41 is the back-facing predicate. A `[[front_facing]] ? 17 : 3` fragment
 *    reads SR41 and selects 17 when that value equals zero, directly pinning
 *    both the register identity and its polarity.
 *
 * SR 34 is the active thread index in the subgroup. Apple's G17P compiler reads
 * it for simd_is_first() in compute, vertex and fragment shaders as
 * `rld SR34 == 0`, and compiles simd_broadcast_first(x) to
 * `simdiballot(SR34 == 0)`, find-first-set, `simdbcast`. A fragment read
 * whose result only reaches the colour output also sets f48; the same read
 * feeding a derivative or a quad reduction, or inside a helper-invocation
 * guard, leaves it clear (agx3_sr_desc::fragment_active_index).
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
         return enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C;
      default:
         return false;
      }
   }
   if (width == AGX3_SR_W32 && sr == 64)
      return enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C;
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
      (enc == AGX_ISA_ENCODING_AGX3B || enc == AGX_ISA_ENCODING_AGX3C);
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
