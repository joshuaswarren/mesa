/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_copy_core.h"

#include "agx3_binder.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}
#define AGX3_COPY_ARRAY_SIZE(a) (sizeof(a) / sizeof((a)[0]))

/*
 * Baselines decoded from Apple's corpus vectors:
 *
 *   agx3b  copy.ep0  f9=2 f10=2 f23=3 f24=2, f2=dest, f17=uniform  (4 B)
 *   agx3c  copy.ep0  the same plus the encoding's f28=3 f29=35 pair (4 B)
 *
 * f9/f10 are the destination's kind (2 = register) and width (2 = 32-bit);
 * f23/f24 are source slot A's kind (3 = uniform) and width.
 */
static const struct agx3_field_kv agx3b_copy_baseline[] = {
   KV(9, 2),
   KV(10, 2),
   KV(23, 3),
   KV(24, 2),
};

static const struct agx3_field_kv agx3c_copy_baseline[] = {
   KV(9, 2),
   KV(10, 2),
   KV(23, 3),
   KV(24, 2),
   KV(28, 3),
   KV(29, 35),
};

/* Native half copies use the same independent destination cache, source
 * cache/discard and scoreboard fields as other copy operands. In particular,
 * a zero wait mask is not evidence that its source dies at this instruction. */
bool
agx3_copy_encode_reg16(enum agx_isa_encoding encoding, uint16_t dest_half,
                       uint16_t src_half, bool dest_cache, bool src_cache,
                       bool src_discard, uint16_t wait_mask,
                       uint8_t out[AGX3_COPY_MAX_LENGTH], uint8_t *len)
{
   const bool agx3c = encoding == AGX_ISA_ENCODING_AGX3C;
   const unsigned slots = agx3c ? 8 : 6;
   if (out == NULL || len == NULL ||
       (encoding != AGX_ISA_ENCODING_AGX3B && !agx3c) ||
       dest_half >= (agx3c ? 256 : AGX3_COPY_REG16_DEST_HALVES) || src_half >= 256 ||
       wait_mask >= (1u << slots) || (src_cache && src_discard))
      return false;

   const struct agx3_field_kv baseline_b[] = {
      KV(9, 2), KV(10, 1), KV(23, 2), KV(24, 1),
   };
   const struct agx3_field_kv baseline_c[] = {
      KV(9, 2), KV(10, 1), KV(23, 2), KV(24, 1), KV(28, 3), KV(29, 35),
   };
   struct agx3_field_kv overrides[13] = {
      KV(2, dest_half), KV(16, src_half), KV(11, dest_cache),
      KV(25, src_cache), KV(27, src_discard),
   };
   unsigned n = 5;
   for (unsigned i = 0; i < slots; ++i) {
      if (wait_mask & (1u << i))
         overrides[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }
   for (int ep = dest_half >= AGX3_COPY_REG16_DEST_HALVES ? 1 : 0;
        ep <= (agx3c ? 1 : 0); ++ep) {
      if (agx3_bind_compose(encoding, AGX3_MN_copy, ep,
                            agx3c ? baseline_c : baseline_b,
                            agx3c ? AGX3_COPY_ARRAY_SIZE(baseline_c)
                                  : AGX3_COPY_ARRAY_SIZE(baseline_b),
                            overrides, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_copy_encode_reg32(enum agx_isa_encoding encoding, uint16_t dest_half,
                       uint16_t src_half, bool dest_cache, bool src_cache,
                       bool src_discard, uint16_t wait_mask,
                       uint8_t out[AGX3_COPY_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL || encoding != AGX_ISA_ENCODING_AGX3B ||
       dest_half >= 128 || src_half >= 256 || ((dest_half | src_half) & 1) ||
       wait_mask >= (1u << 6) || (wait_mask & (wait_mask - 1)) ||
       (src_cache && src_discard))
      return false;

   const struct agx3_field_kv baseline[] = {
      KV(9, 2),
      KV(10, 2),
      KV(23, 2),
      KV(24, 2),
   };
   struct agx3_field_kv overrides[11] = {
      KV(2, dest_half),  KV(16, src_half),    KV(11, dest_cache),
      KV(25, src_cache), KV(27, src_discard),
   };
   unsigned n = 5;
   for (unsigned i = 0; i < 6; ++i) {
      if (wait_mask & (1u << i))
         overrides[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }
   return agx3_bind_compose(encoding, AGX3_MN_copy, 0, baseline,
                            AGX3_COPY_ARRAY_SIZE(baseline), overrides, n,
                            out, len);
}

bool
agx3_copy_encode_reg_to_uniform(enum agx_isa_encoding encoding,
                                uint16_t uniform_half, uint16_t src_half,
                                bool kill, uint16_t wait_mask,
                                uint8_t out[AGX3_COPY_MAX_LENGTH],
                                uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;

   if (encoding == AGX_ISA_ENCODING_AGX3C) {
      if (wait_mask != 0 || (uniform_half & 1) || uniform_half + 2 > 512 ||
          (src_half & 1) || src_half + 2 > 256)
         return false;
      /* Same M5 COPY uniform destination as the captured IOTO publisher.
       * Retain its exact domain tags and caller-owned last-use hint. */
      const struct agx3_field_kv fields[] = {
         KV(3, uniform_half), KV(9, 3), KV(10, 2), KV(16, src_half),
         KV(23, 2), KV(24, 2), KV(27, kill), KV(28, 1), KV(29, 35),
      };
      for (unsigned ep = 0; ep <= 1; ++ep) {
         if (agx3_bind_compose(encoding, AGX3_MN_copy, ep, NULL, 0, fields,
                              AGX3_COPY_ARRAY_SIZE(fields), out, len))
            return true;
      }
      return false;
   }
   /* These forms carry no wait of their own; callers retire producers
    * first. G18G's waiting form is agx3_copy_encode_reg_to_uniform_g18g(). */
   if (encoding != AGX_ISA_ENCODING_AGX3B || wait_mask != 0)
      return false;

   if ((uniform_half & 1) || uniform_half + 2 > 512 || (src_half & 1) ||
       src_half + 2 > 256)
      return false;

   /* Both forms below are written byte by byte rather than through the table
    * composer. In composer terms the instruction is f9=3, f10=2, f23=2,
    * f24=2 with f3 = uniform half, f16 = source half and f27 = kill.
    */
   if (uniform_half < 128) {
      /* Grounded COPY ep0 field deposits. This is the same compact form used
       * by the Ioto load packer, except f27 is the caller's kill bit instead
       * of being forced. Avoid the table-driven composer in per-draw PBE
       * programs, where this exact instruction is emitted thousands of times.
       */
      out[0] = 0x0b | ((uniform_half & 0x1e) << 3);
      out[1] = src_half & 0x7f;
      out[2] = 0x01 | ((uniform_half & 0x60) << 1) | (kill ? 0x08 : 0);
      out[3] = 0x04 | (uniform_half & 1) | ((src_half & 0x80) >> 3);
      *len = 4;
      return true;
   }

   /* f39=1 is present in Apple's extended constant-program COPY record.  The
    * deposits below are the complete aligned f3/f16 domain of that ten-byte
    * ep1 and are exhaustively compared with the table composer by the copy
    * core test.  This form is used twice by every compact graphics launcher;
    * rescanning the generic 160-field table there dominated draw recording. */
   out[0] = 0x0b | ((uniform_half & 0x1e) << 3);
   out[1] = 0x01 | (src_half & 0x7e);
   out[2] = 0x07 | ((uniform_half & 0x60) << 1) | (kill ? 0x08 : 0);
   out[3] = 0x00;
   out[4] = 0x40 | ((uniform_half & 0x01) << 2) |
            ((src_half & 0x01) << 3) |
            ((uniform_half & 0x180) ? 0x02 : 0x00);
   out[5] = ((uniform_half & 0x80) >> 3) | ((src_half & 0x80) >> 7);
   out[6] = (uniform_half & 0x100) >> 6;
   out[7] = 0x00;
   out[8] = 0x60;
   out[9] = 0x00;
   *len = 10;
   return true;
}

bool
agx3_copy_encode_reg_to_uniform_g18g(enum agx_isa_encoding encoding,
                                     uint16_t uniform_half, uint16_t src_half,
                                     bool kill, uint16_t wait_mask,
                                     uint8_t out[AGX3_COPY_MAX_LENGTH],
                                     uint8_t *len)
{
   if (out == NULL || len == NULL || encoding != AGX_ISA_ENCODING_AGX3C)
      return false;

   if ((uniform_half & 1) || uniform_half + 2 > 512 || (src_half & 1) ||
       src_half + 2 > 256)
      return false;

   /* Every G17G fragment preamble ends with this copy: the pointer just
    * loaded into r0 goes to uniform pair 4, waiting for the load in slot
    * 1 (gpu/m5-fragment-depth-reference.metal and four more, all
    * 23 04 07 02 56 a0 a4 0a). It is the G17P form on ep1 with the
    * AGX3c selector pair and f42, and carries its wait itself (f30+i).
    * The form's destination field is nine bits wide (f3 deposits at
    * shifts 0..8), so the geometry preamble's pointer table above half
    * 128 composes in the same witnessed form; the composer round-trips
    * every value through the decoder. */
   if (wait_mask >= (1u << 8))
      return false;
   const struct agx3_field_kv baseline[] = {
      KV(9, 3), KV(10, 2), KV(23, 2), KV(24, 2), KV(28, 1), KV(29, 35),
      KV(42, 1),
   };
   struct agx3_field_kv overrides[11] = {
      KV(3, uniform_half),
      KV(16, src_half),
      KV(27, kill ? 1 : 0),
   };
   unsigned n = 3;
   for (unsigned i = 0; i < 8; ++i) {
      if (wait_mask & (1u << i))
         overrides[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }
   return agx3_bind_compose(encoding, AGX3_MN_copy, 1, baseline,
                            AGX3_COPY_ARRAY_SIZE(baseline), overrides, n,
                            out, len);
}

bool
agx3_copy_encode_uniform_to_reg16(enum agx_isa_encoding encoding,
                                  uint16_t dest_half, uint16_t uniform_half,
                                  bool dest_cache,
                                  uint8_t out[AGX3_COPY_MAX_LENGTH],
                                  uint8_t *len)
{
   const bool m5 = encoding == AGX_ISA_ENCODING_AGX3C;
   if (out == NULL || len == NULL ||
       (encoding != AGX_ISA_ENCODING_AGX3B && !m5) ||
       dest_half >= (m5 ? 256 : 128) || uniform_half >= 512)
      return false;

   /* The 16-bit widths of agx3_copy_encode_reg16() with the uniform source
    * selector of agx3_copy_encode_uniform_to_reg32(). A 16-bit value is one
    * half in either file, so neither index is aligned.
    */
   const struct agx3_field_kv baseline[] = {
      KV(9, 2), KV(10, 1), KV(23, 3), KV(24, 1),
   };
   const struct agx3_field_kv overrides[] = {
      KV(2, dest_half), KV(17, uniform_half), KV(11, dest_cache),
      KV(28, m5 ? 3 : 0), KV(29, m5 ? 35 : 0),
   };

   return agx3_bind_compose(encoding, AGX3_MN_copy,
                            dest_half >= AGX3_COPY_REG16_DEST_HALVES ? 1 : 0,
                            baseline,
                            AGX3_COPY_ARRAY_SIZE(baseline), overrides,
                            AGX3_COPY_ARRAY_SIZE(overrides), out, len);
}

bool
agx3_copy_encode_uniform_to_reg32(enum agx_isa_encoding encoding,
                                  uint16_t dest_half, uint16_t uniform_half,
                                  uint8_t out[AGX3_COPY_MAX_LENGTH],
                                  uint8_t *len)
{
   const struct agx3_field_kv *baseline;
   unsigned nbaseline;

   if (out == NULL || len == NULL)
      return false;

   if (encoding == AGX_ISA_ENCODING_AGX3B) {
      baseline = agx3b_copy_baseline;
      nbaseline = AGX3_COPY_ARRAY_SIZE(agx3b_copy_baseline);
   } else if (agx_encoding_is_g17_class(encoding)) {
      /* G18G (agx3d) spells this copy with the agx3c field vector: the
       * applegpu_g18g build of a kernel storing a constant-buffer float4
       * reads it with copy.ep0 83 00 08 08 .. 83 06 0e 08 (f2 0..6,
       * f17 8..14, f23=3 f24=2 f28=3 f29=35), the G17G build with the same
       * fields. Only those compact destinations are witnessed there. */
      if (encoding == AGX_ISA_ENCODING_AGX3D &&
          dest_half >= AGX3_COPY_REG16_DEST_HALVES)
         return false;
      baseline = agx3c_copy_baseline;
      nbaseline = AGX3_COPY_ARRAY_SIZE(agx3c_copy_baseline);
   } else {
      return false;
   }

   /* A 32-bit value occupies two halves in either file, so both indices are
    * 2-aligned; the GPR file is 256 halves and the uniform file 512. */
   if ((dest_half & 1) || dest_half + 2 > 256)
      return false;
   if ((uniform_half & 1) || uniform_half + 2 > 512)
      return false;

   const struct agx3_field_kv overrides[] = {
      KV(2, dest_half),
      KV(17, uniform_half),
   };

   if (dest_half < AGX3_COPY_REG16_DEST_HALVES)
      return agx3_bind_compose(encoding, AGX3_MN_copy, agx3_bind_ep(encoding, 0),
                               baseline, nbaseline, overrides,
                               AGX3_COPY_ARRAY_SIZE(overrides), out, len);

   /* The compact four-byte copy carries its destination in a seven-bit f2 and
    * truncates at half 127; only the destination is narrow there, as the
    * source witnesses below show. Apple's answer is the ten-byte ep1, which
    * deposits f2's eighth bit -- the same widening agx3_copy_encode_reg_to_
    * uniform() already uses for f3, and NOT the f39 suffix that form needs
    * (these carry f39 = 0).
    *
    * Witnessed with metal-nt -arch applegpu_g17p on MSL written for this,
    * a 100-value loop-carried register rotation (`rotate_pressure`) and a
    * uniform-chain readback under the same pressure (`uni_readback`). Both
    * split at exactly half 128, and neither ever widens for a high source:
    *
    *   ep0 (4 B)   destinations  18..126, sources up to 148
    *   ep1 (10 B)  destinations 128..190, sources 18..178
    *
    *   copy r66, r9.kill   2b 13 0f 00 02 10 00 00 00 00   f2=132 f16=18
    *   copy r67, r10.kill  3b 15 0f 00 02 10 00 00 00 00   f2=134 f16=20
    *   copy r95, r59.kill  fb 77 4f 00 02 10 00 00 00 00   f2=190 f16=118
    *
    * Byte 0 and byte 5 bit 4 carry the destination identically under the
    * uniform source selector, which is why only the encoding point moves
    * here. Ray-query shaders reach this: agx_lower_uniform_sources()
    * materializes every generic-ALU uniform source, and under their register
    * pressure register allocation placed those destinations at halves
    * 134-154, refusing the whole shader.
    *
    * AGX3c carries the same high destination bit in its eight-byte ep1.
    * The source kind and index fields are independent of that destination
    * extension, as with the corresponding extended half-register copy.
    */
   return agx3_bind_compose(encoding, AGX3_MN_copy, 1, baseline, nbaseline,
                            overrides,
                            AGX3_COPY_ARRAY_SIZE(overrides), out, len);
}
