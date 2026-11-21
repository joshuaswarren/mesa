/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_stage_core.h"

#include <string.h>

#include "agx3_binder.h"
#include "agx3_tex_core.h"

#define KV(f, v) {(uint16_t)(f), (uint64_t)(v)}

static bool
stage_alu(enum agx_isa_encoding enc, bool fma, uint16_t dest, uint8_t slot,
          const uint16_t src[3], enum agx3_stage_fma_source source0,
          const bool live[3], const bool discard[3], uint16_t wait_mask,
          uint8_t out[16], uint8_t *len)
{
   /* M5 (G17G) terrain vertex mains write their outputs the same
    * way: staged FMUL R*R, FMA U*R+R and FMA R*R+R with f9=5, the same
    * operand fields, f95=1/f96=55 (FMA) or f78=1/f79=62 (MUL), eight
    * completion slots and an eight-bit wait set. Native witnesses are
    * asserted in test-agx3-uniform-alu.cpp. The coordinate FMA with an
    * immediate stays an AGX3B form. */
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   if ((!m5 && enc != AGX_ISA_ENCODING_AGX3B) || !src || !live || !discard ||
       !out || !len || dest >= 256 || (dest & 1) || slot < 1 ||
       slot > (m5 ? 8 : 6) || wait_mask > (m5 ? 0xff : 0x3f) ||
       (unsigned)source0 > AGX3_STAGE_FMA_IMMEDIATE ||
       (source0 != AGX3_STAGE_FMA_REGISTER && !fma) ||
       (m5 && source0 == AGX3_STAGE_FMA_IMMEDIATE))
      return false;
   for (unsigned s = 0; m5 && s < (fma ? 3 : 2); ++s) {
      const bool uniform = s == 0 && source0 == AGX3_STAGE_FMA_UNIFORM;
      if (!uniform && src[s] >= 252)
         return false;
   }
   for (unsigned s = 0; s < (fma ? 3 : 2); ++s) {
      const bool immediate = s == 0 && source0 == AGX3_STAGE_FMA_IMMEDIATE;
      if (src[s] >= 256 || (!immediate && (src[s] & 1)) ||
          (live[s] && discard[s]))
         return false;
   }
   if (source0 != AGX3_STAGE_FMA_REGISTER && (live[0] || discard[0]))
      return false;

   /* Exact native witnesses are recorded with their full field vectors in
    * vertex/native-terrain-fields.json. 0x360 is a staged FMUL, 0x350 a
    * uniform/register/register FMA, and 0x404 an all-register FMA. */
   const struct agx3_field_kv base[] = {
      KV(9, 5), KV(10, 2), KV(25, 2), KV(26, 2), KV(43, 2),
      KV(60, 2),
   };
   /* Seven base fields, two source hints, six FMA fields, two M5
    * selectors and eight wait bits: the all-register M5 form needs 25. */
   struct agx3_field_kv fields[25] = {
      KV(5, dest),
      KV(12, slot),
      KV(18, src[1]),
      KV(42, source0 == AGX3_STAGE_FMA_UNIFORM     ? 3
             : source0 == AGX3_STAGE_FMA_IMMEDIATE ? 1
                                                   : 2),
      KV(source0 == AGX3_STAGE_FMA_UNIFORM     ? 36
         : source0 == AGX3_STAGE_FMA_IMMEDIATE ? 34
                                               : 35,
         src[0]),
      KV(27, live[1]),
      KV(29, discard[1]),
   };
   unsigned n = 7;
   if (source0 == AGX3_STAGE_FMA_REGISTER) {
      fields[n++] = (struct agx3_field_kv)KV(44, live[0]);
      fields[n++] = (struct agx3_field_kv)KV(46, discard[0]);
   }
   if (fma) {
      fields[n++] = (struct agx3_field_kv)KV(52, src[2]);
      fields[n++] = (struct agx3_field_kv)KV(59, 2);
      fields[n++] = (struct agx3_field_kv)KV(61, live[2]);
      fields[n++] = (struct agx3_field_kv)KV(63, discard[2]);
      fields[n++] = (struct agx3_field_kv)KV(77, 2);
      fields[n++] = (struct agx3_field_kv)KV(91, 2);
   } else {
      fields[n++] = (struct agx3_field_kv)KV(74, 2);
   }
   if (m5) {
      fields[n++] = (struct agx3_field_kv)KV(fma ? 95 : 78, 1);
      fields[n++] = (struct agx3_field_kv)KV(fma ? 96 : 79, fma ? 55 : 62);
   }
   for (unsigned s = 0; s < (m5 ? 8 : 6); ++s)
      fields[n++] = (struct agx3_field_kv)KV((fma ? 97 : 80) + s,
                                           (wait_mask >> s) & 1);
   /* M5 endpoints are not ordered by length: keep the shortest. */
   uint8_t best[16], best_len = 0;
   for (unsigned ep = 0; ep <= (m5 ? (fma ? 7 : 5) : fma ? 5 : 3); ++ep) {
      uint8_t bytes[16], size;
      if (agx3_bind_compose(enc, fma ? AGX3_MN_f32fma : AGX3_MN_f32mul, ep,
                           base, 6, fields, n, bytes, &size)) {
         if (!m5) {
            memcpy(out, bytes, size);
            *len = size;
            return true;
         }
         if (!best_len || size < best_len) {
            memcpy(best, bytes, size);
            best_len = size;
         }
      }
   }
   if (!best_len)
      return false;
   memcpy(out, best, best_len);
   *len = best_len;
   return true;
}

bool
agx3_stage_vertex_alu(enum agx_isa_encoding enc, bool fma, uint16_t dest,
                      uint8_t slot, const uint16_t src[3], bool uniform0,
                      const bool live[3], const bool discard[3],
                      uint16_t wait_mask, uint8_t out[16], uint8_t *len)
{
   return stage_alu(enc, fma, dest, slot, src,
                    uniform0 ? AGX3_STAGE_FMA_UNIFORM : AGX3_STAGE_FMA_REGISTER,
                    live, discard, wait_mask, out, len);
}

bool
agx3_stage_fma(enum agx_isa_encoding enc, uint16_t dest, uint8_t slot,
               const uint16_t src[3], enum agx3_stage_fma_source source0,
               const bool live[3], const bool discard[3], uint16_t wait_mask,
               uint8_t out[16], uint8_t *len)
{
   return stage_alu(enc, true, dest, slot, src, source0, live, discard,
                    wait_mask, out, len);
}

static bool
b_only(enum agx_isa_encoding enc)
{
   return enc == AGX_ISA_ENCODING_AGX3B;
}

/* The staging destination class: index in f5, kill in f13, slot in f12. */
static unsigned
append_stage_dest(struct agx3_field_kv *ov, unsigned n, uint16_t dest,
                  bool kill, uint8_t slot)
{
   ov[n++] = (struct agx3_field_kv)KV(9, 5);
   ov[n++] = (struct agx3_field_kv)KV(5, dest);
   ov[n++] = (struct agx3_field_kv)KV(13, kill ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(12, slot);
   return n;
}

/*
 * staging[dest] <- reg[src], 32-bit, taking scoreboard slot `slot`.
 *
 * This is exactly agx3_tex_stage_copy; it is re-exported here so the writeout
 * path does not have to include the texture header to reach a general encoder.
 */
bool
agx3_stage_copy(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                uint8_t slot, uint16_t wait_mask,
                uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   /* Same encoder, same witnesses; only the name is neutral. */
   return agx3_tex_stage_copy(enc, dest, src, slot, wait_mask, out, len);
}

bool
agx3_stage_copy_source(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                        bool src_kill, uint8_t slot, uint16_t wait_mask,
                        uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   return agx3_stage_copy_liveness(enc, dest, src, false, src_kill, slot,
                                   wait_mask, out, len);
}

bool
agx3_stage_copy_liveness(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                          bool stage_kill, bool src_kill, uint8_t slot,
                          uint16_t wait_mask,
                          uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   const unsigned slots = agx_encoding_is_g17_class(enc) ? 8 : 6;
   if (!out || !len ||
       (enc != AGX_ISA_ENCODING_AGX3B && enc != AGX_ISA_ENCODING_AGX3C) ||
       (stage_kill && enc != AGX_ISA_ENCODING_AGX3B) ||
       (dest & 1) || (src & 1) || dest >= 256 || src >= 256 ||
       slot < 1 || slot > slots || (wait_mask & ~((1u << slots) - 1)))
      return false;

   struct agx3_field_kv base[7] = {
      KV(9, 5), KV(10, 2), KV(23, 2), KV(24, 2), KV(27, src_kill),
   };
   unsigned nb = 5;
   if (enc == AGX_ISA_ENCODING_AGX3C) {
      base[nb++] = (struct agx3_field_kv)KV(28, 1);
      base[nb++] = (struct agx3_field_kv)KV(29, 35);
   }
   struct agx3_field_kv ov[12] = {
      KV(5, dest), KV(16, src), KV(12, slot), KV(13, stage_kill),
   };
   unsigned n = 4;
   for (unsigned i = 0; i < slots; ++i) {
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);
   }

   for (int ep = 0; ep <= 1; ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_copy, ep, base, nb, ov, n, out, len))
         return true;
   }
   return false;
}

bool
agx3_stage_copy_kill(enum agx_isa_encoding enc, uint16_t dest, uint16_t src,
                     bool kill, uint8_t slot, uint16_t wait_mask,
                     uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   if (!kill)
      return agx3_stage_copy(enc, dest, src, slot, wait_mask, out, len);

   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;

   if (!b_only(enc) || (dest & 1) || (src & 1) || dest >= 256 || src >= 256)
      return false;
   if (slot < 1 || slot > 6 || (wait_mask & ~(uint16_t)0x3f))
      return false;

   /* f27 = 1 accompanies the 32-bit register source, exactly as in the
    * texture-coordinate copy this shares an encoding with. */
   const struct agx3_field_kv base[] = {KV(10, 2), KV(23, 2), KV(24, 2),
                                        KV(27, 1)};

   struct agx3_field_kv ov[14];
   unsigned n = append_stage_dest(ov, 0, dest, true, slot);
   ov[n++] = (struct agx3_field_kv)KV(16, src);
   for (unsigned i = 0; i < 6; ++i)
      if (wait_mask & (1u << i))
         ov[n++] = (struct agx3_field_kv)KV(30 + i, 1);

   /* The kill bit does not fit the compact form, so Apple escalates to ep1;
    * the composer's first-success order reproduces that. */
   if (agx3_bind_compose(enc, AGX3_MN_copy, 0, base, 4, ov, n, out, len))
      return true;
   if (agx3_bind_compose(enc, AGX3_MN_copy, 1, base, 4, ov, n, out, len))
      return true;

   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;
   return false;
}

static bool
clr_common(enum agx_isa_encoding enc, const struct agx3_field_kv *ov,
           unsigned n, uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   const struct agx3_field_kv base[] = {KV(10, 2)};
   const struct agx3_field_kv m5_base[] = {
      KV(10, 2), KV(14, 3), KV(15, 25),
   };
   const bool m5 = enc == AGX_ISA_ENCODING_AGX3C;
   /* Native G17G pressure loops clear high registers with ep1. G18G
    * carries the same destination and selectors in its own CLR form. */
   for (unsigned ep = 0; ep <= (m5 ? 1 : 0); ++ep) {
      if (agx3_bind_compose(enc, AGX3_MN_clr, ep, m5 ? m5_base : base,
                           m5 ? 3 : 1, ov, n, out, len))
         return true;
   }
   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;
   return false;
}

bool
agx3_clr(enum agx_isa_encoding enc, uint16_t dest, bool half, bool hint,
         uint8_t out[AGX3_STAGE_MAX_LENGTH], uint8_t *len)
{
   if (out == NULL || len == NULL)
      return false;
   memset(out, 0, AGX3_STAGE_MAX_LENGTH);
   *len = 0;

   if ((!b_only(enc) && enc != AGX_ISA_ENCODING_AGX3C) ||
       dest >= 256 || (!half && (dest & 1)))
      return false;

   struct agx3_field_kv ov[5];
   unsigned n = 0;
   ov[n++] = (struct agx3_field_kv)KV(9, 2);
   ov[n++] = (struct agx3_field_kv)KV(2, dest);
   ov[n++] = (struct agx3_field_kv)KV(11, hint ? 1 : 0);
   ov[n++] = (struct agx3_field_kv)KV(10, half ? 1 : 2);
   return clr_common(enc, ov, n, out, len);
}
