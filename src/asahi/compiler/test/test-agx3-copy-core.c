/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * `copy dest32, uniform32` reproduced byte-for-byte against Apple's own
 * instructions.  Every positive vector below is lifted verbatim from the
 * G16G (AGX3b) and G17S (AGX3c) corpus in agx3_compose_corpus_vectors.c,
 * where these appear as copy.ep0 with f23 = 3 (uniform source kind),
 * f24 = 2 (32-bit), f2 = destination half, f17 = uniform half.
 */

#include "agx3_copy_core.h"

#include "agx3_binder.h"

#include "util/macros.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;
static unsigned failures;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x);          \
         ++failures;                                                           \
      }                                                                        \
   } while (0)

static bool
reference_reg_to_uniform(uint16_t uniform_half, uint16_t src_half, bool kill,
                         uint8_t out[AGX3_COPY_MAX_LENGTH], uint8_t *len)
{
   const struct agx3_field_kv compact[] = {
      {9, 3}, {10, 2}, {23, 2}, {24, 2},
   };
   const struct agx3_field_kv extended[] = {
      {9, 3}, {10, 2}, {23, 2}, {24, 2}, {39, 1},
   };
   const struct agx3_field_kv overrides[] = {
      {3, uniform_half}, {16, src_half}, {27, kill},
   };
   const bool wide = uniform_half >= 128;
   return agx3_bind_compose(
      AGX_ISA_ENCODING_AGX3B, AGX3_MN_copy, wide ? 1 : 0,
      wide ? extended : compact,
      wide ? ARRAY_SIZE(extended) : ARRAY_SIZE(compact), overrides,
      ARRAY_SIZE(overrides), out, len);
}

struct copy_vec {
   enum agx_isa_encoding enc;
   uint16_t dest_half;
   uint16_t uniform_half;
   uint8_t bytes[AGX3_COPY_MAX_LENGTH];
   uint8_t length;
};

static const struct copy_vec vectors[] = {
   {AGX_ISA_ENCODING_AGX3B, 70, 36, {0x3b, 0x24, 0x81, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 72, 38, {0x4b, 0x26, 0x81, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 26, 80, {0xdb, 0x50, 0x01, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 28, 82, {0xeb, 0x52, 0x01, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 22, 120, {0xbb, 0x78, 0x01, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 16, 56, {0x8b, 0x38, 0x01, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 74, 0, {0x5b, 0x00, 0x81, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 76, 2, {0x6b, 0x02, 0x81, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 62, 4, {0xfb, 0x04, 0x41, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 42, 64, {0x5b, 0x40, 0x41, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 96, 80, {0x0b, 0x50, 0xc1, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3B, 6, 56, {0x3b, 0x38, 0x01, 0x08}, 4},
   {AGX_ISA_ENCODING_AGX3C, 52, 36, {0xab, 0x24, 0x41, 0x08}, 4},
};

int
main(void)
{
   for (unsigned i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
      const struct copy_vec *v = &vectors[i];
      uint8_t out[AGX3_COPY_MAX_LENGTH];
      uint8_t len = 0;

      memset(out, 0xcd, sizeof(out));
      const bool ok = agx3_copy_encode_uniform_to_reg32(
         v->enc, v->dest_half, v->uniform_half, out, &len);

      CHECK(ok);
      if (!ok)
         continue;

      CHECK(len == v->length);
      if (len != v->length)
         continue;

      if (memcmp(out, v->bytes, len) != 0) {
         ++failures;
         fprintf(stderr, "FAIL vector %u (r%u <- u%u): got", i, v->dest_half,
                 v->uniform_half);
         for (unsigned b = 0; b < len; ++b)
            fprintf(stderr, " %02x", out[b]);
         fprintf(stderr, ", want");
         for (unsigned b = 0; b < v->length; ++b)
            fprintf(stderr, " %02x", v->bytes[b]);
         fprintf(stderr, "\n");
      }
      ++checks;
   }

   /* The 16-bit register-to-register move, including an odd destination. */
   {
      static const struct {
         uint16_t dest, src;
         uint8_t bytes[4];
      } reg16[] = {
         {0, 5, {0x03, 0x05, 0x29, 0x00}},
         {1, 6, {0x03, 0x06, 0x29, 0x01}},
      };
      for (unsigned i = 0; i < 2; ++i) {
         uint8_t o[AGX3_COPY_MAX_LENGTH];
         uint8_t l = 0;
         CHECK(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, reg16[i].dest,
                                      reg16[i].src, true, false, true, 0, o, &l));
         CHECK(l == 4);
         CHECK(memcmp(o, reg16[i].bytes, 4) == 0);
      }
      uint8_t o[AGX3_COPY_MAX_LENGTH];
      uint8_t l = 0;
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 128, 5, true, false, true, 0, o, &l));
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 0, 256, true, false, true, 0, o, &l));

      /* Register allocation reads AGX3_COPY_REG16_DEST_HALVES to decide when a
       * half victim could become unencodable, and skips the restriction
       * entirely on a register file that never reaches it
       * (agx_register_allocate.c, find_constrained_region).  Pin the boundary
       * from both sides so the constant cannot drift away from the field:
       * one below is the largest encodable destination, the constant itself is
       * the first refused one. */
      CHECK(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B,
                                   AGX3_COPY_REG16_DEST_HALVES - 1, 5, true,
                                   false, true, 0, o, &l));
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B,
                                    AGX3_COPY_REG16_DEST_HALVES, 5, true,
                                    false, true, 0, o, &l));
      CHECK(!agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3B,
                                               AGX3_COPY_REG16_DEST_HALVES, 5,
                                               true, o, &l));
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX2, 0, 5, true, false, true, 0, o, &l));

      /* A slot-1 wait is f30, reproducing an Apple corpus copy exactly. */
      static const uint8_t waited[4] = {0x93, 0x05, 0x21, 0x20};
      CHECK(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 18, 5, true, false, false, 1u << 0, o,
                                   &l));
      CHECK(l == 4 && memcmp(o, waited, 4) == 0);
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, 18, 5, true, false, false, 1u << 6, o,
                                    &l));
   }

   /* Independent AGX3c corpus witnesses: by813, by835 and by836. */
   {
      const struct {
         unsigned dest, src, wait;
         bool discard;
         uint8_t bytes[4];
      } native[] = {
         {16, 1, 1, false, {0x83, 0x01, 0x21, 0x20}},
         {0, 1, 0, true, {0x03, 0x01, 0x29, 0x00}},
         {1, 2, 0, true, {0x03, 0x02, 0x29, 0x01}},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(native); ++i) {
         uint8_t out[AGX3_COPY_MAX_LENGTH], length = 0;
         CHECK(agx3_copy_encode_reg16(
            AGX_ISA_ENCODING_AGX3C, native[i].dest, native[i].src, true,
            false, native[i].discard, native[i].wait, out, &length));
         CHECK(length == 4 && !memcmp(out, native[i].bytes, 4));
      }
      uint8_t out[AGX3_COPY_MAX_LENGTH], length = 0;
      memset(out, 0xcd, sizeof(out));
      /* G17G's extended form reaches all 256 halves and eight wait slots. */
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3C, 256, 1,
                                     true, false, false, 0, out, &length));
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3C, 0, 256,
                                     true, false, false, 0, out, &length));
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3C, 0, 1,
                                     true, true, true, 0, out, &length));
      CHECK(!agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3C, 0, 1,
                                     true, false, false, 1u << 8, out, &length));
      CHECK(length == 0);
      for (unsigned i = 0; i < sizeof(out); ++i)
         CHECK(out[i] == 0xcd);
   }

   /* Fresh Apple copy-live.nat and half-live.nat: ordinary live sources and
    * true dead sources, including odd halves, retain their distinct hints. */
   {
      const struct { unsigned dest, src; bool discard; uint8_t bytes[4]; } v[] = {
         {2, 3, false, {0x13,0x03,0x01,0x00}},
         {0, 5, false, {0x03,0x05,0x01,0x00}},
         {6, 8, true, {0x33,0x08,0x09,0x00}},
         {0, 9, false, {0x03,0x09,0x01,0x00}},
      };
      for (unsigned i=0;i<sizeof(v)/sizeof(v[0]);i++) {
         uint8_t o[AGX3_COPY_MAX_LENGTH], l=0;
         CHECK(agx3_copy_encode_reg16(AGX_ISA_ENCODING_AGX3B, v[i].dest,
               v[i].src, false, false, v[i].discard, 0, o, &l));
         CHECK(l==4 && !memcmp(o,v[i].bytes,4));
      }
   }

   /* The 16-bit uniform-to-register move, byte-exact against the two agx3b
    * copy.ep0 instructions in the Apple corpus that carry f9=2, f10=1,
    * f23=3, f24=1 -- the form the 32-bit encoder below deliberately refuses.
    * Without this, agx_lower_uniform_sources()'s 16-bit move stayed a bitop
    * with a uniform source and the packer refused the whole shader, which is
    * what u_blitter's texel-fetch program hit on G17P.
    */
   {
      static const struct {
         uint16_t dest, uniform;
         uint8_t bytes[4];
      } uni16[] = {
         {7, 114, {0x33, 0x72, 0x21, 0x09}},
         {3, 114, {0x13, 0x72, 0x21, 0x09}},
      };
      for (unsigned i = 0; i < ARRAY_SIZE(uni16); ++i) {
         uint8_t o[AGX3_COPY_MAX_LENGTH];
         uint8_t l = 0;
         CHECK(agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3B,
                                                 uni16[i].dest,
                                                 uni16[i].uniform, true, o,
                                                 &l));
         CHECK(l == 4);
         CHECK(memcmp(o, uni16[i].bytes, 4) == 0);
      }

      /* An odd destination half needs no separate encoding point, and the
       * cache hint is an independent bit, exactly as in the register form. */
      uint8_t o[AGX3_COPY_MAX_LENGTH];
      uint8_t l = 0;
      CHECK(agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3B, 5, 19,
                                              false, o, &l));
      CHECK(l == 4);
      uint8_t cached[AGX3_COPY_MAX_LENGTH];
      uint8_t cached_len = 0;
      CHECK(agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3B, 5, 19,
                                              true, cached, &cached_len));
      CHECK(cached_len == 4 && memcmp(o, cached, 4) != 0);

      /* Fail closed outside the witnessed domain. */
      CHECK(!agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3B, 128, 5,
                                               true, o, &l));
      CHECK(!agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX3B, 0, 512,
                                               true, o, &l));
      CHECK(!agx3_copy_encode_uniform_to_reg16(AGX_ISA_ENCODING_AGX2, 7, 114,
                                               true, o, &l));
   }

   /* Fresh G17P metal-nt containers for half mul, compare/select, and FMA/min
    * fragment pipelines all terminate their constant-program prefix with the
    * same 32-bit copy into an aligned uniform pair. None emits a 16-bit
    * uniform-destination copy; retain Apple's observed width contract. */
   {
      static const uint8_t apple[4] = {0x2b, 0x00, 0x09, 0x04};
      uint8_t o[AGX3_COPY_MAX_LENGTH];
      uint8_t l = 0;
      CHECK(agx3_copy_encode_reg_to_uniform(
         AGX_ISA_ENCODING_AGX3B, 4, 0, true, 0, o, &l));
      CHECK(l == 4 && memcmp(o, apple, 4) == 0);
      CHECK(!agx3_copy_encode_reg_to_uniform(
         AGX_ISA_ENCODING_AGX3B, 4, 0, true, 1, o, &l));

      /* First pair beyond ep0's seven-bit destination uses Apple's extended
       * constant-program COPY and preserves the upper destination bits. */
      static const uint8_t extended[10] = {
         0x0b, 0x03, 0x0f, 0x00, 0x42, 0x10, 0x00, 0x00, 0x60, 0x00,
      };
      CHECK(agx3_copy_encode_reg_to_uniform(
         AGX_ISA_ENCODING_AGX3B, 128, 2, true, 0, o, &l));
      CHECK(l == 10 && memcmp(o, extended, 10) == 0);

      CHECK(!agx3_copy_encode_reg_to_uniform(
         AGX_ISA_ENCODING_AGX3B, 511, 0, true, 0, o, &l));
      CHECK(!agx3_copy_encode_reg_to_uniform(
         AGX_ISA_ENCODING_AGX3B, 4, 256, true, 0, o, &l));
      CHECK(!agx3_copy_encode_reg_to_uniform(
         AGX_ISA_ENCODING_AGX2, 4, 0, true, 0, o, &l));

      /* Apple Metal, applegpu_g17g: the fragment preamble's copy of the
       * freshly loaded r0 pair into uniform pair 4, waiting for slot 1
       * (gpu/m5-fragment-depth-reference.metal, five identical witnesses).
       * The AGX3c form is ep1 with f42 and carries its own wait. */
      static const uint8_t g17g[8] = {0x23, 0x04, 0x07, 0x02,
                                      0x56, 0xa0, 0xa4, 0x0a};
      CHECK(agx3_copy_encode_reg_to_uniform_g18g(
         AGX_ISA_ENCODING_AGX3C, 4, 0, true, 1u << 0, o, &l));
      CHECK(l == 8 && memcmp(o, g17g, 8) == 0);
      static const uint8_t g17g_nowait[8] = {0x23, 0x00, 0x07, 0x02,
                                             0x56, 0xa0, 0xa4, 0x0a};
      CHECK(agx3_copy_encode_reg_to_uniform_g18g(
         AGX_ISA_ENCODING_AGX3C, 4, 0, true, 0, o, &l));
      CHECK(l == 8 && memcmp(o, g17g_nowait, 8) == 0);
      /* Above half 127 the same form deposits f3's upper bits (the
       * geometry preamble publishes its pointer table up to half 148). */
      static const uint8_t g17g_high[8] = {0xa3, 0x00, 0x0f, 0x02,
                                           0x56, 0xa0, 0xa4, 0x0a};
      CHECK(agx3_copy_encode_reg_to_uniform_g18g(
         AGX_ISA_ENCODING_AGX3C, 148, 0, true, 0, o, &l));
      CHECK(l == 8 && memcmp(o, g17g_high, 8) == 0);
      CHECK(!agx3_copy_encode_reg_to_uniform_g18g(
         AGX_ISA_ENCODING_AGX3C, 511, 0, true, 0, o, &l));
      CHECK(!agx3_copy_encode_reg_to_uniform_g18g(
         AGX_ISA_ENCODING_AGX3C, 4, 0, true, 1u << 8, o, &l));
      CHECK(!agx3_copy_encode_reg_to_uniform_g18g(
         AGX_ISA_ENCODING_AGX2, 4, 0, true, 0, o, &l));

      /* Both direct packers must remain byte-identical to the authoritative
       * table composer over the complete aligned CProg/GPR domain. */
      bool mismatch_reported = false;
      for (uint16_t dest = 0; dest < 512; dest += 2) {
         for (uint16_t src = 0; src < 256; src += 2) {
            for (unsigned kill = 0; kill < 2; ++kill) {
               uint8_t fast[AGX3_COPY_MAX_LENGTH] = {0};
               uint8_t reference[AGX3_COPY_MAX_LENGTH] = {0};
               uint8_t fast_len = 0, reference_len = 0;
               const bool match =
                  agx3_copy_encode_reg_to_uniform(
                     AGX_ISA_ENCODING_AGX3B, dest, src, kill, 0, fast,
                     &fast_len) &&
                  reference_reg_to_uniform(dest, src, kill, reference,
                                            &reference_len) &&
                  fast_len == reference_len &&
                  memcmp(fast, reference, fast_len) == 0;
               ++checks;
               if (!match) {
                  ++failures;
                  if (!mismatch_reported) {
                     fprintf(stderr,
                             "FAIL reg-to-uniform exhaustive u%u <- r%u "
                             "kill=%u: fast_len=%u reference_len=%u\n  fast:",
                             dest, src, kill, fast_len, reference_len);
                     for (unsigned b = 0; b < fast_len; ++b)
                        fprintf(stderr, " %02x", fast[b]);
                     fprintf(stderr, "\n  reference:");
                     for (unsigned b = 0; b < reference_len; ++b)
                        fprintf(stderr, " %02x", reference[b]);
                     fprintf(stderr, "\n");
                     mismatch_reported = true;
                  }
               }
            }
         }
      }
   }

   /* Fail closed outside the witnessed domain. */
   uint8_t out[AGX3_COPY_MAX_LENGTH];
   uint8_t len = 0;

   /* A 16-bit half of either file is a different form (Apple sets f11), which
    * agx3_copy_encode_uniform_to_reg16() above encodes. */
   CHECK(!agx3_copy_encode_uniform_to_reg32(AGX_ISA_ENCODING_AGX3B, 7, 114,
                                            out, &len));
   CHECK(!agx3_copy_encode_uniform_to_reg32(AGX_ISA_ENCODING_AGX3B, 8, 115,
                                            out, &len));

   /* Past the end of either file. */
   CHECK(!agx3_copy_encode_uniform_to_reg32(AGX_ISA_ENCODING_AGX3B, 256, 4,
                                            out, &len));
   CHECK(!agx3_copy_encode_uniform_to_reg32(AGX_ISA_ENCODING_AGX3B, 4, 512,
                                            out, &len));

   /*
    * A destination at or above half 128 does not fit the compact four-byte
    * point's seven-bit f2 and moves to the ten-byte ep1, the way Apple's own
    * compiler moves it. metal-nt -arch applegpu_g17p, on a 100-value
    * loop-carried register rotation and on a uniform-chain readback at the
    * same pressure, puts destinations 18..126 in ep0 and 128..190 in ep1 and
    * never widens for a high SOURCE -- its ep0 copies carry sources up to
    * 148. Two of those instructions verbatim:
    *
    *   copy r67, r10.kill  3b 15 0f 00 02 10 00 00 00 00   f2=134 f16=20
    *   copy r95, r59.kill  fb 77 4f 00 02 10 00 00 00 00   f2=190 f16=118
    *
    * The destination deposit below is byte-identical to those: half 134 is
    * byte 0 = 0x3b with byte 5 bit 4 set, half 190 is byte 0 = 0xfb with the
    * same byte 5 bit, and only the source-kind and source-index bytes differ.
    *
    * Ray-query shaders are what reach it. agx_lower_uniform_sources()
    * materializes a move for every generic-ALU uniform source, and under
    * their register pressure register allocation placed those destinations
    * at halves 134-154, so every ray-query shader was refused.
    */
   static const struct copy_vec wide[] = {
      {AGX_ISA_ENCODING_AGX3B, 128, 8,
       {0x0b, 0x09, 0x07, 0x00, 0x82, 0x10, 0x00, 0x00, 0x00, 0x00}, 10},
      {AGX_ISA_ENCODING_AGX3B, 134, 14,
       {0x3b, 0x0f, 0x07, 0x00, 0x82, 0x10, 0x00, 0x00, 0x00, 0x00}, 10},
      {AGX_ISA_ENCODING_AGX3B, 136, 18,
       {0x4b, 0x13, 0x07, 0x00, 0x82, 0x10, 0x00, 0x00, 0x00, 0x00}, 10},
      {AGX_ISA_ENCODING_AGX3B, 154, 12,
       {0xdb, 0x0d, 0x07, 0x00, 0x82, 0x10, 0x00, 0x00, 0x00, 0x00}, 10},
      {AGX_ISA_ENCODING_AGX3B, 190, 12,
       {0xfb, 0x0d, 0x47, 0x00, 0x82, 0x10, 0x00, 0x00, 0x00, 0x00}, 10},
      {AGX_ISA_ENCODING_AGX3B, 254, 510,
       {0xfb, 0x7f, 0xcf, 0x00, 0x82, 0x11, 0x00, 0x00, 0x00, 0x00}, 10},
   };
   for (unsigned i = 0; i < sizeof(wide) / sizeof(wide[0]); ++i) {
      const struct copy_vec *v = &wide[i];
      uint8_t w[AGX3_COPY_MAX_LENGTH];
      uint8_t wl = 0;
      memset(w, 0xcd, sizeof(w));
      CHECK(agx3_copy_encode_uniform_to_reg32(v->enc, v->dest_half,
                                              v->uniform_half, w, &wl));
      CHECK(wl == v->length);
      if (wl != v->length)
         continue;
      if (memcmp(w, v->bytes, wl) != 0) {
         ++failures;
         fprintf(stderr, "FAIL wide vector %u (r%u <- u%u): got", i,
                 v->dest_half, v->uniform_half);
         for (unsigned b = 0; b < wl; ++b)
            fprintf(stderr, " %02x", w[b]);
         fprintf(stderr, ", want");
         for (unsigned b = 0; b < v->length; ++b)
            fprintf(stderr, " %02x", v->bytes[b]);
         fprintf(stderr, "\n");
      }
      ++checks;
   }

   /* The split is exactly the seven-bit boundary, from both sides: the last
    * half the compact point can name stays four bytes, the first one it
    * cannot becomes ten. A regression either way shows up here rather than as
    * a whole refused shader. */
   CHECK(agx3_copy_encode_uniform_to_reg32(
            AGX_ISA_ENCODING_AGX3B, AGX3_COPY_REG16_DEST_HALVES - 2, 12, out,
            &len) &&
         len == 4);
   CHECK(agx3_copy_encode_uniform_to_reg32(
            AGX_ISA_ENCODING_AGX3B, AGX3_COPY_REG16_DEST_HALVES, 12, out,
            &len) &&
         len == 10);

   /* A high uniform SOURCE is encodable at the compact point, as Apple's own
    * ep0 copies with sources past 128 show; it must not drag the instruction
    * to the wide form. */
   CHECK(agx3_copy_encode_uniform_to_reg32(AGX_ISA_ENCODING_AGX3B, 4, 500,
                                           out, &len) &&
         len == 4);

   /* AGX2 has no `copy`; its move is a bitop that reads the uniform file. */
   CHECK(!agx3_copy_encode_uniform_to_reg32(AGX_ISA_ENCODING_AGX2, 4, 4, out,
                                            &len));

   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
