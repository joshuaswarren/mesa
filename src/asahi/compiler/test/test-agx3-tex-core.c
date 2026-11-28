/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx3_tex_core.h"
#include "agx3_compose/agx3_decode.h"

#include <stdio.h>
#include <string.h>

static unsigned checks;

#define CHECK(x)                                                               \
   do {                                                                        \
      ++checks;                                                                \
      if (!(x)) {                                                              \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x);                \
         return 1;                                                             \
      }                                                                        \
   } while (0)

static bool
same(const struct agx3_tex_fetch_desc *a,
     const struct agx3_tex_fetch_desc *b)
{
   return a->dest == b->dest && a->coord == b->coord &&
          a->width == b->width && a->mask == b->mask &&
          a->texture == b->texture && a->slot == b->slot &&
          a->wait_mask == b->wait_mask && a->coord_kill == b->coord_kill &&
          a->final == b->final &&
          a->array == b->array &&
          a->multisample == b->multisample &&
          a->staged_sample == b->staged_sample &&
          a->three_d == b->three_d && a->sample == b->sample &&
          (a->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) ==
             (b->texture_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) &&
          (a->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) ==
             (b->sampler_kind ?: AGX3_TEX_OPERAND_IMMEDIATE) &&
          a->texture_value == b->texture_value &&
          a->sampler_value == b->sampler_value &&
          a->texture_kill == b->texture_kill;
}

static int
roundtrip(enum agx_isa_encoding enc, const struct agx3_tex_fetch_desc *d)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0;
   struct agx3_tex_fetch_desc decoded;
   CHECK(agx3_tex_encode_fetch(enc, d, bytes, &length));
   CHECK(length == (enc == AGX_ISA_ENCODING_AGX3B ?
      ((d->texture_kind <= AGX3_TEX_OPERAND_IMMEDIATE &&
        d->sampler_kind <= AGX3_TEX_OPERAND_IMMEDIATE) ? 14 : 20) : 8));
   CHECK(agx3_tex_decode_fetch(enc, bytes, length, &decoded));
   CHECK(same(d, &decoded));
   return 0;
}

static int
test_exact_witnesses(void)
{
   static const uint8_t b[] = {
      0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x17,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   };
   static const uint8_t c[] = {0x0f, 0x06, 0x1a, 0x00,
                               0x40, 0x80, 0x60, 0x29};
   static const uint8_t b_mask1_d0[] = {
      0x05, 0x80, 0x0c, 0xa0, 0x90, 0x00, 0x17,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   };
   static const uint8_t b_mask1_d2[] = {
      0x0d, 0x80, 0x0c, 0xa0, 0x90, 0x00, 0x17,
      0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00,
   };
   static const uint8_t c_mask1_d0[] = {0x0f, 0x06, 0x1a, 0x00,
                                        0x40, 0x80, 0x00, 0x21};
   static const uint8_t c_mask1_d2[] = {0x1f, 0x06, 0x1a, 0x00,
                                        0x40, 0x80, 0x00, 0x21};
   struct agx3_tex_fetch_desc d = {
      .dest = 0,
      .coord = 0,
      .width = AGX3_TEX_W32,
      .mask = 15,
      .texture = 0,
      .slot = 1,
      .wait_mask = 1,
      .coord_kill = true,
      .final = true,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   struct agx3_tex_fetch_desc decoded;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == sizeof(b) && memcmp(bytes, b, sizeof(b)) == 0);
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, b, sizeof(b), &decoded));
   CHECK(same(&d, &decoded));
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == sizeof(c) && memcmp(bytes, c, sizeof(c)) == 0);
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3C, c, sizeof(c), &decoded));
   CHECK(same(&d, &decoded));

   d.mask = 1;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == sizeof(b_mask1_d0) &&
         memcmp(bytes, b_mask1_d0, sizeof(b_mask1_d0)) == 0);
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == sizeof(c_mask1_d0) &&
         memcmp(bytes, c_mask1_d0, sizeof(c_mask1_d0)) == 0);
   d.dest = 2;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == sizeof(b_mask1_d2) &&
         memcmp(bytes, b_mask1_d2, sizeof(b_mask1_d2)) == 0);
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == sizeof(c_mask1_d2) &&
         memcmp(bytes, c_mask1_d2, sizeof(c_mask1_d2)) == 0);
   return 0;
}

static int
test_supported_axes(void)
{
   for (unsigned ei = 0; ei < 2; ++ei) {
      const enum agx_isa_encoding enc =
         ei ? AGX_ISA_ENCODING_AGX3C : AGX_ISA_ENCODING_AGX3B;
      struct agx3_tex_fetch_desc d = {
         .width = AGX3_TEX_W32,
         .mask = 15,
         .slot = 1,
      };

      for (unsigned v = 0; v < 128; v += 2) {
         d.dest = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.width = AGX3_TEX_W16;
      for (unsigned v = 0; v < 128; ++v) {
         d.dest = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.dest = 0;
      d.width = AGX3_TEX_W32;
      for (unsigned v = 0; v <= 60; v += 4) {
         d.coord = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.coord = 0;
      for (unsigned v = 0; v < 16; ++v) {
         d.texture = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.texture = 0;
      for (unsigned v = 1; v <= (ei ? 8 : 6); ++v) {
         d.slot = v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.slot = 1;
      for (unsigned v = 0; v < (ei ? 7 : 6); ++v) {
         d.wait_mask = 1u << v;
         CHECK(roundtrip(enc, &d) == 0);
      }
      d.wait_mask = 0;
      d.mask = 1;
      CHECK(roundtrip(enc, &d) == 0);
      d.mask = 15;
      d.coord_kill = true;
      CHECK(roundtrip(enc, &d) == 0);
      d.final = true;
      CHECK(roundtrip(enc, &d) == 0);
   }
   return 0;
}

static bool
rejected_and_cleared(enum agx_isa_encoding enc,
                     const struct agx3_tex_fetch_desc *d)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0xff;
   memset(bytes, 0xff, sizeof(bytes));
   if (agx3_tex_encode_fetch(enc, d, bytes, &length) || length != 0)
      return false;
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      if (bytes[i] != 0)
         return false;
   return true;
}

static int
test_fail_closed(void)
{
   struct agx3_tex_fetch_desc d = {
      .width = AGX3_TEX_W32, .mask = 15, .slot = 1,
   };
   struct agx3_tex_fetch_desc out;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;

   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX2, &d));
   d.width = (enum agx3_tex_width)0;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.width = AGX3_TEX_W32;
   d.dest = 1;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.dest = 0;
   d.coord = 2;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.coord = 64;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.coord = 0;
   d.mask = 0;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.mask = 15;
   d.texture = 16;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.texture = 0;
   d.slot = 7;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   d.slot = 9;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   d.slot = 1;
   d.wait_mask = 3;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3B, &d));
   /* Slot eight is G17G's; the compute argument-buffer read waits on it in
    * the ten-byte form, which an immediate texture now reaches as well. */
   d.wait_mask = 0x80;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == 10);
   d.wait_mask = 0x100;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));

   d.wait_mask = 0;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length - 1,
                                &out));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length + 1,
                                &out));
   bytes[12] ^= 0x80; /* reserved/fixed selector bit */
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length, &out));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, NULL, length, &out));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, bytes, length, NULL));
   CHECK(!agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX2, bytes, 0, &out));
   return 0;
}

static int
hex_eq(const uint8_t *bytes, uint8_t len, const char *hex)
{
   char got[2 * (AGX3_TEX_MAX_LENGTH > AGX3_TEX_COPY_MAX_LENGTH
                    ? AGX3_TEX_MAX_LENGTH
                    : AGX3_TEX_COPY_MAX_LENGTH) +
            1] = {0};
   for (unsigned i = 0; i < len; ++i)
      snprintf(got + 2 * i, 3, "%02x", bytes[i]);
   if (strcmp(got, hex) != 0) {
      fprintf(stderr, "got %s expected %s\n", got, hex);
      return 0;
   }
   return 1;
}

/*
 * Byte-exact replay of Apple's filtered-sample encodings. Kernel witnesses
 * (explicit level) are from research/gpu/isa-re/gen-x/texops/corpus; the
 * implicit-LOD fragment witnesses (ts_auto / fs_bufcoord) were compiled from
 * fragment render pipelines with Apple's offline translator
 * (MetalToolchain v27.1.5194.15) and decode with the same tables. AGX3b
 * bytes are g16g (byte-identical on g15g); AGX3c bytes are g17s. The G17P
 * `t_base` bytes are also the exact sample in an A18 Pro vertex pipeline with
 * a VBO plus texture0/sampler0, proving that the explicit-level instruction
 * itself is shared across vertex and compute/fragment stages.
 */
static int
test_sample_witnesses(void)
{
   struct svec {
      const char *name;
      struct agx3_tex_sample_desc d;
      const char *hex_b, *hex_c;
   };
   static const struct svec v[] = {
      {"t_base",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 0, .sampler = 0, .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 0,
        .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true},
       "05800cb8b0000900000010000100",
       "0f041200418060290100010000000000008000000000"},
      {"x_mask_x",
       {.dest = 2, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0d800ca090000900000010000100",
       "1f041200418000210100010000000000008000000000"},
      {"x_mask_xy",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 3,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca8b0000900000010000100", NULL},
      {"x_mask_zw",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 12,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb890000900000050000100",
       "0f041200418060210300010000000000008000000000"},
      {"x_mask_w",
       {.dest = 2, .coord = 0, .width = AGX3_TEX_W32, .mask = 8,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0d800cb890000900000010000100", NULL},
      {"l_lod2 level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000900000010080100",
       "0f041200418060290100010080000000008000000000"},
      {"k_lod7q level(7.25)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 464, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b00009800000101c0100",
       "0f0412004180602901000100d0010000008000000000"},
      {"L_smp2x sampler 1",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .sampler = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .coord_kill = true, .final = true},
       "25800cb830240900000110000100",
       "4f041200c98160090100010000000000008000000000"},
      {"L_tex2x texture 1",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .coord_kill = true, .final = true},
       "25800cb830240900800010000100", NULL},
      {"Ls_slots4 #1",
       {.dest = 16, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 2, .wait_mask = 1u << 1,
        .coord_kill = true},
       "45800c3830250900000010000100",
       "8f041200c90060490100010000000000008000000000"},
      {"Ls_slots4 #2",
       {.dest = 24, .coord = 4, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .wait_mask = 1u << 2, .coord_kill = true},
       "65820c38b0250900800010000100",
       "cf041204c90068690100010000000000008000000000"},
      {"Ls_slots4 #3",
       {.dest = 8, .coord = 8, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 2, .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true},
       "258c0c38b0000900000010000100",
       "4f041208411060290100010000000000008000000000"},
      {"Ls_slots4 #4",
       {.dest = 0, .coord = 12, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 3, .lod = AGX3_TEX_LOD_LEVEL, .slot = 3,
        .wait_mask = 1u << 2, .coord_kill = true, .final = true},
       "058e0cb8b0490900800010000100",
       "0f04120c5190686b0100010000000000008000000000"},
      /* Metal 32023.917 / applegpu_g17p fragment pipelines with two SMPs.
       * Pair the first AUTO with final AUTO for AUTO/AUTO, first AUTO with
       * final LEVEL for AUTO/LEVEL, and the two LEVEL rows for LEVEL/LEVEL.
       * f156 is absent from the first sample and terminates each group on the
       * second. The all-LEVEL pair proves this is a group property rather
       * than an implicit-derivative bit; t_base above is the single-LEVEL
       * control and has no f156. */
      {"fragment group first AUTO",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 2, .wait_mask = 1,
        .sample_group_nonterminal = true},
       "05800c18b0240000000010000100", NULL},
      {"fragment group final AUTO",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .sampler = 1, .lod = AGX3_TEX_LOD_AUTO, .slot = 2,
        .coord_kill = true, .final = true, .sample_group_end = true},
       "25800cb830240000810110000100", NULL},
      {"fragment group first LEVEL",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 2, .wait_mask = 1,
        .sample_group_nonterminal = true},
       "05800c18b0240900000010000100", NULL},
      {"fragment group final LEVEL",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .sampler = 1, .lod = AGX3_TEX_LOD_LEVEL, .slot = 2,
        .coord_kill = true, .final = true, .sample_group_end = true},
       "25800cb830240900810110000100", NULL},
      /* Explicit gradients are outside the aggregate group. In native
       * AUTO/GRADIENT the lone AUTO retains its own f156 while the terminal
       * gradient carries f139=5 and no f156. */
      {"AUTO before terminal GRADIENT",
       {.dest = 0, .coord = 16, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 2, .wait_mask = 2,
        .coord_kill = true},
       "05800c3930250000010010000100", NULL},
      {"terminal GRADIENT outside group",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .sampler = 1, .lod = AGX3_TEX_LOD_GRADIENT, .slot = 2,
        .wait_mask = 1, .coord_kill = true, .final = true},
       "25800cb8b0240401800110000100", NULL},
      /* AUTO/GRADIENT/AUTO proves the gradient is transparent rather than a
       * group boundary: the two AUTO samples still form one group around it. */
      {"AUTO group before transparent GRADIENT",
       {.dest = 8, .coord = 16, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 3, .wait_mask = 2,
        .sample_group_nonterminal = true},
       "25800c1930490000000010000100", NULL},
      {"transparent middle GRADIENT",
       {.dest = 16, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .sampler = 1, .lod = AGX3_TEX_LOD_GRADIENT, .slot = 3,
        .wait_mask = 1, .coord_kill = true},
       "45800c38b0480401800110000100", NULL},
      {"AUTO group after transparent GRADIENT",
       {.dest = 0, .coord = 16, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 2, .sampler = 2, .lod = AGX3_TEX_LOD_AUTO, .slot = 1,
        .coord_kill = true, .final = true, .sample_group_end = true},
       "05880cb931000000010010000100", NULL},
      /* The native three-branch fragment lays its three SMPs out in CFG
       * order. Both possible nonterminal arms clear f156; only the globally
       * final arm carries f156 together with f147. */
      {"branched group first AUTO",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .sample_group_nonterminal = true},
       "05800c38b0000000000010000100", NULL},
      {"branched group middle AUTO",
       {.dest = 14, .coord = 4, .width = AGX3_TEX_W32, .mask = 15,
        .texture = 1, .sampler = 1, .lod = AGX3_TEX_LOD_AUTO, .slot = 1,
        .wait_mask = 2, .coord_kill = true,
        .sample_group_nonterminal = true},
       "3d820c3830010000800110000100", NULL},
      {"branched group final AUTO",
       {.dest = 10, .coord = 4, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 2, .sampler = 2, .lod = AGX3_TEX_LOD_AUTO, .slot = 1,
        .wait_mask = 2, .coord_kill = true, .final = true,
        .sample_group_end = true},
       "2d8a0ca011010000010010000100", NULL},
      {"ts_auto implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000000010010000100",
       "0f061600418060290118010000000000008000000000"},
      /* The same implicit-LOD sample once per result mask. Apple's own
       * applegpu_g17g compiler emits all fifteen from
       * `gpu/m5-sample-mask-reference.metal`, and across them nothing moves
       * but f75: the masks that name one channel take Apple's destination
       * register 2, the rest register 0. GL fragment shaders reach the
       * narrow ones constantly -- vec4(texture(s, uv).rgb, 1.0) is mask 7 --
       * and AGX3C used to admit only 1, 3, 8, 12 and 15. */
      {"ts_auto implicit LOD mask 1 (x)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca090000000010010000100",
       "0f061600418000210118010000000000008000000000"},
      {"ts_auto implicit LOD mask 2 (y)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 2,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca890000000010010000100",
       "0f061600418020210118010000000000008000000000"},
      {"ts_auto implicit LOD mask 3 (xy)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 3,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca8b0000000010010000100",
       "0f061600418020290118010000000000008000000000"},
      {"ts_auto implicit LOD mask 4 (z)",
       {.dest = 2, .coord = 0, .width = AGX3_TEX_W32, .mask = 4,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0d800cb090000000010010000100",
       "1f061600418040210118010000000000008000000000"},
      {"ts_auto implicit LOD mask 5 (xz)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 5,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca090000000010050000100",
       "0f061600418000210318010000000000008000000000"},
      {"ts_auto implicit LOD mask 6 (yz)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 6,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca890000000010050000100",
       "0f061600418020210318010000000000008000000000"},
      {"ts_auto implicit LOD mask 7 (xyz)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 7,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb0b0000000010010000100",
       "0f061600418040290118010000000000008000000000"},
      {"ts_auto implicit LOD mask 8 (w)",
       {.dest = 2, .coord = 0, .width = AGX3_TEX_W32, .mask = 8,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0d800cb890000000010010000100",
       "1f061600418060210118010000000000008000000000"},
      {"ts_auto implicit LOD mask 9 (xw)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 9,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca0b0000000010010000100",
       "0f061600418000290118010000000000008000000000"},
      {"ts_auto implicit LOD mask 10 (yw)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 10,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb090000000010050000100",
       "0f061600418040210318010000000000008000000000"},
      {"ts_auto implicit LOD mask 11 (xyw)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 11,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca8b0000000010050000100",
       "0f061600418020290318010000000000008000000000"},
      {"ts_auto implicit LOD mask 12 (zw)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 12,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb890000000010050000100",
       "0f061600418060210318010000000000008000000000"},
      {"ts_auto implicit LOD mask 13 (xzw)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 13,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800ca0b0000000010050000100",
       "0f061600418000290318010000000000008000000000"},
      {"ts_auto implicit LOD mask 14 (yzw)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 14,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb0b0000000010050000100",
       "0f061600418040290318010000000000008000000000"},
      /* Apple's G17P texture.sample offset differential sets f141/f51 and
       * packs signed x/y nibbles in f43. These three result shapes prove the
       * offset fields are orthogonal to both width and mask:
       *
       *   f32x4 plain       05800cb8b0000000010010001200
       *   f32x4 +(0, 0)     05800cb8b0000000010010001200
       *   f32x4 +(1,-1)     05800cb8b000010201881000120e
       *   f16x4 +(1,-1)     35000cb8b000010201881000120e
       *   f32x1 +(1,-1)     1d800ca09000010201881000120e
       *
       * The expected bytes below retain this encoder's hardware-qualified
       * descriptor profile (f83=0/f112=4); only that independent profile
       * differs from Apple's fragment bytes above.
       */
      {"ts_auto offset +(1,-1)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true,
        .offset_xyz = 0xf1},
       "05800cb8b000010201881000010e", NULL},
      {"ts_auto offset half +(1,-1)",
       {.dest = 6, .coord = 0, .width = AGX3_TEX_W16, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true,
        .offset_xyz = 0xf1},
       "35000cb8b000010201881000010e", NULL},
      {"ts_auto offset x-only result",
       {.dest = 6, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true,
        .offset_xyz = 0xf1},
       "1d800ca09000010201881000010e", NULL},
      {"ts_array direct register layer implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true,
        .array_layer = 22},
       "05800cb8b0000004010010000100", NULL},
      {"texture3d level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true,
        .three_d = true},
       "05800cb8b0003900000010080100", NULL},
      /* G17G's own cube comparisons (gpu/m5-shadow-array-cube-reference.metal,
       * metal-nt -arch applegpu_g17g). The fragment ones terminate their
       * sample group; the vertex one does not. Every witness compares, so a
       * G17G cube sample without a comparison stays refused. */
      {"gpu1 cube comparison implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 0, .sampler = 1, .lod = AGX3_TEX_LOD_AUTO, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .shadow = true,
        .cube = true, .sample_group_end = true},
       NULL, "0f04160045810121010002000000"},
      {"gpu1 cube comparison bias(reg)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 0, .sampler = 1, .lod = AGX3_TEX_LOD_BIAS_REGISTER,
        .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
        .shadow = true, .cube = true, .sample_group_end = true},
       NULL, "0f04160041810321010002000000"},
      {"gpu1 cube comparison level(reg)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 0, .sampler = 1, .lod = AGX3_TEX_LOD_REGISTER, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .shadow = true,
        .cube = true, .sample_group_end = true},
       NULL, "0f05160041810221010002000000"},
      {"gpu1 cube-array comparison implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 0, .sampler = 1, .lod = AGX3_TEX_LOD_AUTO, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .shadow = true,
        .cube = true, .array = true, .sample_group_end = true},
       NULL, "0f04160045810321010002000000"},
      {"gpu1 cube-array comparison level(reg)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 0, .sampler = 1, .lod = AGX3_TEX_LOD_REGISTER, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .shadow = true,
        .cube = true, .array = true, .sample_group_end = true},
       NULL, "0f05160041810021010802000000"},
      {"gpu1 vertex cube-array comparison level(reg)",
       {.dest = 8, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .texture = 0, .sampler = 0, .lod = AGX3_TEX_LOD_REGISTER, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .shadow = true,
        .cube = true, .array = true, .vertex_stage = true},
       NULL, "4f05120041800021010802000000"},
      {"native cube implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0001000010010000100", NULL},
      {"native cube level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0001300010010080100", NULL},
      {"native cube register level",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0000003010010000100", NULL},
      /* metal-nt -arch applegpu_g17p compute kernels close no sample group:
       * the same samples in texturecube kernels at level(2.0) and at a
       * kernel-computed level(l), and a texture2d_array kernel at level(l),
       * leave f156 clear. */
      {"compute cube level(2.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1,
        .wait_mask = 1, .coord_kill = true, .final = true, .cube = true,
        .compute_stage = true},
       "05800cb8b0001300000010080100", NULL},
      {"compute cube register level",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true,
        .compute_stage = true},
       "05800cb8b0000003000010000100", NULL},
      /* metal-nt -arch applegpu_g17p fragment texturecube and
       * texturecube_array samples at bias(b[i]) and bias(0.5). */
      {"native cube register bias",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0001400010010000100", NULL},
      {"native cube bias(0.5)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true},
       "05800cb8b0001101010010020100", NULL},
      /* metal-nt -arch applegpu_g17p, fragment texturecube_array::sample
       * with the layer read from a device buffer into a GPR: the layer is
       * staged with CEM's face (f77=2), with no direct f31/f37 operand. */
      {"native cube-array GPR layer implicit LOD",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b0001004010010000100", NULL},
      {"native cube-array GPR layer level zero",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b0005300010010000100", NULL},
      /* metal-nt -arch applegpu_g17p fragment texturecube_array samples at
       * l[i] with level(b[j]), bias(b[j]) and bias(0.5). */
      {"native cube-array GPR layer register level",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b0000803010010000100", NULL},
      {"native cube-array GPR layer register bias",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b0001800010010000100", NULL},
      {"native cube-array GPR layer bias(0.5)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 22},
       "05800cb8b0009101010010020100", NULL},
      {"native cube comparison",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .cube = true, .shadow = true},
       "05800ca090003000010000000100", NULL},
      {"native cube-array comparison",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .array = true, .cube = true,
        .array_layer = 6, .shadow = true},
       "05800ca090003004010000000100", NULL},
      /*
       * Implicit LOD plus a constant bias.  Apple's own fragment samples of
       * t.sample(s, uv, bias(k)) differ from its plain t.sample(s, uv) in
       * exactly two places -- byte 6 gains 0x07 (f37 = 1, f139 = 1) and the
       * 12-bit x64 field at bytes 7/11 takes k * 64 -- so these rows are the
       * implicit-LOD witness above carrying that same delta:
       *
       *   t.sample(s, uv)              05 80 0c b8 b0 00 00 00 01 00 10 00 12 00
       *   t.sample(s, uv, bias(0.25))  05 80 0c b8 b0 00 07 80 01 00 10 00 12 00
       *   t.sample(s, uv, bias(0.5))   05 80 0c b8 b0 00 07 00 01 00 10 02 12 00
       *   t.sample(s, uv, bias(1.0))   05 80 0c b8 b0 00 07 00 01 00 10 04 12 00
       */
      {"bias(0.25)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 16, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000780010010000100", NULL},
      {"bias(0.5)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000700010010020100", NULL},
      {"bias(1.0)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 64, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b0000700010010040100", NULL},
      {"bias(31.0) saturating field",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 1984, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "05800cb8b00007000100107c0100", NULL},
   };

   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
      /* A shape witnessed on only one encoding names only that one. */
      if (v[i].hex_b != NULL) {
         CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &v[i].d, bytes,
                                      &length));
         CHECK(length == 14 && hex_eq(bytes, length, v[i].hex_b));
      }
      if (v[i].hex_c != NULL) {
         struct agx3_tex_sample_desc g18g = v[i].d;
         g18g.g18g = true;
         CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &g18g, bytes,
                                      &length));
         CHECK((length == 22 || length == 14) &&
               hex_eq(bytes, length, v[i].hex_c));
      }
   }

   /*
    * Apple's f83=4/f112=8 sample is a full fragment ABI, not a stage bit.
    * The A18 Pro hardware matrix with this otherwise identical implicit-LOD
    * oracle was decisive:
    *
    *    f83/f112   result
    *       0/4     exact mip 1 green (driver protocol)
    *       4/4     fixed half-gray
    *       0/8     mip 0 red
    *       4/8     fixed half-gray
    *
    * Keep both complete encodings byte-exact here.  Production descriptors
    * select DRIVER by leaving sample_abi zero; only raw Apple witness tests
    * may opt into APPLE_FRAGMENT until its paired descriptor, coordinate,
    * staging, and derivative ABI is modeled end to end.
    */
   struct agx3_tex_sample_desc abi_sample = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000000010010000100"));

   abi_sample.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000000010010001200"));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &abi_sample,
                                 bytes, &length));

   /* The same hardware differential applies to cube dimensions.  The
    * production driver ABI must keep 0/4; selecting Apple's unrelated full
    * fragment profile 4/8 produces the fixed half-gray failure. */
   abi_sample.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
   abi_sample.cube = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0001000010010000100"));
   abi_sample.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0001000010010001200"));
   abi_sample.cube = false;

   /* Fresh Metal 32023.917 argument-buffer witness: uniform halves20/24
    * carry the texture/sampler pair under the same complete Apple ABI. */
   abi_sample.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   abi_sample.texture_value = 20;
   abi_sample.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   abi_sample.sampler_value = 24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length,
                "05a00cb8b8000000930390011200802000000000"));

   abi_sample.sample_abi = (enum agx3_tex_sample_abi)2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &abi_sample,
                                 bytes, &length));

   /* The plain 2D bias form is witnessed on both encodings (see
    * test_g17g_sample_lod_witnesses for G17G); it refuses a value past the
    * 12-bit field, and refuses a zero bias, which Apple emits as the plain
    * implicit sample. */
   struct agx3_tex_sample_desc bad = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &bad, bytes,
                                &length));
   CHECK(length == 22 &&
         hex_eq(bytes, length, "0f051600458060290100010020000000008000000000"));
   bad.lod_x64 = 0;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.lod_x64 = 0x1000;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   /* A driver array's GPR layer is staged, so the bias moves to f29/f37=1
    * beside f77=2, as in Apple's texture2d_array bias(1.0) fragment
    * function with a runtime layer: 05800cb8b0000701010010040100. */
   bad.lod_x64 = 32;
   bad.array = true;
   bad.array_layer = 22;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000701010010020100"));

   bad = (struct agx3_tex_sample_desc){
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .three_d = true,
   };
   bad.array = true;
   bad.array_layer = 2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.array = false;
   bad.array_layer = 0;
   /* The 3D gradient is witnessed now -- see test_3d_lod_witnesses() -- so
    * it encodes; its six staged scalars are the packer's business. */
   bad.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   bad.lod = AGX3_TEX_LOD_AUTO;
   bad.gather = AGX3_TEX_GATHER_R;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   return 0;
}

/*
 * Explicit gradients and the per-lane LOD bias.
 *
 * The Apple bytes quoted here were regenerated with
 * `xcrun metal-nt -arch applegpu_g17p` (Metal toolchain 27.1) and decomposed
 * with the production compose tables, which only report a field vector that
 * re-composes to the same bytes. The sources are fragment shapes and the
 * compute kernel `t.sample(s, g[0], gradient2d(g[1], g[2]))`.
 *
 * The gradient sample selects f139 = 5 and is the one sample shape that does
 * *not* set the fragment marker f156: the four staged scalars are the
 * derivative the marker would otherwise request. Apple's compute kernel
 * stages six scalars with the compact copy form, all naming one staging slot,
 * into banks 0, 2, 8, 10, 12 and 14 from six consecutive source registers:
 *
 *    copy 0b10210c   bank  0 <- r16      copy 5b16210c   bank 10 <- r22
 *    copy 1b12210c   bank  2 <- r18      copy 6b18210c   bank 12 <- r24
 *    copy 4b14210c   bank  8 <- r20      copy 7b1a210c   bank 14 <- r26
 *    smp  05800cb8b0000401000010000100
 *
 * which is the bank order, the shared slot and the consecutive-source layout
 * this encoder emits. Which gradient component lands in which bank is pinned
 * independently by the fragment kernels that pass literal gradients, where
 * the staged value is the constant itself: gradient2d((0.25, 0.5),
 * (0.75, 1.0)) loads 0.25 into bank 8, 0.75 into bank 10, 0.5 into bank 12
 * and 1.0 into bank 14, so the interleave is (ddx.x, ddy.x, ddx.y, ddy.y).
 * The (1, 2)/(4, 8) and (16, 32)/(64, 128) pairs agree.
 *
 * The per-lane bias is the per-lane level with one field moved. Apple's
 * fragment pair is byte-identical apart from bytes 6 and 7:
 *
 *    t.sample(s, uv, level(in.uv.x))  05800cb8b0000001010010001200   f139 = 4
 *    t.sample(s, uv, bias(in.uv.x))   05800cb8b0000400010010001200   f139 = 3
 *
 * Both set f156. Those two carry the fragment-only f83 = 4 / f112 = 8
 * descriptor-operand profile that this encoder does not emit, so only the
 * delta between them is asserted here, on the encoder's own operand profile.
 */
static int
test_gradient_and_register_bias(void)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;

   /* Apple's own compute gradient sample, reproduced byte for byte:
    * immediate texture and sampler, coordinates in bank 0, waiting on the
    * staging slot the six copies published. */
   const struct agx3_tex_sample_desc grad_apple = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_GRADIENT, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &grad_apple, bytes,
                                &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0000401000010000100"));

   /* The same sample waiting on a different staging slot. */
   const struct agx3_tex_sample_desc grad = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_GRADIENT, .slot = 1, .wait_mask = 1u << 2,
      .coord_kill = true, .final = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &grad, bytes,
                                &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0010401000010000100"));

   /* The same gradient sample reading its texture and sampler out of the
    * bindless heap. Apple's own argument-buffer compute kernel
    * `ts.t.sample(ts.s, g[0], gradient2d(g[1], g[2]))` binds them at uniform
    * halves 20 and 24 and emits the identical six staging copies, so the
    * gradient mode is orthogonal to the descriptor operand class and there is
    * no separate heap gradient form to ground. Byte-exact. */
   struct agx3_tex_sample_desc grad_heap = grad_apple;
   grad_heap.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   grad_heap.texture_value = 20;
   grad_heap.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   grad_heap.sampler_value = 24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &grad_heap, bytes,
                                &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length,
                "05a00cb8b8000401920390010100802000000000"));

   /* The six staging copies this encoder emits, in Apple's bank order. The
    * source-operand class is the packer's own witnessed staging form (f16,
    * f23 = 2), not the compact f17 / f23 = 3 form that particular Apple
    * kernel happened to use, so these are round-trips of the shared helper
    * every already-qualified sample path uses -- what is asserted here is the
    * bank assignment and the shared slot. */
   struct {
      uint16_t dest, src, wait;
      const char *hex;
   } stage[] = {
      {0, 8, 1u << 5, "0b092f00420000880000"},
      {2, 10, 0, "1b0b2f00420000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDX_X, 4, 1u << 0, "4b052f00422000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDY_X, 0, 1u << 1, "5b012f00424000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDX_Y, 6, 0, "6b072f00420000080000"},
      {AGX3_TEX_GRADIENT_BANK_DDY_Y, 2, 0, "7b032f00420000080000"},
   };
   for (unsigned i = 0; i < sizeof(stage) / sizeof(stage[0]); ++i) {
      uint8_t copy[AGX3_TEX_COPY_MAX_LENGTH];
      CHECK(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B, stage[i].dest,
                                      stage[i].src, 3, stage[i].wait, false,
                                      true, copy, &length));
      CHECK(length == 10 && hex_eq(copy, length, stage[i].hex));
   }

   /* The per-lane level, byte-exact, and the per-lane bias derived from it. */
   struct agx3_tex_sample_desc level_reg = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1u << 1,
      .coord_kill = true, .final = true,
   };
   uint8_t level_bytes[AGX3_TEX_MAX_LENGTH], bias_bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t level_len, bias_len;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &level_reg,
                                level_bytes, &level_len));
   CHECK(level_len == 14 &&
         hex_eq(level_bytes, level_len, "05800cb830010001010010000100"));

   struct agx3_tex_sample_desc bias_reg = level_reg;
   bias_reg.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bias_reg, bias_bytes,
                                &bias_len));
   CHECK(bias_len == level_len);

   /* Exactly Apple's own delta: bytes 6 and 7 move (0x00, 0x01) ->
    * (0x04, 0x00), nothing else changes. */
   for (unsigned i = 0; i < level_len; ++i) {
      if (i == 6) {
         CHECK(level_bytes[i] == 0x00 && bias_bytes[i] == 0x04);
      } else if (i == 7) {
         CHECK(level_bytes[i] == 0x01 && bias_bytes[i] == 0x00);
      } else {
         CHECK(level_bytes[i] == bias_bytes[i]);
      }
   }

   /* The immediate bias field f29 is a *signed* 12-bit LOD in 1/64 steps.
    * Sweeping Apple's own `t.sample(s, uv, bias(k))` over these k gives
    * exactly k * 64 in two's complement; the fragment kernels differ from
    * this encoder only in the f83 = 4 / f112 = 8 descriptor-operand profile,
    * which is the trailing byte. Apple saturates at 2047 (bias 32, 63) and
    * at 2048 (bias -33, -64), so a minifloat immediate, which spans
    * [-31, 31], always lands inside the representable range.
    */
   const struct {
      int x64;
      const char *hex;
   } bias_sweep[] = {
      {   16, "05800cb8b0000780010010000100" },   /* bias  +0.015625 * 16 */
      {   32, "05800cb8b0000700010010020100" },   /* bias  +0.5           */
      {   64, "05800cb8b0000700010010040100" },   /* bias  +1             */
      {   96, "05800cb8b0000700010010060100" },   /* bias  +1.5           */
      {  128, "05800cb8b0000700010010080100" },   /* bias  +2             */
      {  256, "05800cb8b0000700010010100100" },   /* bias  +4             */
      {    1, "05800cb8b0000708010010000100" },   /* bias  +1/64          */
      { 1984, "05800cb8b00007000100107c0100" },   /* bias  +31            */
      { 4095, "05801cb8b00007f80100107e0100" },   /* bias  -1/64          */
      { 4064, "05801cb8b00007000100107e0100" },   /* bias  -0.5           */
      { 4032, "05801cb8b00007000100107c0100" },   /* bias  -1             */
      { 3968, "05801cb8b0000700010010780100" },   /* bias  -2             */
      { 3840, "05801cb8b0000700010010700100" },   /* bias  -4             */
      { 2112, "05801cb8b0000700010010040100" },   /* bias  -31            */
   };
   for (unsigned i = 0; i < sizeof(bias_sweep) / sizeof(bias_sweep[0]); ++i) {
      struct agx3_tex_sample_desc b = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = AGX3_TEX_LOD_BIAS, .slot = 1, .wait_mask = 1u << 0,
         .coord_kill = true, .final = true,
         .lod_x64 = (uint16_t)bias_sweep[i].x64,
      };
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &b, bytes,
                                   &length));
      CHECK(length == 14 && hex_eq(bytes, length, bias_sweep[i].hex));
   }

   /* textureGather: f144 = 1, the component in f145, and the f74 mask-group
    * bit every other sample sets is cleared. Apple's five fragment kernels
    * differ from these only in the f83 = 4 / f112 = 8 descriptor-operand
    * profile, which is the trailing byte:
    *
    *    t.gather(s, uv, int2(0), component::x)  05800ca490000000010000001200
    *    t.gather(s, uv, int2(0), component::y)  05800cac90000000010000001200
    *    t.gather(s, uv, int2(0), component::z)  05800cb490000000010000001200
    *    t.gather(s, uv, int2(0), component::w)  05800cbc90000000010000001200
    *
    * `t.gather(s, uv)` with no component is byte-identical to component::x.
    */
   const char *const gathers[] = {
      "05800ca490000000010000000100", "05800cac90000000010000000100",
      "05800cb490000000010000000100", "05800cbc90000000010000000100",
   };
   for (unsigned g = AGX3_TEX_GATHER_R; g <= AGX3_TEX_GATHER_A; ++g) {
      struct agx3_tex_sample_desc gd = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
         .coord_kill = true, .final = true,
         .gather = (enum agx3_tex_gather)g,
      };
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &gd, bytes,
                                   &length));
      CHECK(length == 14 &&
            hex_eq(bytes, length, gathers[g - AGX3_TEX_GATHER_R]));

      /* The array layer composes with the gather -- see
       * test_offset_witnesses() for the byte-exact
       * `texture2d_array::gather(s, uv, layer)` row -- but Metal's gather
       * takes no LOD argument, so no gather carrying one is witnessed. */
      struct agx3_tex_sample_desc array_gather = gd;
      array_gather.array = true;
      array_gather.array_layer = 22;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &array_gather,
                                   bytes, &length));
      struct agx3_tex_sample_desc bad_gather = gd;
      bad_gather.three_d = true;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_gather,
                                    bytes, &length));
      bad_gather = gd;
      bad_gather.cube = true;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_gather,
                                    bytes, &length));
      bad_gather = gd;
      bad_gather.lod = AGX3_TEX_LOD_BIAS;
      bad_gather.lod_x64 = 32;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_gather,
                                    bytes, &length));
   }

   /* Apple's depth2d sample_compare differs from the ordinary sample by
    * f143=1, f75=1 and the absence of f74.  Normalize Apple's destination
    * half6 to half0; the trailing profile byte is Mesa's hardware-qualified
    * fragment profile. */
   struct agx3_tex_sample_desc shadow = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true, .shadow = true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &shadow, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090002000010000000100"));
   struct agx3_tex_sample_desc bad_shadow = shadow;
   bad_shadow.mask = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   /* applegpu-nt 32023.917 / apple-g17p. The array comparison clears the
    * ordinary array register/profile fields and selects packed layer f77=2. */
   struct agx3_tex_sample_desc shadow_array = shadow;
   shadow_array.array = true;
   shadow_array.array_layer = 6;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &shadow_array, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090002004010000000100"));
   /* applegpu-nt 32023.917: cube comparison adds the f76=4 dimension;
    * cube-array comparison also selects the packed layer with f77=2. Both
    * deliberately retain the driver's comparison operand profile. */
   struct agx3_tex_sample_desc shadow_cube = shadow;
   shadow_cube.cube = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &shadow_cube, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090003000010000000100"));
   struct agx3_tex_sample_desc shadow_cube_array = shadow_array;
   shadow_cube_array.cube = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,
                                &shadow_cube_array, bytes, &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800ca090003004010000000100"));
   bad_shadow = shadow_array;
   bad_shadow.array_layer = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   /*
    * The descriptor operand class is orthogonal to the comparison and to the
    * packed layer. metal-nt 27.1 -arch applegpu_g17p, one fragment shader
    * sampling `array<depth2d_array<float>, 8>` through a runtime index and
    * one sampling a fixed depth2d_array, differ only in the four descriptor
    * fields and the wider endpoint they imply:
    *
    *   fixed     1d800ca090002004010000001200
    *   indexed   1dc00ca09c002004930380011200802000000000
    *
    * f143 (the comparison) and f77 = 2 (the packed layer) are byte-identical
    * across the pair. Apple's own destination half6 and trailing profile byte
    * are normalized here exactly as they are for the fixed form above, so the
    * delta between these two vectors is the delta between Apple's two. Before
    * this, every multi-texture shadow-array shader failed closed: one texture
    * instruction gets immediate descriptors from the combined-sampler pass,
    * several fall back to the indexed one, and SuperTuxKart's sunlightshadow
    * samples its cascade four times.
    */
   struct agx3_tex_sample_desc shadow_array_indexed = shadow_array;
   shadow_array_indexed.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   shadow_array_indexed.texture_value = 24;
   shadow_array_indexed.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   shadow_array_indexed.sampler_value = 28;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,
                                &shadow_array_indexed, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "05c00ca09c002004930380010100802000000000"));
   /* The register-descriptor sibling differs only in the two operand-kind
    * fields, the same way it does for every colour sample. */
   shadow_array_indexed.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   shadow_array_indexed.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,
                                &shadow_array_indexed, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "05c00ca09c002004130280010100802000000000"));
   /* An odd descriptor half is still refused. */
   bad_shadow = shadow_array;
   bad_shadow.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   bad_shadow.texture_value = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   bad_shadow = shadow_array;
   bad_shadow.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad_shadow, bytes,
                                 &length));
   /* The plain 2D comparison is witnessed on G17G in its compact form
    * (test_g17g_sample_lod_witnesses). */
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &shadow, bytes,
                                &length));
   CHECK(length == 10 && hex_eq(bytes, length, "0f071600418000210018"));

   /* The per-lane bias is witnessed on G17G too; neither it nor the gradient
    * carries an immediate LOD. */
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &bias_reg, bytes,
                                &length));
   CHECK(length == 22);
   struct agx3_tex_sample_desc bad = grad;
   bad.lod_x64 = 32;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   /* A driver array's GPR layer is staged with f77=2: exactly Apple's
    * texture2d_array gradient2d() fragment function with a runtime layer. */
   bad = grad;
   bad.array = true;
   bad.array_layer = 22;
   bad.wait_mask = 1;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000c01000010000100"));
   bad = bias_reg;
   bad.lod_x64 = 32;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   /* Likewise Apple's texture2d_array bias(b) fragment function with a
    * runtime layer, which publishes layer and bias with one PACK_SMPIR. */
   bad = bias_reg;
   bad.array = true;
   bad.array_layer = 22;
   bad.wait_mask = 1;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000800010010000100"));

   /* Apple's texture2d_array fragment functions with a runtime layer, at
    * level(2.0) and at level(b): the staged layer (f77=2) carries the level
    * in f29/f37 or the packed bank. */
   struct agx3_tex_sample_desc array_level = grad_apple;
   array_level.array = true;
   array_level.array_layer = 22;
   array_level.lod = AGX3_TEX_LOD_LEVEL;
   array_level.lod_x64 = 128;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &array_level, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000901010010080100"));
   array_level.lod = AGX3_TEX_LOD_REGISTER;
   array_level.lod_x64 = 0;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &array_level, bytes,
                                &length));
   CHECK(length == 14 &&
         hex_eq(bytes, length, "05800cb8b0000801010010000100"));

   bad = grad_apple;
   bad.array = true;
   bad.array_layer = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   bad.array_layer = 256;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &bad, bytes,
                                 &length));
   return 0;
}

/*
 * The coordinate-staging copy, replayed against the kernel witnesses
 * (t_base, x_coordreg, Ls_slots4) and the fragment fs_bufcoord witnesses.
 * The wait bit names the producer's scoreboard slot (f30 + slot - 1); a
 * slot above 1 or an AGX3c wait past slot 6 escalates the encoding point,
 * exactly as Apple's own choice does.
 */
static int
test_copy_witnesses(void)
{
   struct cvec {
      const char *name;
      enum agx_isa_encoding enc;
      uint16_t dest, src;
      uint8_t slot;
      uint16_t wait_mask;
      const char *hex;
   };
   static const struct cvec v[] = {
      {"b x_coordreg r0<-r0 wait s6", AGX_ISA_ENCODING_AGX3B, 0, 0, 1,
       1u << 5, "0b0029c4"},
      {"b x_coordreg r1<-r1", AGX_ISA_ENCODING_AGX3B, 2, 2, 1, 0,
       "1b022904"},
      {"b t_base r1<-r0", AGX_ISA_ENCODING_AGX3B, 2, 0, 1, 0, "1b002904"},
      {"b Ls #1 r0<-r4 slot2 wait s6", AGX_ISA_ENCODING_AGX3B, 0, 8, 2,
       1u << 5, "0b092f00420000840000"},
      {"b Ls #2 r1<-r5 slot2", AGX_ISA_ENCODING_AGX3B, 2, 10, 2, 0,
       "1b0b2f00420000040000"},
      {"b Ls #3 r2<-r6 slot3", AGX_ISA_ENCODING_AGX3B, 4, 12, 3, 0,
       "2b0d2f00420000080000"},
      {"b Ls #5 r4<-r0 wait s1", AGX_ISA_ENCODING_AGX3B, 8, 0, 1, 1,
       "4b002924"},
      {"b Ls #6 r5<-r1", AGX_ISA_ENCODING_AGX3B, 10, 2, 1, 0, "5b022904"},
      {"b Ls #7 r6<-r2 slot3", AGX_ISA_ENCODING_AGX3B, 12, 4, 3, 0,
       "6b052f00420000080000"},
      {"c fs_bufcoord r0<-r0 wait s8", AGX_ISA_ENCODING_AGX3C, 0, 0, 1,
       1u << 7, "2b0007022680a402"},
      {"c fs_bufcoord r1<-r1", AGX_ISA_ENCODING_AGX3C, 2, 2, 1, 0,
       "1b022904"},
      {"c t_base r1<-r0", AGX_ISA_ENCODING_AGX3C, 2, 0, 1, 0, "1b002904"},
      {"c Ls #1 r0<-r4 slot2 wait s8", AGX_ISA_ENCODING_AGX3C, 0, 8, 2,
       1u << 7, "2b0007122681a402"},
      {"c Ls #2 r1<-r5 slot2", AGX_ISA_ENCODING_AGX3C, 2, 10, 2, 0,
       "330207122681a402"},
      {"c Ls #3 r2<-r6 slot3", AGX_ISA_ENCODING_AGX3C, 4, 12, 3, 0,
       "2300071a2682a40a"},
      {"c Ls #5 r4<-r0 wait s1", AGX_ISA_ENCODING_AGX3C, 8, 0, 1, 1,
       "4b002924"},
      {"c Ls #7 r6<-r2 slot3", AGX_ISA_ENCODING_AGX3C, 12, 4, 3, 0,
       "2300070a2682a41a"},
   };

   uint8_t bytes[AGX3_TEX_COPY_MAX_LENGTH];
   uint8_t length;
   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
      CHECK(agx3_tex_stage_copy(v[i].enc, v[i].dest, v[i].src, v[i].slot,
                                v[i].wait_mask, bytes, &length));
      if (!hex_eq(bytes, length, v[i].hex)) {
         fprintf(stderr, "copy witness: %s\n", v[i].name);
         return 1;
      }
      checks++;
   }

   /* Current-Xcode applegpu_g17p high-pressure texture kernel. Apple places
    * the float2 at halves 88/90, proving that the old <=60 direct-coordinate
    * policy does not apply to the staged COPY source. */
   CHECK(agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 88, 3, 1u << 2,
                             bytes, &length));
   CHECK(hex_eq(bytes, length, "0b592f00428000080000"));
   CHECK(agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 2, 90, 3, 0, bytes,
                             &length));
   CHECK(hex_eq(bytes, length, "1b5b2f00420000080000"));

   /* COPY's source field is exactly seven bits. Exercise every legal aligned
    * value, then prove that bit 7 refuses instead of aliasing half 128 to 0. */
   for (unsigned src = 0; src < 128; src += 2)
      CHECK(agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, src, 1, 0,
                                bytes, &length));

   memset(bytes, 0xa5, sizeof(bytes));
   length = 0xff;
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 128, 1, 0,
                              bytes, &length));
   CHECK(length == 0);
   for (unsigned i = 0; i < sizeof(bytes); ++i)
      CHECK(bytes[i] == 0);
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3C, 0, 128, 1, 0,
                              bytes, &length));
   CHECK(length == 0);
   return 0;
}

/* AGX3B's 14-byte immediate-descriptor SMP names coordinate banks 0..60. A
 * relocated sample beyond that (the ninth tap of a 3x3 PCF kernel on a
 * shadow array lands on bank 64) takes the 20-byte form, which names the
 * bank and keeps every other field of the short form. */
static int
test_sample_upper_coordinate_banks(void)
{
   for (unsigned variant = 0; variant < 4; ++variant) {
      const bool shadow = variant & 1, array = variant & 2;
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 60, .width = AGX3_TEX_W32,
         .mask = shadow ? 1 : 15, .lod = AGX3_TEX_LOD_AUTO, .slot = 1,
         .wait_mask = 1u << 5, .final = true, .sample_group_end = true,
         .shadow = shadow, .array = array, .array_layer = array ? 44 : 0,
         .sampler = 1, .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      uint8_t short_form[AGX3_TEX_MAX_LENGTH], long_form[AGX3_TEX_MAX_LENGTH];
      uint8_t short_len = 0, long_len = 0;
      uint64_t short_fields[AGX3_MAX_FIELDS], long_fields[AGX3_MAX_FIELDS];
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, short_form,
                                   &short_len));
      CHECK(short_len == 14);
      CHECK(agx3_decode(&agx3b_table, short_form, short_len, AGX3_MN_smp,
                        short_fields) >= 0);
      CHECK(short_fields[19] == 60);

      for (unsigned coord = 64; coord <= 252; coord += 4) {
         d.coord = coord;
         CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, long_form,
                                      &long_len));
         CHECK(long_len == 20);
         CHECK(agx3_decode(&agx3b_table, long_form, long_len, AGX3_MN_smp,
                           long_fields) >= 0);
         CHECK(long_fields[19] == coord);
         /* Destination, slot, mask, dimension, layer, comparison, sampler
          * and wait fields are unchanged by the longer form. */
         static const unsigned same[] = {2, 10, 12, 26, 31, 62 + 5, 75, 76,
                                         77, 123, 143, 147};
         for (unsigned i = 0; i < sizeof(same) / sizeof(same[0]); ++i)
            CHECK(long_fields[same[i]] == short_fields[same[i]]);
      }
   }
   return 0;
}

static int
test_sample_fail_closed(void)
{
   const struct agx3_tex_sample_desc good = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   struct agx3_tex_sample_desc d;

   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &good, bytes,
                                &length));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX2, &good, bytes,
                                 &length));

   d = good;
   d.mask = 0; /* empty samples have no native result */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.sampler = 16;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.lod_x64 = 64; /* AUTO carries no level operand */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 0x1000;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.coord = 2;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.wait_mask = 3;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.offset = true;
   /* Offset (0, 0) is the plain form in native on both encodings. The G17G
    * 2D immediate offset itself is witnessed (test_g17g_sample_lod_witnesses)
    * with the same f43/f51/f141 fields as AGX3B. */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   d.offset_xyz = 0xf1;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(length == 22);
   /* The driver stages the layer, leaving separate immediate fields for
    * level and offset. The direct Apple-fragment profile still collides. */
   d.lod = AGX3_TEX_LOD_LEVEL;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.lod = AGX3_TEX_LOD_AUTO;
   d.array = true;
   d.array_layer = 2;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.lod = AGX3_TEX_LOD_LEVEL;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
   d.lod = AGX3_TEX_LOD_AUTO;
   d.array = false;
   d.array_layer = 0;
   /* Native runtime-indexed descriptors carry the same offset field. */
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 2;
   d.sampler_value = 4;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.offset_xyz = 0x100;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.three_d = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.offset_xyz = 0x1000;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   /* G17G terminates its final fragment sample (a lone immediate level is
    * witnessed with f156) and, like G17P, clears it on the earlier plain 2D
    * members of a group (gpu/m5-shadow-2d-reference.metal); a member marked
    * both nonterminal and terminal is contradictory on either. */
   d = good;
   d.sample_group_end = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   d = good;
   d.sample_group_nonterminal = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   d.sample_group_end = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   d.sample_group_end = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = good;
   d.offset_xyz = 0xf1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));

   /* Stage-copy edges: odd registers, bad slots, wide waits. */
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 1, 0, 1, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 3, 1, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 0, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 7, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 1, 1u << 6,
                              bytes, &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3C, 0, 0, 9, 0, bytes,
                              &length));
   CHECK(!agx3_tex_stage_copy(AGX_ISA_ENCODING_AGX3B, 0, 0, 1, 3, bytes,
                              &length));
   return 0;
}

/*
 * The 2D-array texel fetch. Both vectors are Apple's own bytes, from
 * `texture2d<float, access::read>::read()` and
 * `texture2d_array<float, access::read>::read()` compiled by
 * `metal-nt -arch applegpu_g16g`; they differ in the array selector alone.
 */
static int
test_array_fetch(void)
{
   static const uint8_t plain_2d[14] = {0x05, 0x80, 0x0c, 0xb8, 0x30, 0x01,
                                        0x17, 0x01, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x00};
   static const uint8_t array_2d[14] = {0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00,
                                        0x97, 0x01, 0x00, 0x00, 0x00, 0x00,
                                        0x01, 0x00};

   const struct agx3_tex_fetch_desc plain = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1u << 1, .coord_kill = true,
      .final = true, .array = false,
   };
   const struct agx3_tex_fetch_desc arr = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1u << 0, .coord_kill = true,
      .final = true, .array = true,
   };

   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length = 0;

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &plain, bytes,
                               &length));
   CHECK(length == 14);
   CHECK(memcmp(bytes, plain_2d, 14) == 0);

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &arr, bytes, &length));
   CHECK(length == 14);
   CHECK(memcmp(bytes, array_2d, 14) == 0);

   /* The array flag survives a decode, and the two forms stay distinct. */
   struct agx3_tex_fetch_desc back;
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, array_2d, 14, &back));
   CHECK(back.array);
   CHECK(same(&back, &arr));
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, plain_2d, 14, &back));
   CHECK(!back.array);
   CHECK(same(&back, &plain));

   return 0;
}

/* G17P texture3d::read. The direct instruction is the complete HW-validated
 * c_read_3d bundle from EXP-0016. The register-indexed level-zero and
 * uniform-indexed level-one instructions were freshly compiled for
 * applegpu_g17p from an own-MSL argument-buffer texture array (native SHA256
 * bb14d01a8ef3...). The first is the production Vulkan descriptor shape
 * (killed r3 heap index plus immediate texel-fetch sampler zero). */
static int
test_3d_fetch(void)
{
   static const uint8_t direct[14] = {
      0x05,0x80,0x0c,0xb8,0xb0,0x00,0x79,0x00,0x00,0x00,0x00,0x00,0x01,0x00,
   };
   static const uint8_t indirect[20] = {
      0x05,0xb0,0x0c,0xb8,0xb0,0x00,0x79,0x00,0x00,0x02,
      0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x00,0x00,
   };
   static const uint8_t uniform_lod1[20] = {
      0x05,0xa0,0x0c,0xb8,0xb0,0x00,0x79,0x00,0x90,0x02,
      0x80,0x04,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00,
   };
   struct agx3_tex_fetch_desc d = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1, .coord_kill = true,
      .final = true, .three_d = true,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length = 0;
   struct agx3_tex_fetch_desc back;

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                               &length));
   CHECK(length == sizeof(direct) &&
         hex_eq(bytes, length, "05800cb8b0007900000000000100"));
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, direct,
                               sizeof(direct), &back));
   CHECK(same(&d, &back));

   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 6;
   d.texture_kill = true;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                               &length));
   CHECK(length == sizeof(indirect) &&
         hex_eq(bytes, length,
                "05b00cb8b0007900000280000100002800000000"));
   CHECK(agx3_tex_decode_fetch(AGX_ISA_ENCODING_AGX3B, indirect,
                               sizeof(indirect), &back));
   CHECK(same(&d, &back));

   d.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 20;
   d.texture_kill = false;
   d.lod_x64 = 64;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                               &length));
   CHECK(length == sizeof(uniform_lod1) &&
         hex_eq(bytes, length,
                "05a00cb8b0007900900280040100002000000000"));

   d.array = true;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                &length));
   d.array = false;
   d.multisample = true;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                &length));
   d.multisample = false;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 6;
   d.lod_x64 = 0;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_value = 2;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                &length));
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   d.sampler_value = 0;
   d.register_lod = true;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                               &length));
   d.lod_x64 = 64;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                &length));
   return 0;
}


/*
 * Bindless operand kinds. The texture and sampler each select one of three
 * consecutive fields by kind (95/96/97 and 124/125/126), and immediate vs
 * non-immediate also selects the encoding point: ep3 for immediate, ep4 for
 * register and uniform. All three kinds are witnessed byte-exactly from
 * Apple shaders captured on an A18 Pro; see test-agx3-smp-witness.c.
 */
static int
test_bindless_operand_kinds(void)
{
   struct agx3_tex_fetch_desc d;
   uint8_t out[AGX3_TEX_MAX_LENGTH];
   uint8_t len;

#define FETCH_BASE()                                                           \
   do {                                                                        \
      memset(&d, 0, sizeof(d));                                                \
      d.dest = 8;                                                              \
      d.coord = 16;                                                            \
      d.width = 1;                                                             \
      d.mask = 15;                                                             \
      d.slot = 1;                                                              \
      d.array = true;                                                          \
      len = 0;                                                                 \
   } while (0)

   /* Immediate operands keep the 14-byte ep3 form. A caller that never heard
    * of operand kinds leaves them zero and must encode exactly as before. */
   FETCH_BASE();
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 14);

   /* Register descriptors: the ep4 form, 20 bytes. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 0;
   d.sampler_value = 4;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 20);

   /* Uniform descriptors: also ep4. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 16;
   d.sampler_value = 20;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 20);

   /* Fresh native read() uses an indirect texture with an immediate
    * sampler index. Each operand has its own immediate/indirect class. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));

   /* Fresh MetalToolchain v27.1.5194.15 vertex argument-buffer read. The
    * dynamic VS object (SHA256 2838ec963f7e...) uses SMP.EP4 with its
    * zero-extended texture heap index in uniform half16 and the native read
    * sampler at immediate zero. Its direct texture0 sibling is the ordinary
    * 14-byte EP3 form; this 20-byte instruction is their only SMP delta. */
   memset(&d, 0, sizeof(d));
   d.dest = 0;
   d.coord = 0;
   d.width = AGX3_TEX_W32;
   d.mask = 15;
   d.slot = 1;
   d.wait_mask = 1;
   d.coord_kill = true;
   d.final = true;
   d.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 16;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, out, &len));
   CHECK(len == 20 &&
         hex_eq(out, len,
                "05800cb8b0001701900280000100002000000000"));

   /* Only AGX3b is witnessed for the non-immediate kinds. */
   FETCH_BASE();
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, out, &len));

#undef FETCH_BASE
   return 0;
}

/* Fresh offline G17P Metal argument-buffer samples, September 5 2026.
 * Native UMD 0x33dcc8/0x33db40 stores zero-extended heap indices into these
 * 64-bit argument slots; their contents are not descriptor addresses.
 */
static int
test_indexed_sample_witnesses(void)
{
   const uint8_t vertex_level_bytes[] = {
      0x05,0xa0,0x0c,0xb8,0xb8,0x00,0x09,0x00,0x92,0x03,
      0x90,0x01,0x01,0x00,0x80,0x20,0x00,0x00,0x00,0x00};
   const uint8_t vertex_mixed_bytes[] = {
      0x05,0x80,0x0c,0xb8,0xb0,0x00,0x09,0x00,0x90,0x02,
      0x90,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
   const uint8_t uniform_bytes[] = {
      0x05,0xe0,0x0c,0xb8,0xb0,0x00,0x00,0x00,0x83,0x03,
      0x90,0x01,0x01,0x00,0x80,0x20,0x00,0x00,0x00,0x00};
   const uint8_t register_bytes[] = {
      0x05,0x80,0x0c,0xb8,0x34,0x00,0x00,0x00,0x01,0x02,
      0x90,0x01,0x01,0x00,0xa0,0x28,0x00,0x00,0x40,0x00};
   const uint8_t uniform_array_bytes[] = {
      0x05,0xc0,0x0c,0xb8,0xbc,0x00,0x00,0x04,0x93,0x03,
      0x90,0x01,0x01,0x00,0x80,0x20,0x00,0x00,0x00,0x00};
   struct agx3_tex_sample_desc d = {
      .width=AGX3_TEX_W32, .mask=15, .slot=1, .wait_mask=1,
      .coord_kill=true, .final=true, .lod=AGX3_TEX_LOD_AUTO,
      .texture_kind=AGX3_TEX_OPERAND_UNIFORM,
      .sampler_kind=AGX3_TEX_OPERAND_UNIFORM,
      .texture_value=12, .sampler_value=16,
   };
   uint8_t out[AGX3_TEX_MAX_LENGTH], len;
   /* Fresh v27.1 G17P vertex dynamic argument-buffer sample, object SHA256
    * 49cab707568f.... Texture/sampler heap indices are uniform20/uniform24;
    * the explicit vertex level-zero selector is otherwise the direct sample
    * sibling's exact SMP shape. */
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.texture_value = 20;
   d.sampler_value = 24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len == sizeof(vertex_level_bytes));
   CHECK(memcmp(out, vertex_level_bytes, len) == 0);
   /* The independently compiled mixed sibling (SHA256 6fa7259810a5...)
    * keeps texture uniform16 indirect while selecting sampler0 immediate.
    * Operand class and size selectors are independent in SMP.EP4. */
   d.texture_value = 16;
   d.sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   d.sampler_value = 0;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len == sizeof(vertex_mixed_bytes));
   CHECK(memcmp(out, vertex_mixed_bytes, len) == 0);
   d.lod = AGX3_TEX_LOD_AUTO;
   d.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 12;
   d.sampler_value = 16;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len==sizeof(uniform_bytes));
   CHECK(memcmp(out,uniform_bytes,len)==0);
   d.cube=true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(hex_eq(out,len,
                "05e00cb8b0001000830390010100802000000000"));
   d.cube=false;
   d.array=true;
   d.array_layer=34;
   d.texture_value=24;
   d.sampler_value=28;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len==sizeof(uniform_array_bytes));
   CHECK(memcmp(out,uniform_array_bytes,len)==0);
   d.array=false;
   d.array_layer=0;
   d.texture_kind=d.sampler_kind=AGX3_TEX_OPERAND_REGISTER;
   d.texture_value=0; d.sampler_value=4; d.wait_mask=32;
   d.texture_kill=d.sampler_kill=true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(len==sizeof(register_bytes));
   CHECK(memcmp(out,register_bytes,len)==0);
   d.cube=true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(hex_eq(out,len,
                "05800cb834001000010290010100a02800004000"));
   d.cube=false;
   d.array=true;
   d.array_layer=24;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   CHECK(hex_eq(out,len,
                "05800cb834000004010290010100a02800004000"));
   d.array=false;
   d.array_layer=0;
   d.texture_value=1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   d.texture_value=0; d.texture_kind=AGX3_TEX_OPERAND_UNIFORM;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
   return 0;
}

static int
test_sample_mask_domain(void)
{
   /* Native mask_1..mask_15, every nonempty packed result subset. */
   static const uint8_t native[15][14] = {
      {0x05,0x80,0x0c,0xa0,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa0,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb8,0x90,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa0,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa8,0xb0,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb8,0x90,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xa0,0xb0,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb0,0xb0,0x00,0x00,0x00,0x01,0x00,0x50,0x00,0x01,0x00},
      {0x05,0x80,0x0c,0xb8,0xb0,0x00,0x00,0x00,0x01,0x00,0x10,0x00,0x01,0x00}
   };
   for (unsigned mask=1; mask<16; ++mask) {
      struct agx3_tex_sample_desc d={
         .width=AGX3_TEX_W32,.mask=mask,.slot=1,.wait_mask=1,
         .coord_kill=true,.final=true,.lod=AGX3_TEX_LOD_AUTO,
      };
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==14);
      CHECK(memcmp(out,native[mask-1],len)==0);
   }
   return 0;
}

static int
test_indexed_fetch_native(void)
{
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xe0,0x0c,0xb8,0xb0,0x00,0x17,0x01,0x80,0x03,0x80,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=3,.texture_value=12,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0x30,0x00,0x17,0x01,0x00,0x03,0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x40,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=2,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=32,.coord_kill=true,.array=0,
         .final=1,.texture_kill=1};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xe0,0x0c,0xb8,0xb0,0x00,0x97,0x01,0x80,0x03,0x80,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=3,.texture_value=12,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=1,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xc0,0x0c,0xb8,0xb0,0x00,0x97,0x01,0x00,0x03,0x80,0x00,0x01,0x00,0x00,0x28,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=2,.texture_value=8,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=1,
         .final=1,.texture_kill=1};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0xe0,0x0c,0x38,0xb1,0x00,0x17,0x01,0x80,0x02,0x80,0x00,0x01,0x00,0x00,0x20,0x00,0x00,0x00,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=3,.texture_value=12,.sampler_value=2,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=0,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=1,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=2,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=3,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x0d,0x80,0x0c,0xb0,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=2,.coord=0,.width=AGX3_TEX_W32,
         .mask=4,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=5,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=6,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb0,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=7,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x0d,0x80,0x0c,0xb8,0x90,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=2,.coord=0,.width=AGX3_TEX_W32,
         .mask=8,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=9,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb0,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=10,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa8,0xb0,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=11,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0x90,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=12,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xa0,0xb0,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=13,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb0,0xb0,0x00,0x17,0x01,0x00,0x01,0x40,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=14,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   {
      const uint8_t native[]={0x05,0x80,0x0c,0xb8,0xb0,0x00,0x17,0x01,0x00,0x01,0x00,0x00,0x01,0x00};
      struct agx3_tex_fetch_desc d={.dest=0,.coord=0,.width=AGX3_TEX_W32,
         .mask=15,.texture_kind=1,.texture_value=0,.sampler_value=1,
         .slot=1,.wait_mask=1,.coord_kill=true,.array=0,
         .final=1,.texture_kill=0};
      uint8_t out[AGX3_TEX_MAX_LENGTH],len;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B,&d,out,&len));
      CHECK(len==sizeof(native));
      CHECK(!memcmp(out,native,len));
      CHECK(roundtrip(AGX_ISA_ENCODING_AGX3B,&d)==0);
   }
   return 0;
}

static int
test_staging_source_lifetime(void)
{
   /* Fresh complete native read kernels use coordinates both for TEX and
    * later output addressing/XOR. Their staging producers must not discard. */
   const uint8_t golden[2][2][4]={
      {{0x0b,0x08,0x21,0x24},{0x1b,0x0a,0x21,0x44}},
      {{0x0b,0x88,0x21,0x24},{0x1b,0x8a,0x21,0x44}},
   };
   for(unsigned cache=0;cache<2;cache++)for(unsigned xy=0;xy<2;xy++) {
      uint8_t bytes[AGX3_TEX_COPY_MAX_LENGTH],len;
      CHECK(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B,2*xy,8+2*xy,
         1,1u<<xy,cache,false,bytes,&len));
      CHECK(len==4 && !memcmp(bytes,golden[cache][xy],4));
   }
   uint8_t rejected[AGX3_TEX_COPY_MAX_LENGTH], length=0xff;
   memset(rejected,0xa5,sizeof(rejected));
   CHECK(!agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3B,0,8,1,0,
      true,true,rejected,&length));
   CHECK(length==0);
   for(unsigned i=0;i<sizeof(rejected);i++)CHECK(rejected[i]==0);
   return 0;
}

static int
test_register_lod(void)
{
   /* Fresh native G17P fragment bodies, float-to-half and retained half LOD. */
   const uint8_t dead[] = {0x17,0x05,0x54,0x06,0,0,0,0x04,0xf0,0xc0,0x08,0};
   const uint8_t live[] = {0x17,0x05,0x56,0x06,0,0x10,0,0x04,0x70,0xc0,0x08,0};
   const uint8_t sample[] = {0x05,0x80,0x0c,0xb8,0xb0,0,0,1,1,0,0x10,0,1,0};
   uint8_t b[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B,0,1,0,false,true,b,&len));
   CHECK(len == sizeof(dead) && !memcmp(b,dead,len));
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B,8,1,1u<<5,false,false,b,&len));
   CHECK(len == sizeof(live) && !memcmp(b,live,len));
   struct agx3_tex_sample_desc d = {.dest=0,.coord=0,.width=AGX3_TEX_W32,
      .mask=15,.lod=AGX3_TEX_LOD_REGISTER,.slot=1,.wait_mask=1,
      .coord_kill=true,.final=true};
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B,&d,b,&len));
   CHECK(len == sizeof(sample) && !memcmp(b,sample,len));
   /* Independently decode the full eight-bit field, including the upper
    * bank. Adjacent CACHE/DISCARD and scoreboard bits must stay intact. */
   for (unsigned src = 0; src < 256; ++src) {
      CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B,src,1,0,false,true,b,&len));
      CHECK(len == 12);
      CHECK(((b[5] >> 1) | ((b[6] & 1) << 7)) == src);
      b[5] &= 1;
      b[6] &= ~1;
      CHECK(!memcmp(b,dead,sizeof(dead)));
   }
   return 0;
}

/*
 * The texel fetch's level.
 *
 * Three `metal-nt -arch applegpu_g17p` compute kernels differing only in the
 * level argument to texture2d<float>::read(), from one source file:
 *
 *   read(gid, 0)        f37 = 1, f139 = 2, f29 = 0
 *   read(gid, 2)        f37 = 1, f139 = 2, f29 = 128     -- the level x 64
 *   read(gid, lod[i])   f37 absent, f139 = 4, preceded by a PACK_SMPIR
 *
 * The same delta appears between two fragment programs in texlodfs.metal, so
 * it is not a compute-only shape; those bytes additionally carry the
 * fragment-only f83/f112/f124 selectors this encoder does not model, which is
 * why the vectors here are the compute pair.
 */
static int
test_fetch_lod(void)
{
   const uint8_t lod0[] = {0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x17,
                           0x01, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00};
   const uint8_t lod2[] = {0x05, 0x80, 0x0c, 0xb8, 0xb0, 0x00, 0x17,
                           0x01, 0x00, 0x00, 0x00, 0x08, 0x01, 0x00};
   const uint8_t dynamic[] = {0x05, 0x80, 0x0c, 0xb8, 0x30, 0x01, 0x80,
                              0x24, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00};

   struct agx3_tex_fetch_desc d = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .texture = 0, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true,
   };
   uint8_t b[AGX3_TEX_MAX_LENGTH], len = 0;

   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   CHECK(len == sizeof(lod0) && !memcmp(b, lod0, len));

   d.lod_x64 = 2 * 64;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   CHECK(len == sizeof(lod2) && !memcmp(b, lod2, len));

   /* The per-lane form waits on the staging slot the PACK_SMPIR published,
    * which is why Apple's own program moves the wait from f62 to f63. */
   d.lod_x64 = 0;
   d.register_lod = true;
   d.wait_mask = 1u << 1;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   CHECK(len == sizeof(dynamic) && !memcmp(b, dynamic, len));

   /* An immediate level and a per-lane level are mutually exclusive. The
    * per-lane form's AGX3c witness is the fragment read(coord, level) in
    * gpu/m5-fetch-lod-reference.metal, checked byte-exact below. */
   d.lod_x64 = 64;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));
   d.lod_x64 = 0;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, b, &len));
   CHECK(len == 10);
   d.register_lod = false;
   d.lod_x64 = 2048;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, b, &len));

   /* The level staged for a texel fetch is an INTEGER: PACK_SMPIR's f74 is
    * the only field that moves between the fragment fetch and the fragment
    * filtered sample in texlodtype.metal. */
   const uint8_t integer_level[] = {0x17, 0x05, 0x54, 0x06, 0x00, 0x08,
                                    0x00, 0x04, 0xf0, 0xc4, 0x0c, 0x00};
   const uint8_t float_level[] = {0x17, 0x05, 0x54, 0x06, 0x00, 0x0c,
                                  0x00, 0x04, 0xf0, 0xc0, 0x08, 0x00};
   CHECK(agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 4, 2, 0, false,
                                  true, AGX3_TEX_LEVEL_INTEGER, b, &len));
   CHECK(len == sizeof(integer_level) && !memcmp(b, integer_level, len));
   CHECK(agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 6, 1, 0, false,
                                  true, AGX3_TEX_LEVEL_FLOAT, b, &len));
   CHECK(len == sizeof(float_level) && !memcmp(b, float_level, len));

   /* The default entry point stays the float form every existing caller
    * relies on. */
   uint8_t plain[AGX3_TEX_MAX_LENGTH], plain_len = 0;
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3B, 6, 1, 0, false, true,
                            plain, &plain_len));
   CHECK(plain_len == len && !memcmp(plain, b, len));

   /* No other level type is witnessed. */
   CHECK(!agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 4, 2, 0, false,
                                   true, (enum agx3_tex_level_kind)0, b,
                                   &len));
   CHECK(!agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3B, 4, 2, 0, false,
                                   true, (enum agx3_tex_level_kind)4, b,
                                   &len));

   /* Relocatable 8-half banks 0..248 on AGX3B carry bank + 6 in f5 for both
    * scalar typed LOD and combined array layer + LOD staging. */
   uint64_t f0[AGX3_MAX_FIELDS], fb[AGX3_MAX_FIELDS];
   CHECK(agx3_decode(&agx3b_table, float_level, sizeof(float_level),
                     AGX3_MN_pack_smpir, f0) >= 0);
   CHECK(f0[5] == 6);
   static const uint16_t banks[] = {0, 8, 16, 64, 248};
   for (unsigned i = 0; i < sizeof(banks) / sizeof(banks[0]); ++i) {
      CHECK(agx3_tex_stage_typed_lod_at(AGX_ISA_ENCODING_AGX3B, banks[i], 6, 1,
                                        0, false, true, AGX3_TEX_LEVEL_FLOAT,
                                        b, &len));
      CHECK(len == 12 &&
            agx3_decode(&agx3b_table, b, len, AGX3_MN_pack_smpir, fb) >= 0);
      CHECK(fb[5] == banks[i] + 6);
      fb[5] = f0[5];
      CHECK(memcmp(fb, f0, sizeof(f0)) == 0);

      CHECK(agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3B, banks[i], 8, 9,
                                        1, 0, false, true, false, true, b,
                                        &len));
      CHECK(len == 12 &&
            agx3_decode(&agx3b_table, b, len, AGX3_MN_pack_smpir, fb) >= 0);
      CHECK(fb[5] == banks[i] + 6);
   }
   CHECK(!agx3_tex_stage_typed_lod_at(AGX_ISA_ENCODING_AGX3B, 4, 6, 1, 0, false,
                                      true, AGX3_TEX_LEVEL_FLOAT, b, &len));
   CHECK(!agx3_tex_stage_typed_lod_at(AGX_ISA_ENCODING_AGX3B, 256, 6, 1, 0,
                                      false, true, AGX3_TEX_LEVEL_FLOAT, b,
                                      &len));
   CHECK(!agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3B, 4, 8, 9, 1, 0,
                                      false, true, false, true, b, &len));
   CHECK(!agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3B, 256, 8, 9, 1, 0,
                                      false, true, false, true, b, &len));

   /* Long ep4 escalation for register-LOD / register-bias samples past bank 60. */
   for (unsigned m = 0; m < 2; ++m) {
      struct agx3_tex_sample_desc sd = {
         .dest = 8,
         .coord = 64,
         .width = AGX3_TEX_W32,
         .mask = 15,
         .lod = m ? AGX3_TEX_LOD_BIAS_REGISTER : AGX3_TEX_LOD_REGISTER,
         .slot = 1,
         .wait_mask = 1,
         .final = true,
         .sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER,
      };
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &sd, b, &len));
      CHECK(len == 20 &&
            agx3_decode(&agx3b_table, b, len, AGX3_MN_smp, fb) >= 0);
      CHECK(fb[19] == 64);
   }
   return 0;
}

static int
test_register_array_layer(void)
{
   /* Native direct/uniform sample: discarded r2 publishes on slot 1. */
   const uint8_t native_slot1[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x08,
      0x00, 0x00, 0xf0, 0xc0, 0x04, 0x00,
   };
   /* Native register-indexed sample: discarded r6 publishes on slot 6. */
   const uint8_t native_register[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x18,
      0x00, 0x00, 0xf0, 0xe4, 0x04, 0x00,
   };
   /* applegpu-nt 32023.917 depth2d_array::sample_compare: discarded r3
    * publishes the dynamic layer on slot 1. */
   const uint8_t native_shadow_array[] = {
      0x17, 0x05, 0x54, 0x06, 0x00, 0x0c,
      0x00, 0x00, 0xf0, 0xc0, 0x04, 0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 4, 1, 0,
                                    false, true, bytes, &len));
   CHECK(len == sizeof(native_slot1) && !memcmp(bytes, native_slot1, len));
   CHECK(memcmp(bytes,
                (const uint8_t[]){0x17,0x05,0x54,0x06,0,0x08,0,4,
                                  0xf0,0xc0,0x08,0}, len));
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 12, 6, 0,
                                    false, true, bytes, &len));
   CHECK(len == sizeof(native_register) &&
         !memcmp(bytes, native_register, len));
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3B, 6, 1, 0,
                                    false, true, bytes, &len));
   CHECK(len == sizeof(native_shadow_array) &&
         !memcmp(bytes, native_shadow_array, len));
   return 0;
}

static int
test_cube_coordinate_stage(void)
{
   /* Exact own-MSL texturecube CEM sequence and face publication. */
   static const uint8_t cem[3][12] = {
      {0x17,0x01,0x54,0x00,0x03,0x04,0x0d,0x52,0x54,0x21,0x92,0x00},
      {0x17,0x01,0x54,0x07,0x03,0x04,0x0d,0x52,0x54,0x21,0x92,0x04},
      {0x17,0x01,0x54,0x02,0x03,0x04,0x0c,0x50,0x50,0x2f,0x92,0x08},
   };
   static const uint8_t rcp[10] = {
      0xaf,0x00,0x54,0x00,0x03,0x00,0x10,0x48,0x20,0x00,
   };
   static const uint8_t native[] = {
      0x17,0x05,0x54,0x06,0x00,0x04,
      0x00,0x04,0xf0,0xc0,0x00,0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   for (unsigned axis = 0; axis < 3; ++axis) {
      CHECK(agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3B, 4, 6, 20,
                                   axis, axis == 2, bytes, &len));
      CHECK(len == sizeof(cem[axis]) && !memcmp(bytes, cem[axis], len));
   }
   CHECK(agx3_tex_cube_cem_rcp(AGX_ISA_ENCODING_AGX3B, bytes, &len));
   CHECK(len == sizeof(rcp) && !memcmp(bytes, rcp, len));
   static const uint8_t finalize[2][12] = {
      {0x09,0x0f,0x2e,0x81,0x23,0xa0,0x42,0x00,0x00,0x02,0x02,0x00},
      {0x19,0x05,0x3e,0x01,0x23,0xa0,0x42,0x00,0x00,0x02,0x62,0x00},
   };
   for (unsigned component = 0; component < 2; ++component) {
      CHECK(agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3B, component,
                                       component == 1, bytes, &len));
      CHECK(len == sizeof(finalize[component]) &&
            !memcmp(bytes, finalize[component], len));
   }

   /* Fresh dynamic argument-buffer cube witness. CEM uses allocator-owned
    * halves8/16/12, its reciprocal is in-place, and both finalizers join the
    * descriptor/face transaction on slot6. */
   static const uint8_t dynamic_cem[3][12] = {
      {0x17,0x01,0x54,0x04,0x03,0x0c,0x1d,0x5a,0x54,0x21,0x92,0x00},
      {0x17,0x01,0x54,0x08,0x03,0x0c,0x1d,0x5a,0x54,0x21,0x92,0x04},
      {0x17,0x01,0x54,0x06,0x03,0x0c,0x1c,0x58,0x50,0x2f,0x92,0x08},
   };
   const uint16_t dynamic_dest[3] = {8, 16, 12};
   for (unsigned axis = 0; axis < 3; ++axis) {
      CHECK(agx3_tex_cube_cem_calc_to(
         AGX_ISA_ENCODING_AGX3B, 12, 14, 22, dynamic_dest[axis], axis,
         axis == 2, bytes, &len));
      CHECK(len == sizeof(dynamic_cem[axis]) &&
            !memcmp(bytes, dynamic_cem[axis], len));
   }
   static const uint8_t dynamic_rcp[10] = {
      0xaf,0x00,0x54,0x08,0x03,0x10,0x10,0x48,0x20,0x00,
   };
   CHECK(agx3_tex_cube_cem_rcp_to(AGX_ISA_ENCODING_AGX3B, 8, bytes, &len));
   CHECK(len == sizeof(dynamic_rcp) && !memcmp(bytes, dynamic_rcp, len));
   static const uint8_t dynamic_finalize[2][12] = {
      {0x09,0x11,0x2e,0x89,0x23,0xa0,0x42,0x00,0x00,0x16,0x02,0x00},
      {0x19,0x0d,0x3e,0x09,0x23,0xa0,0x42,0x00,0x00,0x16,0x02,0x00},
   };
   for (unsigned component = 0; component < 2; ++component) {
      CHECK(agx3_tex_cube_cem_finalize_to(
         AGX_ISA_ENCODING_AGX3B, component, 8,
         dynamic_dest[component + 1], 6, bytes, &len));
      CHECK(len == sizeof(dynamic_finalize[component]) &&
            !memcmp(bytes, dynamic_finalize[component], len));
   }
   /* In the application shape XYZ remains live after sampling. Apple's
    * terminal CEM axis then carries neither cache nor discard hints. */
   static const uint8_t retained_axis2[12] = {
      0x17,0x01,0x54,0x09,0x03,0x0e,0x20,0x30,0x50,0x21,0x92,0x08,
   };
   CHECK(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3B, 14, 16, 12,
                                    18, 2, false, bytes, &len));
   CHECK(len == sizeof(retained_axis2) &&
         !memcmp(bytes, retained_axis2, len));
   CHECK(agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3B, 2, 1, 0,
                               false, true, bytes, &len));
   CHECK(len == sizeof(native) && !memcmp(bytes, native, len));
   /* applegpu-nt 32023.917 depthcube_array::sample_compare combines CEM's
    * face and discarded layer half3 in one distinct PACK_SMPIR. */
   static const uint8_t native_shadow_array[] = {
      0x17,0x05,0x54,0x06,0x00,0x06,
      0x08,0x00,0xf0,0xc0,0x29,0x00,
   };
   CHECK(agx3_tex_stage_cube_array_layer(
      AGX_ISA_ENCODING_AGX3B, 2, 3, 1, 0, false, true, bytes, &len));
   CHECK(len == sizeof(native_shadow_array) &&
         !memcmp(bytes, native_shadow_array, len));
   /* G17G carries a cube sample's per-lane level in this same publication,
    * paired with the cube Z -- and for a cube array, with the combined
    * face-and-layer word. Apple's own depthcube and depthcube_array
    * sample_compare with level(reg) (gpu/m5-shadow-array-cube-reference,
    * text-16 and text-08 of the extracted corpus). Staging the level
    * separately instead made every lane read level zero. */
   static const uint8_t g17g_cube_z_lod[] = {
      0x37,0x02,0x00,0x0a,0x25,0x00,0xaa,0x0a,0xc0,0x00,0x60,0x30,0x09,0x00,
   };
   CHECK(agx3_tex_stage_cube_z_lod_g17g(AGX_ISA_ENCODING_AGX3C, false, 2, 3, 1, 0,
                                   false, true, false, true, bytes, &len));
   CHECK(len == sizeof(g17g_cube_z_lod) &&
         !memcmp(bytes, g17g_cube_z_lod, len));
   static const uint8_t g17g_cube_array_z_lod[] = {
      0x37,0x02,0x00,0x4a,0x25,0x00,0xaa,0x0a,0xc0,0x02,0x61,0x40,0x29,0x00,
   };
   CHECK(agx3_tex_stage_cube_z_lod_g17g(AGX_ISA_ENCODING_AGX3C, true, 20, 11, 1, 0,
                                   false, true, false, true, bytes, &len));
   CHECK(len == sizeof(g17g_cube_array_z_lod) &&
         !memcmp(bytes, g17g_cube_array_z_lod, len));
   /* Before that publication, a G17G cube-array comparison normalizes its
    * integer layer into a general register -- the one PACK_SMPIR in this
    * file whose destination is a GPR (f9 = 2, f56 = 3) rather than the
    * staging file. Apple's depthcube_array fragment, text-08. */
   static const uint8_t g17g_cube_array_normalize[] = {
      0x27,0x00,0x00,0x9a,0x2d,0x00,0xaa,0x12,0x00,0x00,0x40,0x30,0x81,0x00,
   };
   CHECK(agx3_tex_normalize_cube_array_layer(AGX_ISA_ENCODING_AGX3C, 8, 16,
                                             false, true, bytes, &len));
   CHECK(len == sizeof(g17g_cube_array_normalize) &&
         !memcmp(bytes, g17g_cube_array_normalize, len));
   CHECK(!agx3_tex_normalize_cube_array_layer(AGX_ISA_ENCODING_AGX3B, 8, 16,
                                              false, true, bytes, &len));
   CHECK(!agx3_tex_normalize_cube_array_layer(AGX_ISA_ENCODING_AGX3C, 9, 16,
                                              false, true, bytes, &len));

   /* G17P has no witness for the pair, and a cube array's word operand must
    * stay even. */
   CHECK(!agx3_tex_stage_cube_z_lod_g17g(AGX_ISA_ENCODING_AGX3B, false, 2, 3, 1, 0,
                                    false, true, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_z_lod_g17g(AGX_ISA_ENCODING_AGX3C, true, 21, 11, 1, 0,
                                    false, true, false, true, bytes, &len));

   /* The G17G forms of both stagers are witnessed below (the shadow
    * array/cube set); only the G17P misalignments stay closed here. */
   /* metal-nt -arch applegpu_g17p, texturecube_array::sample in a kernel at
    * uint(c.w): flt2int writes the 16-bit layer to half11 (2707540b030cb0080300)
    * and PACK_SMPIR names that half beside CEM's face half10. */
   static const uint8_t native_array_face10[] = {
      0x17,0x05,0x54,0x06,0x00,0x16,
      0x28,0x00,0xf0,0xc0,0x29,0x00,
   };
   CHECK(agx3_tex_stage_cube_array_layer(
      AGX_ISA_ENCODING_AGX3B, 10, 11, 1, 0, false, true, bytes, &len));
   CHECK(len == sizeof(native_array_face10) &&
         !memcmp(bytes, native_array_face10, len));

   /* metal-nt -arch applegpu_g17p register level and bias. A cube publishes
    * its face and LOD together; a cube array first packs its layer into r0
    * and publishes that word as the third operand. */
   static const struct {
      const char *name;
      uint16_t face, lod;
      int packed_layer;
      const char *hex;
   } cube_lods[] = {
      {"kernel texturecube level(c.w)", 10, 11, -1,
       "1705540600142c00f0c01100"},
      {"fragment texturecube level/bias(b[i])", 2, 3, -1,
       "1705540600040c00f0c01100"},
      {"kernel texturecube_array level(c.x)", 10, 11, 0,
       "1705540600142c00f0c21700"},
      {"kernel texturecube_array level(lod[i])", 8, 9, 0,
       "1705540600102400f0c21700"},
      {"fragment texturecube_array level/bias(b[j])", 2, 3, 0,
       "1705540600040c00f0c21700"},
   };
   for (unsigned i = 0; i < sizeof(cube_lods) / sizeof(cube_lods[0]); ++i) {
      const bool packed = cube_lods[i].packed_layer < 0
         ? agx3_tex_stage_cube_lod(AGX_ISA_ENCODING_AGX3B, cube_lods[i].face,
                                   cube_lods[i].lod, 1, 0, false, true,
                                   bytes, &len)
         : agx3_tex_stage_cube_array_lod(
              AGX_ISA_ENCODING_AGX3B, cube_lods[i].face, cube_lods[i].lod,
              cube_lods[i].packed_layer, 1, 0, false, true, bytes, &len);
      if (!packed || !hex_eq(bytes, len, cube_lods[i].hex)) {
         fprintf(stderr, "cube LOD staging mismatch: %s\n", cube_lods[i].name);
         return false;
      }
   }
   static const uint8_t native_layer14_r0[] = {
      0x17,0x05,0x54,0x00,0x03,0x1c,
      0x00,0x00,0xf0,0x88,0x04,0x00,
   };
   CHECK(agx3_tex_pack_ms_array_layer(AGX_ISA_ENCODING_AGX3B, 14, 0, false,
                                      true, bytes, &len));
   CHECK(len == sizeof(native_layer14_r0) &&
         !memcmp(bytes, native_layer14_r0, len));
   CHECK(!agx3_tex_stage_cube_lod(AGX_ISA_ENCODING_AGX3B, 3, 3, 1, 0,
                                  false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_lod(AGX_ISA_ENCODING_AGX3B, 2, 3, 1, 0,
                                  true, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_lod(AGX_ISA_ENCODING_AGX3B, 2, 3, 1, 1,
                                        0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_lod(AGX_ISA_ENCODING_AGX3B, 2, 128, 0,
                                        1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3B, 3, 1, 0,
                                false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3B, 2, 1, 0,
                                true, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_shadow(
      AGX_ISA_ENCODING_AGX3B, 7, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_shadow(
      AGX_ISA_ENCODING_AGX3B, 6, 1, 0, true, true, bytes, &len));
   /* G17G runs the identical pipeline. Apple's own depthcube fragment
    * (gpu/m5-shadow-array-cube-reference.metal, text-14) keeps XYZ in
    * halves 4/6/14 and the three axes in 0/8/4, reciprocates the major axis
    * in place, then finalizes both face coordinates onto slot 1. */
   static const uint8_t gpu1_cem[2][14] = {
      {0x27,0x02,0x00,0x0a,0x29,0xb0,0xa2,0x12,0x04,0x84,0x1f,0x00,0x00,0xc0},
      {0x27,0x02,0x00,0x0a,0x29,0xb0,0xa2,0x0a,0x04,0x06,0xe7,0x00,0x00,0xc0},
   };
   const uint16_t gpu1_dest[3] = {0, 8, 4};
   for (unsigned axis = 1; axis < 3; ++axis) {
      CHECK(agx3_tex_cube_cem_calc_to(AGX_ISA_ENCODING_AGX3C, 4, 6, 14,
                                      gpu1_dest[axis], axis, axis == 2,
                                      bytes, &len));
      CHECK(len == sizeof(gpu1_cem[axis - 1]) &&
            !memcmp(bytes, gpu1_cem[axis - 1], len));
   }
   static const uint8_t gpu1_rcp[10] = {
      0x27,0x00,0x04,0x18,0x2a,0x20,0xa1,0x02,0x30,0x00,
   };
   CHECK(agx3_tex_cube_cem_rcp_to(AGX_ISA_ENCODING_AGX3C, 0, bytes, &len));
   CHECK(len == sizeof(gpu1_rcp) && !memcmp(bytes, gpu1_rcp, len));
   static const uint8_t gpu1_finalize[2][12] = {
      {0x09,0x09,0x2e,0x81,0x23,0xa0,0x42,0x00,0x00,0x02,0x02,0x00},
      {0x19,0x05,0x3e,0x01,0x23,0xa0,0x42,0x00,0x00,0x02,0x02,0x00},
   };
   for (unsigned component = 0; component < 2; ++component) {
      CHECK(agx3_tex_cube_cem_finalize_to(
         AGX_ISA_ENCODING_AGX3C, component, 0, gpu1_dest[component + 1], 1,
         bytes, &len));
      CHECK(len == sizeof(gpu1_finalize[component]) &&
            !memcmp(bytes, gpu1_finalize[component], len));
   }
   /* The driver's own fixed-scratch ABI composes on G17G too. */
   CHECK(agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3C, 16, 18, 20, 1, false,
                                bytes, &len));
   CHECK(agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3C, 1, true, bytes,
                                    &len));
   CHECK(!agx3_tex_stage_cube_array_layer(
      AGX_ISA_ENCODING_AGX3B, 2, 128, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_layer(
      AGX_ISA_ENCODING_AGX3B, 3, 6, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_cube_array_layer(
      AGX_ISA_ENCODING_AGX3B, 2, 6, 1, 0, true, true, bytes, &len));
   CHECK(!agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3B, 4, 6, 20, 0,
                                 true, bytes, &len));
   CHECK(!agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3B, 128, 6, 20, 0,
                                 false, bytes, &len));
   CHECK(!agx3_tex_cube_cem_calc(AGX_ISA_ENCODING_AGX3C, 128, 6, 20, 0,
                                 false, bytes, &len));
   CHECK(!agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3B, 0, true,
                                     bytes, &len));
   CHECK(!agx3_tex_cube_cem_finalize(AGX_ISA_ENCODING_AGX3B, 2, false,
                                     bytes, &len));
   /* Both encodings group their finalizers through a three-bit selector. */
   CHECK(!agx3_tex_cube_cem_finalize_to(AGX_ISA_ENCODING_AGX3B, 0, 0, 8, 7,
                                        bytes, &len));
   CHECK(agx3_tex_cube_cem_finalize_to(AGX_ISA_ENCODING_AGX3C, 0, 0, 8, 7,
                                       bytes, &len));
   CHECK(!agx3_tex_cube_cem_finalize_to(AGX_ISA_ENCODING_AGX3C, 0, 0, 8, 8,
                                        bytes, &len));
   return 0;
}

static int
test_multisample_fetch(void)
{
   /* A18 Pro own-MSL texture2d_ms reads, isolated for samples 0..3. */
   static const uint8_t native[4][14] = {
      {0x05,0x80,0x0c,0xb8,0xb0,0x00,0x80,0x14,0,0,0,0,1,0},
      {0x05,0x80,0x0c,0xb8,0xb0,0x02,0x80,0x14,0,0,0,0,1,0},
      {0x05,0x80,0x0c,0xb8,0xb0,0x04,0x80,0x14,0,0,0,0,1,0},
      {0x05,0x80,0x0c,0xb8,0xb0,0x06,0x80,0x14,0,0,0,0,1,0},
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   struct agx3_tex_fetch_desc d = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .multisample = true,
   };

   for (d.sample = 0; d.sample < 4; ++d.sample) {
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
      CHECK(len == sizeof(native[d.sample]));
      CHECK(!memcmp(bytes, native[d.sample], len));
   }

   d.sample = 4;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.staged_sample = true;
   d.sampler_value = 1;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   CHECK(hex_eq(bytes, len, "05800cb8b0008014000100000100"));

   /* Complete native indexed-texture/runtime-sample witness. No extra
    * descriptor-context or anonymous tail bits appear in this form. */
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 0;
   d.texture_kill = true;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   CHECK(hex_eq(bytes, len, "05800cb8b0008014000380000100002800000000"));
   d.sample = 1;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.sampler_value = 0;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.staged_sample = false;
   d.texture_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   d.texture_kill = false;
   d.coord = 16;
   d.array = true;
   d.texture = 1;
   d.wait_mask = 2;
   static const uint8_t native_ms_array[14] = {
      0x05,0x80,0x0c,0xb9,0x30,0x01,0xd9,0x11,0x80,0,0,0,1,0,
   };
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   CHECK(len == sizeof(native_ms_array) &&
         !memcmp(bytes, native_ms_array, len));
   /* Fresh runtime layer+sample native programs use bank0 coordinates and
    * consume their merged operand from staging half4, not literal layer2. */
   d.coord = 0;
   d.texture = 0;
   d.wait_mask = 1;
   d.staged_sample = true;
   d.sampler_value = 1;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   CHECK(hex_eq(bytes, len, "05800cb8b000a014000100000100"));
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 16;
   d.texture_kill = true;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   CHECK(hex_eq(bytes, len, "05800cb8b000a014100380000100002800000000"));
   CHECK(agx3_tex_pack_ms_array_layer(AGX_ISA_ENCODING_AGX3B, 6, 2,
                                     false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "17055402030c0000f0880400"));
   CHECK(agx3_tex_stage_ms_array_sample(AGX_ISA_ENCODING_AGX3B, 0, 2,
                                       false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540400000010d0c20a00"));
   CHECK(!agx3_tex_pack_ms_array_layer(AGX_ISA_ENCODING_AGX3B, 128, 2,
                                      false, true, bytes, &len));
   CHECK(!agx3_tex_stage_ms_array_sample(AGX_ISA_ENCODING_AGX3B, 0, 3,
                                        false, true, bytes, &len));
   d.staged_sample = false;
   d.sampler_value = 0;
   d.texture_kind = AGX3_TEX_OPERAND_IMMEDIATE;
   d.texture_kill = false;
   d.coord = 16;
   d.sample = 1;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.coord = 0;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.array = false;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &len));
   d.multisample = false;
   d.sample = 1;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   d.sample = 0;
   d.multisample = true;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_value = 2;
   CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &d, bytes, &len));
   return 0;
}

static int
test_multisample_sample_stage(void)
{
   /* The native bit_and producer writes half2. PACK_SMPIR reads that same
    * half address with a 16-bit source and publishes it at staging half4.
    * Both fixed and indexed texture witnesses have these exact bytes. */
   static const uint8_t native[] = {
      0x17,0x05,0x54,0x04,0x00,0x04,
      0x00,0x00,0xd0,0xc0,0x08,0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 2, 0,
                                   false, true, bytes, &len));
   CHECK(len == sizeof(native) && !memcmp(bytes, native, len));
   uint8_t other[AGX3_TEX_MAX_LENGTH], other_len;
   CHECK(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 6, 0,
                                  false, true, other, &other_len));
   CHECK(other_len == len && memcmp(other, bytes, len));
   CHECK(other[3] == 4 && bytes[3] == 4 && other[5] == 12);
   /* The store/fetch probe produces sample1 at half12 while half6 is the
    * descriptor's zero high word. Dividing this address aliases sample0. */
   CHECK(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 12, 0,
                                  false, true, other, &other_len));
   CHECK(hex_eq(other, other_len, "1705540400180000d0c00800"));
   static const uint8_t native_r4[] = {
      0x17,0x05,0x54,0x04,0x00,0x10,
      0x00,0x00,0xd0,0xc0,0x08,0x00,
   };
   CHECK(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 8, 0,
                                  false, true, bytes, &len));
   CHECK(len == sizeof(native_r4) && !memcmp(bytes, native_r4, len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 256, 0,
                                   false, false, bytes, &len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 3, 0,
                                   false, true, bytes, &len));
   /* The G17G multisample read itself: the staged-sample form with f76 = 2,
    * colour and depth. */
   {
      struct agx3_tex_fetch_desc ms = {
         .width = AGX3_TEX_W32, .mask = 15, .slot = 1, .wait_mask = 1,
         .coord_kill = true, .final = true, .multisample = true,
         .staged_sample = true, .sampler_value = 1,
      };
      uint8_t msb[AGX3_TEX_MAX_LENGTH], mslen;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &ms, msb, &mslen));
      CHECK(hex_eq(msb, mslen, "0f041a0045816129010802000000"));
      ms.mask = 1;
      CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &ms, msb, &mslen));
      CHECK(hex_eq(msb, mslen, "0f041a0045810121010802000000"));
   }
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 4, 0x40,
                                   false, true, bytes, &len));
   CHECK(!agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 4, 0,
                                   true, true, bytes, &len));

   /* AGX3B relocatable 2D_MS and 2D_MS_ARRAY sample publishers encode bank + 4
    * in f5 and scoreboard slot 1..6 in f12, matching the fixed bank-0 slot-1
    * witnesses at bank == 0, slot == 1. */
   {
      uint8_t base_ms[AGX3_TEX_MAX_LENGTH], reloc_ms[AGX3_TEX_MAX_LENGTH];
      uint8_t base_len = 0, reloc_len = 0;
      CHECK(agx3_tex_stage_ms_sample(AGX_ISA_ENCODING_AGX3B, 2, 0,
                                     false, true, base_ms, &base_len));
      CHECK(agx3_tex_stage_ms_sample_at(AGX_ISA_ENCODING_AGX3B, 0, 2, 1, 0,
                                        false, true, reloc_ms, &reloc_len));
      CHECK(reloc_len == base_len && !memcmp(reloc_ms, base_ms, base_len));

      for (uint16_t bank = 8; bank <= 56; bank += 8) {
         const uint8_t slot = (uint8_t)(((bank / 8) % 6) + 1);
         CHECK(agx3_tex_stage_ms_sample_at(AGX_ISA_ENCODING_AGX3B, bank, 2,
                                           slot, 0, false, true, reloc_ms,
                                           &reloc_len));
         CHECK(reloc_len == base_len && memcmp(reloc_ms, base_ms, base_len) != 0);
         struct agx3_tex_fetch_desc ms = {
            .dest = 8, .coord = bank, .width = AGX3_TEX_W32, .mask = 15,
            .slot = slot, .wait_mask = (uint16_t)(1u << (slot - 1)),
            .coord_kill = true, .final = true, .multisample = true,
            .staged_sample = true, .sampler_value = 1,
         };
         uint8_t msb[AGX3_TEX_MAX_LENGTH], mslen = 0;
         CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &ms, msb, &mslen));
      }
      CHECK(!agx3_tex_stage_ms_sample_at(AGX_ISA_ENCODING_AGX3B, 4, 2, 1, 0,
                                         false, true, reloc_ms, &reloc_len));
      CHECK(!agx3_tex_stage_ms_sample_at(AGX_ISA_ENCODING_AGX3B, 64, 2, 1, 0,
                                         false, true, reloc_ms, &reloc_len));
      CHECK(!agx3_tex_stage_ms_sample_at(AGX_ISA_ENCODING_AGX3B, 8, 2, 7, 0,
                                         false, true, reloc_ms, &reloc_len));
      CHECK(!agx3_tex_stage_ms_sample_at(AGX_ISA_ENCODING_AGX3B, 8, 2, 1, 0x40,
                                         false, true, reloc_ms, &reloc_len));

      uint8_t base_arr[AGX3_TEX_MAX_LENGTH], reloc_arr[AGX3_TEX_MAX_LENGTH];
      uint8_t arr_base_len = 0, arr_reloc_len = 0;
      CHECK(agx3_tex_stage_ms_array_sample(AGX_ISA_ENCODING_AGX3B, 20, 22,
                                           false, true, base_arr, &arr_base_len));
      CHECK(agx3_tex_stage_ms_array_sample_at(AGX_ISA_ENCODING_AGX3B, 0, 20, 22,
                                              1, false, true, reloc_arr,
                                              &arr_reloc_len));
      CHECK(arr_reloc_len == arr_base_len &&
            !memcmp(reloc_arr, base_arr, arr_base_len));

      for (uint16_t bank = 8; bank <= 24; bank += 8) {
         const uint8_t slot = (uint8_t)((bank / 8) + 1);
         CHECK(agx3_tex_stage_ms_array_sample_at(AGX_ISA_ENCODING_AGX3B, bank,
                                                 20, 22, slot, false, true,
                                                 reloc_arr, &arr_reloc_len));
         CHECK(arr_reloc_len == arr_base_len &&
               memcmp(reloc_arr, base_arr, arr_base_len) != 0);
         struct agx3_tex_fetch_desc ms = {
            .dest = 8, .coord = bank, .width = AGX3_TEX_W32, .mask = 15,
            .slot = slot, .wait_mask = (uint16_t)(1u << (slot - 1)),
            .coord_kill = true, .final = true, .array = true,
            .multisample = true, .staged_sample = true, .sampler_value = 1,
         };
         uint8_t msb[AGX3_TEX_MAX_LENGTH], mslen = 0;
         CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3B, &ms, msb, &mslen));
      }
      CHECK(!agx3_tex_stage_ms_array_sample_at(AGX_ISA_ENCODING_AGX3B, 32, 20,
                                               22, 1, false, true, reloc_arr,
                                               &arr_reloc_len));
      CHECK(!agx3_tex_stage_ms_array_sample_at(AGX_ISA_ENCODING_AGX3B, 8, 20,
                                               22, 7, false, true, reloc_arr,
                                               &arr_reloc_len));
   }
   return 0;
}

static int
test_m5_array_sample(void)
{
   /* Apple Metal toolchain 27.1, applegpu_g17g, explicit level(0) from a
    * texture2d_array in an argument buffer. Native code uses uniform halves
    * 16/20 for heap indices and PACK_SMPIR for the per-lane layer. */
   static const uint8_t stage[] = {
      0x37,0x02,0x00,0x0a,0x25,0x00,0xaa,0x0a,0x00,0x00,0x40,0x30,0x81,0x00,
   };
   static const uint8_t sample[] = {
      0x0f,0x04,0x12,0x00,0x05,0x84,0x60,0x28,0x89,0x08,0x01,
      0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length = 0;
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3C, 2, 1, 0,
                                    false, true, bytes, &length));
   CHECK(length == sizeof(stage) && !memcmp(bytes, stage, sizeof(stage)));
   struct agx3_tex_sample_desc d = {
      .dest=0,.coord=0,.width=AGX3_TEX_W32,.mask=15,
      .texture_kind=AGX3_TEX_OPERAND_UNIFORM,.texture_value=16,
      .sampler_kind=AGX3_TEX_OPERAND_UNIFORM,.sampler_value=20,
      .array=true,.lod=AGX3_TEX_LOD_LEVEL,.slot=1,.wait_mask=1,
      .coord_kill=true,.final=true,.g18g=true,
   };
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C,&d,bytes,&length));
   CHECK(length == sizeof(sample) && !memcmp(bytes,sample,sizeof(sample)));
   /* Per-thread argument-buffer indexing keeps both heap indices in GPRs.
    * The native sample waits slot8 while returning on slot1. */
   static const uint8_t indexed[] = {
      0x0f,0x04,0x12,0x00,0x25,0x84,0xe0,0x08,0x01,0x28,0x01,
      0x00,0x00,0x00,0x00,0x00,0x00,0x80,0x00,0x00,0x00,0x00,
   };
   d.texture_kind=AGX3_TEX_OPERAND_REGISTER;
   d.texture_value=0; d.sampler_kind=AGX3_TEX_OPERAND_REGISTER;
   d.sampler_value=4; d.wait_mask=128;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C,&d,bytes,&length));
   CHECK(length == sizeof(indexed) && !memcmp(bytes,indexed,sizeof(indexed)));
   CHECK(!agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3C,128,1,0,
                                     false,true,bytes,&length));
   return 0;
}

/* Apple's native fragment-array instructions use the complete Apple sample
 * ABI: f83/f112=4/8 plus direct f31/f37 layer selection. Honeykrisp's
 * descriptor heaps instead require the independently qualified 0/4 profile
 * paired with the PACK_SMPIR private layer selected by f77=2. Keep both exact
 * encodings without erasing the native immediate/register provenance.
 */
static int
test_array_sample_abi_differential(void)
{
   uint8_t driver[AGX3_TEX_MAX_LENGTH], apple[AGX3_TEX_MAX_LENGTH];
   uint8_t driver_len = 0, apple_len = 0;
   struct agx3_tex_sample_desc d = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .array = true,
      .array_layer = 22,
   };

   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, driver,
                                &driver_len));
   CHECK(driver_len == 14 &&
         hex_eq(driver, driver_len, "05800cb8b0000004010010000100"));
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, apple,
                                &apple_len));
   CHECK(apple_len == driver_len &&
         hex_eq(apple, apple_len, "05800cb8b0000db4010010001200"));
   CHECK(driver[6] == 0x00 && driver[7] == 0x04 &&
         apple[6] == 0x0d && apple[7] == 0xb4 &&
         driver[12] == 0x01 && apple[12] == 0x12);
   for (unsigned i = 0; i < driver_len; ++i)
      CHECK(i == 6 || i == 7 || i == 12 || driver[i] == apple[i]);

   d.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
   d.array_layer = 24;
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 0;
   d.sampler_value = 4;
   d.texture_kill = true;
   d.sampler_kill = true;
   d.wait_mask = 32;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, driver,
                                &driver_len));
   CHECK(driver_len == 20 &&
         hex_eq(driver, driver_len,
                "05800cb834000004010290010100a02800004000"));
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, apple,
                                &apple_len));
   CHECK(apple_len == driver_len &&
         hex_eq(apple, apple_len,
                "05800cb834000dc4010290011200a02800004000"));
   CHECK(driver[6] == 0x00 && driver[7] == 0x04 &&
         apple[6] == 0x0d && apple[7] == 0xc4 &&
         driver[12] == 0x01 && apple[12] == 0x12);
   for (unsigned i = 0; i < driver_len; ++i)
      CHECK(i == 6 || i == 7 || i == 12 || driver[i] == apple[i]);

   return 0;
}

/*
 * The coordinate offset, byte-exact against Apple's own compiler.
 *
 * Produced with the Metal 27.1 offline AIR native translator, one fragment
 * render pipeline per row:
 *
 *   xcrun metal   -mmacosx-version-min=26.0 -c tex2.metal -o tex2.air
 *   xcrun metallib tex2.air -o tex2.metallib
 *   xcrun metal-nt -arch applegpu_g17p -N tex2.mtlp-json \
 *                  -platform_version macos 26.0 26.0 tex2.metallib -o o.nat
 *
 * Every row differs from its offset-free control by exactly three fields --
 * f43 = (y << 4) | x, f51 = 1, f141 = 1 -- recovered by re-composing the
 * bytes through these tables, not by reading a disassembler.  The offset is
 * therefore orthogonal to the LOD shape, the array layer, the gather bit,
 * the depth comparison and the descriptor operand class.
 */
static int
test_offset_witnesses(void)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   const struct agx3_tex_sample_desc base = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT,
   };
   struct agx3_tex_sample_desc d;

   /* The control, and the three-value offset sweep on it. Each nibble is
    * signed and the pair is packed (y << 4) | x, which is exactly what
    * agx_nir_lower_texture() already builds for nir_tex_src_backend2. */
   d = base;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0000000010010001200"));

   static const struct { uint8_t xy; const char *hex; } sweep[] = {
      {0x11, "05800cb8b0000102018810001200"},   /* t.sample(s, uv, int2(1,1)) */
      {0x02, "05800cb8b0000102011010001200"},   /*                  int2(2,0) */
      {0x20, "05800cb8b0000102010010001202"},   /*                  int2(0,2) */
   };
   for (unsigned i = 0; i < sizeof(sweep) / sizeof(sweep[0]); ++i) {
      d = base;
      d.offset = true;
      d.offset_xyz = sweep[i].xy;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                   &length));
      CHECK(length == 14 && hex_eq(bytes, length, sweep[i].hex));
   }

   /* The staged runtime form is the register-source branch of the native
    * PACK_SMPIR offset witness. r2/r3 publish signed X/Y at positions 12/28
    * in operand bank 6, and SMP selects that bank with f141=2. */
   CHECK(agx3_tex_stage_offset_2d(AGX_ISA_ENCODING_AGX3B, 4, 6, 1, 0,
                                  false, true, bytes, &length));
   CHECK(length == 12 &&
         hex_eq(bytes, length, "170554060008180070c00100"));

   d = base;
   d.offset = true;
   d.offset_staged = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                &length));
   d.offset_xyz = 1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                 &length));
   d.offset_xyz = 0;
   d.three_d = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                 &length));

   /* The driver descriptor ABI cannot use Apple's direct array-layer field.
    * Its staged offset is valid only when the layer is already present in the
    * published bank-6 word. */
   d = base;
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
   d.array = true;
   d.array_layer = 12;
   d.offset = true;
   d.offset_staged = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                 &length));
   d.offset_staged_array = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes,
                                &length));

   /* The same offset (1,1) carried by every other LOD shape. f156 on the
    * immediate-level row is the stream group terminator, which this encoder
    * takes from the packer rather than inferring. */
   d = base;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 128;                 /* level(2.0) */
   d.sample_group_end = true;
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0001b02018810081200"));

   d = base;
   d.lod = AGX3_TEX_LOD_BIAS;
   d.lod_x64 = 32;                  /* bias(0.5) */
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0000b03018810021200"));

   d = base;
   d.lod = AGX3_TEX_LOD_REGISTER;   /* level(in.dx.x) */
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0001102018810001200"));

   d = base;
   d.lod = AGX3_TEX_LOD_BIAS_REGISTER;  /* bias(in.dx.x) */
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0008102018810001200"));

   d = base;
   d.lod = AGX3_TEX_LOD_GRADIENT;   /* gradient2d(in.dx, in.dy) */
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0000902008810001200"));

   /* ... by the gather, in two components ... */
   d = base;
   d.gather = AGX3_TEX_GATHER_R;
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800ca490000102018800001200"));
   d.gather = AGX3_TEX_GATHER_G;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cac90000102018800001200"));

   /* ... by a dynamic array layer, alone and with a gradient or a gather ... */
   d = base;
   d.array = true;
   d.array_layer = 22;
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0001bb7018810001200"));
   d.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0000db7008810001200"));
   d.lod = AGX3_TEX_LOD_AUTO;
   d.gather = AGX3_TEX_GATHER_R;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800ca490001bb7018800001200"));
   /* and the same array gather with no offset at all. */
   d.offset = false;
   d.offset_xyz = 0;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800ca490000db4010000001200"));

   /* ... and by a uniform (argument-buffer) descriptor pair, which reaches
    * this encoder as the twenty-byte form. */
   d = base;
   d.texture_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = 20;
   d.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.sampler_value = 24;
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length,
                "05a00cb8b8000102938b90011200802000000000"));
   d.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length,
                "05a00cb8b8000902928b90011200802000000000"));

   /* f43 is one shared immediate slot. An array sample with an immediate
    * level or bias spends it on that immediate, and Apple's
    * `t.sample(s, uv, 3u, bias(0.5f), int2(1,1))` --
    * 05800cb8b000451b010010001202 -- keeps the bias there and selects
    * f141 = 2, staging the offset as a separate twelve-byte operand. That
    * staging is not implemented, so the collision must refuse rather than
    * drop one of the two values. */
   d = base;
   d.array = true;
   d.array_layer = 22;
   d.lod = AGX3_TEX_LOD_BIAS;
   d.lod_x64 = 32;
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.lod = AGX3_TEX_LOD_LEVEL;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   /* The register-layer form leaves f43 free, so it still encodes. */
   d.lod = AGX3_TEX_LOD_AUTO;
   d.lod_x64 = 0;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));

   /* Native 3D offsets use a third nibble. Cubes remain closed and native
    * folds the all-zero offset to a plain sample. */
   d = base;
   d.offset = true;
   d.offset_xyz = 0;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.offset_xyz = 0x11;
   d.three_d = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.three_d = false;
   d.cube = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d.cube = false;
   /* This row carries Apple's fragment ABI, which stays AGX3b-only; the
    * G17G driver-ABI offset is covered by test_g17g_sample_lod_witnesses. */
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   return 0;
}

/*
 * The depth comparison against every LOD shape, byte-exact against Apple.
 *
 * Same harness as above, one `depth2d<float>` fragment per row. Each differs
 * from the corresponding colour sample by f143 = 1, the one-component f75,
 * and the dropped f74 mask-group bit -- and by nothing else, which is what
 * makes the comparison orthogonal to f139.
 */
static int
test_shadow_lod_witnesses(void)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   const struct agx3_tex_sample_desc base = {
      .dest = 6, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true, .shadow = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT,
   };
   struct agx3_tex_sample_desc d;

   /* t.sample_compare(s, uv, 0.5f) -- the control. */
   d = base;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002000010000001200"));

   /* level(0.0f) and level(2.0f): f139 = 2, the level in f29, f156 from the
    * packer's group terminator exactly as on a colour sample. */
   d = base;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 0;
   d.sample_group_end = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002900010000001200"));
   d.lod_x64 = 128;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002900010000081200"));

   /* bias(0.5f): f139 = 1, f29 = 32. */
   d = base;
   d.lod = AGX3_TEX_LOD_BIAS;
   d.lod_x64 = 32;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002700010000021200"));

   /* level(reg) and bias(reg): f139 = 4 and 3, the scalar staged into bank 6
    * exactly as on a colour sample, so the instruction carries no operand. */
   d = base;
   d.lod = AGX3_TEX_LOD_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002001010000001200"));
   d.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002400010000001200"));

   /* gradient2d(dx, dy): f139 = 5, and the gradient form clears f156. */
   d = base;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002401000000001200"));

   /* The comparison composes with the coordinate offset too. */
   d = base;
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "1d800ca090002102018800001200"));

   /* gather_compare returns four texels, so it carries the full f75 mask the
    * ordinary gather does -- with and without an offset. */
   d = base;
   d.dest = 0;
   d.mask = 15;
   d.gather = AGX3_TEX_GATHER_R;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800ca490002000010000001200"));
   d.offset = true;
   d.offset_xyz = 0x11;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800ca490002102018800001200"));

   /* A comparison gather must carry the four-texel mask, and every other
    * comparison exactly one component. */
   d = base;
   d.dest = 0;
   d.mask = 1;
   d.gather = AGX3_TEX_GATHER_R;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.mask = 15;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));

   /* Native compute sample_compare(level(0)) retains the packed layer or
    * CEM face and changes f139 to 2. Other explicit levels remain closed.
    * These kernels carry no fragment sample-group terminator. */
   const uint8_t native_zero[2][14] = {
      {0x05,0x80,0x0c,0xa0,0x90,0x00,0x29,0x01,0x00,0x00,0x00,0x00,0x01,0x00},
      {0x0d,0x80,0x0c,0xa0,0x90,0x00,0x33,0x00,0x00,0x00,0x00,0x00,0x01,0x00},
   };
   for (unsigned cube = 0; cube < 2; ++cube) {
      d = base;
      d.dest = cube ? 2 : 0;
      d.array = !cube;
      d.array_layer = cube ? 0 : 4;
      d.cube = cube;
      d.lod = AGX3_TEX_LOD_LEVEL;
      d.sample_abi = AGX3_TEX_SAMPLE_ABI_DRIVER;
      d.sample_group_nonterminal = true;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
      CHECK(length == sizeof(native_zero[cube]) &&
            !memcmp(bytes, native_zero[cube], length));
   }

   /* Level two and cube gradients do not inherit the level-zero witness. */
   d = base;
   d.array = true;
   d.array_layer = 6;
   d.lod = AGX3_TEX_LOD_LEVEL;
   d.lod_x64 = 128;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.cube = true;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   /* No 3D comparison, and nothing on AGX3c. */
   d = base;
   d.three_d = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   return 0;
}

/*
 * The 3D dimension against every LOD shape, byte-exact against Apple.
 *
 * Five texture3d fragments differing only in the LOD argument, same harness
 * as test_offset_witnesses().  f76 = 3 is orthogonal to f139 exactly as the
 * 2D dimension is, which is what admits the bias and per-lane forms; the
 * gradient is held back because gradient3d stages six scalars rather than
 * four and that layout is not witnessed.
 */
static int
test_3d_lod_witnesses(void)
{
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   const struct agx3_tex_sample_desc base = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1u << 0,
      .coord_kill = true, .final = true, .three_d = true,
      .sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT,
   };
   struct agx3_tex_sample_desc d;

   /* t.sample(s, uvw) */
   d = base;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0008001010010001200"));

   /* t.sample(s, uvw, bias(0.5f)) -- f139 = 1, f29 = 32 */
   d = base;
   d.lod = AGX3_TEX_LOD_BIAS;
   d.lod_x64 = 32;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0003b00010010021200"));

   /* t.sample(s, uvw, bias(in.dx.x)) -- f139 = 3, scalar staged to bank 6 */
   d = base;
   d.lod = AGX3_TEX_LOD_BIAS_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0009001010010001200"));

   /* t.sample(s, uvw, level(in.dx.x)) -- f139 = 4, same bank-6 staging */
   d = base;
   d.lod = AGX3_TEX_LOD_REGISTER;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0008821010010001200"));

   /* t.sample(s, uvw, gradient3d(dx, dy)) -- the ordinary f139 = 5. It stages
    * six scalars rather than four; the extra pair continues the same
    * interleave into banks 16 and 18. Read off two literal-gradient kernels,
    * where each staged value is the constant itself:
    *
    *   gradient3d((1,2,4), (8,16,32))          banks 8/10/12/14/16/18
    *                                             = 1 / 8 / 2 / 16 / 4 / 32
    *   gradient3d((64,128,0.25), (0.5,.75,1.5)) = 64 / 0.5 / 128 / 0.75
    *                                             / 0.25 / 1.5
    *
    * i.e. (ddx.x, ddy.x, ddx.y, ddy.y, ddx.z, ddy.z), the same interleave the
    * 2D control gradient2d((1,2),(4,8)) shows on banks 8/10/12/14. */
   d = base;
   d.lod = AGX3_TEX_LOD_GRADIENT;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "05800cb8b0009c01000010001200"));

   /* No 3D gather, no 3D comparison, no 3D array, nothing on AGX3c. */
   d = base;
   d.gather = AGX3_TEX_GATHER_R;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.shadow = true;
   d.mask = 1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.array = true;
   d.array_layer = 6;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   return 0;
}

static int
test_m5_indexed_reads(void)
{
   /* Native Metal 27.1 m5-texture-read-reference: uniform array layer zero
    * folds to a plain 2D read; lane-indexed array reads stage their layer. */
   struct agx3_tex_fetch_desc d = {
      .width = AGX3_TEX_W32, .mask = 15, .slot = 1,
      .wait_mask = 1, .coord_kill = true, .final = true,
      .texture_kind = AGX3_TEX_OPERAND_UNIFORM, .texture_value = 16,
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "0f061a00418060280800"));
   d.texture_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = 0;
   d.array = true;
   d.wait_mask = 128;
   CHECK(agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "0f061a004580e0080028"));
   CHECK(agx3_tex_stage_array_layer(AGX_ISA_ENCODING_AGX3C, 4, 8, 0,
                                  false, true, bytes, &length));
   CHECK(hex_eq(bytes, length, "3700002a2507aa0a000040308100"));

   d.texture_value = 1;
   CHECK(rejected_and_cleared(AGX_ISA_ENCODING_AGX3C, &d));
   return 0;
}

static int
test_image_write_state0_witnesses(void)
{
   uint8_t bytes[AGX3_IMAGE_WRITE_LENGTH], length = 0;
   struct agx3_image_write_desc d = {
      .image_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      .image_value = 0,
   };

   CHECK(agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(length == 16 &&
         hex_eq(bytes, length, "d71054000040000000040000881a0900"));

   d.normalized = true;
   CHECK(!agx3_tex_encode_image_write_rgba8_2d(
      AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   return 0;
}

static int
test_query_lod_witnesses(void)
{
   /* Native G17P calculate_clamped_lod / calculate_unclamped_lod. The
    * component-only witnesses establish X/Y ordering independently. */
   const struct agx3_tex_sample_desc base = {
      .width = AGX3_TEX_W32,
      .mask = 3,
      .slot = 1,
      .wait_mask = 1,
      .coord_kill = true,
      .final = true,
      .query_lod = true,
   };
   static const struct {
      bool three_d, cube, array;
      unsigned mask;
      const char *hex;
   } rows[] = {
      {false, false, false, 3, "05800ca8b0000000010020000100"},
      {true,  false, false, 3, "05800ca8b0008001010020000100"},
      {false, true,  false, 3, "05800ca8b0001000010020000100"},
      {false, false, true,  3, "05800ca8b0000d00010020000100"},
      {false, true,  true,  3, "05800ca8b0001301010020000100"},
      {false, false, false, 1, "05800ca090000000010020000100"},
      {false, false, false, 2, "05800ca890000000010020000100"},
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
      struct agx3_tex_sample_desc d = base;
      d.three_d = rows[i].three_d;
      d.cube = rows[i].cube;
      d.array = rows[i].array;
      d.mask = rows[i].mask;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
      CHECK(length == 14 && hex_eq(bytes, length, rows[i].hex));
   }
   struct agx3_tex_sample_desc d = base;
   d.texture_kind = d.sampler_kind = AGX3_TEX_OPERAND_UNIFORM;
   d.texture_value = d.sampler_value = 12;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "05e00ca8bc0000008103a0010100802000000000"));
   d.texture_kind = d.sampler_kind = AGX3_TEX_OPERAND_REGISTER;
   d.texture_value = d.sampler_value = 0;
   d.texture_kill = d.sampler_kill = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "05800ca8b00000000102a0010100a02800000000"));

   /* A query shares the existing sample-group lifetime. Native two-query
    * and query-plus-sample programs terminate only the second instruction. */
   d = base;
   d.final = false;
   d.sample_group_nonterminal = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "05800c28b0000000000020000100"));
   d = base;
   d.dest = 4;
   d.coord = 4;
   d.wait_mask = 2;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "15820ca830010000010020000100"));
   d.query_lod = false;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   CHECK(hex_eq(bytes, length, "15820ca830010000010010000100"));

   d = base;
   d.width = AGX3_TEX_W16;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.mask = 4;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.lod = AGX3_TEX_LOD_LEVEL;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.array = true;
   d.array_layer = 4;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.gather = AGX3_TEX_GATHER_R;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.shadow = true;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.offset = true;
   d.offset_xyz = 1;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   d = base;
   d.sample_abi = AGX3_TEX_SAMPLE_ABI_APPLE_FRAGMENT;
   CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
   return 0;
}

/* Own-MSL runtime-indexed argument buffers, metal-nt applegpu_g17p,
 * macOS 27.0. Every literal below independently recomposes to its native
 * twenty-byte SMP. Controls differ only by f43/f51/f141; array controls
 * and offsets both use the staged f77=2 layer. */
static int
test_indexed_offset_witnesses(void)
{
   static const struct {
      const char *name;
      enum agx3_tex_lod lod;
      bool array, three_d, shadow;
      uint16_t level, offset;
      const char *hex;
   } rows[] = {
      {"native_extra/d2_auto_offset", AGX3_TEX_LOD_AUTO,
       false, false, false, 0, 0x87,
       "05800cb834000102013a90010108a02800004000"},
      {"native_extra/d2_level0_offset", AGX3_TEX_LOD_LEVEL,
       false, false, false, 0, 0x87,
       "05800cb834001b02013a90010108a02800004000"},
      {"native_extra/d2_level2_offset", AGX3_TEX_LOD_LEVEL,
       false, false, false, 128, 0x87,
       "05800cb834001b02013a90090108a02800004000"},
      {"native_extra/d2_levelreg_offset", AGX3_TEX_LOD_REGISTER,
       false, false, false, 0, 0x87,
       "05800cb834001102013a90010108a02800004000"},
      {"native_extra/d2_bias_offset", AGX3_TEX_LOD_BIAS_REGISTER,
       false, false, false, 0, 0x87,
       "05800cb834008102013a90010108a02800004000"},
      {"native_extra/d2_gradient_offset", AGX3_TEX_LOD_GRADIENT,
       false, false, false, 0, 0x87,
       "05800cb834000902003a90010108a02800004000"},
      {"native_extra/a2_auto_offset", AGX3_TEX_LOD_AUTO,
       true, false, false, 0, 0x87,
       "05800cb834000103013a90010108a02800004000"},
      {"native_extra/a2_level0_offset", AGX3_TEX_LOD_LEVEL,
       true, false, false, 0, 0x87,
       "05800cb834005b02013a90010108a02800004000"},
      {"native_extra/a2_level2_offset", AGX3_TEX_LOD_LEVEL,
       true, false, false, 128, 0x87,
       "05800cb834005b02013a90090108a02800004000"},
      {"native_extra/a2_levelreg_offset", AGX3_TEX_LOD_REGISTER,
       true, false, false, 0, 0x87,
       "05800cb834005102013a90010108a02800004000"},
      {"native_extra/a2_bias_offset", AGX3_TEX_LOD_BIAS_REGISTER,
       true, false, false, 0, 0x87,
       "05800cb834008103013a90010108a02800004000"},
      {"native_extra/a2_gradient_offset", AGX3_TEX_LOD_GRADIENT,
       true, false, false, 0, 0x87,
       "05800cb834000903003a90010108a02800004000"},
      {"native_extra/d3_auto_offset", AGX3_TEX_LOD_AUTO,
       false, true, false, 0, 0x387,
       "05800cb834009102013a90010138a02800004000"},
      {"native_extra/d3_level0_offset", AGX3_TEX_LOD_LEVEL,
       false, true, false, 0, 0x387,
       "05800cb834009503013a90010138a02800004000"},
      {"native_extra/d3_level2_offset", AGX3_TEX_LOD_LEVEL,
       false, true, false, 128, 0x387,
       "05800cb834009503013a90090138a02800004000"},
      {"native_extra/d3_levelreg_offset", AGX3_TEX_LOD_REGISTER,
       false, true, false, 0, 0x387,
       "05800cb834001902013a90010138a02800004000"},
      {"native_extra/d3_bias_offset", AGX3_TEX_LOD_BIAS_REGISTER,
       false, true, false, 0, 0x387,
       "05800cb834009902013a90010138a02800004000"},
      {"native_extra/d3_gradient_offset", AGX3_TEX_LOD_GRADIENT,
       false, true, false, 0, 0x387,
       "05800cb834001903003a90010138a02800004000"},
      {"native_extra/depth_auto_offset", AGX3_TEX_LOD_AUTO,
       false, false, true, 0, 0x87,
       "05800ca014002102013a80010108a02800004000"},
      {"native_extra/depth_level0_offset", AGX3_TEX_LOD_LEVEL,
       false, false, true, 0, 0x87,
       "05800ca014003b02013a80010108a02800004000"},
      {"native_extra/depth_level2_offset", AGX3_TEX_LOD_LEVEL,
       false, false, true, 128, 0x87,
       "05800ca014003b02013a80090108a02800004000"},
      {"native_extra/depth_levelreg_offset", AGX3_TEX_LOD_REGISTER,
       false, false, true, 0, 0x87,
       "05800ca014003102013a80010108a02800004000"},
      {"native_extra/depth_bias_offset", AGX3_TEX_LOD_BIAS_REGISTER,
       false, false, true, 0, 0x87,
       "05800ca01400a102013a80010108a02800004000"},
      {"native_extra/depth_gradient_offset", AGX3_TEX_LOD_GRADIENT,
       false, false, true, 0, 0x87,
       "05800ca014002902003a80010108a02800004000"},
      {"native_extra/deptha_auto_offset", AGX3_TEX_LOD_AUTO,
       true, false, true, 0, 0x87,
       "05800ca014002103013a80010108a02800004000"},
      {"native_extra/deptha_level0_offset", AGX3_TEX_LOD_LEVEL,
       true, false, true, 0, 0x87,
       "05800ca014007b02013a80010108a02800004000"},
      {"native_extra/deptha_level2_offset", AGX3_TEX_LOD_LEVEL,
       true, false, true, 128, 0x87,
       "05800ca014007b02013a80090108a02800004000"},
      {"native_extra/deptha_levelreg_offset", AGX3_TEX_LOD_REGISTER,
       true, false, true, 0, 0x87,
       "05800ca014007102013a80010108a02800004000"},
      {"native_extra/deptha_bias_offset", AGX3_TEX_LOD_BIAS_REGISTER,
       true, false, true, 0, 0x87,
       "05800ca01400a103013a80010108a02800004000"},
      {"native_extra/deptha_gradient_offset", AGX3_TEX_LOD_GRADIENT,
       true, false, true, 0, 0x87,
       "05800ca014002903003a80010108a02800004000"},
      {"native_ranges/d2_biasimm_offset", AGX3_TEX_LOD_BIAS,
       false, false, false, 32, 0x87,
       "05800cb834000b03013a90030108a02800004000"},
      {"native_ranges/a2_biasimm_offset", AGX3_TEX_LOD_BIAS,
       true, false, false, 32, 0x87,
       "05800cb834004b03013a90030108a02800004000"},
      {"native_ranges/d3_auto_offset", AGX3_TEX_LOD_AUTO,
       false, true, false, 0, 0x837,
       "05802cb83400910201ba90010102a02800004000"},
      {"native_ranges/d3_level0_offset", AGX3_TEX_LOD_LEVEL,
       false, true, false, 0, 0x837,
       "05802cb83400950301ba90010102a02800004000"},
      {"native_ranges/d3_biasimm_offset", AGX3_TEX_LOD_BIAS,
       false, true, false, 32, 0x837,
       "05802cb83400950201ba90030102a02800004000"},
      {"native_ranges/d3_gradient_offset", AGX3_TEX_LOD_GRADIENT,
       false, true, false, 0, 0x837,
       "05802cb83400190300ba90010102a02800004000"},
      {"native_ranges/depth_biasimm_offset", AGX3_TEX_LOD_BIAS,
       false, false, true, 32, 0x87,
       "05800ca014002b03013a80030108a02800004000"},
      {"native_ranges/deptha_biasimm_offset", AGX3_TEX_LOD_BIAS,
       true, false, true, 32, 0x87,
       "05800ca014006b03013a80030108a02800004000"},
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
      const struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32,
         .mask = rows[i].shadow ? 1 : 15,
         .texture_kind = AGX3_TEX_OPERAND_REGISTER,
         .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
         .texture_value = 0, .sampler_value = 4,
         .texture_kill = true, .sampler_kill = true,
         .lod = rows[i].lod, .lod_x64 = rows[i].level,
         .slot = 1, .wait_mask = 1u << 5, .coord_kill = true, .final = true,
         .array = rows[i].array, .array_layer = rows[i].array ? 12 : 0,
         .three_d = rows[i].three_d, .shadow = rows[i].shadow,
         .sample_group_end = rows[i].lod == AGX3_TEX_LOD_LEVEL,
         .offset = true, .offset_xyz = rows[i].offset,
      };
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
      if (!hex_eq(bytes, length, rows[i].hex)) {
         fprintf(stderr, "indexed offset witness: %s\n", rows[i].name);
         return 1;
      }
      checks++;
   }

   /* The layer and register LOD share bank 6. Their single publication
    * matches both native level(reg) and bias(reg), including comparisons. */
   CHECK(agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3B, 12, 13, 6, 0,
                                   false, true, false, true, bytes, &length));
   CHECK(hex_eq(bytes, length, "1705540600183400f0e42500"));
   /* The G17G form is witnessed (layer 8, level 9, slot 1, below); a ninth
    * slot is still out of its range. */
   CHECK(!agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3C, 12, 13, 9, 0,
                                    false, true, false, true, bytes, &length));
   CHECK(!agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3B, 128, 13, 6, 0,
                                    false, true, false, true, bytes, &length));
   CHECK(!agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3B, 12, 256, 6, 0,
                                    false, true, false, true, bytes, &length));
   CHECK(!agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3B, 12, 13, 6, 0,
                                    true, true, false, true, bytes, &length));
   CHECK(!agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3B, 12, 13, 6, 0,
                                    false, true, true, true, bytes, &length));
   CHECK(agx3_tex_stage_array_integer_lod(
      AGX_ISA_ENCODING_AGX3B, 12, 14, 2, 0, false, true, false, true,
      bytes, &length));
   CHECK(length == 12);
   CHECK(!agx3_tex_stage_array_integer_lod(
      AGX_ISA_ENCODING_AGX3B, 12, 13, 2, 0, false, true, false, true,
      bytes, &length));
   return 0;
}

/* Runtime-indexed vertex pipelines from the Metal 27.1 native translator.
 * Every complete SMP recomposes through the AGX3B table. In each pair the
 * offset changes only f43/f51/f141; register level and gradient operands use
 * the same staging ABI as fragments, while vertex level clears fragment-only
 * sample-group terminator f156. */
static int
test_vertex_offset_witnesses(void)
{
   static const struct {
      const char *name;
      enum agx3_tex_lod lod;
      bool array, three_d, shadow;
      uint16_t offset;
      const char *plain, *offset_hex;
   } rows[] = {
      {"2D level", AGX3_TEX_LOD_REGISTER, false, false, false, 0x87,
       "05800cb834000001000290010100a02800004000",
       "05800cb834001102003a90010108a02800004000"},
      {"2D gradient", AGX3_TEX_LOD_GRADIENT, false, false, false, 0x87,
       "05800cb834000401000290010100a02800004000",
       "05800cb834000902003a90010108a02800004000"},
      {"2D array level", AGX3_TEX_LOD_REGISTER, true, false, false, 0x87,
       "05800cb834000801000290010100a02800004000",
       "05800cb834005102003a90010108a02800004000"},
      {"2D array gradient", AGX3_TEX_LOD_GRADIENT, true, false, false, 0x87,
       "05800cb834000c01000290010100a02800004000",
       "05800cb834000903003a90010108a02800004000"},
      {"3D level", AGX3_TEX_LOD_REGISTER, false, true, false, 0x837,
       "05800cb834008821000290010100a02800004000",
       "05802cb83400190200ba90010102a02800004000"},
      {"3D gradient", AGX3_TEX_LOD_GRADIENT, false, true, false, 0x837,
       "05800cb834009c01000290010100a02800004000",
       "05802cb83400190300ba90010102a02800004000"},
      {"depth level", AGX3_TEX_LOD_REGISTER, false, false, true, 0x87,
       "05800ca014002001000280010100a02800004000",
       "05800ca014003102003a80010108a02800004000"},
      {"depth gradient", AGX3_TEX_LOD_GRADIENT, false, false, true, 0x87,
       "05800ca014002401000280010100a02800004000",
       "05800ca014002902003a80010108a02800004000"},
      {"depth array level", AGX3_TEX_LOD_REGISTER, true, false, true, 0x87,
       "05800ca014002801000280010100a02800004000",
       "05800ca014007102003a80010108a02800004000"},
      {"depth array gradient", AGX3_TEX_LOD_GRADIENT, true, false, true, 0x87,
       "05800ca014002c01000280010100a02800004000",
       "05800ca014002903003a80010108a02800004000"},
   };
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
   for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32,
         .mask = rows[i].shadow ? 1 : 15,
         .texture_kind = AGX3_TEX_OPERAND_REGISTER,
         .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
         .texture_value = 0, .sampler_value = 4,
         .texture_kill = true, .sampler_kill = true,
         .lod = rows[i].lod, .slot = 1, .wait_mask = 1u << 5,
         .coord_kill = true, .final = true, .vertex_stage = true,
         .array = rows[i].array,
         .array_layer = rows[i].array ? 12 : 0,
         .three_d = rows[i].three_d, .shadow = rows[i].shadow,
      };
      CHECK(agx3_tex_encode_sample(
         AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
      if (!hex_eq(bytes, length, rows[i].plain)) {
         fprintf(stderr, "vertex sample witness: %s control\n", rows[i].name);
         return 1;
      }
      d.offset = true;
      d.offset_xyz = rows[i].offset;
      CHECK(agx3_tex_encode_sample(
         AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
      if (!hex_eq(bytes, length, rows[i].offset_hex)) {
         fprintf(stderr, "vertex sample witness: %s offset\n", rows[i].name);
         return 1;
      }
      checks += 2;
   }
   return 0;
}

/* Fresh native.metal G17P fragments, independently compiled on 2026-09-13.
 * These retain the native SMP and PACK bytes, not bytes from this encoder. */
static int
test_minimum_lod_witnesses(void)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_typed_lod(enc, 6, 1, 0, false, true,
                                 AGX3_TEX_LEVEL_MINIMUM, bytes, &len));
   CHECK(hex_eq(bytes, len, "17055406000c1004f0c00800"));
   CHECK(agx3_tex_stage_typed_lod(enc, 14, 1, 0, false, true,
                                 AGX3_TEX_LEVEL_MINIMUM, bytes, &len));
   CHECK(hex_eq(bytes, len, "17055406001c1004f0c00800"));
   CHECK(agx3_tex_stage_bias_min(enc, 2, 0, 1, 0, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540600000800f0c02100"));
   CHECK(agx3_tex_stage_bias_min(enc, 10, 8, 1, 0, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540600102800f0c02100"));
   CHECK(!agx3_tex_stage_bias_min(enc, 0, 128, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_bias_min(enc, 256, 0, 1, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_bias_min(enc, 0, 1, 0, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_bias_min(enc, 0, 1, 1, 64, false, true, bytes, &len));
   CHECK(!agx3_tex_stage_bias_min(enc, 0, 1, 1, 0, true, true, bytes, &len));
   /* texture2d_array and texturecube_array bias + min_lod_clamp. */
   CHECK(agx3_tex_stage_array_minimum_lod(enc, 0, 1, 0, false, true, bytes,
                                          &len));
   CHECK(hex_eq(bytes, len, "1705540400001004f0c00800"));
   CHECK(agx3_tex_stage_array_minimum_lod(enc, 4, 1, 0, false, true, bytes,
                                          &len));
   CHECK(hex_eq(bytes, len, "1705540400081004f0c00800"));
   CHECK(agx3_tex_stage_array_lod(enc, 2, 3, 1, 0, false, true, false, true,
                                  bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540600040c00f0c02500"));
   CHECK(!agx3_tex_stage_array_minimum_lod(enc, 256, 1, 0, false, true, bytes,
                                           &len));
   CHECK(!agx3_tex_stage_array_minimum_lod(enc, 0, 1, 0, true, true, bytes,
                                           &len));
   /* depth2d_array sample_compare bias + min_lod_clamp, two in one fragment. */
   CHECK(agx3_tex_pack_minimum(enc, 0, 0, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540002000000d0886000"));
   CHECK(agx3_tex_pack_minimum(enc, 2, 2, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540202040000d0886000"));
   CHECK(!agx3_tex_pack_minimum(enc, 128, 0, false, true, bytes, &len));
   CHECK(!agx3_tex_pack_minimum(enc, 0, 1, false, true, bytes, &len));
   CHECK(!agx3_tex_pack_minimum(enc, 0, 192, false, true, bytes, &len));
   CHECK(!agx3_tex_pack_minimum(enc, 0, 0, true, true, bytes, &len));
   for (unsigned offset = 0; offset < 2; ++offset) {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1, .slot = 1,
         .wait_mask = 1, .coord_kill = true, .final = true, .min_lod = true,
         .min_lod_register = true, .min_lod_value = 0, .min_lod_kill = true,
         .lod = AGX3_TEX_LOD_BIAS_REGISTER, .offset = offset,
         .offset_xyz = offset ? 0xf1 : 0, .shadow = true, .array = true,
         .array_layer = 4, .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
         .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      };
      CHECK(agx3_tex_encode_sample(enc, &d, bytes, &len));
      CHECK(hex_eq(bytes, len, offset ? "05800ca09000020101880008010e"
                                      : "05800ca090004200010000080100"));
      d.min_lod_value = 1;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      d.min_lod_value = 0;
      d.cube = true;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
   }
   /* texture3d bias and gradient3d + min_lod_clamp with an immediate offset
    * (1, -1, 2), and the bias with a runtime offset, whose minimum SMP reads
    * from a packed register;
    * texture2d with a runtime offset, whose bank-6 word carries the minimum.
    * metal-nt -arch applegpu_g17p. */
   static const struct {
      bool three_d, staged, registered, gradient, shadow;
      const char *hex;
   } offsets[] = {
      {true, false, false, false, false, "05800cb8b000d90201881000012e"},
      {true, false, false, true, false, "05800cb8b000590300881000012e"},
      {true, true, true, false, false, "05800cb8b0008200010010080100"},
      {false, true, false, false, false, "05800cb8b0004408010010000100"},
      {false, true, false, true, false, "05800cb8b0005401000010000100"},
      {false, true, false, false, true, "05800ca090006408010000000100"},
      {false, true, false, true, true, "05800ca090007401000000000100"},
   };
   for (unsigned i = 0; i < sizeof(offsets) / sizeof(offsets[0]); ++i) {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32,
         .mask = offsets[i].shadow ? 1 : 15, .slot = 1, .wait_mask = 1,
         .coord_kill = true, .final = true, .min_lod = true,
         .min_lod_register = offsets[i].registered,
         .min_lod_kill = offsets[i].registered,
         .lod = offsets[i].gradient ? AGX3_TEX_LOD_GRADIENT
                                    : AGX3_TEX_LOD_BIAS_REGISTER,
         .offset = true, .offset_staged = offsets[i].staged,
         .offset_xyz = offsets[i].staged ? 0 : 0x2f1,
         .three_d = offsets[i].three_d, .shadow = offsets[i].shadow,
         .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
         .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      };
      CHECK(agx3_tex_encode_sample(enc, &d, bytes, &len));
      CHECK(hex_eq(bytes, len, offsets[i].hex));
      if (offsets[i].three_d && offsets[i].staged) {
         /* Only the bias is witnessed beside a staged 3D offset. */
         d.lod = AGX3_TEX_LOD_GRADIENT;
         d.min_lod_register = false;
         CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      }
   }
   /* A staged 3D offset without the register minimum has no witness. */
   {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15, .slot = 1,
         .wait_mask = 1, .coord_kill = true, .final = true, .min_lod = true,
         .lod = AGX3_TEX_LOD_BIAS_REGISTER, .offset = true,
         .offset_staged = true, .three_d = true,
         .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
         .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      };
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      d.min_lod = false;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
   }
   /* texturecube and depthcube bias + min_lod_clamp: the face packed alone,
    * then minimum and bias beside it. */
   CHECK(agx3_tex_pack_cube_face(enc, 8, 0, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540003100004f0880400"));
   CHECK(agx3_tex_pack_cube_face(enc, 10, 0, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540003140004f0880400"));
   CHECK(!agx3_tex_pack_cube_face(enc, 9, 0, bytes, &len));
   CHECK(!agx3_tex_pack_cube_face(enc, 8, 1, bytes, &len));
   CHECK(agx3_tex_stage_cube_bias_min(enc, 9, 16, 0, 1, 0, false, true, bytes,
                                      &len));
   CHECK(hex_eq(bytes, len, "1705540600202400f0c22300"));
   CHECK(agx3_tex_stage_cube_bias_min(enc, 2, 3, 0, 1, 0, false, true, bytes,
                                      &len));
   CHECK(hex_eq(bytes, len, "1705540600060800f0c22300"));
   CHECK(!agx3_tex_stage_cube_bias_min(enc, 2, 3, 1, 1, 0, false, true, bytes,
                                       &len));
   CHECK(!agx3_tex_stage_cube_bias_min(enc, 2, 3, 0, 1, 3, false, true, bytes,
                                       &len));
   /* texturecube_array bias + min_lod_clamp: layer pack, minimum at half 4
    * (above), then face and bias beside the packed layer. */
   CHECK(agx3_tex_pack_ms_array_layer(enc, 11, 0, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540003160000f0880400"));
   CHECK(agx3_tex_stage_cube_array_lod(enc, 10, 2, 0, 1, 0, false, true, bytes,
                                       &len));
   CHECK(hex_eq(bytes, len, "1705540600140800f0c21700"));
   /* depthcube sample_compare(level(reg)): face and level in bank 6. */
   CHECK(agx3_tex_stage_cube_lod(enc, 10, 8, 1, 0, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540600142000f0c01100"));
   static const struct {
      bool array, shadow, min_lod;
      enum agx3_tex_lod lod;
      const char *hex;
   } cubes[] = {
      {false, false, true, AGX3_TEX_LOD_BIAS_REGISTER, "05800cb8b0001408010010000100"},
      {true, false, true, AGX3_TEX_LOD_BIAS_REGISTER, "05800cb8b0001c00010010000100"},
      {false, true, true, AGX3_TEX_LOD_BIAS_REGISTER, "05800ca090003408010000000100"},
      {false, true, false, AGX3_TEX_LOD_REGISTER, "05800ca090002003010000000100"},
   };
   for (unsigned i = 0; i < sizeof(cubes) / sizeof(cubes[0]); ++i) {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32,
         .mask = cubes[i].shadow ? 1 : 15, .slot = 1, .wait_mask = 1,
         .coord_kill = true, .final = true, .min_lod = cubes[i].min_lod,
         .lod = cubes[i].lod, .shadow = cubes[i].shadow, .cube = true,
         .array = cubes[i].array, .array_layer = cubes[i].array ? 6 : 0,
         .texture_kind = AGX3_TEX_OPERAND_IMMEDIATE,
         .sampler_kind = AGX3_TEX_OPERAND_IMMEDIATE,
      };
      CHECK(agx3_tex_encode_sample(enc, &d, bytes, &len));
      CHECK(hex_eq(bytes, len, cubes[i].hex));
      d.lod = AGX3_TEX_LOD_GRADIENT;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      if (cubes[i].shadow) {
         d.lod = cubes[i].lod;
         d.array = true;
         d.array_layer = 6;
         if (cubes[i].min_lod) {
            /* No cube-array witness clamps a minimum beside its bias. */
            CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
         } else {
            /* The comparison cube array does take a register level. AIR-NT
             * 32023.917 -arch applegpu_g17p compiles
             * depthcube_array.sample_compare(s, dir, uint(layer), ref,
             * level(lod)), every operand a varying, to the auxiliary layer
             * word 1705540003200000f0880400, the face+level word
             * 1705540600040c00f0c21700, the pixel entry 870254000600 and
             * then this SMP. KHR-GL30.ext_texture_shadow_lod's textureLod
             * and bias on samplerCubeArrayShadow are this shape. */
            CHECK(agx3_tex_encode_sample(enc, &d, bytes, &len));
            CHECK(hex_eq(bytes, len, "05800ca090002803010000000100"));
         }
      }
   }
   /* texture2d_array gradient2d + min_lod_clamp. */
   CHECK(agx3_tex_stage_array_minimum_gradient(enc, 14, 15, 1, 0, false, true,
                                               false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "17055406001c3c00f0c02d00"));
   CHECK(!agx3_tex_stage_array_minimum_gradient(enc, 15, 14, 1, 0, false,
                                                true, false, true, bytes,
                                                &len));
   static const struct {
      bool gradient, offset, shadow, three_d, indexed;
      uint16_t texture, sampler;
      const char *hex;
      bool array;
   } rows[] = {
      {false, false, false, false, false, 0, 0, "05800cb8b0000408010010000100"},
      {true, false, false, false, false, 0, 0, "05800cb8b0001401000010000100"},
      {false, true, false, false, false, 0, 0, "05800cb8b000c10201881000010e"},
      {true, true, false, false, false, 0, 0, "05800cb8b000490200881000010e"},
      {false, false, true, false, false, 0, 0, "05800ca090002408010000000100"},
      {true, false, true, false, false, 0, 0, "05800ca090003401000000000100"},
      {false, false, false, true, false, 0, 0, "05800cb8b0009021010010000100"},
      {true, false, false, true, false, 0, 0, "05800cb8b0009c21000010000100"},
      {false, false, false, false, true, 0, 4, "05800cb8b4000408010290010100a02800000000"},
      {true, false, false, false, true, 8, 12, "05c00cb8bc001401000290010100a02800000000"},
      {false, false, false, false, false, 0, 0, "05800cb8b0000c00010010000100", true},
      {true, false, false, false, false, 0, 0, "05800cb8b0001c01000010000100", true},
      {false, true, false, false, false, 0, 0, "05800cb8b000c10301881000010e", true},
      {true, true, false, false, false, 0, 0, "05800cb8b000490300881000010e", true},
      {true, false, true, false, false, 0, 0, "05800ca090003c01000000000100", true},
      {true, true, true, false, false, 0, 0, "05800ca09000690300880000010e", true},
   };
   for (unsigned i = 0; i < sizeof(rows) / sizeof(rows[0]); ++i) {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32,
         .mask = rows[i].shadow ? 1 : 15, .slot = 1, .wait_mask = 1,
         .coord_kill = true, .final = true, .min_lod = true,
         .lod = rows[i].gradient ? AGX3_TEX_LOD_GRADIENT
                                 : AGX3_TEX_LOD_BIAS_REGISTER,
         .offset = rows[i].offset, .offset_xyz = rows[i].offset ? 0xf1 : 0,
         .shadow = rows[i].shadow, .three_d = rows[i].three_d,
         .texture_kind = rows[i].indexed ? AGX3_TEX_OPERAND_REGISTER
                                         : AGX3_TEX_OPERAND_IMMEDIATE,
         .sampler_kind = rows[i].indexed ? AGX3_TEX_OPERAND_REGISTER
                                         : AGX3_TEX_OPERAND_IMMEDIATE,
         .texture_value = rows[i].texture, .sampler_value = rows[i].sampler,
         .texture_kill = rows[i].indexed, .sampler_kill = rows[i].indexed,
         .array = rows[i].array, .array_layer = rows[i].array ? 4 : 0,
      };
      CHECK(agx3_tex_encode_sample(enc, &d, bytes, &len));
      if (!hex_eq(bytes, len, rows[i].hex)) {
         fprintf(stderr, "minimum LOD native row %u\n", i);
         return 1;
      }
      checks++;
      /* Only the biased cube forms are witnessed. */
      const enum agx3_tex_lod lod = d.lod;
      d.cube = true;
      d.lod = AGX3_TEX_LOD_GRADIENT;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      d.cube = false;
      d.lod = lod;
      if (rows[i].array && rows[i].shadow && !rows[i].offset) {
         /* The comparison bias reads its minimum from a register pack. */
         d.lod = AGX3_TEX_LOD_BIAS_REGISTER;
         CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
         d.lod = AGX3_TEX_LOD_GRADIENT;
      }
      if (!rows[i].array || rows[i].shadow) {
         d.min_lod_register = true;
         CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
         d.min_lod_register = false;
      }
      d.vertex_stage = true;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      d.vertex_stage = false;
      d.query_lod = true;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
      d.query_lod = false;
      d.lod = AGX3_TEX_LOD_LEVEL;
      CHECK(!agx3_tex_encode_sample(enc, &d, bytes, &len));
   }
   return 0;
}

/* Fresh G17P runtime-descriptor gather kernels, with native register numbers.
 * These bytes test operand typing, packed layer preservation, f141=2 and the
 * cube dimension independently of the IR packer. */
static int
test_runtime_gather_witnesses(void)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH], len;
   CHECK(agx3_tex_stage_gather_offset(enc, 0, false, 0, 1, false, true,
                                      bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540600000400f0c00100"));
   CHECK(agx3_tex_pack_ms_array_layer(enc, 14, 8, false, true, bytes, &len));
   CHECK(hex_eq(bytes, len, "17055408031c0000f0880400"));
   CHECK(agx3_tex_stage_gather_offset(enc, 16, true, 8, 6, false, true,
                                      bytes, &len));
   CHECK(hex_eq(bytes, len, "1705540600204440f0e60300"));
   CHECK(!agx3_tex_stage_gather_offset(enc, 128, false, 0, 1, false, false,
                                       bytes, &len));
   CHECK(!agx3_tex_stage_gather_offset(enc, 1, false, 0, 1, false, false,
                                       bytes, &len));
   for (unsigned shape = 0; shape < 6; ++shape) {
      struct agx3_tex_sample_desc d = {
         .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
         .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 32,
         .coord_kill = true, .final = true,
         .texture_kind = AGX3_TEX_OPERAND_REGISTER,
         .sampler_kind = AGX3_TEX_OPERAND_REGISTER,
         .texture_value = 0, .sampler_value = 4,
         .texture_kill = true, .sampler_kill = true,
         .gather = AGX3_TEX_GATHER_R, .shadow = shape & 1,
         .offset = shape < 4, .offset_register = shape < 4,
         .array = shape == 2 || shape == 3, .cube = shape >= 4,
      };
      const char *native[] = {
         "05800ca414004000010280010100a02800004000",
         "05800ca414006000010280010100a02800004000",
         "05800ca414004004010280010100a02800004000",
         "05800ca414006004010280010100a02800004000",
         "05800ca414001000010280010100a02800004000",
         "05800ca414003000010280010100a02800004000",
      };
      CHECK(agx3_tex_encode_sample(enc, &d, bytes, &len));
      CHECK(hex_eq(bytes, len, native[shape]));
   }
   return 0;
}

/*
 * G17G (AGX3C) 2D sample LOD, offset and comparison witnesses. Fifteen
 * fragment render pipelines (gpu/m5-sample-lod-reference.metal) compiled by
 * Apple's offline translator for applegpu_g17g (MetalToolchain v27.1.5194.15,
 * -arch applegpu_g17g); the sample bytes were read straight from the native
 * text sections and recomposed byte-exact through the AGX3C tables. Every
 * form carries the AGX3B field values on top of the implicit-LOD control:
 * bias(reg) f139=3, level(reg) f139=4, bias(k) f139=1/f37/f29=k*64,
 * level(k) f139=2/f37/f29, offset f43/f51/f141, comparison f143/f75=1/no f74.
 * Depth comparisons without an offset take the compact ten-byte form.
 */
static int
test_g17g_sample_lod_witnesses(void)
{
   struct gvec {
      const char *name;
      struct agx3_tex_sample_desc d;
      const char *hex;
   };
   static const struct gvec v[] = {
      {"t.sample(s, uv)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0f061600418060290118010000000000008000000000"},
      {"t.sample(s, uv, bias(reg))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0f061600458061290118010000000000008000000000"},
      {"t.sample(s, uv, level(reg))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0f071600458060290118010000000000008000000000"},
      {"t.sample(s, uv, bias(0.5f))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS, .lod_x64 = 32, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true},
       "0f051600458060290100010020000000008000000000"},
      /* A lone fragment immediate level terminates its own group on G17G
       * (f156 set), unlike the compute t_base witness; the stream packer
       * supplies sample_group_end for AGX3C fragment programs. */
      {"t.sample(s, uv, level(2.0f))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 128, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .sample_group_end = true},
       "0f041600418060290100010080000000008000000000"},
      {"t.sample(s, uv, int2(1, 2))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true, .offset_xyz = 0x21},
       "0f061600418063290118010000802100008000000000"},
      {"t.sample(s, uv, bias(reg), int2(1, 2))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
        .lod = AGX3_TEX_LOD_BIAS_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .offset = true, .offset_xyz = 0x21},
       "0f061600458060290100030000802100008000000000"},
      {"depth.sample_compare(s, uv, r)",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .shadow = true},
       "0f071600418000210018"},
      {"depth.sample_compare(s, uv, r, bias(reg))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_BIAS_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .shadow = true},
       "0f071600458001210018"},
      {"depth.sample_compare(s, uv, r, level(reg))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_REGISTER, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .shadow = true},
       "0f041600418001210018"},
      {"depth.sample_compare(s, uv, r, int2(1, 2))",
       {.dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 1,
        .lod = AGX3_TEX_LOD_AUTO, .slot = 1, .wait_mask = 1,
        .coord_kill = true, .final = true, .shadow = true,
        .offset = true, .offset_xyz = 0x21},
       "0f071600418003210118010000802100000000000000"},
   };

   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;
   for (unsigned i = 0; i < sizeof(v) / sizeof(v[0]); ++i) {
      if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &v[i].d, bytes,
                                  &length)) {
         fprintf(stderr, "G17G sample witness refused: %s\n", v[i].name);
         return 1;
      }
      if (!(length == strlen(v[i].hex) / 2 && hex_eq(bytes, length, v[i].hex))) {
         fprintf(stderr, "G17G sample witness mismatch: %s\n", v[i].name);
         return 1;
      }
      ++checks;
   }

   /* The per-lane level and bias publication, identical for every G17G
    * form witnessed (half 6, slot 1, discarded source, float kind). */
   CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3C, 6, 1, 0, false, true,
                            bytes, &length));
   CHECK(length == 14 && hex_eq(bytes, length, "3702002a2500aa0a000040301000"));
   CHECK(!agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3C, 6, 1, 0, true, false,
                             bytes, &length));
   /* The integer kind publishes a fetch level from one half
    * (gpu/m5-fetch-lod-varying-reference.metal: a uint level clamped to 16
    * bits and a ushort level stage identically apart from the half). */
   CHECK(agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3C, 2, 1, 0, false,
                                  true, AGX3_TEX_LEVEL_INTEGER, bytes,
                                  &length));
   CHECK(length == 14 && hex_eq(bytes, length, "3702000a2500aa0a000040301100"));
   CHECK(agx3_tex_stage_typed_lod(AGX_ISA_ENCODING_AGX3C, 0, 1, 0, false,
                                  true, AGX3_TEX_LEVEL_INTEGER, bytes,
                                  &length));
   CHECK(length == 14 && hex_eq(bytes, length, "3700000a2500aa0a000040301100"));

   /* G17G depth2d_array comparisons and vertex-stage samples
    * (gpu/m5-shadow-array-cube-reference.metal): the implicit form is the
    * ten-byte ep1 comparison with the staged layer, a register level or
    * bias moves to the fourteen-byte ep2 form, the vertex stage drops f156
    * and its 2D register level is the 22-byte ep3 form. */
   {
      static const struct {
         const char *name;
         struct agx3_tex_sample_desc d;
         const char *hex;
      } sv[] = {
         {"depth2d_array sample_compare", {.width=AGX3_TEX_W32,.mask=1,
            .sampler=1,.slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .shadow=true,.array=true}, "0f041600458103210018"},
         {"depth2d_array level(reg)", {.width=AGX3_TEX_W32,.mask=1,
            .sampler=1,.slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .shadow=true,.array=true,.lod=AGX3_TEX_LOD_REGISTER},
          "0f05160041810021010002000000"},
         {"depth2d_array bias(reg)", {.width=AGX3_TEX_W32,.mask=1,
            .sampler=1,.slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .shadow=true,.array=true,.lod=AGX3_TEX_LOD_BIAS_REGISTER},
          "0f04160041810121010002000000"},
         {"vertex depth2d_array level(reg)", {.dest=8,.width=AGX3_TEX_W32,
            .mask=1,.slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .shadow=true,.array=true,.vertex_stage=true,
            .lod=AGX3_TEX_LOD_REGISTER}, "4f05120041800021010002000000"},
         {"vertex texture2d level(reg)", {.dest=8,.width=AGX3_TEX_W32,
            .mask=15,.slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .vertex_stage=true,.lod=AGX3_TEX_LOD_REGISTER},
          "4f071200458060290118010000000000008000000000"},
      };
      for (unsigned i = 0; i < sizeof(sv) / sizeof(sv[0]); ++i) {
         if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &sv[i].d, bytes,
                                     &length)) {
            fprintf(stderr, "G17G array witness refused: %s\n", sv[i].name);
            return 1;
         }
         if (!(length == strlen(sv[i].hex) / 2 &&
               hex_eq(bytes, length, sv[i].hex))) {
            fprintf(stderr, "G17G array witness mismatch: %s\n", sv[i].name);
            return 1;
         }
         ++checks;
      }
      /* Their staged operands: layer 8 with level/bias half 9, the cube
       * face publication and the cube-array face/layer word. */
      CHECK(agx3_tex_stage_array_lod(AGX_ISA_ENCODING_AGX3C, 8, 9, 1, 0, false,
                                     true, false, true, bytes, &length));
      CHECK(length == 14 && hex_eq(bytes, length, "3700004a2500aa0a400260302100"));
      {
         uint8_t bank0[AGX3_TEX_MAX_LENGTH], bank8[AGX3_TEX_MAX_LENGTH],
                 bank248[AGX3_TEX_MAX_LENGTH];
         uint8_t len0 = 0, len8 = 0, len248 = 0;
         uint64_t f0[AGX3_MAX_FIELDS], f8[AGX3_MAX_FIELDS],
                  f248[AGX3_MAX_FIELDS];
         CHECK(agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3C, 0, 8, 9, 1,
                                           0, false, true, false, true, bank0,
                                           &len0));
         CHECK(len0 == length && memcmp(bank0, bytes, length) == 0);
         CHECK(agx3_decode(&agx3c_table, bank0, len0, AGX3_MN_pack_smpir, f0) >=
               0);
         CHECK(f0[5] == 6);

         CHECK(agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3C, 8, 8, 9, 1,
                                           0, false, true, false, true, bank8,
                                           &len8));
         CHECK(len8 == 14 &&
               agx3_decode(&agx3c_table, bank8, len8, AGX3_MN_pack_smpir, f8) >=
                  0);
         CHECK(f8[5] == 14);
         f8[5] = f0[5];
         CHECK(memcmp(f8, f0, sizeof(f0)) == 0);

         CHECK(agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3C, 248, 8, 9,
                                           1, 0, false, true, false, true,
                                           bank248, &len248));
         CHECK(len248 == 14 &&
               agx3_decode(&agx3c_table, bank248, len248, AGX3_MN_pack_smpir,
                           f248) >= 0);
         CHECK(f248[5] == 254);
         f248[5] = f0[5];
         CHECK(memcmp(f248, f0, sizeof(f0)) == 0);

         CHECK(!agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3C, 4, 8, 9, 1,
                                            0, false, true, false, true, bank8,
                                            &len8));
         CHECK(!agx3_tex_stage_array_lod_at(AGX_ISA_ENCODING_AGX3C, 256, 8, 9,
                                            1, 0, false, true, false, true,
                                            bank8, &len8));
      }
      CHECK(agx3_tex_stage_cube_z(AGX_ISA_ENCODING_AGX3C, 2, 1, 0, false, true,
                                  bytes, &length));
      CHECK(length == 14 && hex_eq(bytes, length, "3702000a2500aa0a00005a300900"));
      CHECK(agx3_tex_stage_cube_array_shadow(AGX_ISA_ENCODING_AGX3C, 6, 1, 0,
                                             false, true, bytes, &length));
      CHECK(length == 14 && hex_eq(bytes, length, "3703000a2500aa0a800060300100"));
      /* The G17G cube witnesses all compare, over immediate descriptor
       * operands, at the implicit LOD or a per-lane level/bias. A cube
       * sample outside that -- an immediate level, a texel offset -- stays
       * refused rather than extrapolated. */
      struct agx3_tex_sample_desc cube = sv[0].d;
      cube.cube = true;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &cube, bytes,
                                   &length));

      struct agx3_tex_sample_desc closed_cube = cube;
      closed_cube.lod = AGX3_TEX_LOD_LEVEL;
      closed_cube.lod_x64 = 128;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &closed_cube, bytes,
                                    &length));

      closed_cube = cube;
      closed_cube.offset = true;
      closed_cube.offset_xyz = 0xf1;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &closed_cube, bytes,
                                    &length));
   }

   /* Several in-flight G17G comparisons (gpu/m5-shadow-2d-reference.metal,
    * the KHR-GL30 texture_repeat_mode depth fragment: three sample_compare
    * calls against 0.3 / 0.5 / 0.8). Apple stages each sample into its own
    * eight-half block -- coordinates at +0/+2, reference at +4, bias at +6
    * -- names the block with f19 = 0 / 8 / 16, waits on the block's own
    * staging slot, and clears f156 and f150 on every member but the last.
    * The SMPs themselves issued on result slots 1 / 1 / 2; the packer keys
    * both the block and the wait on the sample's own slot, so its second
    * and third samples differ from these bytes in the issue-slot fields
    * alone (2f071208cd0001410018 / 1f071610558001630018 for the bias
    * forms). The level-zero comparison is the eight-byte ep0 form; its
    * staged-layer twin keeps ep1. */
   {
      static const struct {
         const char *name;
         struct agx3_tex_sample_desc d;
         const char *hex;
      } mv[] = {
         {"three sample_compare #1", {.width=AGX3_TEX_W32,.mask=1,.slot=1,
            .wait_mask=1,.coord_kill=true,.shadow=true,
            .sample_group_nonterminal=true}, "0f071200410000210018"},
         {"three sample_compare #2", {.dest=4,.coord=8,.width=AGX3_TEX_W32,
            .mask=1,.slot=1,.wait_mask=2,.coord_kill=true,.shadow=true,
            .sample_group_nonterminal=true}, "2f071208410000410018"},
         {"three sample_compare #3", {.dest=2,.coord=16,.width=AGX3_TEX_W32,
            .mask=1,.slot=2,.wait_mask=4,.coord_kill=true,.final=true,
            .shadow=true}, "1f071610c98000610018"},
         {"three sample_compare bias(reg) #1", {.width=AGX3_TEX_W32,.mask=1,
            .slot=1,.wait_mask=1,.coord_kill=true,.shadow=true,
            .lod=AGX3_TEX_LOD_BIAS_REGISTER,.sample_group_nonterminal=true},
          "0f071200450001210018"},
         {"three sample_compare bias(reg) #2", {.dest=4,.coord=8,
            .width=AGX3_TEX_W32,.mask=1,.slot=1,.wait_mask=2,.coord_kill=true,
            .shadow=true,.lod=AGX3_TEX_LOD_BIAS_REGISTER,
            .sample_group_nonterminal=true}, "2f071208450001410018"},
         {"three sample_compare bias(reg) #3", {.dest=2,.coord=16,
            .width=AGX3_TEX_W32,.mask=1,.slot=2,.wait_mask=4,.coord_kill=true,
            .final=true,.shadow=true,.lod=AGX3_TEX_LOD_BIAS_REGISTER},
          "1f071610cd8001610018"},
         {"packer: bias(reg) #2 on its own slot", {.dest=4,.coord=8,
            .width=AGX3_TEX_W32,.mask=1,.slot=2,.wait_mask=2,.coord_kill=true,
            .shadow=true,.lod=AGX3_TEX_LOD_BIAS_REGISTER,
            .sample_group_nonterminal=true}, "2f071208cd0001410018"},
         {"packer: bias(reg) #3 on its own slot", {.dest=2,.coord=16,
            .width=AGX3_TEX_W32,.mask=1,.slot=3,.wait_mask=4,.coord_kill=true,
            .final=true,.shadow=true,.lod=AGX3_TEX_LOD_BIAS_REGISTER},
          "1f071610558001630018"},
         {"sample_compare level(0)", {.width=AGX3_TEX_W32,.mask=1,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.shadow=true,
            .lod=AGX3_TEX_LOD_LEVEL,.sample_group_end=true},
          "0f05160040800021"},
         {"depth2d_array sample_compare level(0)", {.width=AGX3_TEX_W32,
            .mask=1,.slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .shadow=true,.array=true,.lod=AGX3_TEX_LOD_LEVEL,
            .sample_group_end=true}, "0f051600458000210008"},
         /* Apple's two-texture colour fragment (gpu/m5-texture-abi-
          * reference.metal) clears the same two fields on its first sample
          * and leaves the shared coordinate block alive (f27 = 0). */
         {"two_textures #1", {.width=AGX3_TEX_W32,.mask=15,.slot=2,
            .wait_mask=1,.sample_group_nonterminal=true},
          "0f060200c90060290118010000000000008000000000"},
      };
      for (unsigned i = 0; i < sizeof(mv) / sizeof(mv[0]); ++i) {
         if (!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &mv[i].d, bytes,
                                     &length)) {
            fprintf(stderr, "G17G group witness refused: %s\n", mv[i].name);
            return 1;
         }
         if (!(length == strlen(mv[i].hex) / 2 &&
               hex_eq(bytes, length, mv[i].hex))) {
            fprintf(stderr, "G17G group witness mismatch: %s\n", mv[i].name);
            return 1;
         }
         ++checks;
      }
      /* A member marked both nonterminal and terminal has no witness. */
      struct agx3_tex_sample_desc closed = mv[0].d;
      closed.sample_group_end = true;
      CHECK(!agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &closed, bytes,
                                    &length));

      /* The block copies: one coordinate pair into blocks 0, 8 and 16 on
       * slots 1, 2 and 3, f25 while the pair is still needed and f27 on its
       * last block, and the bias(reg) twin's three block words. */
      static const struct {
         uint16_t dest, src;
         uint8_t slot;
         bool keep;
         const char *hex;
      } cv[] = {
         {0, 2, 1, true, "0b822104"},   {2, 0, 1, true, "1b802104"},
         {8, 2, 2, true, "238207022281a412"},
         {10, 0, 2, true, "338007022281a412"},
         {16, 2, 3, false, "a30207022682a402"},
         {18, 0, 3, false, "b30007022682a402"},
         {0, 4, 1, true, "0b842104"},   {2, 2, 1, true, "1b822104"},
         {6, 0, 1, true, "3b802104"},
         {8, 4, 2, true, "2380070a2281a412"},
         {10, 2, 2, true, "338207022281a412"},
         {14, 0, 2, true, "338007022281a41a"},
         {16, 4, 3, false, "a300070a2682a402"},
         {18, 2, 3, false, "b30207022682a402"},
         {22, 0, 3, false, "b30007022682a40a"},
      };
      for (unsigned i = 0; i < sizeof(cv) / sizeof(cv[0]); ++i) {
         CHECK(agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3C, cv[i].dest,
                                         cv[i].src, cv[i].slot, 0, cv[i].keep,
                                         !cv[i].keep, bytes, &length));
         CHECK(length == strlen(cv[i].hex) / 2 &&
               hex_eq(bytes, length, cv[i].hex));
      }
      /* Neither hint and both hints are unwitnessed on G17G. */
      CHECK(!agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3C, 0, 2, 1, 0,
                                       false, false, bytes, &length));
      CHECK(!agx3_tex_stage_copy_hints(AGX_ISA_ENCODING_AGX3C, 0, 2, 1, 0,
                                       true, true, bytes, &length));

      /* The bias word relay: packed once into r0 from half 6, then copied
       * into each block above. The register destination must be an even
       * half below 128; the bank-6 stager is unchanged. */
      CHECK(agx3_tex_pack_lod_word(AGX_ISA_ENCODING_AGX3C, 0, 6, false, true,
                                   bytes, &length));
      CHECK(length == 14 &&
            hex_eq(bytes, length, "2702003a2d00aa02000040301000"));
      CHECK(!agx3_tex_pack_lod_word(AGX_ISA_ENCODING_AGX3C, 1, 6, false, true,
                                    bytes, &length));
      CHECK(!agx3_tex_pack_lod_word(AGX_ISA_ENCODING_AGX3C, 128, 6, false,
                                    true, bytes, &length));
      CHECK(!agx3_tex_pack_lod_word(AGX_ISA_ENCODING_AGX3C, 0, 6, false, false,
                                    bytes, &length));
      CHECK(!agx3_tex_pack_lod_word(AGX_ISA_ENCODING_AGX3B, 0, 6, false, true,
                                    bytes, &length));
      CHECK(agx3_tex_stage_lod(AGX_ISA_ENCODING_AGX3C, 6, 1, 0, false, true,
                               bytes, &length));
      CHECK(length == 14 &&
            hex_eq(bytes, length, "3702002a2500aa0a000040301000"));
   }

   /* G17G fragment fetches (gpu/m5-fetch-lod-reference.metal): a run-time
    * level takes the ten-byte form with f139 = 4 and no immediate, a
    * constant level 1 the fourteen-byte form with f29 = 64, a depth2d read
    * carries mask x, and an argument-buffer texture names its descriptor
    * through a uniform pair. Apple's fragment fetches name sampler 1. */
   {
      static const struct {
         const char *name;
         struct agx3_tex_fetch_desc d;
         const char *hex;
      } fv[] = {
         {"read(coord, level)", {.width=AGX3_TEX_W32,.mask=15,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.sampler_value=1,
            .register_lod=true}, "0f051a00418161290018"},
         {"depth.read(coord, level)", {.width=AGX3_TEX_W32,.mask=1,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.sampler_value=1,
            .register_lod=true}, "0f051a00418101210018"},
         {"depth.read(coord)", {.width=AGX3_TEX_W32,.mask=1,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.sampler_value=1},
          "0f061a0040810021"},
         {"read(coord, 1)", {.width=AGX3_TEX_W32,.mask=15,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.sampler_value=1,
            .lod_x64=64}, "0f061a0041816029010000004000"},
         {"argument read(coord, level)", {.width=AGX3_TEX_W32,.mask=15,
            .slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .sampler_value=1,.texture_kind=AGX3_TEX_OPERAND_UNIFORM,
            .texture_value=16,.register_lod=true}, "0f051a00418161280818"},
         {"argument depth.read(coord)", {.width=AGX3_TEX_W32,.mask=1,
            .slot=1,.wait_mask=1,.coord_kill=true,.final=true,
            .sampler_value=1,.texture_kind=AGX3_TEX_OPERAND_UNIFORM,
            .texture_value=12}, "0f061a0040e10020"},
         /* Compute texture3d<float>::read(uint3), whole vector and .x
          * (gpu/m5-xfb-alu-reference.metal): the 2D fetch with the 3D
          * selector, sampler 0. */
         {"texture3d read(coord)", {.width=AGX3_TEX_W32,.mask=15,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.three_d=true},
          "0f061a0044806329"},
         {"texture3d read(coord).x", {.width=AGX3_TEX_W32,.mask=1,.slot=1,
            .wait_mask=1,.coord_kill=true,.final=true,.three_d=true},
          "0f061a0044800321"},
      };
      for (unsigned i = 0; i < sizeof(fv) / sizeof(fv[0]); ++i) {
         if (!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &fv[i].d, bytes,
                                    &length)) {
            fprintf(stderr, "G17G fetch witness refused: %s\n", fv[i].name);
            return 1;
         }
         if (!(length == strlen(fv[i].hex) / 2 &&
               hex_eq(bytes, length, fv[i].hex))) {
            fprintf(stderr, "G17G fetch witness mismatch: %s\n", fv[i].name);
            return 1;
         }
         ++checks;
      }
      struct agx3_tex_fetch_desc closed_fetch = fv[0].d;
      closed_fetch.lod_x64 = 64;
      CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &closed_fetch,
                                   bytes, &length));
      closed_fetch = fv[0].d;
      closed_fetch.multisample = true;
      CHECK(!agx3_tex_encode_fetch(AGX_ISA_ENCODING_AGX3C, &closed_fetch,
                                   bytes, &length));
   }

   /* The level-zero comparison is witnessed (0f05160040800021, see the
    * group block above). */
   struct agx3_tex_sample_desc closed = v[7].d;
   closed.lod = AGX3_TEX_LOD_LEVEL;
   closed.sample_group_end = true;
   CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3C, &closed, bytes,
                                &length));
   CHECK(length == 8 && hex_eq(bytes, length, "0f05160040800021"));
   return 0;
}

/*
 * Sparse texel residency.
 *
 * Witness: xcrun metal-nt -arch applegpu_g17p -N k.mtlp-json
 *          -platform_version macos 27.0 27.0 (MetalToolchain v27.1.5194.15),
 * __TEXT,__text of each compute object.  Both arms of each pair bind the
 * texture and sampler through the same argument buffer, so the only source
 * difference is sparse_sample/sparse_read against sample/read:
 *
 *   sample  k_ab_dense  05800cb8b4000900920390010100802000800098
 *           k_ab_sparse 2dc00c38bc000900920392010200802028c00018
 *   fetch   k_ab_dread  05800cb8b0001701900280000100002000800098
 *           k_ab_sread  2dc00c38b0001701900282000200002028c00018
 *
 * Decoding both pairs against the generated AGX3b encoder tables leaves
 * exactly three field differences beyond register allocation and the group
 * terminator: f112 4 -> 8, f120 0 -> 1 and f142 0 -> 1.
 *
 * This encoder reproduces two of those three.  It deliberately leaves f112 at
 * its dense value, because moving it clears operand-profile bit 2 and G17P
 * then fails every sparse fetch and every explicit-LOD sparse sample;
 * agx3_tex_encode_sample() records the hardware measurement.  So byte 12
 * below reads 01 on both arms, where Apple's sparse arm reads 02.
 *
 * The goldens below are this encoder's bytes, which reproduce Apple's
 * through byte 15 apart from that and carry f120 at byte 17 bit 6.  They
 * differ from Apple's in three further ep4 places -- the destination copy in
 * f19's upper byte, f91, and bits 154..159 -- and those three are not
 * specific to sparse: the dense arms below, and every other ep4 texture
 * operation this encoder emits, differ from Apple in exactly the same way.
 * See the KNOWN GAP note at the top of agx3_tex_core.c.  What this test pins
 * is therefore the sparse delta this encoder does emit: byte 10 bit 1 (f142)
 * and byte 17 bit 6 (f120), and nothing else.
 */
static int
test_sparse_residency(void)
{
   const enum agx_isa_encoding enc = AGX_ISA_ENCODING_AGX3B;
   uint8_t bytes[AGX3_TEX_MAX_LENGTH];
   uint8_t length;

   /* Apple's own sparse sample, byte for byte through byte 18. */
   struct agx3_tex_sample_desc sample = {
      .dest = 10, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_LEVEL, .lod_x64 = 0, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = false, .compute_stage = true,
      .texture_kind = AGX3_TEX_OPERAND_UNIFORM, .texture_value = 24,
      .sampler_kind = AGX3_TEX_OPERAND_UNIFORM, .sampler_value = 28,
      .sparse = true,
   };
   CHECK(agx3_tex_encode_sample(enc, &sample, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "2dc00c38bc000900920392010100802000400000"));

   /* The dense sibling differs in exactly the two sparse fields this encoder
    * emits: byte 10 bit 1 (f142) and byte 17 bit 6 (f120).  Byte 12, the
    * operand profile, stays put. */
   struct agx3_tex_sample_desc dense = sample;
   dense.sparse = false;
   CHECK(agx3_tex_encode_sample(enc, &dense, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "2dc00c38bc000900920390010100802000000000"));

   /* The texel fetch carries the same two fields. */
   struct agx3_tex_fetch_desc fetch = {
      .dest = 10, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = false,
      .texture_kind = AGX3_TEX_OPERAND_UNIFORM, .texture_value = 24,
      .sparse = true,
   };
   CHECK(agx3_tex_encode_fetch(enc, &fetch, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "2dc00c38b0001701900282000100002000400000"));

   struct agx3_tex_fetch_desc dense_fetch = fetch;
   dense_fetch.sparse = false;
   CHECK(agx3_tex_encode_fetch(enc, &dense_fetch, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "2dc00c38b0001701900280000100002000000000"));

   /* A sparse fetch round-trips through the strict decoder, and the decoded
    * descriptor reports the residency request. */
   CHECK(agx3_tex_encode_fetch(enc, &fetch, bytes, &length));
   struct agx3_tex_fetch_desc decoded;
   CHECK(agx3_tex_decode_fetch(enc, bytes, length, &decoded));
   CHECK(decoded.sparse && decoded.dest == 10 && decoded.mask == 15 &&
         decoded.texture_kind == AGX3_TEX_OPERAND_UNIFORM &&
         decoded.texture_value == 24);
   CHECK(agx3_tex_encode_fetch(enc, &dense_fetch, bytes, &length));
   CHECK(agx3_tex_decode_fetch(enc, bytes, length, &decoded));
   CHECK(!decoded.sparse);

   /* f120 deposits instruction bit 142, past the 112-bit ep3 form, so a
    * sparse operation takes the 20-byte ep4 form even when both descriptors
    * are immediate -- where the dense sibling stays 14 bytes. */
   struct agx3_tex_sample_desc immediate = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .lod = AGX3_TEX_LOD_LEVEL, .slot = 1, .wait_mask = 1,
      .coord_kill = true, .final = true, .compute_stage = true,
   };
   CHECK(agx3_tex_encode_sample(enc, &immediate, bytes, &length));
   CHECK(length == 14);
   immediate.sparse = true;
   CHECK(agx3_tex_encode_sample(enc, &immediate, bytes, &length));
   CHECK(length == 20 &&
         hex_eq(bytes, length, "05800cb8b0000900000212000100000000400000"));

   /* Negative: a depth comparison and a LOD query have no sparse witness. */
   struct agx3_tex_sample_desc bad = sample;
   bad.shadow = true;
   bad.mask = 1;
   CHECK(!agx3_tex_encode_sample(enc, &bad, bytes, &length));
   bad = sample;
   bad.query_lod = true;
   bad.mask = 3;
   CHECK(!agx3_tex_encode_sample(enc, &bad, bytes, &length));

   /* Negative: no sparse multisample read is witnessed. */
   struct agx3_tex_fetch_desc bad_ms = {
      .dest = 0, .coord = 0, .width = AGX3_TEX_W32, .mask = 15,
      .slot = 1, .wait_mask = 1, .coord_kill = true, .final = true,
      .multisample = true, .sparse = true,
   };
   CHECK(!agx3_tex_encode_fetch(enc, &bad_ms, bytes, &length));

   return 0;
}

/* Local Metal 32023.850, applegpu_g17p, linked Subnautica 7064 VS/FS.
 * The native pipeline publishes distinct four-half coordinate ranges before
 * issuing their samples, including ranges above half 127. These are encoding
 * witnesses, not substituted shader code. */
static int
test_linked_staging_witnesses(void)
{
   const struct { unsigned dest, coord, slot, wait, texture, sampler; bool final;
                  const char *hex; } cases[] = {
      {0, 4, 1, 2, 88, 92, false, "05c20c383c010900da0390010100802000000000"},
      {10, 8, 1, 4, 88, 92, false, "2dc40c383c020900da0390010100802000000000"},
      {18, 12, 1, 8, 88, 92, false, "4dc60c383c000900da0390010100802000001000"},
      {26, 16, 1, 16, 88, 92, true, "6dc00cb93c000900da0390010100802000002000"},
      {34, 84, 5, 16, 96, 100, false, "8d820c3934900900ec0390010100802001002000"},
      {84, 140, 5, 32, 96, 100, false, "55864c3834900900ec0390010100802002004000"},
      {74, 132, 4, 8, 96, 100, false, "2d824c38346c0900ec0390010100802002001000"},
      {66, 128, 4, 4, 96, 100, false, "0d804c38346e0900ec0390010100802002000000"},
      {102, 152, 5, 2, 96, 100, true, "9d844cb934910900ec0390010100802002000000"},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      const struct agx3_tex_sample_desc d = {
         .dest = cases[i].dest, .coord = cases[i].coord,
         .width = AGX3_TEX_W32, .mask = 15, .lod = AGX3_TEX_LOD_LEVEL,
         .slot = cases[i].slot, .wait_mask = cases[i].wait,
         .coord_kill = true, .final = cases[i].final,
         .texture_kind = AGX3_TEX_OPERAND_UNIFORM,
         .texture_value = cases[i].texture,
         .sampler_kind = AGX3_TEX_OPERAND_UNIFORM,
         .sampler_value = cases[i].sampler,
      };
      uint8_t bytes[AGX3_TEX_MAX_LENGTH], length;
      CHECK(agx3_tex_encode_sample(AGX_ISA_ENCODING_AGX3B, &d, bytes, &length));
      CHECK(hex_eq(bytes, length, cases[i].hex));
   }
   return 0;
}

int
main(void)
{
   if (test_linked_staging_witnesses() ||
       test_g17g_sample_lod_witnesses() ||
       test_m5_array_sample() || test_runtime_gather_witnesses() ||
       test_minimum_lod_witnesses() || test_register_lod() || test_fetch_lod() ||
       test_register_array_layer() || test_cube_coordinate_stage() ||
       test_multisample_fetch() ||
       test_multisample_sample_stage() ||
       test_array_sample_abi_differential() ||
       test_staging_source_lifetime() || test_exact_witnesses() ||
       test_supported_axes() || test_fail_closed() ||
       test_sample_witnesses() || test_copy_witnesses() ||
       test_sample_fail_closed() || test_sample_upper_coordinate_banks() ||
       test_array_fetch() || test_3d_fetch() ||
       test_bindless_operand_kinds() || test_indexed_sample_witnesses() ||
       test_sample_mask_domain() || test_indexed_fetch_native() ||
       test_gradient_and_register_bias() ||
       test_offset_witnesses() || test_indexed_offset_witnesses() ||
       test_vertex_offset_witnesses() ||
       test_shadow_lod_witnesses() ||
       test_3d_lod_witnesses() || test_image_write_state0_witnesses() ||
       test_query_lod_witnesses() || test_m5_indexed_reads() || test_sparse_residency())
      return 1;
   printf("agx3 texture core: %u checks passed\n", checks);
   return 0;
}
