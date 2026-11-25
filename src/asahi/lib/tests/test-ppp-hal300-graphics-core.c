/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#include "agx_ppp_hal300_graphics_core.h"
#include "agx_vdm_state_core.h"
#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); return 1; \
} } while (0)

int
main(void)
{
   /* Independently extracted M5/25G76 VDM-referenced payloads. All nine
    * payload boundaries matter: joining these behind one selector is wrong.
    * This witness uses clockwise winding and Ioto VA 0x10000170700. */
   const uint8_t golden[] = {
      0,0,0x10,0x10, 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0,
      0,0,0,1, 0,0,0,0, 0,0,0,0, 0,0,0,0,
      0,0,0,1, 0x1c,0x5c,0,0, 0,0,0,0,
      1,0,0,0, 0xaa,7,0,0, 0,0,0xe0,0xe2, 0,0,0xc0,0xf0,
      0x20,0x30,2,0, 0xff,0xff,1,0,
      0,0,0x28,0x0c, 0,0,0,0, 0,0,0,0, 0,0,0,0, 4,0,0,0,
      0xe0,0,0,2, 0,2,4,0, 0,0x0f,0x20,7, 0,0x0f,0x20,7,
      0,0,0,0x0e, 0,0,0,0x0e, 0,0,0,0,
      0x62,0,0,2, 0,2,0,0, 0,0,0xe0,7, 0,0,0xe0,7, 0xff,0xff,1,0,
      2,0,0x84,0, 0,0,0,0, 0x80,0,0,0,
      0,0,0x80,0, 0x80,4,0,0,
      0,4,0,0, 0,0,0,0, 0,0,0,0,
   };
   const uint16_t offsets[] = {0,20,48,72,92,120,140,152,160};
   const uint8_t lengths[] = {20,28,24,20,28,20,12,8,12};
   struct agx_ppp_hal300_graphics_fields f = {
      .fragment_program = UINT64_C(0x10000170700),
   };
   struct agx_ppp_hal300_graphics_core out;
   CHECK(sizeof(golden) == sizeof(out.bytes));
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(memcmp(out.bytes, golden, sizeof(golden)) == 0);
   CHECK(memcmp(out.offsets, offsets, sizeof(offsets)) == 0);
   CHECK(memcmp(out.lengths, lengths, sizeof(lengths)) == 0);

   /* Native HAL300 disassembly routes context+780c to word1 of the
    * 02000062 secondary record. For pass=1 it retains 20000200 there;
    * every primary-face byte and all other records must stay unchanged. */
   f.fragment_pass = 1;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   for (unsigned i = 0; i < sizeof(golden); ++i)
      CHECK(out.bytes[i] == (i == 127 ? 0x20 : golden[i]));
   const struct agx_ppp_hal300_graphics_core translucent = out;

   /* Punch through is witnessed (see the discard render below), and so is
    * translucent punch through; the pass types above them are not, and a
    * caller asking for one is refused without touching the caller's buffer. */
   f.fragment_pass = 2;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   for (unsigned i = 0; i < sizeof(golden); ++i)
      CHECK(out.bytes[i] == (i == 127 ? 0x40 : golden[i]));
   out = translucent;
   for (unsigned pass = 4; pass < 8; ++pass) {
      f.fragment_pass = pass;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
      CHECK(memcmp(&out, &translucent, sizeof(out)) == 0);
   }
   f.fragment_pass = 0;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(memcmp(out.bytes, golden, sizeof(golden)) == 0);

   /* Relocation must retain all 48 address bits, not only the low word. */
   f.fragment_program = UINT64_C(0xa5c001234540);
   f.cull_mode = 3;
   f.front_face_ccw = true;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   uint64_t address = 0;
   uint32_t raw_template = 1;
   CHECK(agx_vdm_state_vertex_pipeline_pair_unpack(
      AGX_CHIP_G18G, out.bytes + 36, 8, &address, &raw_template));
   CHECK(address == f.fragment_program && raw_template == 0);
   CHECK(out.bytes[156] == 0x83 && out.bytes[157] == 4 && out.bytes[158] == 1);

   const struct agx_ppp_hal300_graphics_core before = out;
   f.fragment_program++;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(memcmp(&out, &before, sizeof(out)) == 0);
   f.fragment_program = UINT64_C(1) << 48;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.fragment_program = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.fragment_program = UINT64_C(0x10000170700);
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G17P, &f, &out));
   f.cull_mode = 4;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(memcmp(&out, &before, sizeof(out)) == 0);
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, NULL, &out));
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, NULL));

   /* Independently captured smooth-color and flat-color/D32 draws. Both
    * native outputs passed the 60928-pixel oracle after these snapshots.
    * The flat reference has one CF binding, exactly as Zink gears does. */
   const uint32_t color_words[] = {
      0x10100000, 4, 0, 0, 0,
      0x01000000, 0x00020100, 0x4c0, 0, 0x01000000, 0x5c1c, 0,
      1, 0x7aa, 0xe2e00000, 0xf0c00000, 0x23020, 0x1ffff,
      0x0c280000, 0x50000, 0, 0, 9,
      0x020000e0, 0x40200, 0x07200f00, 0x07200f00, 0x0e000000, 0x0e000000, 0,
      0x02000062, 0x200, 0x07e00000, 0x07e00000, 0x1ffff,
      0x00840002, 0, 0x80, 0x00800000, 0x480, 0x400, 0, 0,
   };
   /* Native 25G76 colour-free render whose fragment function returns a
    * [[depth(any)]] struct: the pipeline template takes 0x40 and both ISP
    * early-test words drop 0x00c00000. Its varying-count word, pass type and
    * last ISP word are the colour-free and punch-through axes, checked
    * separately; this vector isolates the depth write. */
   const uint32_t zout_words[] = {
      0x10100000, 4, 0, 0, 0,
      0x01000000, 0x00020100, 0x4c0, 0, 0x01000040, 0x5c1c, 0,
      1, 0x7aa, 0xe2e00000, 0xf0c00000, 0x23020, 0x1ffff,
      0x0c280000, 0x50000, 0, 0, 9,
      0x020000e0, 0x40200, 0x07200f00, 0x07200f00, 0x0e000000, 0x0e000000, 0,
      0x02000062, 0x200, 0x07200000, 0x07200000, 0x1ffff,
      0x00840002, 0, 0x80, 0x00800000, 0x480, 0x400, 0, 0,
   };
   const uint32_t flat_depth_words[] = {
      0x10100000, 0x400, 0, 0, 0,
      0x01000000, 0x00010100, 0x480, 0, 0x01000000, 0x6e1c, 0,
      1, 0x7aa, 0xe2e00000, 0xf0c00000, 0x23020, 0x1ffff,
      0x0c280000, 0x50000, 0, 0, 9,
      0x020000e0, 0x200, 0x01000f00, 0x01000f00, 0x0e000000, 0x0e000000, 0,
      0x02000062, 0x200, 0x07e00000, 0x07e00000, 0x1ffff,
      0x00840002, 0, 0x80, 0x00800000, 0x480, 0x400, 0, 0,
   };
   CHECK(sizeof(color_words) == sizeof(out.bytes));
   CHECK(sizeof(flat_depth_words) == sizeof(out.bytes));
   f = (struct agx_ppp_hal300_graphics_fields) {
      .fragment_program = UINT64_C(0x10000170700),
      .varying_table = UINT64_C(0x100000004c0),
      .cf_binding_count = 2,
      .point_size_output = true,
      .vertex_varying_components = 4,
      .fragment_varying_components = 4,
   };
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   for (unsigned i = 0; i < sizeof(out.bytes); ++i)
      CHECK(out.bytes[i] == (uint8_t)(color_words[i / 4] >> (8 * (i % 4))));
   /* Only the depth write moves, against the colour reference above. */
   f.fragment_writes_depth = true;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   for (unsigned i = 0; i < sizeof(out.bytes); ++i)
      CHECK(out.bytes[i] == (uint8_t)(zout_words[i / 4] >> (8 * (i % 4))));
   f.fragment_writes_depth = false;

   f.fragment_program = UINT64_C(0x100001b8700);
   f.varying_table = UINT64_C(0x10000000480);
   f.cf_binding_count = 1;
   f.interpolation = AGX_PPP_HAL300_FLAT;
   f.depth_active = true;
   f.depth_compare = 1; /* less */
   f.depth_write = true;
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   for (unsigned i = 0; i < sizeof(out.bytes); ++i)
      CHECK(out.bytes[i] == (uint8_t)(flat_depth_words[i / 4] >> (8 * (i % 4))));

   /* One controlled render per depth comparison, from
    * gpu/m5-state-matrix-reference.m on 25G76. The face word is the only
    * thing that moves, and the two constants this driver already shipped
    * (always/no-write and less/write) are two rows of the same table.
    *
    * Seven of the eight rows are captured directly. lequal is not: its
    * render was queued when the guest's serial link dropped. It is included
    * because the other seven exhaust every other value of a three-bit field
    * and lequal is the one comparison left, so the table is closed by
    * enumeration rather than by extrapolation. Capture it when the next
    * guest is up and this comment goes away. */
   {
      static const struct { uint8_t compare; uint32_t face; } matrix[] = {
         {0 /* never */, 0x00000f00},    {1 /* less */, 0x01000f00},
         {2 /* equal */, 0x02000f00},    {3 /* lequal */, 0x03000f00},
         {4 /* greater */, 0x04000f00},  {5 /* notequal */, 0x05000f00},
         {6 /* gequal */, 0x06000f00},   {7 /* always */, 0x07000f00},
      };
      struct agx_ppp_hal300_graphics_core probe;
      for (unsigned i = 0; i < sizeof(matrix) / sizeof(matrix[0]); ++i) {
         struct agx_ppp_hal300_graphics_fields d = f;
         d.depth_active = true;
         d.depth_compare = matrix[i].compare;
         d.depth_write = true;
         CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
         const uint8_t *face = probe.bytes + probe.offsets[4] + 8;
         const uint32_t got = face[0] | (face[1] << 8) | (face[2] << 16) |
                              ((uint32_t)face[3] << 24);
         CHECK(got == matrix[i].face);
         const uint8_t *control = probe.bytes + probe.offsets[4] + 4;
         CHECK((control[2] & 0x4) == 0);


         /* Writes off sets 0x200000; an always/no-write render is the
          * inactive depth unit the no-attachment path already encodes. */
         d.depth_write = false;
         CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
         face = probe.bytes + probe.offsets[4] + 8;
         const uint32_t nowrite = face[0] | (face[1] << 8) | (face[2] << 16) |
                                  ((uint32_t)face[3] << 24);
         CHECK(nowrite == (matrix[i].face | 0x00200000));
         /* The test being enabled is what clears 0x40000, whatever the
          * comparison is: dz-always-nw carries 0x200 on hardware. */
         control = probe.bytes + probe.offsets[4] + 4;
         CHECK((control[2] & 0x4) == 0);
      }

      /* Stencil sets 0x80000 and replaces the two keep/always words. */
      struct agx_ppp_hal300_graphics_fields d = f;
      d.depth_active = false;
      d.depth_compare = 0;
      d.depth_write = false;
      d.stencil_active = true;
      d.front_stencil = d.back_stencil = 0x0e02ffff;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      const uint8_t *control = probe.bytes + probe.offsets[4] + 4;
      CHECK((control[2] & 0xc) == 0xc);
      const uint8_t *stencil = probe.bytes + probe.offsets[4] + 16;
      CHECK(stencil[0] == 0xff && stencil[1] == 0xff && stencil[2] == 0x02 &&
            stencil[3] == 0x0e);

      d.depth_active = true;
      d.depth_compare = 8;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      /* A comparison without an active depth unit is refused. */
      d.depth_active = false;
      d.depth_compare = 1;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
   }

   /* Punch through, from the discarding-shader render of the capture
    * matrix: pass 2 in bits 29..31 with the shader's own
    * disable-triangle-merging bit 26 beside it. */
   {
      struct agx_ppp_hal300_graphics_fields d = f;
      struct agx_ppp_hal300_graphics_core probe;
      d.fragment_pass = 2;
      d.disable_tri_merging = true;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      const uint8_t *isp = probe.bytes + probe.offsets[5] + 4;
      const uint32_t word = isp[0] | (isp[1] << 8) | (isp[2] << 16) |
                            ((uint32_t)isp[3] << 24);
      CHECK(word == 0x44000200);

      /* The two halves are independent: neither implies the other. */
      d.disable_tri_merging = false;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      isp = probe.bytes + probe.offsets[5] + 4;
      CHECK((isp[3] & 0xe0) == 0x40 && (isp[3] & 0x04) == 0);
   }

   /* A controlled noperspective render encodes this word exactly like a
    * smooth one; only flat moves the count into the second byte. */
   {
      struct agx_ppp_hal300_graphics_fields d = f;
      struct agx_ppp_hal300_graphics_core probe;
      d.interpolation = AGX_PPP_HAL300_LINEAR;
      d.cf_binding_count = 1;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      const uint8_t *counts = probe.bytes + probe.offsets[0] + 4;
      CHECK(counts[0] == d.fragment_varying_components && counts[1] == 0 &&
            counts[2] == 0);
      d.interpolation = AGX_PPP_HAL300_FLAT;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      counts = probe.bytes + probe.offsets[0] + 4;
      CHECK(counts[0] == 0 && counts[1] == d.fragment_varying_components);
      d.interpolation = (enum agx_ppp_hal300_interpolation)3;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
   }

   /* A colour-free render, from the capture that matches the conformance
    * draw exactly: four noperspective varyings and a discarding shader. */
   {
      struct agx_ppp_hal300_graphics_fields d = f;
      struct agx_ppp_hal300_graphics_core probe;
      d.no_color_attachment = true;
      d.interpolation = AGX_PPP_HAL300_LINEAR;
      d.cf_binding_count = 1;
      d.fragment_pass = 2;
      d.disable_tri_merging = true;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
      const uint8_t *counts = probe.bytes + probe.offsets[0] + 4;
      CHECK(counts[0] == 0 && counts[1] == 0 &&
            counts[2] == d.fragment_varying_components);
      const uint8_t *isp = probe.bytes + probe.offsets[5] + 4;
      const uint32_t word = isp[0] | (isp[1] << 8) | (isp[2] << 16) |
                            ((uint32_t)isp[3] << 24);
      CHECK(word == 0x44000200);
      const uint8_t *last = probe.bytes + probe.offsets[5] + 16;
      CHECK(last[0] == 0 && last[1] == 0 && last[2] == 0);

      /* A flat colour-free varying has no witness. */
      d.interpolation = AGX_PPP_HAL300_FLAT;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &d, &probe));
   }

   const struct agx_ppp_hal300_graphics_core color_before = out;
   f.varying_table++;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.varying_table = UINT64_C(1) << 48;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.varying_table = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.varying_table = UINT64_C(0x10000000480);
   f.cf_binding_count = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.cf_binding_count = 3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.interpolation = AGX_PPP_HAL300_SMOOTH;
   f.cf_binding_count = 4;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.cf_binding_count = 1;
   f.fragment_varying_components = 0;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(memcmp(&out, &color_before, sizeof(out)) == 0);

   /* A constant fragment shader has no CF bindings even when a reusable
    * vertex shader exports float4 color. Preserve its nine UVS words and
    * point-size selection without inventing a fragment interpolation input. */
   f = (struct agx_ppp_hal300_graphics_fields) {
      .fragment_program = UINT64_C(0x10000170700),
      .point_size_output = true,
      .vertex_varying_components = 4,
   };
   CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(out.bytes[4] == 0); /* No interpolated coefficients. */
   for (unsigned i = 24; i < 36; ++i)
      CHECK(out.bytes[i] == 0); /* No VIT pointer or CF binding count. */
   CHECK(out.bytes[78] == 5); /* PSIZ and color vertex outputs. */
   CHECK(out.bytes[88] == 9); /* All vertex output words stay live. */

   /* A clip-distance group is the UVS record's bit 0 and its own words in
    * the live count. Captured on 25G76 against the identical colour render:
    * four varyings and one clip distance publish word 1 = 0x00010001 and a
    * live count of nine, with no point size. */
   {
      struct agx_ppp_hal300_graphics_fields cd = {
         .fragment_program = UINT64_C(0x10000170700),
         .vertex_varying_components = 4,
         .clip_distance_words = 1,
      };
      struct agx_ppp_hal300_graphics_core cd_out;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &cd, &cd_out));
      CHECK(cd_out.bytes[76] == 1); /* Clip-distance group selected. */
      CHECK(cd_out.bytes[78] == 1); /* Colour outputs, no point size. */
      CHECK(cd_out.bytes[88] == 9); /* 4 position + 4 varying + 1 clip. */

      cd.point_size_output = true;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &cd, &cd_out));
      CHECK(cd_out.bytes[76] == 1);
      CHECK(cd_out.bytes[78] == 5);
      CHECK(cd_out.bytes[88] == 10);

      cd.point_size_output = false;
      cd.clip_distance_words = 0;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &cd, &cd_out));
      CHECK(cd_out.bytes[76] == 0);
      CHECK(cd_out.bytes[88] == 8);
   }
   const struct agx_ppp_hal300_graphics_core unused_before = out;
   f.fragment_varying_components = 4;
   f.vertex_varying_components = 0;
   f.varying_table = UINT64_C(0x10000000480);
   f.cf_binding_count = 1;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   CHECK(memcmp(&out, &unused_before, sizeof(out)) == 0);

   /* The Varying Counts ABI counts scalar coefficients: smooth occupies
    * byte 0 and flat byte 1. Changing vector width also changes the UVS
    * word count, but no record boundary, pointer or face-state byte. */
   for (unsigned count = 1; count <= 8; ++count) {
      for (unsigned flat = 0; flat <= 1; ++flat) {
         f = (struct agx_ppp_hal300_graphics_fields) {
            .fragment_program = UINT64_C(0x10000170700),
            .varying_table = UINT64_C(0x100000004c0),
            .cf_binding_count = flat ? 1 : 2,
            .point_size_output = true,
            .vertex_varying_components = count,
            .fragment_varying_components = count,
            .interpolation = flat ? AGX_PPP_HAL300_FLAT : AGX_PPP_HAL300_SMOOTH,
         };
         CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
         for (unsigned i = 0; i < sizeof(out.bytes); ++i) {
            uint8_t expected = color_words[i / 4] >> (8 * (i % 4));
            if (i == 4) expected = flat ? 0 : count;
            if (i == 5) expected = flat ? count : 0;
            if (i == 26) expected = flat ? 1 : 2;
            if (i == 88) expected = 5 + count;
            CHECK(out.bytes[i] == expected);
         }
      }
   }
   f.vertex_varying_components = 9;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));
   f.vertex_varying_components = 2;
   f.fragment_varying_components = 3;
   CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &f, &out));

   /* Patching the fragment program of a packed core must give the bytes of a
    * full pack with only fragment_program changed, on each HAL300 graphics
    * layout: the G17G family's TI-spill template and G18G's depth-write one.
    * Addresses the pair cannot encode, and chips without the pair codec, are
    * refused without touching the core. */
   {
      const struct {
         enum agx_chip chip;
         struct agx_ppp_hal300_graphics_fields fields;
      } layouts[] = {
         {AGX_CHIP_G17S,
          {
             .fragment_program = UINT64_C(0x10000170700),
             .varying_table = UINT64_C(0x100000004c0),
             .cf_binding_count = 2,
             .point_size_output = true,
             .vertex_varying_components = 4,
             .fragment_varying_components = 4,
             .fragment_writes_depth = true,
             .fragment_ti_spill_B = 0x120,
          }},
         {AGX_CHIP_G18G,
          {
             .fragment_program = UINT64_C(0x10000170700),
             .fragment_writes_depth = true,
          }},
      };
      const uint64_t new_addrs[] = {
         UINT64_C(0x100001b8700),
         UINT64_C(0xa5c001234540),
         UINT64_C(0x0000ffffffffffc0),
      };

      for (unsigned l = 0; l < sizeof(layouts) / sizeof(layouts[0]); ++l) {
         const enum agx_chip chip = layouts[l].chip;
         struct agx_ppp_hal300_graphics_fields pf = layouts[l].fields;
         struct agx_ppp_hal300_graphics_core packed, patched;
         CHECK(agx_ppp_hal300_graphics_core_pack(chip, &pf, &patched));

         for (unsigned i = 0; i < sizeof(new_addrs) / sizeof(new_addrs[0]);
              ++i) {
            pf.fragment_program = new_addrs[i];
            CHECK(agx_ppp_hal300_graphics_core_pack(chip, &pf, &packed));
            CHECK(agx_ppp_hal300_graphics_core_patch_program(chip, &patched,
                                                             new_addrs[i]));
            CHECK(memcmp(&packed, &patched, sizeof(packed)) == 0);
         }

         const struct agx_ppp_hal300_graphics_core saved = patched;
         CHECK(!agx_ppp_hal300_graphics_core_patch_program(chip, &patched, 0));
         CHECK(!agx_ppp_hal300_graphics_core_patch_program(
            chip, &patched, UINT64_C(0x100001b8701)));
         CHECK(!agx_ppp_hal300_graphics_core_patch_program(
            chip, &patched, UINT64_C(1) << 48));
         CHECK(!agx_ppp_hal300_graphics_core_patch_program(
            AGX_CHIP_G13G, &patched, UINT64_C(0x100001b8700)));
         CHECK(!agx_ppp_hal300_graphics_core_patch_program(
            chip, NULL, UINT64_C(0x100001b8700)));
         CHECK(memcmp(&saved, &patched, sizeof(saved)) == 0);
      }
   }

   /* G18G carries depth_clip_disable, polygon_mode, viewport_count,
    * layer_output, tag_write_disable, no_color_writes, primitive_msaa, and
    * full clip_distance_words masks in the same HAL300 PPP words as M5. */
   {
      struct agx_ppp_hal300_graphics_fields g = {
         .fragment_program = UINT64_C(0x10000170700),
         .depth_clip_disable = true,
         .polygon_mode = 1,
         .viewport_count = 4,
         .layer_output = true,
         .clip_distance_words = 3,
         .tag_write_disable = true,
         .no_color_writes = true,
         .primitive_msaa = true,
      };
      struct agx_ppp_hal300_graphics_core g_out;
      CHECK(agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &g, &g_out));

      const uint8_t *r3 = g_out.bytes + g_out.offsets[3];
      const uint32_t osel = r3[4] | ((uint32_t)r3[5] << 8) |
                            ((uint32_t)r3[6] << 16) | ((uint32_t)r3[7] << 24);
      const uint32_t vp_count = r3[12] | ((uint32_t)r3[13] << 8) |
                                ((uint32_t)r3[14] << 16) | ((uint32_t)r3[15] << 24);
      const uint32_t uvs = r3[16] | ((uint32_t)r3[17] << 8) |
                           ((uint32_t)r3[18] << 16) | ((uint32_t)r3[19] << 24);
      CHECK(osel == (0x180000u | 0x7u));
      CHECK(vp_count == 3u);
      CHECK(uvs == 8u);

      const uint8_t *r4 = g_out.bytes + g_out.offsets[4];
      const uint32_t face = r4[8] | ((uint32_t)r4[9] << 8) |
                            ((uint32_t)r4[10] << 16) | ((uint32_t)r4[11] << 24);
      CHECK(((face >> 18) & 0x3u) == 1u);

      const uint8_t *r5 = g_out.bytes + g_out.offsets[5];
      const uint32_t isp = r5[4] | ((uint32_t)r5[5] << 8) |
                           ((uint32_t)r5[6] << 16) | ((uint32_t)r5[7] << 24);
      const uint32_t mask = r5[16] | ((uint32_t)r5[17] << 8) |
                            ((uint32_t)r5[18] << 16) | ((uint32_t)r5[19] << 24);
      CHECK((isp & 0x00200000u) != 0);
      CHECK((isp & 0x04000000u) != 0);
      CHECK(mask == 0u);

      const uint8_t *r7 = g_out.bytes + g_out.offsets[7];
      const uint32_t cull = r7[4] | ((uint32_t)r7[5] << 8) |
                            ((uint32_t)r7[6] << 16) | ((uint32_t)r7[7] << 24);
      CHECK((cull & 0xc00u) == 0x800u);
      CHECK((cull & 0x8000u) != 0);

      g.viewport_count = 17;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &g, &g_out));
      g.viewport_count = 1;
      g.polygon_mode = 3;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &g, &g_out));
      g.polygon_mode = 0;
      g.clip_distance_words = 9;
      CHECK(!agx_ppp_hal300_graphics_core_pack(AGX_CHIP_G18G, &g, &g_out));
   }

   puts("M5 native PPP graphics records: PASS");
   return 0;
}
