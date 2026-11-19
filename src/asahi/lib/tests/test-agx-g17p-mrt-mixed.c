/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P render passes with three to eight colour targets of arbitrary formats.
 *
 * Apple's builtin end-of-tile factory (metal-nt -arch applegpu_g17p, BTNF
 * root field 2 = 5) emits the same program for two to eight same-partition 2D
 * targets whatever their formats: the raw32 text for that target count in
 * agx3_g17p_quad_eot.h with each target's IMGWBLK selector byte replaced.
 * Sweeping internal formats 0..23 and components 1..4 shows the selector is
 * a function of the target's own format alone, and 42 mixed combinations of
 * two to four targets reproduce template-plus-selectors byte for byte. The
 * layouts below are the nine dEQP-VK renderpasses attachment.{3,4}.* cases,
 * followed by layouts of five to eight targets.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/format/u_format.h"
#include "agx3_g17p_quad_eot.h"
#include "agx3_sr_core.h"
#include "agx_bg_eot_key.h"
#include "agx_abi.h"
#include "agx_compile.h"
#include "agx_debug.h"
#include "agx_linker.h"
#include "agx_nir.h"
#include "agx_nir_texture.h"
#include "agx_tilebuffer.h"
#include "layout.h"
#include "nir_builder.h"

static unsigned checks;

#define CHECK(cond)                                                            \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #cond);               \
         return 1;                                                             \
      }                                                                        \
   } while (0)

struct layout {
   const char *name;
   unsigned count;
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
   uint8_t offsets[AGX_MAX_RENDER_TARGETS];
   uint8_t sample_size_B;
};

static const struct layout layouts[] = {
   {"3.296", 3,
    {PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16_SNORM, PIPE_FORMAT_R16_FLOAT},
    {0, 4, 6}, 8},
   {"3.314", 3,
    {PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8_SINT},
    {0, 1, 5}, 8},
   {"3.336", 3,
    {PIPE_FORMAT_R16G16B16A16_SNORM, PIPE_FORMAT_R16_FLOAT,
     PIPE_FORMAT_R32G32B32A32_SINT},
    {16, 24, 0}, 32},
   {"3.397", 3,
    {PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R16G16_FLOAT,
     PIPE_FORMAT_R16G16B16A16_UINT},
    {0, 4, 8}, 16},
   {"4.418", 4,
    {PIPE_FORMAT_R8G8B8A8_SNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
     PIPE_FORMAT_R32G32_UINT, PIPE_FORMAT_R16_UNORM},
    {10, 14, 0, 8}, 24},
   {"4.465", 4,
    {PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R16_UNORM,
     PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R8G8B8A8_UNORM},
    {0, 20, 4, 22}, 32},
   {"4.541", 4,
    {PIPE_FORMAT_R16G16B16A16_UINT, PIPE_FORMAT_R8_SINT,
     PIPE_FORMAT_R8G8_SNORM, PIPE_FORMAT_R8G8_SINT},
    {0, 8, 9, 11}, 16},
   {"4.568", 4,
    {PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16G16B16A16_SNORM,
     PIPE_FORMAT_R8G8B8A8_SINT, PIPE_FORMAT_B8G8R8A8_UNORM},
    {0, 8, 16, 20}, 24},
   {"4.587", 4,
    {PIPE_FORMAT_R8G8B8A8_SRGB, PIPE_FORMAT_B10G10R10A2_UNORM,
     PIPE_FORMAT_B8G8R8A8_UNORM, PIPE_FORMAT_R16G16_UINT},
    {0, 4, 12, 8}, 16},
   {"rgb9e5x3", 3,
    {PIPE_FORMAT_R9G9B9E5_FLOAT, PIPE_FORMAT_R9G9B9E5_FLOAT,
     PIPE_FORMAT_R9G9B9E5_FLOAT},
    {0, 4, 8}, 16},
};

#define RGBA8 PIPE_FORMAT_R8G8B8A8_UNORM

/* Five to eight targets, all admitted by the mixed-format programs. The
 * homogeneous RGBA8 layouts match Apple's fragment witnesses. */
static const struct layout wide_layouts[] = {
   {"rgba8x5", 5, {RGBA8, RGBA8, RGBA8, RGBA8, RGBA8}, {0, 4, 8, 12, 16}, 24},
   {"mixed6", 6,
    {RGBA8, PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R8_UNORM,
     PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R8G8_UNORM,
     PIPE_FORMAT_R32G32B32A32_FLOAT},
    {24, 16, 28, 29, 33, 0}, 40},
   {"rgba16x7", 7,
    {PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
     PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
     PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16G16B16A16_UNORM,
     PIPE_FORMAT_R16G16B16A16_UNORM},
    {0, 8, 16, 24, 32, 40, 48}, 56},
   {"rgba8x8", 8, {RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8},
    {0, 4, 8, 12, 16, 20, 24, 28}, 32},
   {"mixed8", 8,
    {PIPE_FORMAT_R32G32B32A32_FLOAT, RGBA8, PIPE_FORMAT_R16G16B16A16_FLOAT,
     PIPE_FORMAT_R8G8B8A8_UINT, PIPE_FORMAT_R16G16_SNORM, PIPE_FORMAT_R8_UNORM,
     PIPE_FORMAT_R8G8_UNORM, PIPE_FORMAT_R32_FLOAT},
    {0, 32, 20, 36, 28, 40, 41, 16}, 48},
};

/* The store format the end-of-tile builder hands IMGWBLK. */
static enum pipe_format
eot_store_format(enum pipe_format format)
{
   return (format == PIPE_FORMAT_R11G11B10_FLOAT ||
           format == PIPE_FORMAT_R9G9B9E5_FLOAT)
             ? PIPE_FORMAT_R32_UINT
          : util_format_is_srgb(format) ? PIPE_FORMAT_R8_UNORM
                                        : ail_pixel_format[format].renderable;
}

static bool
compile_eot(unsigned count, const enum pipe_format *formats,
            const uint8_t *offsets, const struct agx_shader_key *base_key,
            struct agx_shader_part *part)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE,
                                                  &agx_nir_options, "eot");
   nir_def *xy = nir_trim_vector(&b, nir_load_global_invocation_id(&b, 32), 2);
   for (unsigned rt = 0; rt < count; ++rt) {
      nir_image_store_block_agx(&b, nir_imm_intN_t(&b, rt, 16),
                                nir_imm_int(&b, offsets[rt]), xy,
                                .format = eot_store_format(formats[rt]),
                                .image_dim = GLSL_SAMPLER_DIM_2D,
                                .explicit_coord = true);
   }
   agx_preprocess_nir(b.shader);

   struct agx_shader_key key = *base_key;
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

/* Every admitted format's multi-target selector is the selector of that
 * format's own single-target program. */
static int
test_selectors_match_single_target_programs(void)
{
   unsigned admitted = 0;
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      const uint8_t selector = agx_g17p_mrt_eot_selector(f);
      if (!selector)
         continue;

      struct agx_shader_key key = {
         .dev.encoding = AGX_ISA_ENCODING_AGX3B,
         .no_preamble = true,
      };
      agx_g17p_configure_rt0_eot_key(f, &key);
      const enum pipe_format format = f;
      const uint8_t offset = 0;
      struct agx_shader_part part = {0};
      CHECK(compile_eot(1, &format, &offset, &key, &part));
      CHECK(part.info.binary_size == 68);
      CHECK(((uint8_t *)part.binary)[61] == selector);
      free(part.binary);
      admitted++;
   }

   /* The selector names the lane class Apple's own fragment stores the
    * format in (metal-nt render pipelines per MTLPixelFormat, matched to the
    * builtin background clears of each internal format). */
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R32G32_UINT) == 0x35);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R32G32_SINT) == 0x35);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R16G16B16A16_UNORM) == 0x95);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R16G16B16A16_SINT) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B5G6R5_UNORM) == 0x15);
   /* ABGR4Unorm, A1BGR5Unorm and BGR5A1Unorm fragments store the same W16
    * float LSTCOL as RGBA16Float, RGB10A2Uint the same W16 integer LSTCOL as
    * RGBA16Uint, and RGBA8Snorm the same W8 LSTCOL as RGBA8Unorm. Channel
    * order is the PBE's, as for the BGRA forms already listed. */
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_A4B4G4R4_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_A4R4G4B4_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_A1B5G5R5_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_A1R5G5B5_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B5G5R5A1_UNORM) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B10G10R10A2_UINT) == 0x15);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B8G8R8A8_SNORM) == 0x75);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R10G10B10X2_UNORM) == 0x75);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_B10G10R10X2_UNORM) == 0x75);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R11G11B10_FLOAT) == 0x35);
   CHECK(agx_g17p_mrt_eot_selector(PIPE_FORMAT_R9G9B9E5_FLOAT) == 0x35);
   CHECK(admitted > 40);
   return 0;
}

static int
check_layout_classification(const struct layout *l)
{
   {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);

      CHECK(tib.sample_size_B == l->sample_size_B);
      for (unsigned rt = 0; rt < l->count; ++rt)
         CHECK(agx_tilebuffer_offset_B(&tib, rt) == l->offsets[rt]);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
      CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_tilebuffer_layout bad = tib;
      bad.spilled[l->count - 1] = true;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad._offset_B[l->count - 1] = bad._offset_B[0];
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad._offset_B[l->count - 1] = tib.sample_size_B - 1;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      bad = tib;
      bad.logical_format[l->count - 1] = PIPE_FORMAT_R32G32B32_FLOAT;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      /* Multisample layouts take the mixed programs; other counts none. */
      bad = tib;
      bad.nr_samples = 2;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_MIXED);

      bad = tib;
      bad.nr_samples = 3;
      CHECK(agx_g17p_classify_mrt(&bad) == AGX_G17P_MRT_NONE);

      /* A layered layout takes the same mixed programs. Apple's factory emits
       * a 2D-ARRAY end-of-tile program that differs from the 2D one only in
       * also publishing the layer wire, and the per-target IMGWBLK selector is
       * still the only format-dependent byte; those texts are recorded in
       * agx3_g17p_array_eot.h. Verified on G17P: the 4,050-case
       * renderpasses.*.multisample_resolve.layers_{3,6} family runs 3,168
       * Pass / 882 NotSupported / 0 Fail with this admitted. */
      struct agx_tilebuffer_layout layered = tib;
      layered.layered = true;
      CHECK(agx_g17p_classify_mrt(&layered) == AGX_G17P_MRT_MIXED);

      layered.nr_samples = 2;
      CHECK(agx_g17p_classify_mrt(&layered) == AGX_G17P_MRT_MIXED);
   }
   return 0;
}

static int
test_layout_classification(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      if (check_layout_classification(&layouts[i]))
         return 1;
   }
   for (unsigned i = 0; i < ARRAY_SIZE(wide_layouts); ++i) {
      if (check_layout_classification(&wide_layouts[i]))
         return 1;
   }

   /* Targets that no longer fit one 64-byte sample spill, which G17P does not
    * support: eight RGBA32 targets. */
   enum pipe_format spill[AGX_MAX_RENDER_TARGETS];
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      spill[rt] = PIPE_FORMAT_R32G32B32A32_UINT;
   struct agx_tilebuffer_layout spilled =
      agx_build_tilebuffer_layout(spill, AGX_MAX_RENDER_TARGETS, 1, false);
   CHECK(agx_g17p_classify_mrt(&spilled) == AGX_G17P_MRT_NONE);
   CHECK(!agx_g17p_rt0_tile_layout_supported(&spilled));

   /* Homogeneous layouts keep their own programs. */
   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM};
   enum pipe_format rgba32[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT};
   enum pipe_format r32[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R32_FLOAT};
   for (unsigned count = 3; count <= 4; ++count) {
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rgba8, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_RGBA8);
      tib = agx_build_tilebuffer_layout(rgba32, count, 1, false);
      CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_RGBA32);
   }
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(r32, 4, 1, false);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_R32_QUAD);

   /* Two targets outside the two-target families use the same programs. */
   enum pipe_format pair[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R16_UNORM};
   tib = agx_build_tilebuffer_layout(pair, 2, 1, false);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

   /* A full sample selects the 16x16 tile, as for four RGBA32 targets. */
   enum pipe_format wide[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT, PIPE_FORMAT_R32G32B32A32_FLOAT};
   tib = agx_build_tilebuffer_layout(wide, 4, 1, false);
   CHECK(tib.tile_size == 16 * 16);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

   /* The failing CTS render pass leaves attachment zero unused. Compacting
    * its three packed-float targets preserves their physical offsets and the
    * 16-byte sample while selecting the generic mixed program. */
   enum pipe_format sparse_r11[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_NONE, PIPE_FORMAT_R11G11B10_FLOAT,
      PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R11G11B10_FLOAT};
   tib = agx_build_tilebuffer_layout(sparse_r11, 4, 1, false);
   CHECK(tib.sample_size_B == 16);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));
   return 0;
}

/* Expected multi-target EOT: the native raw32 text with each target's
 * IMGWBLK selector at the byte metal-nt changes for it. */
static const uint8_t *
expected_eot(unsigned count, const uint8_t *selectors, uint8_t *out)
{
   static const unsigned selector_offsets[][AGX_MAX_RENDER_TARGETS] = {
      [2] = {101, 113},
      [3] = {133, 145, 157},
      [4] = {165, 177, 189, 201},
      [5] = {197, 209, 221, 233, 245},
      [6] = {229, 241, 253, 265, 277, 289},
      [7] = {69, 273, 285, 297, 309, 321, 333},
      [8] = {101, 113, 317, 329, 341, 353, 365, 377},
   };
   static const uint8_t *const texts[] = {
      [2] = agx3_g17p_dual_raw32_eot,  [3] = agx3_g17p_triple_raw32_eot,
      [4] = agx3_g17p_quad_eot,        [5] = agx3_g17p_penta_raw32_eot,
      [6] = agx3_g17p_hexa_raw32_eot,  [7] = agx3_g17p_hepta_raw32_eot,
      [8] = agx3_g17p_octa_raw32_eot,
   };
   memcpy(out, texts[count], 32 + 44 * count);
   for (unsigned rt = 0; rt < count; ++rt)
      out[selector_offsets[count][rt]] = selectors[rt];
   return out;
}

/* The multisample program for texture type 4 attachments changes three
 * bytes per target: the wire width byte of its PACK_IMGWIR gains 2, and its
 * IMGWBLK clears bit 7 of byte 8 and takes dimension class 2 in the selector's
 * low nibble. */
static const uint8_t *
expected_ms_eot(unsigned count, const uint8_t *selectors, uint8_t *out)
{
   static const unsigned wire_offsets[][AGX_MAX_RENDER_TARGETS] = {
      [2] = {43, 63},
      [3] = {43, 63, 95},
      [4] = {43, 63, 95, 127},
      [5] = {43, 63, 95, 127, 159},
      [6] = {43, 63, 95, 127, 159, 191},
      [7] = {43, 75, 107, 139, 171, 203, 235},
      [8] = {43, 63, 119, 151, 183, 215, 247, 279},
   };
   static const unsigned selector_offsets[][AGX_MAX_RENDER_TARGETS] = {
      [2] = {101, 113},
      [3] = {133, 145, 157},
      [4] = {165, 177, 189, 201},
      [5] = {197, 209, 221, 233, 245},
      [6] = {229, 241, 253, 265, 277, 289},
      [7] = {69, 273, 285, 297, 309, 321, 333},
      [8] = {101, 113, 317, 329, 341, 353, 365, 377},
   };
   expected_eot(count, selectors, out);
   for (unsigned rt = 0; rt < count; ++rt) {
      out[wire_offsets[count][rt]] |= 0x02;
      out[selector_offsets[count][rt] - 1] &= 0x7f;
      out[selector_offsets[count][rt]] = (selectors[rt] & 0xf0) | 0x02;
   }
   return out;
}

static unsigned
decode_hex(const char *hex, uint8_t *out, unsigned capacity)
{
   unsigned n = 0;
   for (; hex[0] && hex[1] && n < capacity; hex += 2)
      sscanf(hex, "%2hhx", &out[n++]);
   return n;
}

/* metal-nt -arch applegpu_g17p builtin end-of-tile BTNF: two, seven and eight
 * RGBA8 (internal format 2) targets in partition zero, texture type 4. */
static const struct {
   unsigned count;
   const char *hex;
} ms_eot_witnesses[] = {
   {2,
      "1c9c100624a814060c9d102624a910269f1054020304140050260a009f20540003001000"
      "50260a009705540600080000d04010000b8221041b8021049705540e00100000d0441000"
      "4b032f004200000400005b012f0042000004000057105400000000802872000057205408"
      "00100090287200000e000000"},
   {7,
      "1c9c100624a814060c9d102624a910269f1054020304140050260a009f20540003001000"
      "50260a009705540600080000d04010000b8221041b802104571054000000008028720000"
      "9705540e00100000d04410004b8327004200000400005b81270042000004000097055416"
      "00180000d05010008b8327004200000800009b8127004200000800009705541e00200000"
      "d0541000cb8327004200000c0000db8127004200000c00009705542600280000d0601000"
      "0b8367004200001000001b8167004200001000009705542e00300000d06010004b836700"
      "4200001000005b8167004200001000009705543600380000d06410008b036f0042000014"
      "00009b016f0042000014000057205408001000902872000057405410002000a028720000"
      "57805418003000a02872000057005520004000b02872000057005428005000d028720000"
      "57005630006000e0287200000e000000"},
   {8,
      "1c9c100624a814060c9d102624a910269f1054020304140050260a009f20540003001000"
      "50260a009705540600080000d04010000b8221041b8021049705540e00100000d0441000"
      "4b8327004200000400005b81270042000004000057105400000000802872000057205408"
      "00100080287200009705541600180000d05010008b8327004200000800009b8127004200"
      "000800009705541e00200000d0541000cb8327004200000c0000db8127004200000c0000"
      "9705542600280000d06010000b8367004200001000001b8167004200001000009705542e"
      "00300000d06410004b8367004200001400005b8167004200001400009705543600380000"
      "d06410008b8367004200001400009b8167004200001400009705543e00400000d0441000"
      "cb036f00420000040000db016f0042000004000057405410002000a02872000057805418"
      "003000b02872000057005520004000b02872000057005628005000d02872000057005430"
      "006000e0287200005720543800700090287200000e000000"},
};

static int
test_multisample_eot_witnesses(void)
{
   const uint8_t rgba8[AGX_MAX_RENDER_TARGETS] = {
      0x75, 0x75, 0x75, 0x75, 0x75, 0x75, 0x75, 0x75,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(ms_eot_witnesses); ++i) {
      const unsigned count = ms_eot_witnesses[i].count;
      uint8_t native[sizeof(agx3_g17p_octa_raw32_eot)];
      uint8_t expected[sizeof(agx3_g17p_octa_raw32_eot)];
      CHECK(decode_hex(ms_eot_witnesses[i].hex, native, sizeof(native)) ==
            32 + 44 * count);
      CHECK(!memcmp(expected_ms_eot(count, rgba8, expected), native,
                    32 + 44 * count));
   }
   return 0;
}

static int
check_eot_subset(unsigned count, const enum pipe_format *formats,
                 const uint8_t *offsets)
{
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
      .g17p_mrt_eot_count = count,
   };
   agx_g17p_configure_rt0_eot_key(formats[0], &key);
   for (unsigned rt = 0; rt < count; ++rt)
      key.g17p_mrt_eot_selectors[rt] = agx_g17p_mrt_eot_selector(formats[rt]);

   struct agx_shader_part part = {0};
   CHECK(compile_eot(count, formats, offsets, &key, &part));
   uint8_t expected[sizeof(agx3_g17p_octa_raw32_eot)];
   CHECK(part.info.binary_size == 32 + 44 * count);
   CHECK(part.info.nr_gprs == 8);
   CHECK(part.info.agx3_staging_known &&
         part.info.agx3_staging_halfs == 8 * count);
   CHECK(!memcmp(part.binary,
                 expected_eot(count, key.g17p_mrt_eot_selectors, expected),
                 part.info.binary_size));
   free(part.binary);

   struct agx_shader_key ms_key = key;
   ms_key.g17p_rt0_eot_multisample = true;
   part = (struct agx_shader_part){0};
   CHECK(compile_eot(count, formats, offsets, &ms_key, &part));
   CHECK(part.info.binary_size == 32 + 44 * count);
   CHECK(!memcmp(part.binary,
                 expected_ms_eot(count, key.g17p_mrt_eot_selectors, expected),
                 part.info.binary_size));
   free(part.binary);

   /* A store whose format cannot carry its target's selector is refused
    * rather than published with a selector from another format. */
   for (unsigned rt = 0; rt < count; ++rt) {
      struct agx_shader_key wrong = key;
      wrong.g17p_mrt_eot_selectors[rt] =
         key.g17p_mrt_eot_selectors[rt] == 0x55 ? 0x95 : 0x55;
      part = (struct agx_shader_part){0};
      CHECK(!compile_eot(count, formats, offsets, &wrong, &part));
   }

   struct agx_shader_key short_key = key;
   short_key.g17p_mrt_eot_count = count - 1;
   part = (struct agx_shader_part){0};
   CHECK(!compile_eot(count, formats, offsets, &short_key, &part));
   return 0;
}

static int
check_eot_programs(const struct layout *l, bool every_subset)
{
   {

      /* One stored target keeps its format's single-target program at its
       * own offset. */
      for (unsigned rt = 0; rt < l->count; ++rt) {
         struct agx_shader_key key = {
            .dev.encoding = AGX_ISA_ENCODING_AGX3B,
            .no_preamble = true,
         };
         agx_g17p_configure_rt0_eot_key(l->formats[rt], &key);
         struct agx_shader_part part = {0};
         CHECK(compile_eot(1, &l->formats[rt], &l->offsets[rt], &key, &part));
         CHECK(part.info.binary_size == 68);
         CHECK(((uint8_t *)part.binary)[61] ==
               agx_g17p_mrt_eot_selector(l->formats[rt]));
         free(part.binary);
      }

      for (unsigned mask = 1; mask < BITFIELD_MASK(l->count); ++mask) {
         if (util_bitcount(mask) < 2 ||
             (!every_subset && util_bitcount(mask) != l->count - 1))
            continue;
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
         uint8_t offsets[AGX_MAX_RENDER_TARGETS];
         unsigned count = 0;
         u_foreach_bit(rt, mask) {
            formats[count] = l->formats[rt];
            offsets[count++] = l->offsets[rt];
         }
         if (check_eot_subset(count, formats, offsets))
            return 1;
      }
      if (check_eot_subset(l->count, l->formats, l->offsets))
         return 1;
   }
   return 0;
}

static int
test_eot_programs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      if (check_eot_programs(&layouts[i], true))
         return 1;
   }
   /* Wider layouts: every target count through eight, without each subset. */
   for (unsigned i = 0; i < ARRAY_SIZE(wide_layouts); ++i) {
      if (check_eot_programs(&wide_layouts[i], false))
         return 1;
   }

   /* Packed R11G11B10 reaches EOT as one raw R32 word per target. Exercise
    * the exact three compacted targets and physical offsets from the sparse
    * CTS render pass against the native raw32 template. */
   const enum pipe_format r11[3] = {
      PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R11G11B10_FLOAT,
      PIPE_FORMAT_R11G11B10_FLOAT};
   const uint8_t r11_offsets[3] = {4, 8, 12};
   if (check_eot_subset(3, r11, r11_offsets))
      return 1;

   /* Homogeneous RGBA8 targets reproduce the separately captured programs. */
   const uint8_t rgba8[] = {0x75, 0x75, 0x75, 0x75};
   uint8_t expected[sizeof(agx3_g17p_octa_raw32_eot)];
   CHECK(!memcmp(expected_eot(3, rgba8, expected), agx3_g17p_rgba8_3_eot,
                 sizeof(agx3_g17p_rgba8_3_eot)));
   CHECK(!memcmp(expected_eot(4, rgba8, expected), agx3_g17p_rgba8_4_eot,
                 sizeof(agx3_g17p_rgba8_4_eot)));
   return 0;
}

static bool
lower_tex_handle_to_u0(nir_builder *b, nir_intrinsic_instr *intr, void *data)
{
   if (intr->intrinsic != nir_intrinsic_load_texture_handle_agx)
      return false;

   b->cursor = nir_instr_remove(&intr->instr);
   nir_def_rewrite_uses(
      &intr->def,
      nir_bindless_image_agx(b, nir_imul_imm(b, intr->src[0].ssa, 32),
                             .desc_set = 0));
   return true;
}

/* Mirror agx_build_background_shader for reloads followed by clears. */
static bool
compile_background(const struct agx_tilebuffer_layout *layout,
                   unsigned reloads, bool mrt_bg, unsigned reserved_preamble,
                   struct agx_shader_part *part)
{
   struct agx_tilebuffer_layout tib = *layout;
   unsigned count = 0;
   while (count < AGX_MAX_RENDER_TARGETS &&
          tib.logical_format[count] != PIPE_FORMAT_NONE)
      count++;

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P mixed BG");
   b.shader->info.fs.untyped_color_outputs = true;
   for (unsigned rt = 0; rt < count; ++rt) {
      const unsigned nr = util_format_get_nr_components(tib.logical_format[rt]);
      nir_def *value;
      if (rt < reloads) {
         nir_def *coord = nir_u2u32(&b, nir_load_pixel_coord(&b));
         if (tib.layered)
            coord = nir_vec3(&b, nir_channel(&b, coord, 0),
                             nir_channel(&b, coord, 1), nir_load_layer_id(&b));
         nir_def *tex = nir_txf(&b, coord, .texture_index = rt * 2,
                                .dim = GLSL_SAMPLER_DIM_2D,
                                .is_array = tib.layered,
                                .dest_type = nir_type_uint32,
                                .backend_flags = AGX_TEXTURE_FLAG_NO_CLAMP);
         value = nir_trim_vector(&b, tex, nr);
      } else {
         value = nir_load_preamble(&b, nr, 32, 4 + rt * 8);
      }
      nir_store_output(&b, value, nir_imm_int(&b, 0),
                       .write_mask = BITFIELD_MASK(nr),
                       .src_type = nir_type_uint32,
                       .io_semantics.location = FRAG_RESULT_DATA0 + rt,
                       .io_semantics.num_slots = 1);
      b.shader->info.outputs_written |= BITFIELD64_BIT(FRAG_RESULT_DATA0 + rt);
   }

   agx_nir_lower_texture(b.shader);
   agx_preprocess_nir(b.shader);
   unsigned bindless_base = 0;
   NIR_PASS(_, b.shader, agx_nir_lower_tilebuffer, &tib, NULL, &bindless_base,
            NULL, NULL, false, true);
   NIR_PASS(_, b.shader, agx_nir_lower_monolithic_msaa, tib.nr_samples);
   NIR_PASS(_, b.shader, agx_nir_lower_multisampled_image_store);
   NIR_PASS(_, b.shader, agx_nir_lower_texture);
   NIR_PASS(_, b.shader, nir_shader_intrinsics_pass, lower_tex_handle_to_u0,
            nir_metadata_control_flow, NULL);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, tib.nr_samples);

   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.ignore_tib_dependencies = true,
      .g17p_mrt_bg = mrt_bg,
      .reserved_preamble = reserved_preamble,
      .no_preamble = true,
   };
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
has_background_layer_read(const struct agx_shader_part *part)
{
   const uint8_t *code = part->binary;
   for (unsigned reg = 0; reg < 64; ++reg) {
      for (unsigned slot = 1; slot <= 6; ++slot) {
         struct agx3_sr_desc sr = {
            .dest = reg, .width = AGX3_SR_W16, .sr = 12, .slot = slot,
         };
         uint8_t expected[AGX3_SR_MAX_LENGTH], len;
         if (!agx3_sr_encode(AGX_ISA_ENCODING_AGX3B, &sr, expected, &len))
            continue;
         for (unsigned at = 0; at + len <= part->info.binary_size; at += 2) {
            if (!memcmp(code + at, expected, len))
               return true;
         }
      }
   }
   return false;
}

static int
test_background_programs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts) + ARRAY_SIZE(wide_layouts);
        ++i) {
      const struct layout *l = i < ARRAY_SIZE(layouts)
                                  ? &layouts[i]
                                  : &wide_layouts[i - ARRAY_SIZE(layouts)];
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);

      for (unsigned reloads = 0; reloads <= l->count; ++reloads) {
         struct agx_shader_part part = {0};
         CHECK(compile_background(&tib, reloads, true,
                                  reloads < l->count ? 4 + 8 * l->count : 0,
                                  &part));
         CHECK(part.info.uses_txf == (reloads > 0));
         CHECK(part.info.push_count ==
               (reloads < l->count ? 4 + 8 * l->count : 0));
         CHECK(!part.info.scratch_size);
         CHECK(!part.info.agx3_colour_link.entry_emitted);
         free(part.binary);
      }
   }
   /* Layered render-pass reloads must not use compute-grid Z as the layer.
    * Pin the native BG half-register read in the compiled shader, beyond the
    * isolated SR encoder test. */
   const enum pipe_format layered_formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_tilebuffer_layout layered =
      agx_build_tilebuffer_layout(layered_formats, 1, 1, true);
   struct agx_shader_part part = {0};
   CHECK(compile_background(&layered, 1, false, 0, &part));
   CHECK(has_background_layer_read(&part));
   free(part.binary);
   return 0;
}

static bool
compile_epilog_samples_shared(unsigned count, const enum pipe_format *formats,
                       unsigned written, uint64_t partitions, bool additive,
                       bool write_z, unsigned samples, unsigned shared,
                       struct agx_shader_part *part)
{
   const enum pipe_blendfactor dst =
      additive ? PIPE_BLENDFACTOR_ONE : PIPE_BLENDFACTOR_ZERO;
   struct agx_fs_epilog_key epilog = {
      .link.loc_written = written,
      .link.size_32 = written,
      .link.write_z = write_z,
      .nr_samples = samples,
      .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
   };
   memset(epilog.remap, -1, sizeof(epilog.remap));
   for (unsigned rt = 0; rt < count; ++rt) {
      epilog.remap[rt] = rt;
      epilog.rt_formats[rt] = formats[rt];
      epilog.blend.rt[rt].colormask =
         BITFIELD_MASK(util_format_get_nr_components(formats[rt]));
      epilog.blend.rt[rt].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, dst, PIPE_BLEND_ADD,
         PIPE_BLENDFACTOR_ONE, dst);
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &epilog);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P MRT epilog");
   agx_nir_fs_epilog(&b, &epilog);
   /* Model a monolithic fragment exporting the same value to several RTs.
    * Leave packing, CSE, liveness and register allocation to the compiler. */
   if (shared) {
      const unsigned base = AGX_ABI_FOUT_COLOUR(ffs(shared) - 1);
      nir_foreach_function_impl(impl, b.shader) {
         nir_foreach_block(block, impl) {
            nir_foreach_instr(instr, block) {
               if (instr->type != nir_instr_type_intrinsic)
                  continue;
               nir_intrinsic_instr *intr = nir_instr_as_intrinsic(instr);
               if (intr->intrinsic != nir_intrinsic_load_exported_agx)
                  continue;
               u_foreach_bit(rt, shared) {
                  if (nir_intrinsic_base(intr) == AGX_ABI_FOUT_COLOUR(rt)) {
                     nir_intrinsic_set_base(intr, base);
                     break;
                  }
               }
            }
         }
      }
   }
   agx_nir_lower_g17p_tile_samples(b.shader, samples);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                .fs.nr_samples = samples,
                                .secondary = true};
   u_foreach_bit64(offset, partitions)
      BITSET_SET(key.g17p_tile_partitions, offset);
   /* Match Honeykrisp's keyed physical partitions even when every target
    * is written. Unkeyed GL shaders now carry logical attachment IDs, which
    * differ from physical order for the reordered mixed-format witnesses. */
   if (!partitions && count > 1) {
      enum pipe_format full_formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(full_formats, formats, count * sizeof(*formats));
      const struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout_bounded(
         full_formats, count, samples, false, AGX_G17P_MAX_BYTES_PER_SAMPLE);
      for (unsigned rt = 0; rt < count; ++rt) {
         if (full_formats[rt] != PIPE_FORMAT_NONE && !tib.spilled[rt])
            BITSET_SET(key.g17p_tile_partitions, tib._offset_B[rt]);
      }
   }
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_epilog_samples(unsigned count, const enum pipe_format *formats,
                       unsigned written, uint64_t partitions, bool additive,
                       bool write_z, unsigned samples,
                       struct agx_shader_part *part)
{
   return compile_epilog_samples_shared(count, formats, written, partitions,
                                         additive, write_z, samples, 0, part);
}

static bool
compile_sample_shaded_epilog(unsigned count,
                             const enum pipe_format *formats,
                             unsigned samples,
                             struct agx_shader_part *part)
{
   const unsigned written = BITFIELD_MASK(count);
   struct agx_fs_epilog_key epilog = {
      .link.loc_written = written,
      .link.size_32 = written,
      .link.sample_shading = true,
      .link.write_z = true,
      .nr_samples = samples,
      .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
   };
   memset(epilog.remap, -1, sizeof(epilog.remap));
   for (unsigned rt = 0; rt < count; ++rt) {
      epilog.remap[rt] = rt;
      epilog.rt_formats[rt] = formats[rt];
      epilog.blend.rt[rt].colormask =
         rt == 1 ? 1 : BITFIELD_MASK(util_format_get_nr_components(formats[rt]));
      const enum pipe_blendfactor dst =
         rt == 1 ? PIPE_BLENDFACTOR_ZERO : PIPE_BLENDFACTOR_ONE;
      epilog.blend.rt[rt].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, dst,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE, dst);
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &epilog);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options,
      "G17P sample-shaded MRT epilog");
   struct agx_fs_epilog_key pixel = epilog;
   pixel.link.sample_shading = false;
   pixel.defer_monolithic_msaa = true;
   agx_nir_fs_epilog(&b, &pixel);
   NIR_PASS(_, b.shader, agx_nir_lower_to_per_sample);
   NIR_PASS(_, b.shader, agx_nir_unroll_g17p_samples, samples, written);
   NIR_PASS(_, b.shader, nir_opt_constant_folding);
   NIR_PASS(_, b.shader, agx_nir_hoist_g17p_zs_writes);
   NIR_PASS(_, b.shader, agx_nir_lower_g17p_tile_samples, samples);
   agx_preprocess_nir(b.shader);

   enum pipe_format full_formats[AGX_MAX_RENDER_TARGETS] = {0};
   memcpy(full_formats, formats, count * sizeof(*formats));
   const struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout_bounded(
      full_formats, count, samples, false, AGX_G17P_MAX_BYTES_PER_SAMPLE);
   struct agx_shader_key key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .fs.nr_samples = samples,
      .secondary = true,
   };
   for (unsigned rt = 0; rt < count; ++rt) {
      if (!tib.spilled[rt])
         BITSET_SET(key.g17p_tile_partitions, tib._offset_B[rt]);
   }

   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_epilog_written(unsigned count, const enum pipe_format *formats,
                       unsigned written, uint64_t partitions, bool additive,
                       bool write_z, struct agx_shader_part *part)
{
   return compile_epilog_samples(count, formats, written, partitions,
                                 additive, write_z, 1, part);
}

static bool
compile_epilog(unsigned count, const enum pipe_format *formats, bool additive,
               bool write_z, struct agx_shader_part *part)
{
   return compile_epilog_written(count, formats, BITFIELD_MASK(count), 0,
                                 additive, write_z, part);
}

/* Find each pixel instruction of a native witness, in order. */
static bool
contains_in_order(const struct agx_shader_part *part,
                  const uint8_t (*ops)[6], unsigned count)
{
   const uint8_t *text = part->binary;
   size_t at = 0;
   for (unsigned i = 0; i < count; ++i) {
      while (at + 6 <= part->info.binary_size &&
             memcmp(text + at, ops[i], 6))
         at++;
      if (at + 6 > part->info.binary_size)
         return false;
      at += 6;
   }
   return true;
}

/* The pixel transactions of Apple's store-only MRT fragments for these
 * layouts, from metal-nt render pipelines with the same colour formats and a
 * [[depth(any)]] output: the entry, the class-1 depth transaction, then the
 * colour partitions exactly as without depth. */
static const uint8_t entry_512[6] = {0x87, 0x02, 0x54, 0x00, 0x06, 0x00};
static const uint8_t wait_1[6] = {0x87, 0x02, 0x54, 0x01, 0x00, 0x00};
static const uint8_t depthf[6] = {0xd7, 0x14, 0x54, 0x00, 0x00, 0x03};
static const uint8_t release_1[6] = {0x07, 0x02, 0x54, 0x01, 0x00, 0x00};
static const uint8_t wait_192[6] = {0x87, 0x02, 0x54, 0xc0, 0x08, 0x00};
static const uint8_t release_192[6] = {0x07, 0x02, 0x54, 0xc0, 0x00, 0x00};
static const uint8_t wait_48[6] = {0x87, 0x02, 0x54, 0x30, 0x08, 0x00};
static const uint8_t release_48[6] = {0x07, 0x02, 0x54, 0x30, 0x00, 0x00};
static const uint8_t wait_12[6] = {0x87, 0x02, 0x54, 0x0c, 0x08, 0x00};
static const uint8_t release_12[6] = {0x07, 0x02, 0x54, 0x0c, 0x00, 0x00};
static const uint8_t release_524[6] = {0x07, 0x02, 0x54, 0x0c, 0x02, 0x00};
static const uint8_t release_560[6] = {0x07, 0x02, 0x54, 0x30, 0x02, 0x00};
static const uint8_t release_704[6] = {0x07, 0x02, 0x54, 0xc0, 0x02, 0x00};

/* lstcol f13 of each colour store, in store order. */
static unsigned
store_f13_mask(const struct agx_shader_part *part, unsigned *stores)
{
   const uint8_t *text = part->binary;
   unsigned mask = 0, n = 0;
   for (size_t at = 0; at + 12 <= part->info.binary_size;) {
      if (text[at] == 0xe7 && (text[at + 1] & 0x07) == 0x06 &&
          (text[at + 2] & 0xfc) == 0x54) {
         mask |= ((text[at + 7] >> 7) & 1) << n++;
         at += 12;
      } else {
         at++;
      }
   }
   *stores = n;
   return mask;
}

/* A repeated tile payload must remain valid through its later store. This
 * follows the eight-attachment write-mask hardware regression; it does not
 * assign an undocumented architectural meaning to f13. Partial sharing also
 * checks that independent leading stores retain the witnessed hint. */
static int
test_reused_mrt_payloads(void)
{
   static const enum pipe_format formats[8] = {
      RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8, RGBA8,
   };
   static const struct {
      unsigned shared, expected_f13;
   } cases[] = {
      {0, 7},                     /* All unique: preserve native hints. */
      {0xff, 0},                  /* The payload remains live until RT0. */
      {BITFIELD_BIT(7) | 1, 6},
      {BITFIELD_BIT(6) | 1, 5},
      {BITFIELD_BIT(5) | 1, 3},
      {BITFIELD_BIT(7) | BITFIELD_BIT(6), 6}, /* Last shared use is leading. */
   };
   for (unsigned c = 0; c < ARRAY_SIZE(cases); ++c) {
      struct agx_shader_part part = {0};
      CHECK(compile_epilog_samples_shared(8, formats, 0xff, 0, false, false,
                                            1, cases[c].shared, &part));
      unsigned stores;
      unsigned f13 = store_f13_mask(&part, &stores);
      if (f13 != cases[c].expected_f13)
         fprintf(stderr, "MRT shared=%02x: f13=%02x expected=%02x\n",
                 cases[c].shared, f13, cases[c].expected_f13);
      CHECK(stores == 8);
      CHECK(f13 == cases[c].expected_f13);

      /* In these low-register stores byte 3 names the payload half-register
       * and byte 5 names the two-byte offset (as in the native tile-op vectors).
       * Assert actual reuse, so fresh copies cannot make the regression pass
       * without exercising a live source. The homogeneous layout is emitted
       * in descending physical order, RT7 through RT0. */
      const uint8_t *text = part.binary;
      unsigned n = 0, first_shared = ~0u;
      for (size_t at = 0; at + 12 <= part.info.binary_size;) {
         if (text[at] == 0xe7 && (text[at + 1] & 0x07) == 0x06 &&
             (text[at + 2] & 0xfc) == 0x54) {
            CHECK(n < 8);
            const unsigned rt = 7 - n++;
            CHECK(text[at + 5] == rt * 2);
            CHECK(text[at + 3] < 64);
            if (cases[c].shared & BITFIELD_BIT(rt)) {
               if (first_shared == ~0u)
                  first_shared = text[at + 3];
               else
                  CHECK(text[at + 3] == first_shared);
            }
            at += 12;
         } else {
            at++;
         }
      }
      CHECK(n == 8);
      free(part.binary);
   }
   return 0;
}

static int
test_epilogs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts) + ARRAY_SIZE(wide_layouts);
        ++i) {
      const struct layout *l = i < ARRAY_SIZE(layouts)
                                  ? &layouts[i]
                                  : &wide_layouts[i - ARRAY_SIZE(layouts)];
      for (unsigned write_z = 0; write_z < 2; ++write_z) {
         struct agx_shader_part part = {0};
         CHECK(compile_epilog(l->count, l->formats, false, write_z, &part));
         CHECK(part.info.agx3_colour_link.colour_classes == 7);
         CHECK(part.info.agx3_colour_link.entry_emitted);

         unsigned n = 0;
         uint8_t sequence[16][6];
         memcpy(sequence[n++], entry_512, 6);
         if (write_z) {
            memcpy(sequence[n++], wait_1, 6);
            memcpy(sequence[n++], depthf, 6);
            memcpy(sequence[n++], release_1, 6);
         }
         if (l->count == 3) {
            memcpy(sequence[n++], wait_192, 6);
            memcpy(sequence[n++], release_192, 6);
            memcpy(sequence[n++], wait_48, 6);
            memcpy(sequence[n++], release_48, 6);
            memcpy(sequence[n++], wait_12, 6);
            memcpy(sequence[n++], release_524, 6);
         } else {
            memcpy(sequence[n++], wait_12, 6);
            memcpy(sequence[n++], wait_192, 6);
            memcpy(sequence[n++], release_192, 6);
            memcpy(sequence[n++], wait_48, 6);
            memcpy(sequence[n++], release_48, 6);
            memcpy(sequence[n++], release_524, 6);
         }
         if (!contains_in_order(&part, (const uint8_t (*)[6])sequence, n)) {
            fprintf(stderr, "MRT epilog mismatch: layout=%u targets=%u depth=%u\n",
                    i, l->count, write_z);
            const uint8_t *bytes = part.binary;
            for (unsigned j = 0; j < part.info.binary_size; ++j)
               fprintf(stderr, "%02x", bytes[j]);
            fprintf(stderr, "\n");
         }
         CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));

         /* Wider than five targets, the leading stores set f13. */
         unsigned stores;
         const unsigned f13 = store_f13_mask(&part, &stores);
         CHECK(stores == l->count);
         CHECK(f13 == BITFIELD_MASK(MAX2(l->count, 5) - 5));
         free(part.binary);
      }
   }
   return 0;
}

/* The pixel transactions of Apple's store-only fragments writing some of the
 * colour targets, from metal-nt -arch applegpu_g17p render pipelines of two to
 * four targets for every written subset, indexed by the written partitions.
 * Each written partition keeps the classes of its position among all targets,
 * partitions store in descending order, the last store also releases class
 * 512, and class 12 stays held from partition 3 to partition 0 only when both
 * are written. A lone store to partition 0 is the one-target protocol.
 */
enum pixel_op { END, W12, W48, W192, R12, R48, R192, R524, R560, R704 };

static const struct {
   unsigned count, written;
   enum pixel_op ops[8];
} subset_witnesses[] = {
   {3, 0x1, {W12, R524}},
   {3, 0x2, {W48, R560}},
   {3, 0x4, {W192, R704}},
   {3, 0x3, {W48, R48, W12, R524}},
   {3, 0x5, {W192, R192, W12, R524}},
   {3, 0x6, {W192, R192, W48, R560}},
   {4, 0x1, {W12, R524}},
   {4, 0x2, {W48, R560}},
   {4, 0x4, {W192, R704}},
   {4, 0x8, {W12, R524}},
   {4, 0x3, {W48, R48, W12, R524}},
   {4, 0x5, {W192, R192, W12, R524}},
   {4, 0x9, {W12, R524}},
   {4, 0x6, {W192, R192, W48, R560}},
   {4, 0xa, {W12, R12, W48, R560}},
   {4, 0xc, {W12, R12, W192, R704}},
   {4, 0x7, {W192, R192, W48, R48, W12, R524}},
   {4, 0xb, {W12, W48, R48, R524}},
   {4, 0xd, {W12, W192, R192, R524}},
   {4, 0xe, {W12, R12, W192, R192, W48, R560}},
};

/* The same for eight RGBA8 targets, from metal-nt -arch applegpu_g17p render
 * pipelines writing these subsets, with the number of leading stores that set
 * lstcol f13: every partition from 3 on shares class 12 with partition 0, and
 * a transaction storing more than five targets including 1 and 2 sets f13 on
 * all but its last five stores. */
static const struct {
   unsigned written, f13;
   enum pixel_op ops[12];
} octa_subset_witnesses[] = {
   {0x01, 0, {W12, R524}},
   {0x02, 0, {W48, R560}},
   {0x08, 0, {W12, R524}},
   {0x80, 0, {W12, R524}},
   {0x0c, 0, {W12, R12, W192, R704}},
   {0x19, 0, {W12, R524}},
   {0x21, 0, {W12, R524}},
   {0x46, 0, {W12, R12, W192, R192, W48, R560}},
   {0x88, 0, {W12, R524}},
   {0x8e, 0, {W12, R12, W192, R192, W48, R560}},
   {0x87, 0, {W12, W192, R192, W48, R48, R524}},
   {0xc7, 0, {W12, W192, R192, W48, R48, R524}},
   {0xf0, 0, {W12, R524}},
   {0x1f, 0, {W12, W192, R192, W48, R48, R524}},
   {0x3e, 0, {W12, R12, W192, R192, W48, R560}},
   {0xe6, 0, {W12, R12, W192, R192, W48, R560}},
   {0xf1, 0, {W12, R524}},
   {0xf8, 0, {W12, R524}},
   {0xf9, 0, {W12, R524}},
   {0xfa, 0, {W12, R12, W48, R560}},
   {0xfb, 0, {W12, W48, R48, R524}},
   {0xfc, 0, {W12, R12, W192, R704}},
   {0xfd, 0, {W12, W192, R192, R524}},
   {0x3f, 1, {W12, W192, R192, W48, R48, R524}},
   {0xe7, 1, {W12, W192, R192, W48, R48, R524}},
   {0xf6, 1, {W12, R12, W192, R192, W48, R560}},
   {0xf7, 2, {W12, W192, R192, W48, R48, R524}},
   {0xfe, 2, {W12, R12, W192, R192, W48, R560}},
   {0xff, 3, {W12, W192, R192, W48, R48, R524}},
};

/* Layouts whose one-channel 32-bit targets reach the epilog in their fixed
 * export registers, GPR half 8 + 8n, beyond a lone store's register bank:
 * dEQP-VK.renderpasses attachment.3.286, 4.430 and 4.420. */
static const struct layout scalar_layouts[] = {
   {"3.286", 3,
    {PIPE_FORMAT_R16_SINT, PIPE_FORMAT_R16_UINT, PIPE_FORMAT_R32_UINT},
    {4, 6, 0}, 8},
   {"4.430", 4,
    {PIPE_FORMAT_R8G8_SINT, PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32G32_UINT,
     PIPE_FORMAT_R8G8B8A8_UNORM},
    {12, 0, 4, 14}, 24},
   {"4.420", 4,
    {PIPE_FORMAT_R16G16B16A16_SINT, PIPE_FORMAT_R8G8B8A8_UNORM,
     PIPE_FORMAT_B10G10R10A2_UNORM, PIPE_FORMAT_R32_UINT},
    {8, 16, 0, 4}, 24},
};

static int
check_subset_epilogs(const struct layout *l, unsigned samples)
{
   static const uint8_t *const bytes[] = {
      [W12] = wait_12,     [W48] = wait_48,     [W192] = wait_192,
      [R12] = release_12,  [R48] = release_48,  [R192] = release_192,
      [R524] = release_524, [R560] = release_560, [R704] = release_704,
   };

   uint64_t partitions = 0;
   for (unsigned rt = 0; rt < l->count; ++rt)
      partitions |= BITFIELD64_BIT(l->offsets[rt]);

   for (unsigned written = 1; written < BITFIELD_MASK(l->count); ++written) {
      unsigned ranks = 0;
      u_foreach_bit(rt, written) {
         ranks |= BITFIELD_BIT(
            util_bitcount64(partitions & BITFIELD64_MASK(l->offsets[rt])));
      }

      unsigned w = 0;
      while (subset_witnesses[w].count != l->count ||
             subset_witnesses[w].written != ranks)
         w++;

      unsigned classes = 0, n = 0;
      u_foreach_bit(rank, ranks)
         classes |= BITFIELD_BIT(rank % 3);
      uint8_t sequence[8][6];
      for (; n < 8 && subset_witnesses[w].ops[n] != END; ++n)
         memcpy(sequence[n], bytes[subset_witnesses[w].ops[n]], 6);

      struct agx_shader_part part = {0};
      CHECK(compile_epilog_samples(l->count, l->formats, written, partitions,
                                   false, false, samples, &part));
      CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));
      CHECK(part.info.agx3_colour_link.colour_classes == classes);
      free(part.binary);
   }
   return 0;
}

/* Apple's 2x/4x store-only fragments of every target acquire all their
 * classes in one PIXWAIT and release them once around stores through the
 * coverage register, with no lstcol f13 even for eight targets; fragments
 * writing only some targets bracket each partition exactly as at one sample
 * (metal-nt -arch applegpu_g17p render pipelines of two to eight RGBA8
 * targets, and of targets 1, 2, 1-2 of three and 1, 3 of four at 1x, 2x and
 * 4x: msaa/native/persample/{msmrt,msmrt2,mssubset}.py). */
static const uint8_t wait_60[6] = {0x87, 0x02, 0x54, 0x3c, 0x08, 0x00};
static const uint8_t wait_252[6] = {0x87, 0x02, 0x54, 0xfc, 0x08, 0x00};
static const uint8_t release_572[6] = {0x07, 0x02, 0x54, 0x3c, 0x02, 0x00};
static const uint8_t release_764[6] = {0x07, 0x02, 0x54, 0xfc, 0x02, 0x00};

static int
test_multisample_epilogs(void)
{
   for (unsigned samples = 2; samples <= 4; samples += 2) {
      for (unsigned i = 0; i < ARRAY_SIZE(layouts) + ARRAY_SIZE(wide_layouts);
           ++i) {
         const struct layout *l = i < ARRAY_SIZE(layouts)
                                     ? &layouts[i]
                                     : &wide_layouts[i - ARRAY_SIZE(layouts)];
         uint8_t sequence[3][6];
         memcpy(sequence[0], entry_512, 6);
         memcpy(sequence[1], l->count == 2 ? wait_60 : wait_252, 6);
         memcpy(sequence[2], l->count == 2 ? release_572 : release_764, 6);

         /* A layout whose tile would spill at this count is not stored. */
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
         memcpy(formats, l->formats, sizeof(l->formats));
         struct agx_tilebuffer_layout tib =
            agx_build_tilebuffer_layout(formats, l->count, samples, false);
         if (agx_tilebuffer_spills(&tib))
            continue;

         struct agx_shader_part part = {0};
         CHECK(compile_epilog_samples(l->count, l->formats,
                                      BITFIELD_MASK(l->count), 0, false, false,
                                      samples, &part));
         CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, 3));
         unsigned stores = 0;
         CHECK(store_f13_mask(&part, &stores) == 0 && stores == l->count);
         free(part.binary);
      }

      for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
         enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
         memcpy(formats, layouts[i].formats, sizeof(layouts[i].formats));
         struct agx_tilebuffer_layout tib = agx_build_tilebuffer_layout(
            formats, layouts[i].count, samples, false);
         if (!agx_tilebuffer_spills(&tib) &&
             check_subset_epilogs(&layouts[i], samples))
            return 1;
      }
   }

   /* A sample-shaded MRT epilog repeats the same descending target set once
    * per sample inside one combined colour transaction. This is the exact
    * shape used by the multisample-render-to-single-sample CTS family. */
   const enum pipe_format sample_shaded[] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R16G16B16A16_SINT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_shader_part part = {0};
   CHECK(compile_sample_shaded_epilog(ARRAY_SIZE(sample_shaded),
                                      sample_shaded, 2, &part));
   unsigned stores = 0;
   CHECK(store_f13_mask(&part, &stores) == 0);
   CHECK(stores == ARRAY_SIZE(sample_shaded) * 2);
   CHECK(part.info.agx3_colour_link.colour_classes == 7);
   free(part.binary);

   const enum pipe_format sample_shaded_4x[] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R16_SINT,
   };
   CHECK(compile_sample_shaded_epilog(ARRAY_SIZE(sample_shaded_4x),
                                      sample_shaded_4x, 4, &part));
   free(part.binary);

   return 0;
}

static int
test_octa_subset_epilogs(void)
{
   static const uint8_t *const bytes[] = {
      [W12] = wait_12,     [W48] = wait_48,     [W192] = wait_192,
      [R12] = release_12,  [R48] = release_48,  [R192] = release_192,
      [R524] = release_524, [R560] = release_560, [R704] = release_704,
   };
   const struct layout *l = &wide_layouts[3];
   CHECK(l->count == 8 && l->formats[7] == RGBA8);

   uint64_t partitions = 0;
   for (unsigned rt = 0; rt < l->count; ++rt)
      partitions |= BITFIELD64_BIT(l->offsets[rt]);

   for (unsigned w = 0; w < ARRAY_SIZE(octa_subset_witnesses); ++w) {
      const unsigned written = octa_subset_witnesses[w].written;
      unsigned classes = 0, n = 0;
      u_foreach_bit(rt, written)
         classes |= BITFIELD_BIT(rt < 3 ? rt : 0);
      uint8_t sequence[12][6];
      for (; n < 12 && octa_subset_witnesses[w].ops[n] != END; ++n)
         memcpy(sequence[n], bytes[octa_subset_witnesses[w].ops[n]], 6);

      struct agx_shader_part part = {0};
      CHECK(compile_epilog_written(l->count, l->formats, written,
                                   written == 0xff ? 0 : partitions, false,
                                   false, &part));
      CHECK(contains_in_order(&part, (const uint8_t (*)[6])sequence, n));
      CHECK(part.info.agx3_colour_link.colour_classes == classes);
      unsigned stores;
      CHECK(store_f13_mask(&part, &stores) ==
            BITFIELD_MASK(octa_subset_witnesses[w].f13));
      CHECK(stores == util_bitcount(written));
      free(part.binary);
   }
   return 0;
}

static int
test_subset_epilogs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(layouts); ++i) {
      if (check_subset_epilogs(&layouts[i], 1))
         return 1;
   }

   for (unsigned i = 0; i < ARRAY_SIZE(scalar_layouts); ++i) {
      const struct layout *l = &scalar_layouts[i];
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      memcpy(formats, l->formats, sizeof(l->formats));
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, l->count, 1, false);
      CHECK(tib.sample_size_B == l->sample_size_B);
      for (unsigned rt = 0; rt < l->count; ++rt)
         CHECK(agx_tilebuffer_offset_B(&tib, rt) == l->offsets[rt]);
      if (check_subset_epilogs(l, 1))
         return 1;
   }
   return 0;
}

static bool
compile_epilog_blended_preserve(unsigned count, enum pipe_format format,
                                unsigned blended, unsigned preserve_rt,
                                struct agx_shader_part *part)
{
   struct agx_fs_epilog_key epilog = {
      .link.loc_written = (uint8_t)BITFIELD_MASK(count),
      .link.size_32 = (uint8_t)BITFIELD_MASK(count),
      .nr_samples = 1,
      .max_sample_size_B = AGX_G17P_MAX_BYTES_PER_SAMPLE,
      .preserve_zero_mask = true,
      .g17p_tile_formats = true,
      .preserve_rt = preserve_rt,
   };
   memset(epilog.remap, -1, sizeof(epilog.remap));
   for (unsigned rt = 0; rt < count; ++rt) {
      const bool blend = blended & BITFIELD_BIT(rt);
      epilog.remap[rt] = rt;
      epilog.rt_formats[rt] = format;
      epilog.blend.rt[rt].colormask = 0xf;
      epilog.blend.rt[rt].mode = agx_pack_blend_standard(
         PIPE_BLEND_ADD,
         blend ? PIPE_BLENDFACTOR_SRC_ALPHA : PIPE_BLENDFACTOR_ONE,
         blend ? PIPE_BLENDFACTOR_INV_SRC_ALPHA : PIPE_BLENDFACTOR_ZERO,
         PIPE_BLEND_ADD, PIPE_BLENDFACTOR_ONE,
         blend ? PIPE_BLENDFACTOR_INV_SRC_ALPHA : PIPE_BLENDFACTOR_ZERO);
   }
   agx_fs_epilog_apply_device_policy(AGX_CHIP_G17P, &epilog);

   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "G17P blended MRT epilog");
   agx_nir_fs_epilog(&b, &epilog);
   agx_nir_lower_g17p_tile_samples(b.shader, 1);
   agx_preprocess_nir(b.shader);
   struct agx_shader_key key = {.dev.encoding = AGX_ISA_ENCODING_AGX3B,
                                .fs.nr_samples = 1,
                                .secondary = true};
   if (preserve_rt) {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {0};
      for (unsigned rt = 0; rt < count; ++rt)
         formats[rt] = format;
      const struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, count, 1, false);
      for (unsigned rt = 0; rt < count; ++rt)
         BITSET_SET(key.g17p_tile_partitions, tib._offset_B[rt]);
   }
   const bool ok = agx_compile_shader_nir(b.shader, &key, part);
   ralloc_free(b.shader);
   return ok;
}

static bool
compile_epilog_blended(unsigned count, enum pipe_format format,
                       unsigned blended, struct agx_shader_part *part)
{
   return compile_epilog_blended_preserve(count, format, blended, 0, part);
}

/* The colour transactions of a fragment in order: "W<class>" and
 * "R<class>" for PIXWAIT and PIXREL, "L<offset>" and "S<offset>" for the
 * loads and stores at a tile byte offset, "S<offset>*" for a store carrying
 * lstcol f13. */
static void
colour_transactions(const struct agx_shader_part *part, char *out,
                    size_t size)
{
   const uint8_t *text = part->binary;
   size_t used = 0;
   out[0] = 0;
   for (size_t at = 0; at + 6 <= part->info.binary_size;) {
      const bool pixel = (text[at] == 0x87 || text[at] == 0x07) &&
                         text[at + 1] == 0x02 && text[at + 2] == 0x54;
      const bool tile = at + 12 <= part->info.binary_size &&
                        (text[at] == 0x67 || text[at] == 0xe7) &&
                        (text[at + 1] & 0x07) == 0x06 &&
                        (text[at + 2] & 0xfc) == 0x54;
      int n = 0;
      if (pixel) {
         n = snprintf(out + used, size - used, "%s%c%u", used ? " " : "",
                      text[at] == 0x87 ? 'W' : 'R',
                      text[at + 3] | ((text[at + 4] & 3) << 8));
         at += 6;
      } else if (tile) {
         const bool store = text[at] == 0xe7;
         n = snprintf(out + used, size - used, "%s%c%u%s", used ? " " : "",
                      store ? 'S' : 'L', text[at + 5] << 1,
                      store && (text[at + 7] & 0x80) ? "*" : "");
         at += 12;
      } else {
         at++;
      }
      if (n > 0)
         used = MIN2(used + n, size - 1);
   }
}

/* Split a transaction string at its first colour store's write wait: the
 * loads, with the read classes acquired for them, and everything after. */
static const char *
store_phase(const char *ops)
{
   const char *s = strchr(ops, 'S');
   if (!s)
      return ops;
   while (s > ops && s[-1] == ' ')
      --s;
   while (s > ops && s[-1] != ' ')
      --s;
   return (s[0] == 'W') ? s : strchr(s, 'S');
}

/* Every load must hold its partition's read class: 8 for partition 0 and
 * the fourth partition on, 32 and 128 for partitions 1 and 2. */
static bool
loads_hold_classes(const char *ops, unsigned stride, char *loads,
                   size_t size)
{
   unsigned held = 0;
   size_t used = 0;
   loads[0] = 0;
   const char *end = store_phase(ops);
   for (const char *p = ops; p < end && *p;) {
      unsigned value = strtoul(p + 1, NULL, 10);
      if (p[0] == 'W') {
         held |= value;
      } else if (p[0] == 'L') {
         const unsigned rt = value / stride;
         const unsigned read = rt == 1 ? 32 : rt == 2 ? 128 : 8;
         if (!(held & read))
            return false;
         used += snprintf(loads + used, size - used, "%u,", value);
      }
      p = strchr(p, ' ');
      if (!p)
         break;
      ++p;
   }
   return true;
}

/* The pixel transactions of Apple's blending fragments: metal-nt -arch
 * applegpu_g17p -platform_version macos 26.0 26.0 render pipelines of
 * `fs<N>` storing N float4 [[color(i)]] outputs, with blending
 * (SourceAlpha / OneMinusSourceAlpha, Add) enabled on the attachments in
 * `blended` and disabled on the rest. RGBA8Unorm witnesses load in
 * descending target order; RGBA16Float ones load in another order under the
 * same classes, so those compare their loads and store phases only. */
static const struct {
   unsigned count;
   enum pipe_format format;
   unsigned blended;
   bool exact;
   const char *ops;
} blend_witnesses[] = {
   {2, RGBA8, 0x3, true, "W512 W32 L4 W8 L0 W16 S4 R48 W4 S0 R524"},
   {3, RGBA8, 0x7, true,
    "W512 W128 L8 W32 L4 W8 L0 W64 S8 R192 W16 S4 R48 W4 S0 R524"},
   {4, RGBA8, 0xf, true,
    "W512 W8 L12 W128 L8 W32 L4 L0 W4 S12 W64 S8 R192 W16 S4 R48 S0 R524"},
   {5, RGBA8, 0x1f, true,
    "W512 W8 L16 L12 W128 L8 W32 L4 L0 W4 S16 S12 W64 S8 R192 W16 S4 R48 "
    "S0 R524"},
   {6, RGBA8, 0x3f, true,
    "W512 W8 L20 L16 L12 W128 L8 W32 L4 L0 W4 S20* S16 S12 W64 S8 R192 "
    "W16 S4 R48 S0 R524"},
   {7, RGBA8, 0x7f, true,
    "W512 W8 L24 L20 L16 L12 W128 L8 W32 L4 L0 W4 S24* S20* S16 S12 W64 "
    "S8 R192 W16 S4 R48 S0 R524"},
   {8, RGBA8, 0xff, true,
    "W512 W8 L28 L24 L20 L16 L12 W128 L8 W32 L4 L0 W4 S28* S24* S20* S16 "
    "S12 W64 S8 R192 W16 S4 R48 S0 R524"},
   {4, RGBA8, 0x01, true,
    "W512 W8 L0 W4 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {4, RGBA8, 0x03, true,
    "W512 W32 L4 W8 L0 W4 S12 W192 S8 R192 W16 S4 R48 S0 R524"},
   {4, RGBA8, 0x06, true,
    "W512 W128 L8 W32 L4 W12 S12 W64 S8 R192 W16 S4 R48 S0 R524"},
   {4, RGBA8, 0x07, true,
    "W512 W128 L8 W32 L4 W8 L0 W4 S12 W64 S8 R192 W16 S4 R48 S0 R524"},
   {4, RGBA8, 0x08, true,
    "W512 W8 L12 W4 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {4, RGBA8, 0x09, true,
    "W512 W8 L12 L0 W4 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {4, RGBA8, 0x0c, true,
    "W512 W8 L12 W128 L8 W4 S12 W64 S8 R192 W48 S4 R48 S0 R524"},
   {4, RGBA8, 0x0e, true,
    "W512 W8 L12 W128 L8 W32 L4 W4 S12 W64 S8 R192 W16 S4 R48 S0 R524"},
   {5, RGBA8, 0x10, true,
    "W512 W8 L16 W4 S16 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {5, RGBA8, 0x11, true,
    "W512 W8 L16 L0 W4 S16 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {8, RGBA8, 0x01, true,
    "W512 W8 L0 W4 S28* S24* S20* S16 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {8, RGBA8, 0x0f, true,
    "W512 W8 L12 W128 L8 W32 L4 L0 W4 S28* S24* S20* S16 S12 W64 S8 R192 "
    "W16 S4 R48 S0 R524"},
   {8, RGBA8, 0x10, true,
    "W512 W8 L16 W4 S28* S24* S20* S16 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {8, RGBA8, 0x21, true,
    "W512 W8 L20 L0 W4 S28* S24* S20* S16 S12 W192 S8 R192 W48 S4 R48 S0 "
    "R524"},
   {8, RGBA8, 0x80, true,
    "W512 W8 L28 W4 S28* S24* S20* S16 S12 W192 S8 R192 W48 S4 R48 S0 R524"},
   {8, RGBA8, 0x88, true,
    "W512 W8 L28 L12 W4 S28* S24* S20* S16 S12 W192 S8 R192 W48 S4 R48 S0 "
    "R524"},
   {8, RGBA8, 0xf0, true,
    "W512 W8 L28 L24 L20 L16 W4 S28* S24* S20* S16 S12 W192 S8 R192 W48 S4 "
    "R48 S0 R524"},
   {3, PIPE_FORMAT_R16G16B16A16_FLOAT, 0x7, false,
    "W512 W32 L8 W128 L16 W8 L0 W64 S16 R192 W16 S8 R48 W4 S0 R524"},
   {3, PIPE_FORMAT_R32G32B32A32_FLOAT, 0x7, true,
    "W512 W128 L32 W32 L16 W8 L0 W64 S32 R192 W16 S16 R48 W4 S0 R524"},
   {4, PIPE_FORMAT_R16G16B16A16_FLOAT, 0xf, false,
    "W512 W128 L16 W32 L8 W8 L24 L0 W4 S24 W64 S16 R192 W16 S8 R48 S0 R524"},
   {5, PIPE_FORMAT_R16G16B16A16_FLOAT, 0x1f, false,
    "W512 W8 L24 W128 L16 W32 L8 L0 L32 W4 S32 S24 W64 S16 R192 W16 S8 R48 "
    "S0 R524"},
   {8, PIPE_FORMAT_R16G16B16A16_FLOAT, 0xff, false,
    "W512 W8 L48 L40 L32 L24 W128 L16 W32 L8 L56 L0 W4 S56* S48* S40* S32 "
    "S24 W64 S16 R192 W16 S8 R48 S0 R524"},
};

static int
test_blended_epilogs(void)
{
   for (unsigned i = 0; i < ARRAY_SIZE(blend_witnesses); ++i) {
      const unsigned count = blend_witnesses[i].count;
      const enum pipe_format format = blend_witnesses[i].format;
      const unsigned stride = util_format_get_blocksize(format);
      struct agx_shader_part part = {0};
      CHECK(compile_epilog_blended(count, format, blend_witnesses[i].blended,
                                   &part));

      char ops[512], want_loads[128], got_loads[128];
      colour_transactions(&part, ops, sizeof(ops));
      free(part.binary);
      if (blend_witnesses[i].exact) {
         if (strcmp(ops, blend_witnesses[i].ops))
            fprintf(stderr, "blend %u:\n  got  %s\n  want %s\n", i, ops,
                    blend_witnesses[i].ops);
         CHECK(!strcmp(ops, blend_witnesses[i].ops));
         continue;
      }

      /* Loads in either order under the same held classes, the same set of
       * loads, then an identical store phase. */
      CHECK(loads_hold_classes(ops, stride, got_loads, sizeof(got_loads)));
      CHECK(loads_hold_classes(blend_witnesses[i].ops, stride, want_loads,
                               sizeof(want_loads)));
      unsigned got_set = 0, want_set = 0;
      for (char *p = got_loads; *p; p = strchr(p, ',') + 1)
         got_set |= BITFIELD_BIT(strtoul(p, NULL, 10) / stride);
      for (char *p = want_loads; *p; p = strchr(p, ',') + 1)
         want_set |= BITFIELD_BIT(strtoul(p, NULL, 10) / stride);
      CHECK(got_set == want_set);
      if (strcmp(store_phase(ops), store_phase(blend_witnesses[i].ops)))
         fprintf(stderr, "blend %u stores:\n  got  %s\n  want %s\n", i,
                 store_phase(ops), store_phase(blend_witnesses[i].ops));
      CHECK(!strcmp(store_phase(ops), store_phase(blend_witnesses[i].ops)));
   }

   return 0;
}

static int
test_preserved_blended_epilogs(void)
{
   /* VK_EXT_shader_object changes color-write enable dynamically. The CTS
    * path disables RT0 while blending RT1..3, so production reloads and stores
    * RT0 unchanged instead of dropping the store from a nonempty HAL200 tile. */
   const enum pipe_format preserve_formats[] = {
      PIPE_FORMAT_R32_FLOAT, PIPE_FORMAT_R32G32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT};
   for (unsigned i = 0; i < ARRAY_SIZE(preserve_formats); ++i) {
      struct agx_shader_part part = {0};
      CHECK(compile_epilog_blended_preserve(
         4, preserve_formats[i], 0xe, BITFIELD_BIT(0), &part));

      if (preserve_formats[i] == PIPE_FORMAT_R32G32B32A32_FLOAT) {
         char ops[256];
         colour_transactions(&part, ops, sizeof(ops));
         CHECK(!strcmp(ops, "W512 W8 L48 W128 L32 W32 L16 W4 S48 R12 "
                            "W64 S32 R192 W16 S16 R560"));
      }
      free(part.binary);
   }
   return 0;
}

/* G17P keeps colour targets in the imageblock past 64 bytes a sample. The
 * witnesses are metal-nt -arch applegpu_g17p render pipelines storing float4
 * [[color(i)]] outputs (blending as in blend_witnesses where noted), and
 * builtin BG constructors for five RGBA32Float targets at sample size 80. */
static int
test_wide_samples(void)
{
   const enum pipe_format rgba32f = PIPE_FORMAT_R32G32B32A32_FLOAT;
   enum pipe_format f5[AGX_MAX_RENDER_TARGETS] = {rgba32f, rgba32f, rgba32f,
                                                  rgba32f, rgba32f};
   enum pipe_format f8[AGX_MAX_RENDER_TARGETS] = {rgba32f, rgba32f, rgba32f,
                                                  rgba32f, rgba32f, rgba32f,
                                                  rgba32f, rgba32f};
   /* An eight-target CTS attachment set of 65 bytes. */
   enum pipe_format m8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R16_FLOAT,
      PIPE_FORMAT_R16G16B16A16_UNORM, PIPE_FORMAT_R16_SINT,
      PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R8G8B8A8_SNORM,
      PIPE_FORMAT_R32G32B32A32_SINT, PIPE_FORMAT_R8_SNORM,
   };
   const uint8_t bound = AGX_G17P_MAX_BYTES_PER_SAMPLE;

   /* The AGX2 bound is unchanged. */
   const struct agx_tilebuffer_layout agx2 =
      agx_build_tilebuffer_layout(f5, 5, 1, false);
   CHECK(agx2.spilled[4] && agx2.sample_size_B == 64 && !agx2.max_sample_size_B);

   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout_bounded(f5, 5, 1, false, bound);
   CHECK(!agx_tilebuffer_spills(&tib) && tib.sample_size_B == 80 &&
         tib.tile_size == 16 * 16);
   for (unsigned rt = 0; rt < 5; ++rt)
      CHECK(tib._offset_B[rt] == 16 * rt);
   CHECK(agx_g17p_classify_mrt(&tib) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&tib));

   /* A layout completed at draw time keeps its bound. */
   struct agx_tilebuffer_layout deferred =
      agx_build_tilebuffer_layout_bounded(f5, 5, 0, false, bound);
   agx_tilebuffer_set_samples(&deferred, 1);
   CHECK(!agx_tilebuffer_spills(&deferred) && deferred.sample_size_B == 80 &&
         deferred.max_sample_size_B == bound);

   /* 128 bytes fill a 16x16 tile to exactly 32 KiB, and two samples of 80
    * bytes overflow it: both still spill. */
   struct agx_tilebuffer_layout full =
      agx_build_tilebuffer_layout_bounded(f8, 8, 1, false, bound);
   CHECK(agx_tilebuffer_spills(&full) && !agx_g17p_rt0_tile_layout_supported(&full));
   struct agx_tilebuffer_layout two =
      agx_build_tilebuffer_layout_bounded(f5, 5, 2, false, bound);
   CHECK(agx_tilebuffer_spills(&two));

   struct agx_tilebuffer_layout mixed =
      agx_build_tilebuffer_layout_bounded(m8, 8, 1, false, bound);
   CHECK(!agx_tilebuffer_spills(&mixed) && mixed.sample_size_B == 72 &&
         mixed.tile_size == 16 * 16);
   CHECK(agx_g17p_classify_mrt(&mixed) == AGX_G17P_MRT_MIXED);
   CHECK(agx_g17p_rt0_tile_layout_supported(&mixed));

   /* The end-of-tile programs take each offset as a uniform. Apple's builtin
    * storing any one target of eight RGBA32Float targets is the 68-byte
    * single-target program, and Mesa's for the target at byte 64 matches it
    * byte for byte. */
   static const uint8_t apple_single_rgba32f[68] = {
      0x1c, 0x9c, 0x10, 0x06, 0x24, 0xa8, 0x14, 0x06, 0x0c, 0x9d, 0x10, 0x26,
      0x24, 0xa9, 0x10, 0x26, 0x9f, 0x10, 0x54, 0x00, 0x00, 0x04, 0x14, 0x00,
      0x50, 0x06, 0x0b, 0x00, 0x9f, 0x20, 0x54, 0x02, 0x00, 0x00, 0x10, 0x00,
      0x50, 0x06, 0x0b, 0x00, 0x97, 0x05, 0x54, 0x04, 0x00, 0x08, 0x00, 0x00,
      0xd0, 0x40, 0x10, 0x00, 0x57, 0x10, 0x54, 0x00, 0x00, 0x00, 0x00, 0x80,
      0xa8, 0x35, 0x00, 0x00, 0x0e, 0x00, 0x00, 0x00,
   };
   struct agx_shader_key eot_key = {
      .dev.encoding = AGX_ISA_ENCODING_AGX3B,
      .no_preamble = true,
   };
   agx_g17p_configure_rt0_eot_key(rgba32f, &eot_key);
   const uint8_t offset64 = 64;
   struct agx_shader_part eot = {0};
   CHECK(compile_eot(1, &rgba32f, &offset64, &eot_key, &eot));
   CHECK(eot.info.binary_size == sizeof(apple_single_rgba32f) &&
         !memcmp(eot.binary, apple_single_rgba32f,
                 sizeof(apple_single_rgba32f)));
   free(eot.binary);

   struct layout wide = {"5.rgba32f", 5, {0}, {0}, tib.sample_size_B};
   struct layout wide_mixed = {"8.mixed", 8, {0}, {0}, mixed.sample_size_B};
   for (unsigned rt = 0; rt < 8; ++rt) {
      if (rt < 5) {
         wide.formats[rt] = f5[rt];
         wide.offsets[rt] = tib._offset_B[rt];
      }
      wide_mixed.formats[rt] = m8[rt];
      wide_mixed.offsets[rt] = mixed._offset_B[rt];
   }
   CHECK(!check_eot_programs(&wide, false));
   CHECK(!check_eot_programs(&wide_mixed, false));

   /* Apple's clear and reload builtins store every target at its offset:
    * lstcol f57 = 0, 16, 32, 48 and 64. */
   for (unsigned reloads = 0; reloads <= 5; reloads += 5) {
      struct agx_shader_part part = {0};
      CHECK(compile_background(&tib, reloads, true,
                               reloads < 5 ? 4 + 8 * 5 : 0, &part));
      char ops[256];
      colour_transactions(&part, ops, sizeof(ops));
      free(part.binary);
      unsigned stored = 0;
      for (const char *p = ops; (p = strchr(p, 'S')); ++p)
         stored |= BITFIELD_BIT(strtoul(p + 1, NULL, 10) / 16);
      CHECK(stored == BITFIELD_MASK(5) && !strchr(ops, 'W'));
   }

   /* dEQP-VK.renderpasses.*.attachment.8.620: eight targets of 80 bytes.
    * Apple's clear and reload builtins for this layout (BTNF, sample size 80,
    * these offsets) store the same targets, the reload in the same order, and
    * a metal-nt render pipeline declaring the attachments in offset order
    * stores them with the same transactions, f13 and classes. */
   enum pipe_format a620[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_UINT, PIPE_FORMAT_R32G32B32A32_SINT,
      PIPE_FORMAT_R16G16B16A16_SINT, PIPE_FORMAT_R16G16_UNORM,
      PIPE_FORMAT_R16G16B16A16_UINT, PIPE_FORMAT_R16G16_SNORM,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R16G16B16A16_UNORM,
   };
   static const uint8_t offsets620[8] = {0, 16, 48, 56, 60, 68, 32, 72};
   struct agx_tilebuffer_layout t620 =
      agx_build_tilebuffer_layout_bounded(a620, 8, 1, false, bound);
   CHECK(!agx_tilebuffer_spills(&t620) && t620.sample_size_B == 80 &&
         t620.tile_size == 16 * 16 &&
         !memcmp(t620._offset_B, offsets620, sizeof(offsets620)));
   CHECK(agx_g17p_rt0_tile_layout_supported(&t620));
   {
      char ops[512];
      struct agx_shader_part part = {0};
      CHECK(compile_background(&t620, 8, true, 0, &part));
      colour_transactions(&part, ops, sizeof(ops));
      free(part.binary);
      CHECK(!strcmp(ops, "S0 S16 S48 S56 S60 S68 S32 S72"));

      part = (struct agx_shader_part){0};
      CHECK(compile_background(&t620, 0, true, 4 + 8 * 8, &part));
      colour_transactions(&part, ops, sizeof(ops));
      free(part.binary);
      uint32_t stored = 0, expected = 0;
      unsigned stores = 0;
      for (const char *p = ops; (p = strchr(p, 'S')); ++p, ++stores)
         stored |= BITFIELD_BIT(strtoul(p + 1, NULL, 10) / 4);
      for (unsigned rt = 0; rt < 8; ++rt)
         expected |= BITFIELD_BIT(offsets620[rt] / 4);
      CHECK(stores == 8 && stored == expected && !strchr(ops, 'W'));

      part = (struct agx_shader_part){0};
      CHECK(compile_epilog(8, a620, false, false, &part));
      colour_transactions(&part, ops, sizeof(ops));
      free(part.binary);
      CHECK(!strcmp(ops, "W512 W12 S72* S68* S60* S56 S48 W192 S32 R192 W48 "
                         "S16 R48 S0 R524"));
   }

   /* Store-only and blending fragments of five RGBA32Float targets. */
   struct agx_shader_part part = {0};
   CHECK(compile_epilog(5, f5, false, false, &part));
   char ops[512];
   colour_transactions(&part, ops, sizeof(ops));
   free(part.binary);
   CHECK(!strcmp(ops, "W512 W12 S64 S48 W192 S32 R192 W48 S16 R48 S0 R524"));

   part = (struct agx_shader_part){0};
   CHECK(compile_epilog_blended(5, rgba32f, BITFIELD_MASK(5), &part));
   colour_transactions(&part, ops, sizeof(ops));
   free(part.binary);
   char loads[128];
   CHECK(loads_hold_classes(ops, 16, loads, sizeof(loads)));
   CHECK(!strcmp(store_phase(ops),
                 store_phase("W512 W8 L64 L48 W128 L32 W32 L16 L0 W4 S64 S48 "
                             "W64 S32 R192 W16 S16 R48 S0 R524")));
   return 0;
}

int
main(void)
{
   if (test_selectors_match_single_target_programs() ||
       test_layout_classification() || test_eot_programs() ||
       test_background_programs() || test_epilogs() ||
       test_reused_mrt_payloads() ||
       test_subset_epilogs() || test_octa_subset_epilogs() ||
       test_multisample_eot_witnesses() || test_multisample_epilogs() ||
       test_blended_epilogs() || test_preserved_blended_epilogs() ||
       test_wide_samples())
      return 1;

   printf("G17P mixed-format MRT: %u checks PASS\n", checks);
   return 0;
}
