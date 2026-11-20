/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "agx_native_descriptor.h"
#include "asahi/genxml/agx_pack.h"
#include "asahi/lib/agx_abi.h"
#include "asahi/compiler/agx_nir.h"
#include "asahi/lib/agx_pbe_agx3_core.h"
#include "asahi/vulkan/hk_native_descriptor.h"
#include "util/macros.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned checks;

#define CHECK(cond)                                                            \
   do {                                                                        \
      ++checks;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__,   \
                 #cond);                                                       \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static void
test_chip_routing(void)
{
   static const struct {
      enum agx_chip chip;
      enum agx_descriptor_generation generation;
      unsigned size;
   } cases[] = {
      {AGX_CHIP_G13G, AGX_DESCRIPTOR_GENERATION_7, 24},
      {AGX_CHIP_G13X, AGX_DESCRIPTOR_GENERATION_7, 24},
      {AGX_CHIP_G14G, AGX_DESCRIPTOR_GENERATION_8, 24},
      {AGX_CHIP_G14X, AGX_DESCRIPTOR_GENERATION_8, 24},
      {AGX_CHIP_G15G, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G15S, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G15C, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G15D, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G16P, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G16G, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G16S, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G16C, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G17P, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G17A, AGX_DESCRIPTOR_GENERATION_9, 32},
      {AGX_CHIP_G17G, AGX_DESCRIPTOR_GENERATION_10, 32},
      {AGX_CHIP_G17S, AGX_DESCRIPTOR_GENERATION_10, 32},
      {AGX_CHIP_G17C, AGX_DESCRIPTOR_GENERATION_10, 32},
   };

   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
      const enum agx_descriptor_generation generation =
         agx_chip_descriptor_generation(cases[i].chip);
      CHECK(generation == cases[i].generation);
      CHECK(agx_gallium_texture_descriptor_size(generation) == cases[i].size);
      CHECK(agx_gallium_pbe_descriptor_size(generation) == cases[i].size);

      const struct agx_gallium_storage_descriptor_layout storage =
         agx_gallium_storage_descriptor_layout(generation);
      CHECK(storage.texture == 0);
      CHECK(storage.pbe == cases[i].size);
      CHECK(storage.stride == 2 * cases[i].size);
   }

   CHECK(agx_chip_descriptor_generation(AGX_CHIP_UNKNOWN) ==
         AGX_DESCRIPTOR_GENERATION_UNKNOWN);
   CHECK(agx_chip_descriptor_generation(AGX_CHIP_G15P) ==
         AGX_DESCRIPTOR_GENERATION_UNKNOWN);
   CHECK(agx_gallium_storage_descriptor_layout(
            agx_chip_descriptor_generation(AGX_CHIP_G15P))
            .stride == 0);
   CHECK(agx_gallium_texture_descriptor_size(
            AGX_DESCRIPTOR_GENERATION_UNKNOWN) == 0);
   CHECK(agx_gallium_pbe_descriptor_size(
            AGX_DESCRIPTOR_GENERATION_UNKNOWN) == 0);
}

static void
test_feature_gates(void)
{
   CHECK(agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_7, true, 99));
   CHECK(agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_8, true, 99));
   CHECK(agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_9, false, 0));
   CHECK(agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_9, true, 0));
   CHECK(!agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_9, false, 1));
   CHECK(agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_10, true, 0));
   CHECK(!agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_10, false, 1));
   CHECK(!agx_gallium_native_texture_supported(
      AGX_DESCRIPTOR_GENERATION_UNKNOWN, false, 0));

   CHECK(agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_7, 2));
   CHECK(agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_8, 4));
   CHECK(agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_9, 2));
   CHECK(agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_9, 4));
   CHECK(agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_9, 8));
   CHECK(!agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_9, 16));
   CHECK(agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_10, 8));
   CHECK(!agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_10, 16));
   CHECK(!agx_gallium_native_texture_sample_count_representable(
      AGX_DESCRIPTOR_GENERATION_UNKNOWN, 1));

   CHECK(agx_gallium_native_pbe_supported(AGX_DESCRIPTOR_GENERATION_7));
   CHECK(agx_gallium_native_pbe_supported(AGX_DESCRIPTOR_GENERATION_8));
   CHECK(agx_gallium_native_pbe_supported(AGX_DESCRIPTOR_GENERATION_9));
   CHECK(agx_gallium_native_pbe_supported(AGX_DESCRIPTOR_GENERATION_10));
   CHECK(!agx_gallium_native_pbe_supported(AGX_DESCRIPTOR_GENERATION_UNKNOWN));

   CHECK(agx_gallium_shared_compression_supported(
      AGX_DESCRIPTOR_GENERATION_7));
   CHECK(agx_gallium_shared_compression_supported(
      AGX_DESCRIPTOR_GENERATION_8));
   CHECK(!agx_gallium_shared_compression_supported(
      AGX_DESCRIPTOR_GENERATION_9));
   CHECK(!agx_gallium_shared_compression_supported(
      AGX_DESCRIPTOR_GENERATION_10));
   CHECK(!agx_gallium_shared_compression_supported(
      AGX_DESCRIPTOR_GENERATION_UNKNOWN));
}

static void
test_g17p_txf_sampler(void)
{
   /* Same byte-exact sampler-zero vector as Honeykrisp's known-good G17P
    * texture.read() path. The full hardware heap element is 32 bytes. */
   static const uint8_t expected[AGX_GALLIUM_SAMPLER_AGX3_SIZE] = {
      0x00, 0x00, 0x0e, 0x68, 0x1b, 0x00, 0x00, 0x00,
   };
   struct agx_gallium_sampler_descriptor sampler;
   memset(&sampler, 0xa5, sizeof(sampler));
   CHECK(agx_gallium_pack_g17p_txf_sampler(AGX_CHIP_G17P, &sampler));
   CHECK(memcmp(&sampler, expected, sizeof(expected)) == 0);

   /* G15S packs the Gen4 layout, which differs from HAL200 only above bit
    * 42 (bias, border colour, seamful cubes); sampler zero leaves those
    * clear, so the bytes are the same. */
   memset(&sampler, 0xa5, sizeof(sampler));
   CHECK(agx_gallium_pack_g17p_txf_sampler(AGX_CHIP_G15S, &sampler));
   CHECK(memcmp(&sampler, expected, sizeof(expected)) == 0);

   /* G17G's native fragment fetches address the texture state table
    * directly, so it also takes a texel-fetch sampler, in its Gen5 record. */
   memset(&sampler, 0xa5, sizeof(sampler));
   CHECK(agx_gallium_pack_g17p_txf_sampler(AGX_CHIP_G17G, &sampler));
   CHECK(memcmp(&sampler,
                &(struct agx_gallium_sampler_descriptor){0},
                sizeof(sampler)) != 0);

   /* G18G packs the same fetch sampler as G17G's Gen5 record (the generation
    * hk_pack_native_sampler emits there): the compare-mode selector is set
    * even without a comparison. */
   static const uint8_t expected_g18g[AGX_GALLIUM_SAMPLER_AGX3_SIZE] = {
      0x00, 0x00, 0x0f, 0x68, 0x1b, 0x04, 0x00, 0x00,
   };
   memset(&sampler, 0xa5, sizeof(sampler));
   CHECK(agx_gallium_pack_g17p_txf_sampler(AGX_CHIP_G18G, &sampler));
   if (memcmp(&sampler, expected_g18g, sizeof(expected_g18g)) != 0) {
      const uint8_t *b = (const uint8_t *)&sampler;
      fprintf(stderr, "G18G txf sampler bytes:");
      for (unsigned i = 0; i < sizeof(expected_g18g); ++i)
         fprintf(stderr, " %02x", b[i]);
      fprintf(stderr, "\n");
   }
   CHECK(memcmp(&sampler, expected_g18g, sizeof(expected_g18g)) == 0);
   CHECK(!agx_gallium_pack_g17p_txf_sampler(AGX_CHIP_G18G, NULL));
   CHECK(!agx_gallium_pack_g17p_txf_sampler(AGX_CHIP_G17P, NULL));
}

/* The same API state lands in the chip's own sampler word: G17P keeps the
 * bias in the HAL200 field, G15S's Gen4 word has none. */
static void
test_native_sampler_generations(void)
{
   const struct agx_gallium_native_sampler state = {
      .minimum_lod = 1.0f,
      .maximum_lod = 100.0f,
      .lod_bias = -1.015625f,
      .maximum_anisotropy = 8,
      .magnify = AGX_FILTER_LINEAR,
      .mip_filter = AGX_MIP_FILTER_LINEAR,
      .wrap_s = AGX_WRAP_REPEAT,
      .compare_enable = true,
      .compare_func = AGX_COMPARE_FUNC_GEQUAL,
      .border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE,
      .seamful_cube_maps = true,
   };

   struct agx_gallium_sampler_descriptor hal200, gen4;
   uint32_t expected[AGX_GALLIUM_SAMPLER_AGX3_SIZE / 4] = {0};
   CHECK(agx_gallium_pack_native_sampler(AGX_CHIP_G17P, &hal200, &state));
   agx_pack(expected, SAMPLER_HAL200, cfg) {
      cfg.minimum_lod = 1.0f;
      cfg.maximum_lod = 100.0f;
      cfg.lod_bias = -1.015625f;
      cfg.maximum_anisotropy = 8;
      cfg.magnify = AGX_FILTER_LINEAR;
      cfg.mip_filter = AGX_MIP_FILTER_LINEAR;
      cfg.wrap_s = AGX_WRAP_REPEAT;
      cfg.compare_enable = true;
      cfg.compare_func = AGX_COMPARE_FUNC_GEQUAL;
      cfg.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
      cfg.seamful_cube_maps = true;
   }
   CHECK(memcmp(&hal200, expected, sizeof(expected)) == 0);

   memset(expected, 0, sizeof(expected));
   CHECK(agx_gallium_pack_native_sampler(AGX_CHIP_G15S, &gen4, &state));
   agx_pack(expected, SAMPLER, cfg) {
      cfg.minimum_lod = 1.0f;
      cfg.maximum_lod = 100.0f;
      cfg.maximum_anisotropy = 8;
      cfg.magnify = AGX_FILTER_LINEAR;
      cfg.mip_filter = AGX_MIP_FILTER_LINEAR;
      cfg.wrap_s = AGX_WRAP_REPEAT;
      cfg.compare_enable = true;
      cfg.compare_func = AGX_COMPARE_FUNC_GEQUAL;
      cfg.border_colour = AGX_BORDER_COLOUR_OPAQUE_WHITE;
      cfg.seamful_cube_maps = true;
   }
   CHECK(memcmp(&gen4, expected, sizeof(expected)) == 0);
   CHECK(memcmp(&gen4, &hal200, sizeof(gen4)) != 0);

   CHECK(!agx_gallium_pack_native_sampler(AGX_CHIP_G13G, &gen4, &state));
   CHECK(!agx_gallium_pack_native_sampler(AGX_CHIP_G17P, &gen4, NULL));
}

static void
test_g17p_sampler_handles(void)
{
   static const struct {
      unsigned api;
      uint16_t native;
   } cases[] = {
      {0, 1},
      {AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS - 1,
       AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS},
      {AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS,
       AGX_GALLIUM_G17P_IMMEDIATE_API_SAMPLERS + 1},
      {AGX_GALLIUM_G17P_API_SAMPLERS - 1,
       AGX_GALLIUM_G17P_API_SAMPLERS},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      uint16_t native = UINT16_MAX;
      CHECK(agx_gallium_g17p_sampler_handle(AGX_CHIP_G17P, cases[i].api,
                                            &native));
      CHECK(native == cases[i].native);

      uint16_t custom = UINT16_MAX;
      CHECK(agx_gallium_g17p_custom_border_handle(
         AGX_CHIP_G17P, cases[i].api, &custom));
      CHECK(custom == AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + cases[i].api);

      native = custom = UINT16_MAX;
      CHECK(agx_gallium_g17p_sampler_handle(AGX_CHIP_G15S, cases[i].api,
                                            &native));
      CHECK(agx_gallium_g17p_custom_border_handle(
         AGX_CHIP_G15S, cases[i].api, &custom));
      CHECK(native == cases[i].native &&
            custom == AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + cases[i].api);

      /* G18G shares the G17P sampler table layout: its API samplers follow
       * the fetch sampler at slot 0 the same way. */
      native = custom = UINT16_MAX;
      CHECK(agx_gallium_g17p_sampler_handle(AGX_CHIP_G18G, cases[i].api,
                                            &native));
      CHECK(agx_gallium_g17p_custom_border_handle(
         AGX_CHIP_G18G, cases[i].api, &custom));
      CHECK(native == cases[i].native &&
            custom == AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE + cases[i].api);
   }

   uint16_t untouched = 0x5555;
   CHECK(!agx_gallium_g17p_sampler_handle(
      AGX_CHIP_G17P, AGX_GALLIUM_G17P_API_SAMPLERS, &untouched));
   CHECK(untouched == 0x5555);
   untouched = 0x5555;
   CHECK(agx_gallium_g17p_sampler_handle(AGX_CHIP_G17G, 0, &untouched));
   CHECK(untouched == 1);
   CHECK(!agx_gallium_g17p_sampler_handle(AGX_CHIP_G17P, 0, NULL));
   untouched = 0x5555;
   CHECK(!agx_gallium_g17p_custom_border_handle(
      AGX_CHIP_G17P, AGX_GALLIUM_G17P_API_SAMPLERS, &untouched));
   CHECK(untouched == 0x5555);
   untouched = 0x5555;
   CHECK(agx_gallium_g17p_custom_border_handle(AGX_CHIP_G17G, 0,
                                               &untouched));
   CHECK(untouched == AGX_GALLIUM_G17P_CUSTOM_BORDER_BASE);
   CHECK(!agx_gallium_g17p_custom_border_handle(AGX_CHIP_G17P, 0, NULL));
   untouched = 0x5555;
   CHECK(!agx_gallium_g17p_sampler_handle(
      AGX_CHIP_G18G, AGX_GALLIUM_G17P_API_SAMPLERS, &untouched));
   CHECK(!agx_gallium_g17p_custom_border_handle(
      AGX_CHIP_G18G, AGX_GALLIUM_G17P_API_SAMPLERS, &untouched));
   CHECK(untouched == 0x5555);
   CHECK(!agx_gallium_g17p_sampler_handle(AGX_CHIP_G18G, 0, NULL));
   CHECK(!agx_gallium_g17p_custom_border_handle(AGX_CHIP_G18G, 0, NULL));
}

static void
test_standard_border_colours(void)
{
   enum agx_border_colour result = AGX_BORDER_COLOUR_CUSTOM;
   uint32_t color[4] = {0, 0x80000000, 0, 0};
   CHECK(agx_gallium_classify_standard_border(color, false, &result));
   CHECK(result == AGX_BORDER_COLOUR_TRANSPARENT_BLACK);

   memcpy(color, (uint32_t[4]){0, 0, 0, 0x3f800000}, sizeof(color));
   CHECK(agx_gallium_classify_standard_border(color, false, &result));
   CHECK(result == AGX_BORDER_COLOUR_OPAQUE_BLACK);

   memcpy(color,
          (uint32_t[4]){0x3f800000, 0x3f800000, 0x3f800000, 0x3f800000},
          sizeof(color));
   CHECK(agx_gallium_classify_standard_border(color, false, &result));
   CHECK(result == AGX_BORDER_COLOUR_OPAQUE_WHITE);

   memcpy(color, (uint32_t[4]){1, 1, 1, 1}, sizeof(color));
   CHECK(agx_gallium_classify_standard_border(color, true, &result));
   CHECK(result == AGX_BORDER_COLOUR_OPAQUE_WHITE);

   memcpy(color, (uint32_t[4]){0x3e800000, 0, 0, 0x3f800000},
          sizeof(color));
   result = AGX_BORDER_COLOUR_CUSTOM;
   CHECK(!agx_gallium_classify_standard_border(color, false, &result));
   CHECK(result == AGX_BORDER_COLOUR_CUSTOM);
   CHECK(!agx_gallium_classify_standard_border(NULL, false, &result));
   CHECK(!agx_gallium_classify_standard_border(color, false, NULL));
}

/* A plain 2D render target: exactly what the end-of-tile store binds. */
static struct AGX_PBE
render_target_pbe(void)
{
   struct AGX_PBE cfg = {AGX_PBE_header};
   cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
   cfg.layout = AGX_LAYOUT_TWIDDLED;
   cfg.channels = AGX_CHANNELS_R8G8B8A8;
   cfg.type = AGX_TEXTURE_TYPE_UNORM;
   cfg.swizzle_r = AGX_CHANNEL_R;
   cfg.swizzle_g = AGX_CHANNEL_G;
   cfg.swizzle_b = AGX_CHANNEL_B;
   cfg.swizzle_a = AGX_CHANNEL_A;
   cfg.width = 4;
   cfg.height = 8;
   cfg.buffer = UINT64_C(0x4000010000);
   return cfg;
}

/*
 * The AGX3 render-target store. Before this existed, agx_batch_upload_pbe()
 * returned false for every Gen9/Gen10 image and the end-of-tile program's
 * caller aborted the process through agx_native_descriptor_rejected(), so
 * native Gallium could not store a single pixel on G17P.
 */
static void
test_agx3_pbe(void)
{
   const struct AGX_PBE rt = render_target_pbe();
   struct agx_gallium_pbe_descriptor native;

   memset(&native, 0xa5, sizeof(native));
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &rt));

   /* The Gen9 descriptor is the shared byte-grounded codec's, not a copy of
    * the AGX2 packet: assert equality with it rather than restating bytes.
    */
   const struct agx_pbe_agx3_2d_fields fields = {
      .layout = rt.layout,
      .channels = rt.channels,
      .type = rt.type,
      .swizzle_r = rt.swizzle_r,
      .swizzle_g = rt.swizzle_g,
      .swizzle_b = rt.swizzle_b,
      .swizzle_a = rt.swizzle_a,
      .width = rt.width,
      .height = rt.height,
      .buffer = rt.buffer,
      .stride_B = 0,
      .page_aligned_layers = rt.page_aligned_layers,
   };
   struct agx_pbe_agx3_descriptor expected;
   CHECK(agx_pbe_agx3_2d_mip_pack(AGX_PBE_AGX3_GENERATION_9, &fields, 0, 1,
                                  &expected));
   CHECK(memcmp(native.opaque, expected.bytes, sizeof(expected.bytes)) == 0);

   /* Persisting an MSAA attachment uses the native multisample PBE codec,
    * retaining the logical extent and encoding the sample count separately. */
   for (unsigned samples = 2; samples <= 4; samples *= 2) {
      struct AGX_PBE msaa = rt;
      msaa.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
      msaa.samples = samples == 4 ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2;
      CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9,
                                         &native, &msaa));
      CHECK(agx_pbe_agx3_2d_multisample_pack(
         AGX_PBE_AGX3_GENERATION_9, &fields, samples, &expected));
      CHECK(memcmp(native.opaque, expected.bytes, sizeof(expected.bytes)) == 0);
      CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10,
                                         &native, &msaa));
      CHECK(agx_pbe_agx3_2d_multisample_pack(
         AGX_PBE_AGX3_GENERATION_10, &fields, samples, &expected));
      CHECK(memcmp(native.opaque, expected.bytes, sizeof(expected.bytes)) == 0);
      CHECK(agx_gallium_pack_g18_native_pbe(&native, &msaa));
      msaa.levels = 2;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9,
                                          &native, &msaa));
      msaa.levels = 1;
      msaa.samples = AGX_SAMPLE_COUNT_8;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9,
                                          &native, &msaa));
   }

   /* Native gl_fbo_readback's full-size render target. The width/height split
    * is pinned by the G17P PBE capture corpus, including independent 8192x8,
    * 100x60 and 40x24 witnesses; this checks the exact EOT descriptor the
    * production allocation must reach after compression is ruled out.
    */
   static const uint8_t fullscreen_bytes[32] = {
      0x22, 0x0a, 0xe4, 0x67, 0x49, 0x78, 0x01, 0x00,
      0x00, 0x10, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_PBE fullscreen = rt;
   fullscreen.layout = AGX_LAYOUT_GPU;
   fullscreen.width = 2408;
   fullscreen.height = 1506;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &fullscreen));
   CHECK(memcmp(native.opaque, fullscreen_bytes, sizeof(fullscreen_bytes)) ==
         0);

   fullscreen.compressed = true;
   memset(&native, 0xff, sizeof(native));
   CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                      &fullscreen));
   CHECK(native.opaque[0] == 0 && native.opaque[7] == 0);

   /* A single-level 2D image happens to share its first words with the AGX2
    * packet -- Gen9 keeps dimension, layout, channels, type, swizzles, width
    * and the shifted address where Gen7 had them. The encodings are not the
    * same thing, though: Gen9's mip extension has no AGX2 counterpart, and
    * it is what makes a mipmapped render target encodable at all.
    */
   struct agx_gallium_pbe_descriptor agx2;
   memset(&agx2, 0, sizeof(agx2));
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_7, &agx2, &rt));
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &rt));
   CHECK(memcmp(native.opaque, agx2.opaque, AGX_PBE_LENGTH) == 0);

   struct AGX_PBE mipped = rt;
   mipped.level = 1;
   mipped.levels = 4;
   struct agx_gallium_pbe_descriptor native_mip;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native_mip,
                                     &mipped));
   CHECK(memcmp(native_mip.opaque, native.opaque, sizeof(native.opaque)) != 0);

   /* Gen10 has no mip extension: the same mipped image is refused there, and
    * the single-level one still encodes. */
   CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10,
                                      &native_mip, &mipped));
   memset(&native, 0xa5, sizeof(native));
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &native,
                                     &rt));

   /* The null descriptor must encode on AGX3 too -- unbound storage images
    * write it, and a refusal there aborts the process.
    */
   memset(&native, 0xa5, sizeof(native));
   CHECK(agx_gallium_set_null_native_pbe(AGX_DESCRIPTOR_GENERATION_9,
                                         &native));
   CHECK(agx_gallium_set_null_native_pbe(AGX_DESCRIPTOR_GENERATION_10,
                                         &native));

   /* Everything outside the recovered subset stays refused, writing zeroes
    * rather than a half-encoded descriptor. Each of these is one axis away
    * from the accepted render target above.
    */
   static const char *const names[] = {
      "software sideband (image atomics)", "MSAA sideband", "layer stride",
      "compressed", "extended word", "multisampled", "1D",
      "sparse image mode",
   };
   struct AGX_PBE bad[ARRAY_SIZE(names)];
   for (unsigned i = 0; i < ARRAY_SIZE(names); ++i)
      bad[i] = rt;

   bad[0].level_offset_sw = 0x80;
   bad[1].aligned_width_msaa_sw = 64;
   bad[2].layer_stride_sw = 0x4000;
   bad[3].compressed = true;
   bad[4].extended = true;
   bad[5].samples = AGX_SAMPLE_COUNT_4;
   bad[6].dimension = AGX_TEXTURE_DIMENSION_1D;
   bad[7].mode = AGX_IMAGE_MODE_SPARSE;

   /* sRGB on store is Gen9-only, as in hk_pack_native_pbe(). */
   struct AGX_PBE srgb = rt;
   srgb.srgb = true;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &srgb));
   CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &native,
                                      &srgb));

   for (unsigned i = 0; i < ARRAY_SIZE(bad); ++i) {
      memset(&native, 0xff, sizeof(native));
      if (agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                      &bad[i])) {
         fprintf(stderr, "AGX3 PBE accepted %s\n", names[i]);
         exit(1);
      }
      ++checks;
      CHECK(native.opaque[0] == 0 && native.opaque[7] == 0);

      /* The same descriptor is still legal on AGX2, so these gates are
       * about the missing Gen9 placement, not about the image itself. */
      if (i < 3) {
         struct agx_gallium_pbe_descriptor still_ok;
         CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_7,
                                           &still_ok, &bad[i]));
      }
   }

   /* Match the exact shared/Honeykrisp Gen9 layered EOT descriptor. Array
    * storage is tiled, base-level-only and Gen9-only; every adjacent shape
    * stays closed. */
   struct AGX_PBE array = rt;
   array.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   array.layers = 2;
   array.page_aligned_layers = true;
   struct agx_pbe_agx3_descriptor expected_array;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &array));
   struct agx_pbe_agx3_2d_fields array_fields = fields;
   array_fields.page_aligned_layers = true;
   CHECK(agx_pbe_agx3_2d_array_pack(AGX_PBE_AGX3_GENERATION_9,
                                    &array_fields, 2, &expected_array));
   CHECK(memcmp(native.opaque, expected_array.bytes,
                sizeof(expected_array.bytes)) == 0);

   array.layout = AGX_LAYOUT_LINEAR;
   array.stride = (array.width * 4) - 4;
   CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                      &array));
   /* Gen10 adds the layered packets, base level only. */
   array = rt;
   array.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   array.layers = 2;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &native,
                                     &array));
   CHECK(agx_pbe_agx3_2d_array_mip_pack(AGX_PBE_AGX3_GENERATION_10, &fields,
                                        2, 0, 1, &expected_array));
   CHECK(memcmp(native.opaque, expected_array.bytes,
                sizeof(expected_array.bytes)) == 0);
   /* A mipmapped array level and a 3D level take the Gen9 mip packets
    * Honeykrisp stores storage images with. */
   array.level = 1;
   array.levels = 2;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &array));
   CHECK(agx_pbe_agx3_2d_array_mip_pack(AGX_PBE_AGX3_GENERATION_9, &fields,
                                        2, 1, 2, &expected_array));
   CHECK(memcmp(native.opaque, expected_array.bytes,
                sizeof(expected_array.bytes)) == 0);
   CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &native,
                                      &array));

   struct AGX_PBE volume = rt;
   volume.dimension = AGX_TEXTURE_DIMENSION_3D;
   volume.layers = 4;
   volume.levels = 3;
   CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                     &volume));
   CHECK(agx_pbe_agx3_3d_mip_pack(AGX_PBE_AGX3_GENERATION_9, &fields, 4, 0,
                                  3, &expected_array));
   CHECK(memcmp(native.opaque, expected_array.bytes,
                sizeof(expected_array.bytes)) == 0);
   CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10, &native,
                                      &volume));
}

/* Image atomics decode one packed addressing word from the G17P storage
 * tail; Gallium and Honeykrisp must write it identically. */
/* G17P's compressed colour render target as agx_batch_upload_pbe() builds it
 * for an end-of-tile store: GPU tiling, one level, one layer, the metadata
 * address in the extended word. These are the native bytes
 * test-hk-native-descriptor.c pins for Honeykrisp: a live 1024x1024 RGBA8
 * record and independent packets at display sizes.
 */
static void
test_g17p_compressed_render_target(void)
{
   static const struct {
      unsigned width, height;
      uint64_t buffer, metadata;
      uint64_t words[4];
   } targets[] = {
      {1024, 1024, UINT64_C(0x10000080000), UINT64_C(0x10000480000),
       {UINT64_C(0x0800ffc3ffe40a22), UINT64_C(0x8000001000008000),
        UINT64_C(0x0000001000048000), 0}},
      {1920, 1080, UINT64_C(0x10000000000), UINT64_C(0x100000a0000),
       {UINT64_C(0x08010dc77fe40a22), UINT64_C(0x8000001000000000),
        UINT64_C(0x000000100000a000), 0}},
      {2408, 1506, UINT64_C(0x10000000000), UINT64_C(0x100000a0000),
       {UINT64_C(0x0801784967e40a22), UINT64_C(0x8000001000000000),
        UINT64_C(0x000000100000a000), 0}},
      {2560, 1536, UINT64_C(0x10000000000), UINT64_C(0x100000a0000),
       {UINT64_C(0x08017fc9ffe40a22), UINT64_C(0x8000001000000000),
        UINT64_C(0x000000100000a000), 0}},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(targets); ++i) {
      struct AGX_PBE rt = render_target_pbe();
      rt.layout = AGX_LAYOUT_GPU;
      rt.width = targets[i].width;
      rt.height = targets[i].height;
      rt.buffer = targets[i].buffer;
      rt.levels = 1;
      rt.layers = 1;
      rt.compressed = true;
      rt.extended = true;
      rt.acceleration_buffer = targets[i].metadata;

      struct agx_gallium_pbe_descriptor native;
      memset(&native, 0xa5, sizeof(native));
      CHECK(agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                        &rt));
      CHECK(memcmp(native.opaque, targets[i].words,
                   sizeof(targets[i].words)) == 0);

      /* Outside the compressed single-layer 2D packet nothing is encoded. */
      const struct AGX_PBE original = rt;
      rt.acceleration_buffer = 0;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                         &rt));
      rt = original;
      rt.extended = false;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                         &rt));
      rt = original;
      rt.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
      rt.layers = 2;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                         &rt));
      rt = original;
      rt.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
      rt.samples = AGX_SAMPLE_COUNT_4;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                         &rt));
      rt = original;
      rt.layout = AGX_LAYOUT_TWIDDLED;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_9, &native,
                                         &rt));
      rt = original;
      CHECK(!agx_gallium_pack_native_pbe(AGX_DESCRIPTOR_GENERATION_10,
                                         &native, &rt));
   }
}

static void
test_g17p_image_atomic_sw(void)
{
   static const struct {
      uint64_t level_offset_B, layer_stride_B;
      unsigned tile_w, tile_h, samples;
   } cases[] = {
      {0, 0, 1, 1, 1},           {0x180, 0x4000, 64, 32, 4},
      {0x80, 0x80, 16, 16, 1},   {0x7fff80, 0x123480, 128, 64, 8},
      {1, 0, 64, 64, 1},         {0, 0, 48, 64, 1},
      {0, 0, 64, 64, 3},         {(uint64_t)1 << 34, 0, 1, 1, 1},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      uint64_t a = ~0ull, b = ~0ull;
      const bool ok_a = agx_g17p_pack_image_atomic_sw(
         cases[i].level_offset_B, cases[i].tile_w, cases[i].tile_h,
         cases[i].layer_stride_B, cases[i].samples, &a);
      const bool ok_b = hk_pack_g17p_image_atomic_sw(
         cases[i].level_offset_B, cases[i].tile_w, cases[i].tile_h,
         cases[i].layer_stride_B, cases[i].samples, &b);
      CHECK(ok_a == ok_b);
      CHECK(!ok_a || a == b);
   }

   /* The tail sits where libagx_image_texel_address_gen9() looks for it:
    * 40 bytes past the PBE, the second of three 32-byte records. */
   CHECK(AGX_G17P_IMAGE_ATOMIC_SW_OFFSET ==
         AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE + AGX_G17P_IMAGE_ATOMIC_SW_FROM_PBE);
   CHECK(AGX_G17P_EMRT_ALIGNED_WIDTH_OFFSET + 4 <=
         3 * AGX_GALLIUM_DESCRIPTOR_AGX3_SIZE);
}

static void
test_native_packers(void)
{
   struct AGX_TEXTURE semantic = {AGX_TEXTURE_header};
   semantic.dimension = AGX_TEXTURE_DIMENSION_2D;
   semantic.layout = AGX_LAYOUT_TWIDDLED;
   semantic.channels = AGX_CHANNELS_R8G8B8A8;
   semantic.type = AGX_TEXTURE_TYPE_UNORM;
   semantic.swizzle_r = AGX_CHANNEL_R;
   semantic.swizzle_g = AGX_CHANNEL_G;
   semantic.swizzle_b = AGX_CHANNEL_B;
   semantic.swizzle_a = AGX_CHANNEL_A;
   semantic.width = 256;
   semantic.height = 128;
   semantic.address = 0x1000008000ull;

   struct agx_gallium_texture_descriptor native;
   struct agx_texture_packed legacy = {0};
   AGX_TEXTURE_pack(legacy.opaque, &semantic);

   memset(&native, 0xa5, sizeof(native));
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_7, &native, &semantic));
   CHECK(memcmp(&native, &legacy, sizeof(legacy)) == 0);
   CHECK(native.opaque[6] == 0xa5a5a5a5 && native.opaque[7] == 0xa5a5a5a5);

   memset(&native, 0xa5, sizeof(native));
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_8, &native, &semantic));
   CHECK(memcmp(&native, &legacy, sizeof(legacy)) == 0);
   CHECK(native.opaque[6] == 0xa5a5a5a5 && native.opaque[7] == 0xa5a5a5a5);

   const struct AGX_TEXTURE null_legacy_semantic = {
      AGX_TEXTURE_header,
      .layout = AGX_LAYOUT_TWIDDLED,
      .channels = AGX_CHANNELS_R8,
      .type = AGX_TEXTURE_TYPE_UNORM,
      .swizzle_r = AGX_CHANNEL_0,
      .swizzle_g = AGX_CHANNEL_0,
      .swizzle_b = AGX_CHANNEL_0,
      .swizzle_a = AGX_CHANNEL_0,
      .address = AGX_ZERO_PAGE_ADDRESS,
   };
   struct agx_texture_packed expected_null_texture = {0};
   AGX_TEXTURE_pack(expected_null_texture.opaque, &null_legacy_semantic);

   for (enum agx_descriptor_generation generation =
           AGX_DESCRIPTOR_GENERATION_7;
        generation <= AGX_DESCRIPTOR_GENERATION_8; ++generation) {
      memset(&native, 0xa5, sizeof(native));
      CHECK(agx_gallium_set_null_native_texture(generation, &native));
      CHECK(memcmp(&native, &expected_null_texture,
                   sizeof(expected_null_texture)) == 0);
      CHECK(native.opaque[6] == 0xa5a5a5a5 &&
            native.opaque[7] == 0xa5a5a5a5);
   }

   memset(&native, 0xa5, sizeof(native));
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &semantic));
   CHECK(native.opaque[6] == 0 && native.opaque[7] == 0);

   /* HAL200's exact 2x2 RGBA8 descriptor clears word1[62] for an ordinary
    * one-level 2D texture.  AIL's page-aligned backing-allocation flag is not
    * that native layer-mode bit, so both allocation choices must produce the
    * same packet here.  This is the ubiquitous compositor/WebGL sampling
    * shape, not an array-layer policy.
    */
   static const uint8_t gen9_plain_2d_bytes[32] = {
      0x22, 0x0a, 0x88, 0x16, 0x00, 0x04, 0x00, 0x00,
      0x00, 0xfc, 0xfe, 0x7f, 0x08, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_TEXTURE plain_2d = semantic;
   plain_2d.layout = AGX_LAYOUT_GPU;
   plain_2d.width = 2;
   plain_2d.height = 2;
   plain_2d.address = UINT64_C(0x87ffefc000);
   struct agx_gallium_texture_descriptor plain_unaligned, plain_aligned;
   plain_2d.page_aligned_layers = false;
   CHECK(agx_gallium_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9,
                                          &plain_unaligned, &plain_2d));
   plain_2d.page_aligned_layers = true;
   CHECK(agx_gallium_pack_native_texture(AGX_DESCRIPTOR_GENERATION_9,
                                          &plain_aligned, &plain_2d));
   CHECK(memcmp(&plain_aligned, gen9_plain_2d_bytes,
                sizeof(gen9_plain_2d_bytes)) == 0);
   CHECK(memcmp(&plain_aligned, &plain_unaligned, sizeof(plain_aligned)) == 0);

   /* A18 Pro EXP-0015 type_cube has compression enabled in bit 59.
    * Preserve its captured bytes and represent that compression state;
    * uncompressed cube views are checked separately below. */
   static const uint8_t gen9_cube_bytes[32] = {
      0x26, 0x0a, 0x88, 0xf6, 0x00, 0x3c, 0x00, 0x08,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0xd0, 0x1d, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_TEXTURE cube = semantic;
   cube.dimension = AGX_TEXTURE_DIMENSION_CUBE;
   cube.layout = AGX_LAYOUT_GPU;
   cube.width = 16;
   cube.height = 16;
   cube.address = UINT64_C(0x1000001c500);
   cube.extended = true;
   cube.compressed = true;
   cube.acceleration_buffer = UINT64_C(0x1000001dd00);
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &cube));
   CHECK(memcmp(native.opaque, gen9_cube_bytes, sizeof(gen9_cube_bytes)) == 0);

   /* A18 Pro EXP-0015 arr_2 also has compression enabled. Its secondary VA
    * follows the two 0x400-byte layers in this captured allocation. */
   static const uint8_t gen9_array2_bytes[32] = {
      0x23, 0x0a, 0x88, 0xf6, 0x00, 0x3c, 0x00, 0x08,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x40, 0x00, 0x80,
      0xd0, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_TEXTURE array2 = semantic;
   array2.dimension = AGX_TEXTURE_DIMENSION_2D_ARRAY;
   array2.layout = AGX_LAYOUT_GPU;
   array2.width = 16;
   array2.height = 16;
   array2.depth = 2;
   array2.address = UINT64_C(0x1000001c500);
   array2.extended = true;
   array2.compressed = true;
   array2.acceleration_buffer = UINT64_C(0x1000001cd00);
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &array2));
   CHECK(memcmp(native.opaque, gen9_array2_bytes,
                sizeof(gen9_array2_bytes)) == 0);

   /* The correction is deliberately plain-2D-only.  A valid extended array
    * still preserves the inherited layer policy, changing exactly bit 126.
    */
   struct agx_gallium_texture_descriptor array2_aligned;
   array2.page_aligned_layers = true;
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &array2_aligned, &array2));
   array2_aligned.opaque[3] ^= BITFIELD_BIT(30);
   CHECK(memcmp(&array2_aligned, &native, sizeof(array2_aligned)) == 0);

   /* A18 Pro EXP-0015 mip_2 has both the mip-level and compression bits set.
    * Keep the captured bytes unchanged and represent its compression state;
    * uncompressed mip views are checked separately below. */
   static const uint8_t gen9_mip2_bytes[32] = {
      0x22, 0x0a, 0x88, 0xf6, 0x00, 0x3c, 0x00, 0x0c,
      0x50, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x00, 0x80,
      0xb8, 0x1c, 0x00, 0x00, 0x10, 0x00, 0x01, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
   };
   struct AGX_TEXTURE mip2 = semantic;
   mip2.layout = AGX_LAYOUT_GPU;
   mip2.width = 16;
   mip2.height = 16;
   mip2.address = UINT64_C(0x1000001c500);
   mip2.last_level = 1;
   mip2.compressed = true;
   mip2.acceleration_buffer = UINT64_C(0x1000001cb80);
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &mip2));
   CHECK(memcmp(native.opaque, gen9_mip2_bytes,
                sizeof(gen9_mip2_bytes)) == 0);

   mip2.acceleration_buffer = 0;
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &mip2));

   /* Gallium can sample only level zero from a render target whose backing
    * allocation still contains a complete miptree.  unk_mipmapped makes that
    * base-only view extended too, so it requires the allocation's end VA. */
   struct AGX_TEXTURE mipmapped_base = semantic;
   mipmapped_base.layout = AGX_LAYOUT_GPU;
   mipmapped_base.unk_mipmapped = true;
   mipmapped_base.acceleration_buffer = UINT64_C(0x100000c000);
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &mipmapped_base));
   mipmapped_base.acceleration_buffer = 0;
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &mipmapped_base));

   cube.extended = false;
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &cube));
   cube.extended = true;
   cube.acceleration_buffer = 0;
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &cube));

   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &native, &semantic));
   CHECK(native.opaque[6] == 0 && native.opaque[7] == 0);

   struct agx_gallium_texture_descriptor null9 = {0}, null9_dirty;
   struct agx_gallium_texture_descriptor null10 = {0}, null10_dirty;
   struct AGX_TEXTURE_GEN9 null9_unpacked = {0};
   struct AGX_TEXTURE_GEN10 null10_unpacked = {0};

   CHECK(agx_gallium_set_null_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &null9));
   memset(&null9_dirty, 0xa5, sizeof(null9_dirty));
   CHECK(agx_gallium_set_null_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &null9_dirty));
   CHECK(memcmp(&null9, &null9_dirty, sizeof(null9)) == 0);
   CHECK(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)null9.opaque,
                                 &null9_unpacked));
   CHECK(null9_unpacked.width == 1 && null9_unpacked.height == 1 &&
         null9_unpacked.depth == 1);
   CHECK(null9_unpacked.address == AGX_ZERO_PAGE_ADDRESS);
   CHECK(null9.opaque[6] == 0 && null9.opaque[7] == 0);

   CHECK(agx_gallium_set_null_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &null10));
   memset(&null10_dirty, 0xa5, sizeof(null10_dirty));
   CHECK(agx_gallium_set_null_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &null10_dirty));
   CHECK(memcmp(&null10, &null10_dirty, sizeof(null10)) == 0);
   CHECK(AGX_TEXTURE_GEN10_unpack(NULL, (const uint8_t *)null10.opaque,
                                  &null10_unpacked));
   CHECK(null10_unpacked.width == 1 && null10_unpacked.height == 1 &&
         null10_unpacked.depth == 1);
   CHECK(null10_unpacked.address == AGX_ZERO_PAGE_ADDRESS);
   CHECK(null10.opaque[6] == 0 && null10.opaque[7] == 0);

   struct AGX_TEXTURE unsupported = semantic;
   unsupported.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   static const struct {
      enum agx_sample_count samples;
      unsigned selector;
   } gen9_msaa[] = {
      {AGX_SAMPLE_COUNT_2, 0},
      {AGX_SAMPLE_COUNT_4, 1},
      {AGX_SAMPLE_COUNT_8, 2},
   };

   for (unsigned i = 0; i < sizeof(gen9_msaa) / sizeof(gen9_msaa[0]); ++i) {
      unsupported.samples = gen9_msaa[i].samples;
      CHECK(agx_gallium_pack_native_texture(
         AGX_DESCRIPTOR_GENERATION_9, &native, &unsupported));

      uint64_t gen9_msaa_word0;
      memcpy(&gen9_msaa_word0, native.opaque, sizeof(gen9_msaa_word0));
      CHECK(((gen9_msaa_word0 >> 56) & 3) == gen9_msaa[i].selector);

      struct AGX_TEXTURE_GEN9 unpacked = {0};
      CHECK(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)native.opaque,
                                    &unpacked));
      CHECK(unpacked.samples == gen9_msaa[i].samples);
   }

   unsupported.samples = (enum agx_sample_count)3;
   memset(&native, 0xff, sizeof(native));
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &unsupported));
   CHECK(memcmp(&native, &(struct agx_gallium_texture_descriptor){0},
                sizeof(native)) == 0);

   unsupported = semantic;
   unsupported.samples = AGX_SAMPLE_COUNT_4;
   memset(&native, 0xff, sizeof(native));
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_9, &native, &unsupported));
   CHECK(memcmp(&native, &(struct agx_gallium_texture_descriptor){0},
                sizeof(native)) == 0);

   unsupported = semantic;
   unsupported.dimension = AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED;
   unsupported.samples = AGX_SAMPLE_COUNT_8;
   CHECK(agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &native, &unsupported));
   uint64_t m5_msaa_word0;
   memcpy(&m5_msaa_word0, native.opaque, sizeof(m5_msaa_word0));
   CHECK(((m5_msaa_word0 >> 58) & 3) == AGX_SAMPLE_COUNT_8);

   unsupported.samples = (enum agx_sample_count)3;
   memset(&native, 0xff, sizeof(native));
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &native, &unsupported));
   CHECK(native.opaque[0] == 0 && native.opaque[7] == 0);

   unsupported = semantic;
   unsupported.samples = AGX_SAMPLE_COUNT_8;
   memset(&native, 0xff, sizeof(native));
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &native, &unsupported));
   CHECK(memcmp(&native, &(struct agx_gallium_texture_descriptor){0},
                sizeof(native)) == 0);

   unsupported = semantic;
   unsupported.buffer_offset_sw = 1;
   memset(&native, 0xff, sizeof(native));
   CHECK(!agx_gallium_pack_native_texture(
      AGX_DESCRIPTOR_GENERATION_10, &native, &unsupported));
   CHECK(native.opaque[0] == 0 && native.opaque[7] == 0);

   /* AGX2 still packs the legacy 24-byte packet verbatim, and only those 24
    * bytes: the AGX3 stride's tail is untouched. */
   const struct AGX_PBE pbe_semantic = render_target_pbe();
   struct agx_pbe_packed pbe;
   AGX_PBE_pack(pbe.opaque, &pbe_semantic);

   struct agx_gallium_pbe_descriptor native_pbe;
   memset(&native_pbe, 0xa5, sizeof(native_pbe));
   CHECK(agx_gallium_pack_native_pbe(
      AGX_DESCRIPTOR_GENERATION_8, &native_pbe, &pbe_semantic));
   CHECK(memcmp(&native_pbe, &pbe, sizeof(pbe)) == 0);
   CHECK(native_pbe.opaque[6] == 0xa5a5a5a5 &&
         native_pbe.opaque[7] == 0xa5a5a5a5);

   /* The AGX2 null PBE's bytes are unchanged by the AGX3 work: same 1D
    * dimension, same everything. */
   struct agx_pbe_packed expected_null_pbe;
   agx_pack(&expected_null_pbe, PBE, cfg) {
      cfg.width = 1;
      cfg.height = 1;
      cfg.levels = 1;
      cfg.layout = AGX_LAYOUT_TWIDDLED;
      cfg.channels = AGX_CHANNELS_R8;
      cfg.type = AGX_TEXTURE_TYPE_UNORM;
      cfg.swizzle_r = AGX_CHANNEL_R;
      cfg.swizzle_g = AGX_CHANNEL_R;
      cfg.swizzle_b = AGX_CHANNEL_R;
      cfg.swizzle_a = AGX_CHANNEL_R;
      cfg.buffer = AGX_SCRATCH_PAGE_ADDRESS;
   }

   memset(&native_pbe, 0xa5, sizeof(native_pbe));
   CHECK(agx_gallium_set_null_native_pbe(
      AGX_DESCRIPTOR_GENERATION_7, &native_pbe));
   CHECK(memcmp(&native_pbe, &expected_null_pbe, sizeof(expected_null_pbe)) ==
         0);
   CHECK(native_pbe.opaque[6] == 0xa5a5a5a5 &&
         native_pbe.opaque[7] == 0xa5a5a5a5);

   /* A refused AGX3 descriptor leaves the whole native stride zeroed rather
    * than a partially-written one. The render target above carries the
    * software addressing sideband here, which Gen10 has no home for. */
   struct AGX_PBE sideband = pbe_semantic;
   sideband.level_offset_sw = 0x80;
   memset(&native_pbe, 0xff, sizeof(native_pbe));
   CHECK(!agx_gallium_pack_native_pbe(
      AGX_DESCRIPTOR_GENERATION_10, &native_pbe, &sideband));
   CHECK(native_pbe.opaque[0] == 0 && native_pbe.opaque[7] == 0);

   /* An unknown generation has no descriptor at all: nothing is written and
    * nothing is claimed. */
   CHECK(!agx_gallium_pack_native_pbe(
      AGX_DESCRIPTOR_GENERATION_UNKNOWN, &native_pbe, &pbe_semantic));
}

static void
test_gen9_uncompressed_mip_views(void)
{
   /* An uncompressed miptree must not enable compression metadata reads.
    * Include the base-only render-target reload view used by GL CTS. */
   const unsigned first[] = {0, 0, 2};
   const unsigned last[] = {0, 5, 5};
   for (unsigned i = 0; i < ARRAY_SIZE(first); ++i) {
      struct AGX_TEXTURE semantic = {AGX_TEXTURE_header};
      semantic.dimension = AGX_TEXTURE_DIMENSION_2D;
      semantic.layout = AGX_LAYOUT_GPU;
      semantic.width = 49;
      semantic.height = 23;
      semantic.depth = 1;
      semantic.address = UINT64_C(0x5fffe4c000);
      semantic.acceleration_buffer = UINT64_C(0x5fffe50000);
      semantic.unk_mipmapped = true;
      semantic.first_level = first[i];
      semantic.last_level = last[i];
      semantic.page_aligned_layers = true;
      struct agx_gallium_texture_descriptor native;
      CHECK(agx_gallium_pack_native_texture(
         AGX_DESCRIPTOR_GENERATION_9, &native, &semantic));
      struct AGX_TEXTURE_GEN9 decoded;
      CHECK(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)native.opaque,
                                     &decoded));
      CHECK(!decoded.compressed);
      CHECK(decoded.has_levels && decoded.extended);
      CHECK(decoded.first_level == first[i] && decoded.last_level == last[i]);
   }
}

static void
test_gen9_uncompressed_dimensions(void)
{
   const enum agx_texture_dimension dimensions[] = {
      AGX_TEXTURE_DIMENSION_2D_ARRAY,
      AGX_TEXTURE_DIMENSION_3D,
      AGX_TEXTURE_DIMENSION_CUBE,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(dimensions); ++i) {
      struct AGX_TEXTURE semantic = {AGX_TEXTURE_header};
      semantic.dimension = dimensions[i];
      semantic.layout = AGX_LAYOUT_GPU;
      semantic.channels = AGX_CHANNELS_R16;
      semantic.type = AGX_TEXTURE_TYPE_UNORM;
      semantic.width = semantic.height = 128;
      semantic.depth = dimensions[i] == AGX_TEXTURE_DIMENSION_CUBE ? 1 : 4;
      semantic.address = UINT64_C(0x5ffff20000);
      semantic.acceleration_buffer = UINT64_C(0x5ffff40000);
      semantic.extended = true;
      semantic.page_aligned_layers = true;
      struct agx_gallium_texture_descriptor native;
      CHECK(agx_gallium_pack_native_texture(
         AGX_DESCRIPTOR_GENERATION_9, &native, &semantic));
      struct AGX_TEXTURE_GEN9 decoded;
      CHECK(AGX_TEXTURE_GEN9_unpack(NULL, (const uint8_t *)native.opaque,
                                     &decoded));
      CHECK(!decoded.compressed);
      CHECK(decoded.dimension == dimensions[i] && decoded.extended);
      CHECK(decoded.depth == semantic.depth);
   }
}

int
main(void)
{
   test_chip_routing();
   test_feature_gates();
   test_g17p_txf_sampler();
   test_native_sampler_generations();
   test_g17p_sampler_handles();
   test_standard_border_colours();
   test_gen9_uncompressed_mip_views();
   test_gen9_uncompressed_dimensions();
   test_native_packers();
   test_agx3_pbe();
   test_g17p_compressed_render_target();
   test_g17p_image_atomic_sw();
   printf("Gallium AGX native descriptors: %u checks passed\n", checks);
   return 0;
}
