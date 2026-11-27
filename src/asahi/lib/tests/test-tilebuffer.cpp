/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "util/format/u_format.h"
#include "agx_tilebuffer.h"
#include "agx_bg_eot_key.h"
#include "agx_g17p_compression.h"
#include "layout/layout.h"

#include <vector>
#include <gtest/gtest.h>

extern "C" {
#include "asahi/compiler/agx_compile.h"
#include "compiler/nir/nir_builder.h"
}

struct test {
   const char *name;
   uint8_t nr_samples;
   enum pipe_format formats[8];
   struct agx_tilebuffer_layout layout;
   uint32_t total_size;
};

/* clang-format off */
struct test tests[] = {
   {
      "Simple test",
      1,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "MSAA 2x",
      2,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 2,
         .tile_size = 32 * 32,
      },
      16384
   },
   {
      "MSAA 4x",
      4,
      { PIPE_FORMAT_R8G8B8A8_UNORM },
      {
         ._offset_B = { 0 },
         .sample_size_B = 8,
         .nr_samples = 4,
         .tile_size = 32 * 16,
      },
      16384
   },
   {
      "MRT",
      1,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      24576
   },
   {
      "MRT with MSAA 2x",
      2,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 2,
         .tile_size = 32 * 16,
      },
      24576
   },
   {
      "MRT with MSAA 4x",
      4,
      {
         PIPE_FORMAT_R16_SINT,
         PIPE_FORMAT_R32G32_FLOAT,
         PIPE_FORMAT_R8_SINT,
         PIPE_FORMAT_R32G32_SINT,
      },
      {
         ._offset_B = { 16, 0, 18, 8 },
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size = 16 * 16,
      },
      24576
   },
   {
      "MRT test requiring 2 alignment on the second RT",
      1,
      { PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R16G16_SNORM },
      {
         ._offset_B = { 4, 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "Simple MRT test requiring 4 alignment on the second RT",
      1,
      { PIPE_FORMAT_R8_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM },
      {
         ._offset_B = { 4, 0 },
         .sample_size_B = 8,
         .nr_samples = 1,
         .tile_size = 32 * 32,
      },
      8192
   },
   {
      "MRT test that requires spilling to consider alignment requirements",
      4,
      {
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
         PIPE_FORMAT_R32_FLOAT,
      },
      {
         .spilled = { false, false, false, false, false, false, true, true },
         ._offset_B = { 0, 4, 8, 12, 16, 20, 0, 0},
         .sample_size_B = 24,
         .nr_samples = 4,
         .tile_size = 16 * 16,
      },
      24576
   },

};
/* clang-format on */

TEST(Tilebuffer, Layouts)
{
   for (unsigned i = 0; i < ARRAY_SIZE(tests); ++i) {
      unsigned nr_cbufs;

      for (nr_cbufs = 0; nr_cbufs < ARRAY_SIZE(tests[i].formats) &&
                         tests[i].formats[nr_cbufs] != PIPE_FORMAT_NONE;
           ++nr_cbufs)
         ;

      struct agx_tilebuffer_layout actual = agx_build_tilebuffer_layout(
         tests[i].formats, nr_cbufs, tests[i].nr_samples, false);

      ASSERT_EQ(tests[i].layout.sample_size_B, actual.sample_size_B)
         << tests[i].name;
      ASSERT_EQ(tests[i].layout.nr_samples, actual.nr_samples) << tests[i].name;
      ASSERT_EQ(tests[i].layout.tile_size, actual.tile_size) << tests[i].name;
      ASSERT_EQ(tests[i].total_size, agx_tilebuffer_total_size(&tests[i].layout))
         << tests[i].name;

      for (unsigned j = 0; j < 8; ++j) {
         ASSERT_EQ(tests[i].layout._offset_B[j], actual._offset_B[j])
            << tests[i].name << ", render target " << j;
      }
   }
}

TEST(Tilebuffer, G17PUnorm8MultisampleLayouts)
{
   /* The allocator keeps a minimal MSAA slot even without colour output. */
   const enum pipe_format none = PIPE_FORMAT_NONE;
   for (unsigned samples : {2u, 4u}) {
      auto tib = agx_build_tilebuffer_layout(&none, 0, samples, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      EXPECT_EQ(tib.sample_size_B, 8u);
      tib.sample_size_B = 16;
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8G8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   for (enum pipe_format format : formats) {
      for (unsigned samples : {2u, 4u}) {
         auto tib = agx_build_tilebuffer_layout(&format, 1, samples, false);
         EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
         EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), PIPE_FORMAT_R8_UNORM);
         EXPECT_EQ(tib.sample_size_B, 8u);
         EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0u);
         EXPECT_LT(agx_tilebuffer_total_size(&tib), 32768u);
      }
   }
}

TEST(Tilebuffer, G17PDualRenderTargets)
{
   enum pipe_format wboit[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(wboit, 2, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_WBOIT);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0u);
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 8u);

   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba8, 2, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_RGBA8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 4u);

   enum pipe_format rgba8_aliases[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8X8_UNORM,
      PIPE_FORMAT_B8G8R8A8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(rgba8_aliases, 2, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_RGBA8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 4u);

   /* Multisample layouts take the mixed-format programs. */
   tib = agx_build_tilebuffer_layout(wboit, 2, 2, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   tib._offset_B[1] = 4;
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);

   tib = agx_build_tilebuffer_layout(wboit, 2, 1, false);
   tib.spilled[1] = true;
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   /* A third target of another format uses the mixed-format layout. */
   enum pipe_format three_targets[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8_UNORM,
      PIPE_FORMAT_R8_UNORM,
   };
   tib = agx_build_tilebuffer_layout(three_targets, 3, 1, false);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   EXPECT_EQ(agx_g17p_classify_mrt(nullptr), AGX_G17P_MRT_NONE);
}

TEST(Tilebuffer, G17PCompactsSparseRGBA8Attachments)
{
   enum pipe_format sparse[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_NONE,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_NONE,
      PIPE_FORMAT_B8G8R8A8_UNORM,
   };
   const struct agx_tilebuffer_layout source =
      agx_build_tilebuffer_layout(sparse, 4, 1, false);
   struct agx_tilebuffer_layout compact;
   uint8_t map[AGX_MAX_RENDER_TARGETS] = {0};
   ASSERT_EQ(agx_tilebuffer_compact(&source, &compact, map), 2u);
   EXPECT_EQ(map[0], 1u);
   EXPECT_EQ(map[1], 3u);
   EXPECT_EQ(compact.logical_format[0], PIPE_FORMAT_R8G8B8A8_UNORM);
   EXPECT_EQ(compact.logical_format[1], PIPE_FORMAT_B8G8R8A8_UNORM);
   EXPECT_EQ(compact._offset_B[0], 0u);
   EXPECT_EQ(compact._offset_B[1], 4u);
   EXPECT_EQ(agx_g17p_classify_mrt(&source), AGX_G17P_MRT_RGBA8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&source));

   for (unsigned count : {3u, 4u}) {
      enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
         PIPE_FORMAT_R8G8B8A8_UNORM,
      };
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rgba8, count, 1, false);
      EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_RGBA8);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   enum pipe_format none[AGX_MAX_RENDER_TARGETS] = {PIPE_FORMAT_NONE};
   struct agx_tilebuffer_layout attachmentless =
      agx_build_tilebuffer_layout(none, 1, 0, false);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&attachmentless));
   EXPECT_EQ(attachmentless.nr_samples, 0u);
}

TEST(Tilebuffer, G17PNarrowSRGB)
{
   for (enum pipe_format format : {PIPE_FORMAT_R8_SRGB,
                                   PIPE_FORMAT_R8G8_SRGB}) {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {format};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, 1, 1, false);

      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                PIPE_FORMAT_R8G8B8A8_SRGB);
      EXPECT_EQ(agx_g17p_rt0_pbe_format(format), util_format_linear(format));

      tib.spilled[0] = true;
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

      tib = agx_build_tilebuffer_layout(formats, 1, 1, true);
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

      /* These narrow sRGB targets share the R8G8B8A8_SRGB physical tile, and
       * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_color_config
       * exercises that tile at four samples through GL_SRGB8_ALPHA8. */
      for (unsigned samples : {2u, 4u}) {
         tib = agx_build_tilebuffer_layout(formats, 1, samples, false);
         EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      }

      formats[1] = format;
      tib = agx_build_tilebuffer_layout(formats, 2, 1, false);
      EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   }
}

TEST(Tilebuffer, G17PAllRGBA8AttachmentMasksUseCompactStorage)
{
   for (unsigned mask = 1; mask < 256; ++mask) {
      SCOPED_TRACE(mask);
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {};
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
         if (mask & (1u << rt))
            formats[rt] = PIPE_FORMAT_R8G8B8A8_UNORM;
      }
      auto tib = agx_build_tilebuffer_layout(formats, 8, 1, false);
      EXPECT_EQ(agx_g17p_rgba8_mrt_mask(&tib), mask);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      unsigned active = 0, last = 0;
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt) {
         if (!(mask & (1u << rt)))
            continue;
         EXPECT_EQ(agx_tilebuffer_offset_B(&tib, rt), 4 * active++);
         EXPECT_LT(agx_tilebuffer_offset_B(&tib, rt), tib.sample_size_B);
         last = rt;
      }
      tib._offset_B[last] += 4;
      EXPECT_EQ(agx_g17p_rgba8_mrt_mask(&tib), 0u);
   }
   EXPECT_EQ(agx_g17p_rgba8_mrt_mask(nullptr), 0u);
}

TEST(Tilebuffer, G17PDenseWideRGBA8RenderTargets)
{
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
   for (auto &format : formats)
      format = PIPE_FORMAT_R8G8B8A8_UNORM;

   for (unsigned count = 3; count <= AGX_MAX_RENDER_TARGETS; ++count) {
      auto tib = agx_build_tilebuffer_layout(formats, count, 1, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      EXPECT_EQ(tib.sample_size_B, ((4 * count + 7) & ~7u));
      EXPECT_EQ(tib.tile_size, count >= 7 ? 32 * 16 : 32 * 32);
      for (unsigned rt = 0; rt < count; ++rt)
         EXPECT_EQ(agx_tilebuffer_offset_B(&tib, rt), 4 * rt);
   }
}

TEST(Tilebuffer, G17PRGB565EOTKey)
{
   struct agx_shader_key key = {};

   /* Seed an unrelated format first: configuring a new EOT format must clear
    * the I32 admission rather than inheriting a stale cache key. */
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R32G32_UINT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 2);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);

   /* RGB565 keeps RGBA16F's two 16-bit float words and its selector. */
   for (enum pipe_format format : {PIPE_FORMAT_R5G6B5_UNORM,
                                   PIPE_FORMAT_B5G6R5_UNORM,
                                   PIPE_FORMAT_R16G16B16A16_FLOAT}) {
      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
      EXPECT_EQ(agx_g17p_mrt_eot_selector(format), 0x15);
   }

   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R8G8B8A8_UNORM, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 0);
}

TEST(Tilebuffer, G17PPacked16EOTKey)
{
   struct agx_shader_key key = {};
   const enum pipe_format packed[] = {
      /* Vulkan R5G5B5A1/B5G5R5A1 use these low-alpha Gallium forms. */
      PIPE_FORMAT_A1B5G5R5_UNORM,
      PIPE_FORMAT_A1R5G5B5_UNORM,
      PIPE_FORMAT_R5G5B5A1_UNORM,
      PIPE_FORMAT_B5G5R5A1_UNORM,
      PIPE_FORMAT_R5G5B5X1_UNORM,
      PIPE_FORMAT_B5G5R5X1_UNORM,
      PIPE_FORMAT_R4G4B4A4_UNORM,
      PIPE_FORMAT_B4G4R4A4_UNORM,
      PIPE_FORMAT_A4B4G4R4_UNORM,
      PIPE_FORMAT_A4R4G4B4_UNORM,
   };

   for (enum pipe_format format : packed) {
      enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {format};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(formats, 1, 1, false);

      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                PIPE_FORMAT_R16_FLOAT);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
   }
}

TEST(Tilebuffer, G17PSpecialFormatEOTKey)
{
   static const struct {
      enum pipe_format logical;
      enum pipe_format physical;
   } formats[] = {
      {PIPE_FORMAT_R16G16B16X16_FLOAT, PIPE_FORMAT_R16_FLOAT},
      {PIPE_FORMAT_R16G16B16X16_UINT, PIPE_FORMAT_R16_UINT},
      {PIPE_FORMAT_R16G16B16X16_SINT, PIPE_FORMAT_R16_UINT},
   };

   for (const auto &format : formats) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format.logical};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rt, 1, 1, false);
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), format.physical);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_shader_key key = {};
      agx_g17p_configure_rt0_eot_key(format.logical, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 2);
      EXPECT_EQ(agx_g17p_mrt_eot_selector(format.logical), 0x15);

      /* Multisample admission follows the physical tilebuffer format, and
       * R16_FLOAT is one of the families the four-sample hardware sweep
       * covers. R16_UINT is not. */
      tib = agx_build_tilebuffer_layout(rt, 1, 2, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   enum pipe_format packed_float[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R11G11B10_FLOAT,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(packed_float, 1, 1, false);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
             PIPE_FORMAT_R11G11B10_FLOAT);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   struct agx_shader_key key = {};
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R11G11B10_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 1);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R11G11B10_FLOAT),
             PIPE_FORMAT_R32_UINT);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R8G8B8A8_SRGB),
             PIPE_FORMAT_R8G8B8A8_UNORM);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R10G10B10A2_UNORM),
             PIPE_FORMAT_R10G10B10A2_UNORM);

   /* The packed float tile is eight bytes per sample at any sample count, and
    * KHR-GL30.framebuffer_blit.multisampled_to_singlesampled_blit_color_config
    * exercises it at four samples through GL_R11F_G11F_B10F. Layering is a
    * separate contract and stays refused. */
   tib = agx_build_tilebuffer_layout(packed_float, 1, 2, false);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   tib = agx_build_tilebuffer_layout(packed_float, 1, 1, true);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   /* Beside other targets at four samples it takes the mixed multi-target
    * programs with the W32 selector, as RGB9E5 does (Transport Fever 3). */
   enum pipe_format packed_mrt[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R11G11B10_FLOAT,
   };
   tib = agx_build_tilebuffer_layout_with_limits(
      packed_mrt, 2, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_EQ(agx_g17p_classify_mrt(&tib), AGX_G17P_MRT_MIXED);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

   /* RGB9E5 is packed natively into the same raw tile word. */
   enum pipe_format rgb9e5[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R9G9B9E5_FLOAT,
   };
   tib = agx_build_tilebuffer_layout(rgb9e5, 1, 1, false);
   EXPECT_EQ(tib.sample_size_B, 8);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   key = {};
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R9G9B9E5_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 1);
   EXPECT_TRUE(key.g17p_rt0_eot_i32);
   EXPECT_EQ(agx_g17p_rt0_pbe_format(PIPE_FORMAT_R9G9B9E5_FLOAT),
             PIPE_FORMAT_R32_UINT);
   tib = agx_build_tilebuffer_layout(rgb9e5, 1, 4, false);
   EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
   tib = agx_build_tilebuffer_layout(rgb9e5, 1, 1, true);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));
}

TEST(Tilebuffer, G17PRGBX8IntegerEOTKey)
{
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8G8B8X8_UINT,
      PIPE_FORMAT_R8G8B8A8_UINT,
      PIPE_FORMAT_B8G8R8X8_UINT,
      PIPE_FORMAT_B8G8R8A8_UINT,
      PIPE_FORMAT_R8G8B8X8_SINT,
      PIPE_FORMAT_R8G8B8A8_SINT,
      PIPE_FORMAT_B8G8R8X8_SINT,
      PIPE_FORMAT_B8G8R8A8_SINT,
   };

   for (enum pipe_format format : formats) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rt, 1, 1, false);
      EXPECT_EQ(tib.sample_size_B, 8);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                PIPE_FORMAT_R8_UINT);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_shader_key key = {};
      agx_g17p_configure_rt0_eot_key(format, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 1);
      EXPECT_FALSE(key.g17p_rt0_eot_i32);
   }
}

TEST(Tilebuffer, G17PRGBX8Multisample)
{
   for (enum pipe_format format : {PIPE_FORMAT_R8G8B8X8_UNORM,
                                   PIPE_FORMAT_B8G8R8X8_UNORM}) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format};
      for (unsigned samples : {2u, 4u}) {
         struct agx_tilebuffer_layout tib =
            agx_build_tilebuffer_layout(rt, 1, samples, false);
         EXPECT_EQ(tib.sample_size_B, 8);
         EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0),
                   PIPE_FORMAT_R8_UNORM);
         EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      }
   }
}

TEST(Tilebuffer, G17PRGB32EOTKey)
{
   static const struct {
      enum pipe_format logical;
      bool signed_format;
   } formats[] = {
      {PIPE_FORMAT_R32G32B32X32_FLOAT, false},
      {PIPE_FORMAT_R32G32B32X32_UINT, false},
      {PIPE_FORMAT_R32G32B32X32_SINT, true},
   };

   for (const auto &format : formats) {
      enum pipe_format rt[AGX_MAX_RENDER_TARGETS] = {format.logical};
      struct agx_tilebuffer_layout tib =
         agx_build_tilebuffer_layout(rt, 1, 1, false);
      EXPECT_EQ(tib.sample_size_B, 16);
      EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), PIPE_FORMAT_R32_UINT);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));

      struct agx_shader_key key = {};
      agx_g17p_configure_rt0_eot_key(format.logical, &key);
      EXPECT_EQ(key.g17p_rt0_eot_words, 4);
      EXPECT_TRUE(key.g17p_rt0_eot_i32);
      EXPECT_EQ(key.g17p_rt0_eot_i32_four_words, !format.signed_format);
      EXPECT_EQ(key.g17p_rt0_eot_i32_signed_four_words,
                format.signed_format);

      /* Apple stores the 32-bit tile at every sample count with the same
       * lstcol it uses at one: metal-nt -arch applegpu_g17p on one
       * RGBA32Float attachment fed by a multisample read emits
       * e71654000000012e00000000 at one sample and e71654000000082e08000008
       * at four. KHR-GL33.texture_size_promotion renders to it. */
      tib = agx_build_tilebuffer_layout(rt, 1, 2, false);
      EXPECT_TRUE(agx_g17p_rt0_tile_layout_supported(&tib));
      tib = agx_build_tilebuffer_layout(rt, 1, 1, true);
      EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));
   }

   /* Compact three-lane RGB32 is an emulated texture-buffer-only format, not
    * the four-lane RGBX render-target family admitted above. */
   enum pipe_format compact[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32_FLOAT,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout(compact, 1, 1, false);
   EXPECT_EQ(agx_tilebuffer_physical_format(&tib, 0), PIPE_FORMAT_NONE);
   EXPECT_FALSE(agx_g17p_rt0_tile_layout_supported(&tib));

   struct agx_shader_key key = {};
   agx_g17p_configure_rt0_eot_key(PIPE_FORMAT_R32G32B32_FLOAT, &key);
   EXPECT_EQ(key.g17p_rt0_eot_words, 0);
   EXPECT_FALSE(key.g17p_rt0_eot_i32);
}

TEST(Tilebuffer, IndependentNativeStorageAndSampleLimits)
{
   enum pipe_format mixed[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   const auto legacy = agx_build_tilebuffer_layout_bounded(
      mixed, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE);
   EXPECT_TRUE(legacy.spilled[0]);

   const auto native = agx_build_tilebuffer_layout_with_limits(
      mixed, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_FALSE(agx_tilebuffer_spills(&native));
   EXPECT_EQ(native.sample_size_B, 32u);
   EXPECT_EQ(native.tile_size, 16u * 16u);
   EXPECT_EQ(native.sample_size_B * native.nr_samples * native.tile_size, 32768u);
   EXPECT_EQ(native._offset_B[2], 0u);
   EXPECT_EQ(native._offset_B[1], 16u);
   EXPECT_EQ(native._offset_B[0], 24u);

   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   for (unsigned samples : {1u, 2u, 4u}) {
      const auto tib = agx_build_tilebuffer_layout_with_limits(
         rgba8, 1, samples, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
      EXPECT_EQ(tib.tile_size, samples == 4 ? 32u * 16u : 32u * 32u);
      EXPECT_LE(tib.nr_samples * tib.tile_size, 2048u);
   }

   /* Limits must survive attachmentless rendering's deferred sample count. */
   auto deferred = agx_build_tilebuffer_layout_with_limits(
      mixed, 3, 0, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   agx_tilebuffer_set_samples(&deferred, 4);
   EXPECT_EQ(deferred.max_bytes_per_tile, 32768u);
   EXPECT_EQ(deferred.max_samples_per_tile, 2048u);
   EXPECT_EQ(deferred.tile_size, native.tile_size);
   EXPECT_EQ(deferred.sample_size_B, native.sample_size_B);
   EXPECT_FALSE(agx_tilebuffer_spills(&deferred));

   /* True over-capacity layouts still spill. */
   enum pipe_format wide[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_UINT,
   };
   const auto spilling = agx_build_tilebuffer_layout_with_limits(
      wide, 3, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_TRUE(spilling.spilled[2]);
   EXPECT_FALSE(spilling.spilled[0]);
   EXPECT_FALSE(spilling.spilled[1]);
}

/* hk_tilebuffer_layout() builds a layout with the 48 KiB budget when the
 * kernel reports DRM_ASAHI_FEATURE_TILE_BUFFER_48K, and keeps it only when it
 * is 4x, 32x16 and spills nothing; otherwise it builds with 32 KiB. These are
 * the layouts it chooses between.
 */
TEST(Tilebuffer, NativeTileBuffer48KLayouts)
{
   const auto build = [](const enum pipe_format *formats, unsigned count,
                         unsigned samples, uint32_t budget) {
      return agx_build_tilebuffer_layout_with_limits_resident(
         formats, count, samples, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, budget,
         2048, 0);
   };

   /* A deferred G-buffer pass and its lighting target: 24-byte samples. */
   enum pipe_format gbuffer[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM, PIPE_FORMAT_R32_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   for (unsigned count : {4u, 5u}) {
      const auto narrow =
         build(gbuffer, count, 4, AGX_NATIVE_TILE_BUFFER_BYTES);
      EXPECT_FALSE(agx_tilebuffer_spills(&narrow));
      EXPECT_EQ(narrow.sample_size_B, 24u);
      EXPECT_EQ(narrow.tile_size, 16u * 16u);

      /* The shape the 48 KiB budget is taken for. */
      const auto wide =
         build(gbuffer, count, 4, AGX_NATIVE_TILE_BUFFER_48K_BYTES);
      EXPECT_FALSE(agx_tilebuffer_spills(&wide));
      EXPECT_EQ(wide.nr_samples, 4u);
      EXPECT_EQ(wide.sample_size_B, 24u);
      EXPECT_EQ(wide.tile_size, 32u * 16u);
      EXPECT_EQ(wide.sample_size_B * wide.nr_samples * wide.tile_size,
                AGX_NATIVE_TILE_BUFFER_48K_BYTES);
   }

   /* Eight 2x RGBA32 targets spill under either budget, so they keep the
    * 32 KiB layout: with 48 KiB six would stay resident, and the native
    * background and end-of-tile programs take four raw 32-bit targets.
    */
   enum pipe_format rgba32[AGX_MAX_RENDER_TARGETS];
   for (unsigned i = 0; i < AGX_MAX_RENDER_TARGETS; ++i)
      rgba32[i] = PIPE_FORMAT_R32G32B32A32_FLOAT;

   const auto narrow =
      build(rgba32, AGX_MAX_RENDER_TARGETS, 2, AGX_NATIVE_TILE_BUFFER_BYTES);
   const auto wide =
      build(rgba32, AGX_MAX_RENDER_TARGETS, 2, AGX_NATIVE_TILE_BUFFER_48K_BYTES);
   EXPECT_TRUE(agx_tilebuffer_spills(&narrow));
   EXPECT_TRUE(agx_tilebuffer_spills(&wide));

   unsigned narrow_resident = 0, wide_resident = 0;
   for (unsigned i = 0; i < AGX_MAX_RENDER_TARGETS; ++i) {
      narrow_resident += !narrow.spilled[i];
      wide_resident += !wide.spilled[i];
   }
   EXPECT_EQ(narrow_resident, 4u);
   EXPECT_EQ(wide_resident, 6u);
}

TEST(Tilebuffer, ResidentAttachmentsStayInTheTilebuffer)
{
   /* M5 at 4x: 64 bytes per sample. Four RGBA32F targets fill it, so the
    * plain descending-alignment order spills the RGBA8 target. */
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,     PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT,
   };
   const auto plain = agx_build_tilebuffer_layout_resident(
      formats, 5, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048, 0);
   EXPECT_TRUE(plain.spilled[0]);
   EXPECT_EQ(plain.resident_mask, 0u);

   /* Asking for RT0 to stay resident moves it first; a filler spills. */
   const auto resident = agx_build_tilebuffer_layout_resident(
      formats, 5, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048, 0x1);
   EXPECT_FALSE(resident.spilled[0]);
   EXPECT_EQ(resident._offset_B[0], 0u);
   EXPECT_EQ(resident._offset_B[1], 4u);
   EXPECT_TRUE(resident.spilled[4]);
   EXPECT_EQ(resident.resident_mask, 0x1u);

   /* Rebuilding with the recorded mask reproduces the layout. */
   const auto again = agx_build_tilebuffer_layout_resident(
      formats, 5, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048,
      resident.resident_mask);
   EXPECT_EQ(memcmp(&again, &resident, sizeof(again)), 0);

   /* A mask is ignored when the plain order keeps those targets resident. */
   const auto fits = agx_build_tilebuffer_layout_resident(
      formats, 2, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048, 0x1);
   EXPECT_FALSE(agx_tilebuffer_spills(&fits));
   EXPECT_EQ(fits.resident_mask, 0u);
   EXPECT_EQ(fits._offset_B[1], 0u);
   EXPECT_EQ(fits._offset_B[0], 16u);
}

TEST(Tilebuffer, G17PCompressedTwoSampleTargetsStayResident)
{
   /* G17P at 2x: a 32 KiB tile of 16x16 pixels holds 64 bytes a sample.
    * Four RGBA32F targets fill it; the descending-alignment order places
    * them first and spills the two (compressed) RGBA16F targets, which G17P
    * cannot do. */
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   const auto plain = agx_build_tilebuffer_layout_with_limits_resident(
      formats, 6, 2, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048, 0);
   EXPECT_TRUE(plain.spilled[0] && plain.spilled[5]);
   EXPECT_FALSE(plain.spilled[4]);
   const auto limits = agx_build_tilebuffer_layout_with_limits(
      formats, 6, 2, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_EQ(memcmp(&plain, &limits, sizeof(plain)), 0);

   /* Allocated first they stay, and a 16-byte target spills instead. */
   const auto resident = agx_build_tilebuffer_layout_with_limits_resident(
      formats, 6, 2, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048, 0x21);
   EXPECT_FALSE(resident.spilled[0] || resident.spilled[5]);
   EXPECT_TRUE(resident.spilled[4]);
   EXPECT_EQ(resident.resident_mask, 0x21u);
   EXPECT_EQ(resident.sample_size_B, 64u);

   /* Eight eight-byte targets, the most a 2x pass can compress, fit. */
   enum pipe_format rgba16f[AGX_MAX_RENDER_TARGETS];
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      rgba16f[rt] = PIPE_FORMAT_R16G16B16A16_FLOAT;
   const auto eight = agx_build_tilebuffer_layout_with_limits_resident(
      rgba16f, 8, 2, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048, 0xff);
   EXPECT_FALSE(agx_tilebuffer_spills(&eight));

   /* Multiview's eight-byte budget still spills the second one. */
   const auto multiview = agx_build_tilebuffer_layout_with_limits_resident(
      formats, 6, 2, false, AGX_G17P_MAX_UNCOVERED_TILE_BYTES_PER_SAMPLE,
      32768, 2048, 0x21);
   EXPECT_FALSE(multiview.spilled[0]);
   EXPECT_TRUE(multiview.spilled[5]);
}

TEST(Tilebuffer, M5CompressedFourSampleInputAttachmentStaysResident)
{
   /* M5 at 4x: a 64 KiB tile holds 64 bytes per sample. Seven RGBA32F
    * uncompressed attachments (112 B/sample) plus a compressed RGBA16F MSAA4
    * input attachment on RT7 exceed the tilebuffer; plain descending-alignment
    * order spills RT7. Pinning RT7 via resident_mask keeps it resident and
    * spills wider uncompressed RGBA32F targets instead. */
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   const auto plain = agx_build_tilebuffer_layout_resident(
      formats, 8, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048, 0);
   EXPECT_TRUE(plain.spilled[7]);
   EXPECT_FALSE(plain.spilled[0]);
   EXPECT_FALSE(plain.spilled[3]);
   EXPECT_TRUE(plain.spilled[4]);

   const auto resident = agx_build_tilebuffer_layout_resident(
      formats, 8, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048,
      BITFIELD_BIT(7));
   EXPECT_FALSE(resident.spilled[7]);
   EXPECT_EQ(resident._offset_B[7], 0u);
   EXPECT_FALSE(resident.spilled[0]);
   EXPECT_FALSE(resident.spilled[1]);
   EXPECT_FALSE(resident.spilled[2]);
   EXPECT_TRUE(resident.spilled[3]);
   EXPECT_TRUE(resident.spilled[4]);
   EXPECT_TRUE(resident.spilled[5]);
   EXPECT_TRUE(resident.spilled[6]);
   EXPECT_EQ(resident.resident_mask, BITFIELD_BIT(7));
   EXPECT_EQ(resident.sample_size_B, 56u);

   /* All eight 8-byte compressed MSAA4 targets fit without spilling. */
   enum pipe_format rgba16f[AGX_MAX_RENDER_TARGETS];
   for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
      rgba16f[rt] = PIPE_FORMAT_R16G16B16A16_FLOAT;
   const auto eight = agx_build_tilebuffer_layout_resident(
      rgba16f, 8, 4, false, AGX_MAX_RENDER_TARGETS * 16, 65536, 2048, 0xff);
   EXPECT_FALSE(agx_tilebuffer_spills(&eight));
   EXPECT_EQ(eight.sample_size_B, 64u);
}

class SampleDepthOrder : public ::testing::Test {
 protected:
   SampleDepthOrder()
   {
      glsl_type_singleton_init_or_ref();
   }
   ~SampleDepthOrder()
   {
      glsl_type_singleton_decref();
   }
};
static std::vector<char>
zs_tile_order(unsigned samples, bool overlap, bool stencil)
{
   nir_builder b = nir_builder_init_simple_shader(
      MESA_SHADER_FRAGMENT, &agx_nir_options, "sample depth order");
   for (unsigned i = 0; i < samples; ++i) {
      nir_def *mask = nir_imm_intN_t(&b, 1u << (overlap ? 0 : i), 16);
      nir_sample_mask_agx(&b, mask, nir_imm_intN_t(&b, 0, 16));
      auto *zs =
         nir_store_zs_agx(&b, mask, nir_imm_float(&b, float(i) / samples),
                          nir_imm_intN_t(&b, i, 16));
      nir_intrinsic_set_base(zs, stencil ? 2 : 1);
      auto *tile = nir_store_local_pixel_agx(&b, nir_imm_int(&b, i), mask,
                                             nir_undef(&b, 2, 16));
      nir_intrinsic_set_base(tile, 0);
      nir_intrinsic_set_write_mask(tile, 1);
      nir_intrinsic_set_format(tile, PIPE_FORMAT_R32_UINT);
   }
   agx_nir_hoist_g17p_zs_writes(b.shader);
   nir_validate_shader(b.shader, "sample depth ordering");
   std::vector<char> order;
   nir_foreach_block(block, b.impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         switch (nir_instr_as_intrinsic(instr)->intrinsic) {
         case nir_intrinsic_sample_mask_agx:
            order.push_back('K');
            break;
         case nir_intrinsic_store_zs_agx:
            order.push_back('Z');
            break;
         case nir_intrinsic_store_local_pixel_agx:
            order.push_back('T');
            break;
         default:
            break;
         }
      }
   }
   ralloc_free(b.shader);
   return order;
}

TEST_F(SampleDepthOrder, KeepDisjointDepthWithItsSampleColour)
{
   for (unsigned samples : {2u, 4u}) {
      std::vector<char> expected{'K'};
      for (unsigned i = 0; i < samples; ++i) {
         expected.push_back('Z');
         expected.push_back('T');
      }
      EXPECT_EQ(zs_tile_order(samples, false, false), expected);
   }
}

TEST_F(SampleDepthOrder, OverlappingDepthSamplesKeepExistingHoist)
{
   EXPECT_EQ(zs_tile_order(2, true, false),
             (std::vector<char>{'K', 'Z', 'Z', 'T', 'T'}));
}

TEST_F(SampleDepthOrder, StencilSamplesKeepExistingHoist)
{
   EXPECT_EQ(zs_tile_order(2, false, true),
             (std::vector<char>{'K', 'Z', 'Z', 'T', 'T'}));
}

static struct agx_tilebuffer_layout
g17p_single_sample_layout(const enum pipe_format *formats, unsigned count,
                          bool layered)
{
   return agx_build_tilebuffer_layout_with_limits(
      formats, count, 1, layered, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
}

/* At single-sample (1x), every renderable format takes at most 16 bytes of a
 * sample and every compressed G17P colour format at most 8, so eight targets
 * need at most 7 * 16 + 8 = 120 bytes, the G17P per-sample limit, and never
 * spill. Check the 1x bound and every placement. (At 2x/4x MSAA, wide MRT
 * framebuffers can exceed the tilebuffer byte budget and spill; Gallium
 * pre-decompresses those targets via agx_decompress() in
 * agx_set_framebuffer_state() instead of using in-place decompression.)
 */
TEST(Tilebuffer, G17PCompressedColourNeverSpills)
{
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      const enum pipe_format format = (enum pipe_format)f;
      if (ail_pixel_format[format].renderable == PIPE_FORMAT_NONE ||
          util_format_is_depth_or_stencil(format))
         continue;

      enum pipe_format one[AGX_MAX_RENDER_TARGETS] = {format};
      const auto tib = g17p_single_sample_layout(one, 1, false);
      EXPECT_FALSE(tib.spilled[0]) << util_format_name(format);
      EXPECT_LE(tib.sample_size_B, 16u) << util_format_name(format);
   }

   static const enum pipe_format widest[] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_R32G32B32A32_SINT,
   };

   unsigned admitted = 0;
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      const enum pipe_format format = (enum pipe_format)f;
      if (!agx_g17p_compression_color_format(format))
         continue;

      ++admitted;
      const char *name = util_format_name(format);
      EXPECT_NE(ail_pixel_format[format].renderable, PIPE_FORMAT_NONE) << name;
      EXPECT_FALSE(util_format_is_depth_or_stencil(format)) << name;
      EXPECT_LE(util_format_get_blocksize(format), 8u) << name;
      EXPECT_TRUE(ail_can_compress(format, 16, 16, 1)) << name;
      EXPECT_FALSE(ail_can_compress(format, 15, 16, 1)) << name;

      /* The end-of-tile store class is the image's own class. */
      EXPECT_TRUE(ail_formats_compatible(
         format, agx_g17p_rt0_pbe_format(util_format_linear(format))))
         << name;

      enum pipe_format one[AGX_MAX_RENDER_TARGETS] = {format};
      EXPECT_LE(g17p_single_sample_layout(one, 1, false).sample_size_B, 8u)
         << name;

      for (enum pipe_format wide : widest) {
         for (unsigned slot = 0; slot < AGX_MAX_RENDER_TARGETS; ++slot) {
            enum pipe_format formats[AGX_MAX_RENDER_TARGETS];
            for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
               formats[rt] = rt == slot ? format : wide;

            for (bool layered : {false, true}) {
               const auto tib = g17p_single_sample_layout(
                  formats, AGX_MAX_RENDER_TARGETS, layered);
               EXPECT_FALSE(tib.spilled[slot])
                  << name << " in slot " << slot << " next to "
                  << util_format_name(wide);
            }
         }
      }
   }

   /* Honeykrisp's 46 Vulkan formats are 41 pipe formats (the five
    * A8B8G8R8 PACK32 formats are R8G8B8A8); Gallium adds 12 X variants. */
   EXPECT_EQ(admitted, 53u);

   /* An X variant is its A variant's channel class, type and tilebuffer
    * format, so it compresses identically. */
   static const enum pipe_format x_variants[][2] = {
      {PIPE_FORMAT_R8G8B8X8_UNORM, PIPE_FORMAT_R8G8B8A8_UNORM},
      {PIPE_FORMAT_R8G8B8X8_SNORM, PIPE_FORMAT_R8G8B8A8_SNORM},
      {PIPE_FORMAT_R8G8B8X8_UINT, PIPE_FORMAT_R8G8B8A8_UINT},
      {PIPE_FORMAT_R8G8B8X8_SINT, PIPE_FORMAT_R8G8B8A8_SINT},
      {PIPE_FORMAT_R8G8B8X8_SRGB, PIPE_FORMAT_R8G8B8A8_SRGB},
      {PIPE_FORMAT_B8G8R8X8_UNORM, PIPE_FORMAT_B8G8R8A8_UNORM},
      {PIPE_FORMAT_B8G8R8X8_SRGB, PIPE_FORMAT_B8G8R8A8_SRGB},
      {PIPE_FORMAT_R10G10B10X2_UNORM, PIPE_FORMAT_R10G10B10A2_UNORM},
      {PIPE_FORMAT_B10G10R10X2_UNORM, PIPE_FORMAT_B10G10R10A2_UNORM},
      {PIPE_FORMAT_R16G16B16X16_UINT, PIPE_FORMAT_R16G16B16A16_UINT},
      {PIPE_FORMAT_R16G16B16X16_SINT, PIPE_FORMAT_R16G16B16A16_SINT},
      {PIPE_FORMAT_R16G16B16X16_FLOAT, PIPE_FORMAT_R16G16B16A16_FLOAT},
   };
   for (const auto &pair : x_variants) {
      const char *name = util_format_name(pair[0]);
      EXPECT_TRUE(agx_g17p_compression_color_format(pair[0])) << name;
      EXPECT_TRUE(agx_g17p_compression_color_format(pair[1])) << name;
      EXPECT_EQ(ail_pixel_format[pair[0]].channels,
                ail_pixel_format[pair[1]].channels) << name;
      EXPECT_EQ(ail_pixel_format[pair[0]].type,
                ail_pixel_format[pair[1]].type) << name;
      EXPECT_EQ(ail_pixel_format[pair[0]].renderable,
                ail_pixel_format[pair[1]].renderable) << name;
      EXPECT_EQ(util_format_get_blocksize(pair[0]),
                util_format_get_blocksize(pair[1])) << name;
   }

   /* No packed-float, shared-exponent, 16-byte, depth or stencil format. */
   static const enum pipe_format refused[] = {
      PIPE_FORMAT_R11G11B10_FLOAT, PIPE_FORMAT_R9G9B9E5_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_R32G32B32A32_UINT,
      PIPE_FORMAT_Z32_FLOAT, PIPE_FORMAT_Z16_UNORM, PIPE_FORMAT_S8_UINT,
      PIPE_FORMAT_R4G4B4A4_UNORM, PIPE_FORMAT_B5G5R5A1_UNORM,
      PIPE_FORMAT_A8_UNORM, PIPE_FORMAT_L8_UNORM,
   };
   for (enum pipe_format format : refused)
      EXPECT_FALSE(agx_g17p_compression_color_format(format))
         << util_format_name(format);

   /* 16-byte RGBA32 colour formats live in their own helper so wide-MRT spill
    * invariants on the <=8-byte set stay explicit. */
   unsigned admitted_16b = 0;
   for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
      const enum pipe_format format = (enum pipe_format)f;
      if (!agx_g17p_compression_color_format_16b(format))
         continue;

      ++admitted_16b;
      const char *name = util_format_name(format);
      EXPECT_FALSE(agx_g17p_compression_color_format(format)) << name;
      EXPECT_NE(ail_pixel_format[format].renderable, PIPE_FORMAT_NONE) << name;
      EXPECT_FALSE(util_format_is_depth_or_stencil(format)) << name;
      EXPECT_EQ(util_format_get_blocksize(format), 16u) << name;
      EXPECT_TRUE(ail_can_compress(format, 16, 16, 1)) << name;
      EXPECT_FALSE(ail_can_compress(format, 15, 16, 1)) << name;
      EXPECT_TRUE(ail_formats_compatible(
         format, agx_g17p_rt0_pbe_format(util_format_linear(format))))
         << name;

      enum pipe_format one[AGX_MAX_RENDER_TARGETS] = {format};
      const auto tib1 = g17p_single_sample_layout(one, 1, false);
      EXPECT_FALSE(tib1.spilled[0]) << name;
      EXPECT_EQ(tib1.sample_size_B, 16u) << name;

      /* Eight 16-byte targets exceed G17P's 120-byte budget and spill the
       * eighth target, which agx_set_framebuffer_state() pre-decompresses. */
      enum pipe_format eight[AGX_MAX_RENDER_TARGETS];
      for (unsigned rt = 0; rt < AGX_MAX_RENDER_TARGETS; ++rt)
         eight[rt] = format;
      const auto tib8 =
         g17p_single_sample_layout(eight, AGX_MAX_RENDER_TARGETS, false);
      EXPECT_TRUE(agx_tilebuffer_spills(&tib8)) << name;
      EXPECT_TRUE(tib8.spilled[AGX_MAX_RENDER_TARGETS - 1]) << name;
   }
   EXPECT_EQ(admitted_16b, 6u);

   static const enum pipe_format x_variants_16b[][2] = {
      {PIPE_FORMAT_R32G32B32X32_FLOAT, PIPE_FORMAT_R32G32B32A32_FLOAT},
      {PIPE_FORMAT_R32G32B32X32_UINT, PIPE_FORMAT_R32G32B32A32_UINT},
      {PIPE_FORMAT_R32G32B32X32_SINT, PIPE_FORMAT_R32G32B32A32_SINT},
   };
   for (const auto &pair : x_variants_16b) {
      const char *name = util_format_name(pair[0]);
      EXPECT_TRUE(agx_g17p_compression_color_format_16b(pair[0])) << name;
      EXPECT_TRUE(agx_g17p_compression_color_format_16b(pair[1])) << name;
      EXPECT_EQ(ail_pixel_format[pair[0]].channels,
                ail_pixel_format[pair[1]].channels) << name;
      EXPECT_EQ(ail_pixel_format[pair[0]].type,
                ail_pixel_format[pair[1]].type) << name;
      EXPECT_EQ(ail_pixel_format[pair[0]].renderable,
                ail_pixel_format[pair[1]].renderable) << name;
   }
}

TEST(Tilebuffer, PackFirstIfFitsMovesClearedRTIntoFirst8Bytes)
{
   /* Default descending-alignment order puts 16B RGBA32F at [0, 16) and
    * 4B RGBA8 at [16, 20). Packing RT1 first moves it into [0, 4) without
    * changing sample_size_B, tile_size, or spilling. */
   enum pipe_format formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R32G32B32A32_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   auto tib = agx_build_tilebuffer_layout_with_limits(
      formats, 2, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 0u);
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 16u);
   EXPECT_EQ(agx_tilebuffer_format_size_B(PIPE_FORMAT_R32G32B32A32_FLOAT), 16u);
   EXPECT_EQ(agx_tilebuffer_format_size_B(PIPE_FORMAT_R8G8B8A8_UNORM), 4u);

   const uint16_t orig_tile = tib.tile_size;
   EXPECT_TRUE(agx_tilebuffer_pack_first_if_fits(&tib, BITFIELD_BIT(1), 8));
   EXPECT_EQ(tib.first_mask, BITFIELD_BIT(1));
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 1), 0u);
   EXPECT_EQ(agx_tilebuffer_offset_B(&tib, 0), 4u);
   EXPECT_EQ(tib.tile_size, orig_tile);
   EXPECT_FALSE(agx_tilebuffer_spills(&tib));

   /* Rebuilding + applying first_mask reproduces the layout byte-for-byte. */
   auto rebuilt = agx_build_tilebuffer_layout_with_limits(
      formats, 2, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   agx_tilebuffer_apply_first_mask(&rebuilt, tib.first_mask);
   EXPECT_EQ(memcmp(&rebuilt, &tib, sizeof(tib)), 0);

   /* Already within [0, 8): no-op and first_mask stays 0. */
   enum pipe_format fits_formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   auto already = agx_build_tilebuffer_layout_with_limits(
      fits_formats, 2, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   EXPECT_FALSE(agx_tilebuffer_pack_first_if_fits(&already, BITFIELD_BIT(0), 8));
   EXPECT_EQ(already.first_mask, 0u);

   /* Moving RT1 (4B) ahead of RT0 (8B) fits RT1 into [0, 4). */
   EXPECT_TRUE(agx_tilebuffer_pack_first_if_fits(&already, BITFIELD_BIT(1), 8));
   EXPECT_EQ(already.first_mask, BITFIELD_BIT(1));
   EXPECT_EQ(agx_tilebuffer_offset_B(&already, 1), 0u);
   EXPECT_EQ(agx_tilebuffer_offset_B(&already, 0), 4u);

   /* Combined cleared size > 8B: refused and layout stays untouched. */
   auto wide = agx_build_tilebuffer_layout_with_limits(
      fits_formats, 2, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768, 2048);
   const auto wide_orig = wide;
   EXPECT_FALSE(agx_tilebuffer_pack_first_if_fits(
      &wide, BITFIELD_BIT(0) | BITFIELD_BIT(1), 8));
   EXPECT_EQ(memcmp(&wide, &wide_orig, sizeof(wide)), 0);

   /* Deferred MSAA4 tile pass: 3 transient G-buffer targets (DONT_CARE) plus
    * 1 stored RGBA8 lighting output at RT3. Packing only the stored RT3 puts
    * it at offset 0 within the 8B uncovered-tile limit without spilling. */
   enum pipe_format deferred_formats[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   auto def = agx_build_tilebuffer_layout_with_limits(
      deferred_formats, 4, 4, false, AGX_G17P_MAX_BYTES_PER_SAMPLE, 32768,
      2048);
   EXPECT_GT(def.sample_size_B, 8u);
   EXPECT_GT(agx_tilebuffer_offset_B(&def, 3), 8u);
   EXPECT_TRUE(agx_tilebuffer_pack_first_if_fits(&def, BITFIELD_BIT(3), 8));
   EXPECT_EQ(agx_tilebuffer_offset_B(&def, 3), 0u);
   EXPECT_FALSE(agx_tilebuffer_spills(&def));
}

/* Interchange is advertised exactly where the shared Gen9 2D compression
 * capability is enabled, and for the eight plain 32-bit 8-bit and 10-bit
 * RGB/RGBA formats. Every other chip and format (including FP16 and 10-bit
 * UINT) keeps the modifier list it had before. */
TEST(Interchange, ModifierSupport)
{
   static const uint64_t all_modifiers[] = {
      DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
      DRM_FORMAT_MOD_APPLE_GPU_TILED,
      DRM_FORMAT_MOD_APPLE_INTERCHANGE,
      DRM_FORMAT_MOD_LINEAR,
   };
   const std::vector<uint64_t> base_list = {
      DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
      DRM_FORMAT_MOD_APPLE_GPU_TILED,
      DRM_FORMAT_MOD_LINEAR,
   };
   const std::vector<uint64_t> interchange_list = {
      DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
      DRM_FORMAT_MOD_APPLE_GPU_TILED,
      DRM_FORMAT_MOD_APPLE_INTERCHANGE,
      DRM_FORMAT_MOD_LINEAR,
   };

   /* Explicit sanity checks: AGX2 (M1/M2), unmeasured Gen9 parts (G15/G16),
    * and Gen10 (M5/M6) refuse Interchange; G17P admits it via the shared
    * AGX_CAP_GEN9_2D_COMPRESSION capability. */
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G13G));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G14G));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G15P));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G15G));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G15S));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G15C));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G16G));
   EXPECT_TRUE(agx_chip_supports_interchange(AGX_CHIP_G17P));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G17G));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_G18G));
   EXPECT_FALSE(agx_chip_supports_interchange(AGX_CHIP_UNKNOWN));

   /* Explicit negative checks for unsupported display/HDR formats. */
   EXPECT_FALSE(agx_interchange_modifier_supported(AGX_CHIP_G17P,
                                                   PIPE_FORMAT_R16G16B16A16_FLOAT));
   EXPECT_FALSE(agx_interchange_modifier_supported(AGX_CHIP_G17P,
                                                   PIPE_FORMAT_R10G10B10A2_UINT));
   EXPECT_FALSE(agx_interchange_modifier_supported(AGX_CHIP_G17P,
                                                   PIPE_FORMAT_B10G10R10A2_UINT));
   EXPECT_FALSE(agx_interchange_modifier_supported(AGX_CHIP_G17P,
                                                   PIPE_FORMAT_R11G11B10_FLOAT));
   EXPECT_FALSE(agx_interchange_modifier_supported(AGX_CHIP_G17P,
                                                   PIPE_FORMAT_B8G8R8A8_SRGB));

   for (unsigned c = 0; c <= AGX_CHIP_UNKNOWN; ++c) {
      const enum agx_chip chip = (enum agx_chip)c;
      const bool capable = agx_chip_has_cap(chip, AGX_CAP_GEN9_2D_COMPRESSION);

      EXPECT_EQ(agx_chip_supports_interchange(chip), capable) << c;

      for (unsigned f = 1; f < PIPE_FORMAT_COUNT; ++f) {
         const enum pipe_format format = (enum pipe_format)f;
         const bool supported = format == PIPE_FORMAT_B8G8R8A8_UNORM ||
                                format == PIPE_FORMAT_B8G8R8X8_UNORM ||
                                format == PIPE_FORMAT_R8G8B8A8_UNORM ||
                                format == PIPE_FORMAT_R8G8B8X8_UNORM ||
                                format == PIPE_FORMAT_B10G10R10A2_UNORM ||
                                format == PIPE_FORMAT_B10G10R10X2_UNORM ||
                                format == PIPE_FORMAT_R10G10B10A2_UNORM ||
                                format == PIPE_FORMAT_R10G10B10X2_UNORM;

         EXPECT_EQ(agx_interchange_modifier_supported(chip, format),
                   capable && supported)
            << c << " " << util_format_name(format);

         for (bool nocompress : {false, true}) {
            std::vector<uint64_t> advertised;
            for (uint64_t mod : all_modifiers) {
               if (agx_dmabuf_modifier_supported(chip, nocompress, mod, format))
                  advertised.push_back(mod);
            }
            EXPECT_EQ(advertised,
                      (capable && supported && !nocompress) ? interchange_list
                                                            : base_list)
               << c << " " << util_format_name(format) << " " << nocompress;
         }
      }
   }
}

TEST(Tilebuffer, G18GNativeMixedEpilogMask)
{
   enum pipe_format rgba8[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R8G8B8A8_UNORM,
   };
   struct agx_tilebuffer_layout tib =
      agx_build_tilebuffer_layout_resident(rgba8, 3, 1, false, 64, 65536, 2048,
                                           0);
   EXPECT_EQ(agx_g18g_word_mrt_mask(&tib), 0x7u);
   EXPECT_EQ(agx_g18g_native_mixed_mask(&tib), 0u);
   EXPECT_EQ(agx_g18g_native_mixed_epilog_mask(&tib), 0x7u);

   /* Non-native1130 formats (e.g. sRGB) stay excluded. */
   enum pipe_format srgb[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_SRGB,
      PIPE_FORMAT_R8G8B8A8_SRGB,
      PIPE_FORMAT_R8G8B8A8_SRGB,
   };
   struct agx_tilebuffer_layout srgb_tib =
      agx_build_tilebuffer_layout_resident(srgb, 3, 1, false, 64, 65536, 2048,
                                           0);
   EXPECT_EQ(agx_g18g_native_mixed_epilog_mask(&srgb_tib), 0u);

   /* Layered and multisample layouts stay excluded. */
   struct agx_tilebuffer_layout layered_tib =
      agx_build_tilebuffer_layout_resident(rgba8, 3, 1, true, 32, 32768, 2048,
                                           0);
   EXPECT_EQ(agx_g18g_native_mixed_epilog_mask(&layered_tib), 0u);

   struct agx_tilebuffer_layout msaa_tib =
      agx_build_tilebuffer_layout_resident(rgba8, 3, 4, false, 64, 65536, 2048,
                                           0);
   EXPECT_EQ(agx_g18g_native_mixed_epilog_mask(&msaa_tib), 0u);

   /* Genuine mixed layouts match both masks. */
   enum pipe_format mixed[AGX_MAX_RENDER_TARGETS] = {
      PIPE_FORMAT_R8G8B8A8_UNORM,
      PIPE_FORMAT_R16G16B16A16_FLOAT,
   };
   struct agx_tilebuffer_layout mixed_tib =
      agx_build_tilebuffer_layout_g18_mixed(mixed, 2);
   EXPECT_EQ(agx_g18g_native_mixed_mask(&mixed_tib), 0x3u);
   EXPECT_EQ(agx_g18g_native_mixed_epilog_mask(&mixed_tib), 0x3u);
}
