/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Verify the precompiled in-place decompression kernels
 * (libagx_agx3_decompress, libagx_reset_compression_metadata) of HAL200 (G17P
 * and G16G) and of Gen10 private compression (AGX_CAP_GEN10_2D_COMPRESSION,
 * the M5's g17g library; not g18g), hk_g17p_decompress_pack_mode(),
 * hk_g17p_decompress_image_type_supported(), Gen9 compressed texture
 * descriptor packing across all admitted colour formats and shapes, and the
 * Gen10 descriptor and 16x8 metadata tile of the single-sample 16-byte M5
 * spill.
 */
#include "asahi/compiler/agx_compile.h"
#include "asahi/layout/layout.h"
#include "asahi/lib/agx_chip_caps.h"
#include "asahi/lib/agx_device.h"
#include "asahi/vulkan/hk_g17p_compression.h"
#include "asahi/vulkan/hk_g17p_decompress.h"
#include "asahi/vulkan/hk_native_descriptor_pack.h"
#include "libagx.h"
#include "libagx_shaders.h"
#include "vk_format.h"

#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* A texture packet's defaults, filled in field by field below (designated
 * initializers after the header would override its fields). */
static struct AGX_TEXTURE
texture_header(void)
{
   struct AGX_TEXTURE cfg = {AGX_TEXTURE_header};
   return cfg;
}

static void
check_kernel(const uint32_t *bin, const char *name)
{
   const struct agx_precompiled_kernel_info *info = (const void *)bin;
   assert(info && name);
   assert(info->main_size && info->nr_gprs);
   assert(info->main_offset <= info->binary_size);
   assert(info->main_size <= info->binary_size - info->main_offset);
   assert(info->agx3_staging_known);
   assert(info->nr_preamble_gprs == 0 && info->preamble_size == 0);
   assert(info->workgroup_size[0] == 32);
   assert(info->workgroup_size[1] == 1 && info->workgroup_size[2] == 1);
   assert(info->printf_uniform == 0);
   assert(info->scratch_size == 0);
}

int
main(void)
{
   /* The library of a chip with Gen10 private compression (the M5's g17g)
    * carries the spill programs; g18g compiles as G17G without it. */
   assert(agx_chip_has_cap(AGX_CHIP_G17G, AGX_CAP_GEN10_2D_COMPRESSION));
   assert(!agx_chip_has_cap(AGX_CHIP_G18G, AGX_CAP_GEN10_2D_COMPRESSION));

   /* Verify libagx_reset_compression_metadata on HAL200 (G17P, G16G), Gen10
    * (g17g) and AGX2, and its absence from g18g. */
   check_kernel(libagx_g17p[LIBAGX_RESET_COMPRESSION_METADATA],
                "libagx_reset_compression_metadata (g17p)");
   check_kernel(libagx_g16g[LIBAGX_RESET_COMPRESSION_METADATA],
                "libagx_reset_compression_metadata (g16g)");
   check_kernel(libagx_g17g[LIBAGX_RESET_COMPRESSION_METADATA],
                "libagx_reset_compression_metadata (g17g)");
   assert(libagx_g18g[LIBAGX_RESET_COMPRESSION_METADATA] == NULL);
   assert(libagx_g13g[LIBAGX_RESET_COMPRESSION_METADATA] != NULL);
   assert(libagx_g13x[LIBAGX_RESET_COMPRESSION_METADATA] != NULL);

   /* Verify all 6 variants of libagx_agx3_decompress (1x/2x/4x x 2D/array)
    * on G17P, G16G and g17g, and exclusion from the other targets. */
   for (unsigned s = 0; s < 3; ++s) {
      for (unsigned layered = 0; layered < 2; ++layered) {
         const unsigned prog = LIBAGX_AGX3_DECOMPRESS(s, layered);
         check_kernel(libagx_g17p[prog], "libagx_agx3_decompress (g17p)");
         check_kernel(libagx_g16g[prog], "libagx_agx3_decompress (g16g)");
         check_kernel(libagx_g17g[prog], "libagx_agx3_decompress (g17g)");
         assert(libagx_g13g[prog] == NULL);
         assert(libagx_g13x[prog] == NULL);
         assert(libagx_g15g[prog] == NULL);
         assert(libagx_g15s[prog] == NULL);
         assert(libagx_g18g[prog] == NULL);
      }
   }

   /* The M5 spill of a single-sample 16-byte target: its metadata tile is
    * 16x8 (ail_metadata_tile(), the tile_height_sa hk_begin_rendering()
    * passes), and the canonical UINT read descriptor packs as Gen10. */
   {
      struct ail_layout m5 = {
         .width_px = 100,
         .height_px = 60,
         .depth_px = 1,
         .sample_count_sa = 1,
         .levels = 1,
         .policy = AIL_LAYOUT_POLICY_M5,
         .tiling = AIL_TILING_GPU,
         .compressed = true,
         .format = PIPE_FORMAT_R32G32B32A32_FLOAT,
         .renderable = true,
      };
      ail_make_miptree(&m5);
      assert(ail_is_level_logically_compressed(&m5, 0));
      assert(ail_metadata_tile(&m5).width_el == 16);
      assert(ail_metadata_tile(&m5).height_el == 8);
      assert(ail_metadata_width_tl(&m5, 0) == 7);
      assert(ail_metadata_height_tl(&m5, 0) == 8);

      /* One to eight byte texels keep the 16x16 tile. */
      struct ail_layout m5_rgba8 = m5;
      memset(m5_rgba8.level_offsets_B, 0, sizeof(m5_rgba8.level_offsets_B));
      m5_rgba8.format = PIPE_FORMAT_R8G8B8A8_UNORM;
      ail_make_miptree(&m5_rgba8);
      assert(ail_metadata_tile(&m5_rgba8).height_el == 16);

      uint32_t pack_mode = UINT32_MAX;
      const enum agx_channels channels = ail_pixel_format[m5.format].channels;
      assert(hk_g17p_decompress_pack_mode(channels, &pack_mode));
      assert(pack_mode == LIBAGX_AGX3_DECOMPRESS_R32G32B32A32);

      struct AGX_TEXTURE cfg = texture_header();
      cfg.dimension = AGX_TEXTURE_DIMENSION_2D;
      cfg.layout = AGX_LAYOUT_GPU;
      cfg.channels = channels;
      cfg.type = AGX_TEXTURE_TYPE_UINT;
      cfg.swizzle_r = AGX_CHANNEL_R;
      cfg.swizzle_g = AGX_CHANNEL_G;
      cfg.swizzle_b = AGX_CHANNEL_B;
      cfg.swizzle_a = AGX_CHANNEL_A;
      cfg.address = 0x100000;
      cfg.width = m5.width_px;
      cfg.height = m5.height_px;
      cfg.compressed = true;
      cfg.extended = true;
      cfg.acceleration_buffer = 0x100000 + m5.metadata_offset_B;
      cfg.depth = 1;
      cfg.page_aligned_layers = m5.page_aligned_layers;
      struct hk_native_texture_descriptor packed;
      assert(hk_pack_native_texture(10, &packed, &cfg));
   }

   /* Verify every G17P compressed colour format maps to a valid pack mode in
    * hk_g17p_decompress_pack_mode() and packs a valid canonical Gen9
    * compressed texture descriptor across 1x/2x/4x and 2D/2D_ARRAY. */
   unsigned format_count = 0;
   uint32_t pack_mode_mask = 0;
   for (unsigned f = 1; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; ++f) {
      const VkFormat vk_fmt = (VkFormat)f;
      if (!hk_g17p_compression_color_format(vk_fmt) &&
          !hk_g17p_compression_color_format_16b(vk_fmt))
         continue;
      format_count++;

      const enum pipe_format p_fmt = vk_format_to_pipe_format(vk_fmt);
      const enum agx_channels channels = ail_pixel_format[p_fmt].channels;
      uint32_t pack_mode = UINT32_MAX;
      assert(hk_g17p_decompress_pack_mode(channels, &pack_mode));
      assert(pack_mode <= LIBAGX_AGX3_DECOMPRESS_R32G32B32A32);
      pack_mode_mask |= 1u << pack_mode;
      const enum agx_texture_type type =
         channels == AGX_CHANNELS_R5G6B5 ? AGX_TEXTURE_TYPE_UNORM
                                         : AGX_TEXTURE_TYPE_UINT;

      for (unsigned samples = 1; samples <= 4; samples *= 2) {
         for (unsigned layered = 0; layered < 2; ++layered) {
            const bool msaa = samples > 1;
            struct AGX_TEXTURE cfg = texture_header();
            cfg.dimension =
               msaa ? (layered ? AGX_TEXTURE_DIMENSION_2D_ARRAY_MULTISAMPLED
                               : AGX_TEXTURE_DIMENSION_2D_MULTISAMPLED)
                    : (layered ? AGX_TEXTURE_DIMENSION_2D_ARRAY
                               : AGX_TEXTURE_DIMENSION_2D);
            cfg.layout = AGX_LAYOUT_GPU;
            cfg.channels = channels;
            cfg.type = type;
            cfg.swizzle_r = AGX_CHANNEL_R;
            cfg.swizzle_g = AGX_CHANNEL_G;
            cfg.swizzle_b = AGX_CHANNEL_B;
            cfg.swizzle_a = AGX_CHANNEL_A;
            cfg.address = 0x100000;
            cfg.width = 256;
            cfg.height = 128;
            cfg.first_level = 0;
            cfg.last_level = 0;
            cfg.compressed = true;
            cfg.extended = true;
            cfg.acceleration_buffer = 0x200000;
            cfg.depth = layered ? 6 : 1;
            cfg.page_aligned_layers = layered != 0;
            if (msaa)
               cfg.samples =
                  samples == 4 ? AGX_SAMPLE_COUNT_4 : AGX_SAMPLE_COUNT_2;

            struct hk_native_texture_descriptor packed;
            assert(hk_pack_native_texture(9, &packed, &cfg));
         }
      }
   }
   assert(format_count == 49);
   assert(pack_mode_mask == ((1u << (LIBAGX_AGX3_DECOMPRESS_R32G32B32A32 + 1)) - 1u));
   uint32_t bad_mode = 0;
   assert(!hk_g17p_decompress_pack_mode(AGX_CHANNELS_R11G11B10, &bad_mode));
   assert(hk_g17p_decompress_image_type_supported(VK_IMAGE_TYPE_2D));
   assert(!hk_g17p_decompress_image_type_supported(VK_IMAGE_TYPE_3D));

   printf("AGX3 decompression: 21 kernels (G17P+G16G+G17G) + %u formats x 6 "
          "shapes + M5 16-byte spill passed\n",
          format_count);
   return 0;
}
