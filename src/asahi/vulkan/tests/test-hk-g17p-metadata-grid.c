/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * The metadata tile indexing the G17P compute mip path writes with
 * (hk_g17p_metadata_tile_index) against libagx_fast_clear's index_metadata,
 * and against the metadata AIL allocates for each compressed level: every
 * tile of a level has its own word inside that level's metadata.
 */
#include "hk_g17p_meta_copy_core.h"
#include "layout.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned failures;

#define EXPECT(cond, ...)                                                      \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, __VA_ARGS__);                                         \
         fprintf(stderr, "\n");                                                \
         failures++;                                                           \
      }                                                                        \
   } while (0)

/* libagx/compression.cl index_metadata(), with a bit-by-bit interleave */
static uint32_t
reference_index(uint32_t x, uint32_t y, uint32_t width, uint32_t height)
{
   const uint32_t major = width > height ? x : y;
   const uint32_t minor = width < height ? width : height;
   uint32_t bits = 0;
   while ((1u << bits) < minor)
      bits++;
   const uint32_t mask = (1u << bits) - 1;

   uint32_t interleaved = 0;
   for (uint32_t i = 0; i < 16; ++i) {
      interleaved |= ((x & mask) >> i & 1) << (2 * i);
      interleaved |= ((y & mask) >> i & 1) << (2 * i + 1);
   }
   return interleaved + ((major & ~mask) << bits);
}

static void
check_grid(uint32_t width_tl, uint32_t height_tl)
{
   struct hk_g17p_metadata_grid grid;
   EXPECT(hk_g17p_metadata_grid_init(width_tl, height_tl, &grid),
          "grid %ux%u refused", width_tl, height_tl);
   for (uint32_t y = 0; y < height_tl; ++y) {
      for (uint32_t x = 0; x < width_tl; ++x) {
         const uint32_t got = hk_g17p_metadata_tile_index(&grid, x, y);
         const uint32_t want = reference_index(x, y, width_tl, height_tl);
         if (got != want) {
            EXPECT(false, "grid %ux%u tile (%u,%u): index %u, want %u",
                   width_tl, height_tl, x, y, got, want);
            return;
         }
      }
   }
}

/* Every logically compressed level of an RGBA8 image: distinct words, all
 * inside the level's own metadata. */
static void
check_layout(uint32_t width, uint32_t height)
{
   struct ail_layout layout = {
      .width_px = width,
      .height_px = height,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = util_logbase2(MAX2(width, height)) + 1,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .compressed = true,
   };
   ail_make_miptree(&layout);

   for (uint32_t l = 0; l < layout.levels; ++l) {
      if (!ail_is_level_logically_compressed(&layout, l))
         continue;

      const uint64_t end_B =
         l + 1 < layout.levels && ail_is_level_allocated_compressed(&layout, l + 1)
            ? layout.level_offsets_compressed_B[l + 1]
            : layout.compression_layer_stride_B;
      const uint32_t words =
         (uint32_t)((end_B - layout.level_offsets_compressed_B[l]) / 8);

      struct hk_g17p_metadata_grid grid;
      const uint32_t w_tl = ail_metadata_width_tl(&layout, l);
      const uint32_t h_tl = ail_metadata_height_tl(&layout, l);
      EXPECT(hk_g17p_metadata_grid_init(w_tl, h_tl, &grid),
             "%ux%u level %u: grid refused", width, height, l);

      uint8_t *seen = calloc(words, 1);
      for (uint32_t y = 0; y < h_tl; ++y) {
         for (uint32_t x = 0; x < w_tl; ++x) {
            const uint32_t index = hk_g17p_metadata_tile_index(&grid, x, y);
            if (index >= words || seen[index]) {
               EXPECT(false,
                      "%ux%u level %u tile (%u,%u): word %u of %u%s", width,
                      height, l, x, y, index, words,
                      index < words ? " written twice" : "");
               free(seen);
               return;
            }
            seen[index] = 1;
         }
      }
      free(seen);
   }
}

/* Which levels of common chains are stored compressed by a render pass */
static void
check_mip_split(void)
{
   EXPECT(hk_g17p_mip_level_through_pbe(1920, 1080),
          "3840x2160 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(2048, 2048),
          "4096x4096 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(1280, 720),
          "2560x1440 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(960, 540),
          "1920x1080 level 1 should take a render pass");
   EXPECT(hk_g17p_mip_level_through_pbe(640, 360),
          "1280x720 level 1 should take a render pass");
   EXPECT(!hk_g17p_mip_level_through_pbe(480, 270),
          "1920x1080 level 2 should take compute");
   EXPECT(!hk_g17p_mip_level_through_pbe(256, 256),
          "512x512 level 1 should take compute");
   EXPECT(!hk_g17p_mip_level_through_pbe(1, 1), "1x1 should take compute");

   /* Exact 2x2 compressed levels stay in compute via compressed block-mip on
    * HAL200/AGX3B (G17P and G16G) at all sizes, while odd levels and AGX2
    * keep the raw-compute render-pass crossover. */
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 1280, 720, 640, 360),
          "G17P exact 2x2 640x360 should use compute block-mip");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 2560, 1440, 1280, 720),
          "G17P exact 2x2 1280x720 should use compute block-mip");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 2048, 2048, 1024, 1024),
          "G17P exact 2x2 1024x1024 should use compute block-mip");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 3840, 2160, 1920, 1080),
          "G17P exact 2x2 1920x1080 should use compute block-mip");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 4096, 4096, 2048, 2048),
          "G17P exact 2x2 2048x2048 should use compute block-mip");
   EXPECT(hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 1281, 720, 640, 360),
          "G17P odd 640x360 should keep render pass");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G17P, 961, 540, 480, 270),
          "G17P odd 480x270 below crossover should use raw compute");

   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G16G, 2048, 2048, 1024, 1024),
          "G16G exact 2x2 1024x1024 should use compute block-mip");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G16G, 4096, 4096, 2048, 2048),
          "G16G exact 2x2 2048x2048 should use compute block-mip");
   EXPECT(hk_compressed_mip_level_through_pbe(AGX_CHIP_G16G, 2049, 2048, 1024, 1024),
          "G16G odd 1024x1024 should keep render pass");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G16G, 2561, 1440, 1280, 720),
          "G16G odd 1280x720 below classic crossover should use raw compute");

   EXPECT(hk_compressed_mip_level_through_pbe(AGX_CHIP_G14G, 2048, 2048, 1024, 1024),
          "AGX2 1024x1024 without block-mip should keep render pass");
   EXPECT(!hk_compressed_mip_level_through_pbe(AGX_CHIP_G14G, 1280, 720, 640, 360),
          "AGX2 640x360 below classic crossover should use raw compute");
}

/* A layer owns a complete metadata miptree, including 3D Z slots whose
 * higher levels are unused. Samples expand X/Y, never the layer count. */
static void
check_layered_layout(uint32_t width, uint32_t height, unsigned layers,
                     unsigned samples, unsigned levels, bool three_d)
{
   struct ail_layout layout = {
      .width_px = width,
      .height_px = height,
      .depth_px = layers,
      .sample_count_sa = samples,
      .levels = levels,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .compressed = true,
      .renderable = true,
      .mipmapped_z = three_d,
   };
   ail_make_miptree(&layout);

   unsigned a = (width * (samples == 4 ? 2 : 1) + 15) & ~15u;
   unsigned b = (height * (samples >= 2 ? 2 : 1) + 15) & ~15u;
   uint64_t stride = 0;
   unsigned allocated = 0;
   for (unsigned l = 0; l < levels && MAX2(a, b) >= 16; ++l) {
      const unsigned tw = MAX2(util_next_power_of_two(a), 16) / 16;
      const unsigned th = MAX2(util_next_power_of_two(b), 16) / 16;
      const uint64_t size = ((uint64_t)tw * th * 8 + 127) & ~UINT64_C(127);
      EXPECT(layout.level_offsets_compressed_B[l] == stride,
             "%ux%u, %ux, level %u: offset mismatch", width, height, samples,
             l);
      stride += size;
      a = (a + 1) / 2;
      b = (b + 1) / 2;
      allocated++;
   }
   EXPECT(layout.compression_layer_stride_B == stride,
          "%ux%u, %ux: metadata stride mismatch", width, height, samples);
   EXPECT(layout.size_B == layout.metadata_offset_B + layers * stride,
          "%ux%u, %ux, %u layers: metadata reserve mismatch", width, height,
          samples, layers);
   EXPECT(layers == 1 ||
             (layout.page_aligned_layers && layout.layer_stride_B % 16384 == 0),
          "renderable data layers must be page aligned");
   for (unsigned l = 0; l < allocated; ++l) {
      const unsigned valid_layers = three_d ? MAX2(layers >> l, 1) : layers;
      const uint64_t level_offset = layout.level_offsets_compressed_B[l];
      for (unsigned k = 0; k < valid_layers; ++k) {
         const uint64_t address =
            layout.metadata_offset_B + k * stride + level_offset;
         EXPECT(address % 128 == 0 && address < layout.size_B,
                "metadata layer/level outside allocation");
         EXPECT(k == 0 || address >= layout.metadata_offset_B + k * stride,
                "metadata layers overlap");
      }
   }
}

static void
check_layered_layouts(void)
{
   static const unsigned sizes[][2] = {
      {16, 16}, {17, 31}, {129, 127}, {300, 17}, {17, 4096}, {1920, 1080},
   };
   for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
      const unsigned w = sizes[i][0], h = sizes[i][1];
      for (unsigned samples = 1; samples <= 4; samples *= 2) {
         check_layered_layout(w, h, 1, samples, 1, false);
         check_layered_layout(w, h, 6, samples, 1, false);
         check_layered_layout(w, h, 12, samples, 1, false);
      }
      const unsigned levels = util_logbase2(MAX2(w, h)) + 1;
      check_layered_layout(w, h, 12, 1, levels, false);
      check_layered_layout(w, h, 9, 1, levels, true);
   }
}

/* Exact 2x2 fast-path rounding in build_g17p_mip_shader() must match the
 * general fixed4 ties-toward-128 formula bit-for-bit across all 2x2 sums. */
static void
check_exact_2x2_rounding(void)
{
   for (uint32_t sum = 0; sum <= 4u * 255u; ++sum) {
      const uint32_t value = sum * 16384u;
      const uint32_t fixed4 = (value + 2048u) >> 12;
      const uint32_t ref_out = fixed4 >> 4;
      const uint32_t rem = fixed4 & 15u;
      const uint32_t up = rem > 8u || (rem == 8u && ref_out < 128u);
      const uint32_t want = ref_out + up;
      const uint32_t got = (sum + 2u - (sum >> 9)) >> 2;
      if (got != want) {
         EXPECT(false, "exact_2x2 sum %u: got %u, want %u", sum, got, want);
         return;
      }
   }
}

static uint32_t
compact_bits_8(uint32_t v)
{
   v &= 0x55u;
   v = (v | (v >> 1)) & 0x33u;
   return (v | (v >> 2)) & 0x0fu;
}

/* Verify that build_g17p_mip_shader()'s 16x16 Morton workgroup dispatch maps
 * 1:1 to 16x16 metadata tiles and matches AIL's 64x64 tiled element offset
 * for every pixel without spread_bits. */
static void
check_mip_16x16_tile64_morton(void)
{
   static const uint32_t dims[][2] = {
      {2560, 1440}, {1920, 1080}, {1280, 720}, {160, 90},
   };

   for (unsigned d = 0; d < sizeof(dims) / sizeof(dims[0]); ++d) {
      struct ail_layout layout = {
         .width_px = dims[d][0],
         .height_px = dims[d][1],
         .depth_px = 1,
         .sample_count_sa = 1,
         .levels = util_logbase2(MAX2(dims[d][0], dims[d][1])) + 1,
         .tiling = AIL_TILING_GPU,
         .format = PIPE_FORMAT_R8G8B8A8_UNORM,
         .compressed = true,
      };
      ail_make_miptree(&layout);

      for (uint32_t l = 1; l < layout.levels; ++l) {
         const uint32_t w = u_minify(layout.width_px, l);
         const uint32_t h = u_minify(layout.height_px, l);
         const uint32_t gx = DIV_ROUND_UP(w, 16u);
         const uint32_t gy = DIV_ROUND_UP(h, 16u);

         if (ail_is_level_logically_compressed(&layout, l)) {
            EXPECT(gx == ail_metadata_width_tl(&layout, l) &&
                      gy == ail_metadata_height_tl(&layout, l),
                   "%ux%u level %u: 16x16 groups (%ux%u) != metadata grid (%ux%u)",
                   dims[d][0], dims[d][1], l, gx, gy,
                   ail_metadata_width_tl(&layout, l),
                   ail_metadata_height_tl(&layout, l));
         }

         const struct ail_tile tile = layout.tilesize_el[l];
         if (tile.width_el != 64 || tile.height_el != 64)
            continue;

         struct hk_g17p_tile_address_config cfg;
         EXPECT(hk_g17p_tile_address_config_init(layout.stride_el[l],
                                                 tile.width_el, tile.height_el,
                                                 &cfg),
                "%ux%u level %u: tile config refused", dims[d][0], dims[d][1],
                l);

         /* Check full pixel coverage and address identity on levels up to
          * 320x180, and border/interior workgroups on larger levels. */
         const bool full_scan = w <= 320 && h <= 180;
         uint8_t *seen = full_scan ? calloc((size_t)w * h, 1) : NULL;
         for (uint32_t wy = 0; wy < gy; ++wy) {
            if (!full_scan && wy > 4 && wy + 2 < gy)
               continue;
            for (uint32_t wx = 0; wx < gx; ++wx) {
               if (!full_scan && wx > 4 && wx + 2 < gx)
                  continue;
               const uint32_t tile_idx =
                  (wy >> 2) * cfg.tiles_per_row + (wx >> 2);
               const uint32_t morton_hi =
                  ((wx & 1u) << 8) | ((wy & 1u) << 9) |
                  ((wx & 2u) << 9) | ((wy & 2u) << 10);
               for (uint32_t idx = 0; idx < 256u; ++idx) {
                  const uint32_t x = (wx << 4) | compact_bits_8(idx);
                  const uint32_t y = (wy << 4) | compact_bits_8(idx >> 1);
                  if (x >= w || y >= h)
                     continue;
                  const uint32_t got = (tile_idx << 12) | morton_hi | idx;
                  const uint32_t ref_tile =
                     (y >> 6) * cfg.tiles_per_row + (x >> 6);
                  const uint32_t ref_morton =
                     ail_space_bits(x & 63u) | (ail_space_bits(y & 63u) << 1);
                  const uint32_t want = (ref_tile << 12) | ref_morton;
                  if (got != want) {
                     EXPECT(false,
                            "%ux%u level %u (%u,%u): element %u, want %u",
                            dims[d][0], dims[d][1], l, x, y, got, want);
                     free(seen);
                     return;
                  }
                  if (seen)
                     seen[y * w + x]++;
               }
            }
         }
         if (seen) {
            for (uint32_t p = 0; p < w * h; ++p) {
               if (seen[p] != 1) {
                  EXPECT(false, "%ux%u level %u pixel %u hit %u times",
                         dims[d][0], dims[d][1], l, p, seen[p]);
                  free(seen);
                  return;
               }
            }
            free(seen);
         }
      }
   }
}

static void
check_g18g_meta_copy_admission(void)
{
   const struct hk_g17p_buffer_to_image_copy comp = {
      .chip = AGX_CHIP_G18G,
      .format = VK_FORMAT_ASTC_4x4_SRGB_BLOCK,
      .plane_format_renderable = false,
      .layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      .image_type = VK_IMAGE_TYPE_3D,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .plane_count = 1,
      .mip_levels = 1,
      .array_layers = 1,
      .image_extent = {16, 16, 4},
      .aspect = VK_IMAGE_ASPECT_COLOR_BIT,
      .mip_level = 0,
      .base_array_layer = 0,
      .layer_count = 1,
      .image_offset = {0, 0, 0},
      .copy_extent = {16, 16, 4},
   };
   struct hk_g17p_compressed_copy_plan plan;
   EXPECT(hk_g17p_compressed_buffer_to_image_copy_plan(&comp, 0, 16384, &plan),
          "G18G compressed 3D buffer-to-image plan refused");

   struct hk_g17p_buffer_to_image_copy color = comp;
   color.format = VK_FORMAT_R8G8B8A8_UNORM;
   color.plane_format_renderable = true;
   EXPECT(hk_g17p_buffer_to_image_copy_supported(&color),
          "G18G graphics buffer-to-image copy refused");

   const VkImageCopy2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .extent = {16, 16, 4},
   };
   VkImageCopy2 slice;
   EXPECT(hk_g17p_image_copy_slice(AGX_CHIP_G18G, VK_IMAGE_TYPE_3D,
                                   VK_IMAGE_TYPE_3D, 4, 4, &region, 2, &slice) &&
             slice.extent.depth == 1 && slice.srcOffset.z == 2 &&
             slice.dstOffset.z == 2,
          "G18G 3D image copy slice helper refused");
}

static void
check_resolve_slice(void)
{
   /* Multi-layer 2D array resolve on native AGX3: splits into single-layer
    * passes. The sample count comes from srcSubresource.layerCount. */
   const VkImageResolve2 region = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_RESOLVE_2,
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 3},
      .srcOffset = {0, 0, 0},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 2, 3},
      .dstOffset = {0, 0, 0},
      .extent = {64, 64, 1},
   };
   VkImageResolve2 out;

   EXPECT(hk_g17p_image_resolve_slice(AGX_CHIP_G16G, VK_IMAGE_TYPE_2D,
                                      VK_IMAGE_TYPE_2D, 3, 3, &region, 0,
                                      &out) &&
             out.srcSubresource.baseArrayLayer == 2 &&
             out.srcSubresource.layerCount == 1 &&
             out.dstSubresource.baseArrayLayer == 2 &&
             out.dstSubresource.layerCount == 1 &&
             out.extent.width == 64 && out.extent.height == 64,
          "G16G resolve slice helper refused slice 0");

   EXPECT(hk_g17p_image_resolve_slice(AGX_CHIP_G17P, VK_IMAGE_TYPE_2D,
                                      VK_IMAGE_TYPE_2D, 5, 5, &region, 2,
                                      &out) &&
             out.srcSubresource.baseArrayLayer == 4 &&
             out.srcSubresource.layerCount == 1 &&
             out.dstSubresource.baseArrayLayer == 4 &&
             out.dstSubresource.layerCount == 1,
          "G17P resolve slice helper refused slice 2");

   /* Non-native-AGX3 chips, 3D image types, mismatched layer counts and
    * out-of-range slices are refused. */
   EXPECT(!hk_g17p_image_resolve_slice(AGX_CHIP_G13G, VK_IMAGE_TYPE_2D,
                                       VK_IMAGE_TYPE_2D, 3, 3, &region, 0,
                                       &out),
          "AGX2 should refuse resolve slice helper");
   EXPECT(!hk_g17p_image_resolve_slice(AGX_CHIP_G17P, VK_IMAGE_TYPE_3D,
                                       VK_IMAGE_TYPE_2D, 3, 3, &region, 0,
                                       &out),
          "3D source should refuse resolve slice helper");
   EXPECT(!hk_g17p_image_resolve_slice(AGX_CHIP_G17P, VK_IMAGE_TYPE_2D,
                                       VK_IMAGE_TYPE_2D, 3, 4, &region, 0,
                                       &out),
          "mismatched layer counts should refuse resolve slice helper");
   EXPECT(!hk_g17p_image_resolve_slice(AGX_CHIP_G17P, VK_IMAGE_TYPE_2D,
                                       VK_IMAGE_TYPE_2D, 3, 3, &region, 3,
                                       &out),
          "slice past layer count should refuse resolve slice helper");

   /* Non-zero z offsets are refused: VkImageResolve addresses 2D arrays
    * through baseArrayLayer, not z. */
   VkImageResolve2 bad_z_src = region;
   bad_z_src.srcOffset.z = 1;
   EXPECT(!hk_g17p_image_resolve_slice(AGX_CHIP_G17P, VK_IMAGE_TYPE_2D,
                                       VK_IMAGE_TYPE_2D, 3, 3, &bad_z_src, 0,
                                       &out),
          "non-zero srcOffset.z should refuse resolve slice helper");
}

int
main(void)
{
   check_g18g_meta_copy_admission();
   check_resolve_slice();
   check_layered_layouts();
   check_mip_split();
   check_exact_2x2_rounding();
   check_mip_16x16_tile64_morton();
   for (uint32_t h = 1; h <= 70; ++h) {
      for (uint32_t w = 1; w <= 70; ++w)
         check_grid(w, h);
   }
   check_grid(240, 135);
   check_grid(135, 240);
   check_grid(4096, 1);

   static const uint32_t sizes[][2] = {
      {1920, 1080}, {3840, 2160}, {256, 256}, {129, 127}, {1000, 40},
      {40, 1000},   {300, 17},    {16, 16},   {17, 4096}, {960, 540},
   };
   for (unsigned i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i)
      check_layout(sizes[i][0], sizes[i][1]);

   if (failures) {
      fprintf(stderr, "%u failure(s)\n", failures);
      return 1;
   }
   printf("hk_g17p_metadata_grid: OK\n");
   return 0;
}
