/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Host unit tests for the full-slice linear tiled copy eligibility checks,
 * cross-checked against independent element-wise Morton addressing.
 */
#include "hk_g17p_meta_copy_core.h"
#include "layout.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned failures;

#define EXPECT(cond, ...)                                                      \
   do {                                                                        \
      if (!(cond)) {                                                           \
         fprintf(stderr, __VA_ARGS__);                                         \
         fprintf(stderr, "\n");                                                \
         failures++;                                                           \
      }                                                                        \
   } while (0)

static uint32_t
space_bits(uint32_t x)
{
   x &= 0x3fffu;
   x = (x | (x << 8)) & 0x00ff00ffu;
   x = (x | (x << 4)) & 0x0f0f0f0fu;
   x = (x | (x << 2)) & 0x33333333u;
   x = (x | (x << 1)) & 0x55555555u;
   return x;
}

/* Independent element-wise Morton offset matching libagx_copy_image. */
static uint64_t
morton_element_offset_B(const struct ail_layout *layout, uint32_t level,
                        uint32_t x, uint32_t y, uint32_t raw_block_B)
{
   const struct ail_tile tile = layout->tilesize_el[level];
   const uint32_t tiles_per_row =
      DIV_ROUND_UP(layout->stride_el[level], tile.width_el);
   const uint32_t tile_idx =
      (y / tile.height_el) * tiles_per_row + (x / tile.width_el);
   const uint32_t tile_area = tile.width_el * tile.height_el;
   const uint32_t morton =
      space_bits(x & (tile.width_el - 1)) |
      (space_bits(y & (tile.height_el - 1)) << 1);
   return ((uint64_t)tile_idx * tile_area + morton) * raw_block_B;
}

static struct hk_g17p_linear_tiled_slice_copy
make_desc_from_layouts(const struct ail_layout *src, uint32_t src_level,
                       const struct ail_layout *dst, uint32_t dst_level,
                       VkSampleCountFlagBits samples)
{
   const uint32_t src_w_el = DIV_ROUND_UP(
      u_minify(src->width_px, src_level), util_format_get_blockwidth(src->format));
   const uint32_t src_h_el = DIV_ROUND_UP(
      u_minify(src->height_px, src_level),
      util_format_get_blockheight(src->format));
   const uint32_t dst_w_el = DIV_ROUND_UP(
      u_minify(dst->width_px, dst_level), util_format_get_blockwidth(dst->format));
   const uint32_t dst_h_el = DIV_ROUND_UP(
      u_minify(dst->height_px, dst_level),
      util_format_get_blockheight(dst->format));
   struct hk_g17p_tile_address_config src_cfg = {0};
   struct hk_g17p_tile_address_config dst_cfg = {0};
   hk_g17p_tile_address_config_init(src->stride_el[src_level],
                                    src->tilesize_el[src_level].width_el,
                                    src->tilesize_el[src_level].height_el,
                                    &src_cfg);
   hk_g17p_tile_address_config_init(dst->stride_el[dst_level],
                                    dst->tilesize_el[dst_level].width_el,
                                    dst->tilesize_el[dst_level].height_el,
                                    &dst_cfg);

   return (struct hk_g17p_linear_tiled_slice_copy){
      .src_tiled = src->tiling == AIL_TILING_GPU,
      .dst_tiled = dst->tiling == AIL_TILING_GPU,
      .src_compressed = src->compressed,
      .dst_compressed = dst->compressed,
      .src_type = VK_IMAGE_TYPE_2D,
      .dst_type = VK_IMAGE_TYPE_2D,
      .src_samples = samples,
      .dst_samples = samples,
      .src_w_el = src_w_el,
      .src_h_el = src_h_el,
      .dst_w_el = dst_w_el,
      .dst_h_el = dst_h_el,
      .src_tile_w_el = src->tilesize_el[src_level].width_el,
      .src_tile_h_el = src->tilesize_el[src_level].height_el,
      .dst_tile_w_el = dst->tilesize_el[dst_level].width_el,
      .dst_tile_h_el = dst->tilesize_el[dst_level].height_el,
      .src_tiles_per_row = src_cfg.tiles_per_row,
      .dst_tiles_per_row = dst_cfg.tiles_per_row,
      .raw_block_size_B = ail_get_blocksize_B(src) * samples,
      .src_level = src_level,
      .src_levels = src->levels,
      .dst_level = dst_level,
      .dst_levels = dst->levels,
      .src_level_size_B = ail_get_level_size_B(src, src_level),
      .dst_level_size_B = ail_get_level_size_B(dst, dst_level),
      .max_groups_x = 0x7fffffff,
      .plan =
         {
            .block_size_B = ail_get_blocksize_B(src),
            .width_el = src_w_el,
            .height_el = src_h_el,
         },
   };
}

/* Verify that whenever the fast path accepts a pair, every element (x, y) maps
 * to the exact same byte offset in both images and stays inside the slice span
 * and both levels' allocated bounds. */
static void
verify_pair_against_morton(const struct ail_layout *src, uint32_t src_level,
                           const struct ail_layout *dst, uint32_t dst_level,
                           VkSampleCountFlagBits samples, const char *label)
{
   struct hk_g17p_linear_tiled_slice_copy d =
      make_desc_from_layouts(src, src_level, dst, dst_level, samples);
   const uint64_t span_B = hk_g17p_linear_tiled_slice_bytes(&d);
   if (!span_B)
      return;

   EXPECT((span_B & 15) == 0, "%s: span %llu not 16B aligned", label,
          (unsigned long long)span_B);
   EXPECT(span_B <= ail_get_level_size_B(src, src_level),
          "%s: span exceeds src level", label);
   EXPECT(span_B <= ail_get_level_size_B(dst, dst_level),
          "%s: span exceeds dst level", label);

   for (uint32_t layer = 0; layer < src->depth_px && layer < dst->depth_px;
        ++layer) {
      const uint64_t src_off = ail_get_layer_level_B(src, layer, src_level);
      const uint64_t dst_off = ail_get_layer_level_B(dst, layer, dst_level);
      EXPECT(src_off + span_B <= src->size_B, "%s: src layer %u overflow",
             label, layer);
      EXPECT(dst_off + span_B <= dst->size_B, "%s: dst layer %u overflow",
             label, layer);
   }

   for (uint32_t y = 0; y < d.plan.height_el; ++y) {
      for (uint32_t x = 0; x < d.plan.width_el; ++x) {
         const uint64_t so =
            morton_element_offset_B(src, src_level, x, y, d.raw_block_size_B);
         const uint64_t doff =
            morton_element_offset_B(dst, dst_level, x, y, d.raw_block_size_B);
         if (so != doff || so + d.raw_block_size_B > span_B) {
            EXPECT(false,
                   "%s (%u,%u): src_off=%llu dst_off=%llu span=%llu", label, x,
                   y, (unsigned long long)so, (unsigned long long)doff,
                   (unsigned long long)span_B);
            return;
         }
      }
   }
}

static void
test_cross_format_and_pot_morton(void)
{
   const struct {
      uint32_t src_w, src_h;
      enum pipe_format src_fmt;
      uint32_t dst_w, dst_h;
      enum pipe_format dst_fmt;
      VkSampleCountFlagBits samples;
      bool expect_level0_accept;
      const char *name;
   } cases[] = {
      /* Same-format POT and non-POT dimensions, including MSAA 2x/4x */
      {128, 128, PIPE_FORMAT_R8G8B8A8_UNORM, 128, 128,
       PIPE_FORMAT_R32_UINT, VK_SAMPLE_COUNT_1_BIT, true, "rgba8-r32-128"},
      {100, 16, PIPE_FORMAT_R8G8B8A8_UNORM, 100, 16,
       PIPE_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT, true, "pot-rect-100x16"},
      {97, 53, PIPE_FORMAT_R16G16B16A16_FLOAT, 97, 53,
       PIPE_FORMAT_R32G32_UINT, VK_SAMPLE_COUNT_1_BIT, true, "rgba16f-rg32-odd"},
      {64, 48, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 48,
       PIPE_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_2_BIT, true, "msaa2x"},
      {64, 48, PIPE_FORMAT_R8G8B8A8_UNORM, 64, 48,
       PIPE_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_4_BIT, true, "msaa4x"},
      {33, 19, PIPE_FORMAT_R8_UNORM, 33, 19,
       PIPE_FORMAT_R8_UNORM, VK_SAMPLE_COUNT_1_BIT, true, "r8-odd"},
      {33, 19, PIPE_FORMAT_R16_UINT, 33, 19,
       PIPE_FORMAT_R16_UINT, VK_SAMPLE_COUNT_1_BIT, true, "r16-odd"},
      /* Cross-format BC <-> RGBA32/RG32 (exact multiple and padded stride) */
      {256, 128, PIPE_FORMAT_BPTC_RGBA_UNORM, 64, 32,
       PIPE_FORMAT_R32G32B32A32_UINT, VK_SAMPLE_COUNT_1_BIT, true, "bc7-rgba32-pot"},
      {260, 132, PIPE_FORMAT_BPTC_RGBA_UNORM, 65, 33,
       PIPE_FORMAT_R32G32B32A32_UINT, VK_SAMPLE_COUNT_1_BIT, true, "bc7-rgba32-odd"},
      {256, 128, PIPE_FORMAT_DXT1_RGBA, 64, 32,
       PIPE_FORMAT_R32G32_UINT, VK_SAMPLE_COUNT_1_BIT, true, "bc1-rg32-pot"},
      {260, 132, PIPE_FORMAT_DXT1_RGBA, 65, 33,
       PIPE_FORMAT_R32G32_UINT, VK_SAMPLE_COUNT_1_BIT, true, "bc1-rg32-odd"},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      struct ail_layout src = {
         .width_px = cases[i].src_w,
         .height_px = cases[i].src_h,
         .depth_px = 3,
         .sample_count_sa = cases[i].samples,
         .levels = cases[i].samples == 1 ? 6 : 1,
         .tiling = AIL_TILING_GPU,
         .format = cases[i].src_fmt,
         .policy = AIL_LAYOUT_POLICY_M5,
      };
      struct ail_layout dst = {
         .width_px = cases[i].dst_w,
         .height_px = cases[i].dst_h,
         .depth_px = 3,
         .sample_count_sa = cases[i].samples,
         .levels = cases[i].samples == 1 ? 6 : 1,
         .tiling = AIL_TILING_GPU,
         .format = cases[i].dst_fmt,
         .policy = AIL_LAYOUT_POLICY_M5,
      };
      ail_make_miptree(&src);
      ail_make_miptree(&dst);

      if (cases[i].expect_level0_accept) {
         struct hk_g17p_linear_tiled_slice_copy d0 =
            make_desc_from_layouts(&src, 0, &dst, 0, cases[i].samples);
         EXPECT(hk_g17p_linear_tiled_slice_bytes(&d0) > 0,
                "%s: expected level 0 acceptance", cases[i].name);
      }

      const uint32_t nlevels = MIN2(src.levels, dst.levels);
      for (uint32_t l = 0; l < nlevels; ++l) {
         verify_pair_against_morton(&src, l, &dst, l, cases[i].samples,
                                    cases[i].name);
         verify_pair_against_morton(&dst, l, &src, l, cases[i].samples,
                                    cases[i].name);
      }
   }
}

static void
test_rejections(void)
{
   struct ail_layout a = {
      .width_px = 512,
      .height_px = 512,
      .depth_px = 2,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .policy = AIL_LAYOUT_POLICY_M5,
   };
   ail_make_miptree(&a);

   const struct hk_g17p_linear_tiled_slice_copy base =
      make_desc_from_layouts(&a, 0, &a, 0, VK_SAMPLE_COUNT_1_BIT);
   const uint64_t base_B = hk_g17p_linear_tiled_slice_bytes(&base);
   EXPECT(base_B == 512u * 512u * 4u, "base rejected");

   struct hk_g17p_linear_tiled_slice_copy d;

   d = base; d.src_tiled = false;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "linear src accepted");

   d = base; d.dst_compressed = true;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "compressed dst accepted");

   d = base; d.src_sparse = true;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "sparse src accepted");

   d = base; d.src_type = VK_IMAGE_TYPE_3D;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "3D src accepted");

   d = base; d.dst_samples = VK_SAMPLE_COUNT_2_BIT;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "sample mismatch accepted");

   d = base; d.dst_tile_w_el /= 2;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "tile mismatch accepted");

   d = base; d.dst_tiles_per_row += 1;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "stride mismatch accepted");

   d = base; d.plan.src_x_el = 1;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "offset copy accepted");

   d = base; d.plan.width_el -= 1;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "partial copy accepted");

   d = base; d.src_level_size_B = base_B - 16;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "level overflow accepted");

   d = base; d.max_groups_x = 1;
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "grid overflow accepted");

   /* 1x1 R8 slice is 1 byte (not 16-byte aligned), must fall back. */
   struct ail_layout tiny = {
      .width_px = 1,
      .height_px = 1,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8_UNORM,
      .policy = AIL_LAYOUT_POLICY_M5,
   };
   ail_make_miptree(&tiny);
   d = make_desc_from_layouts(&tiny, 0, &tiny, 0, VK_SAMPLE_COUNT_1_BIT);
   EXPECT(hk_g17p_linear_tiled_slice_bytes(&d) == 0, "1x1 R8 slice accepted");
}

static void
test_address_validation(void)
{
   const uint64_t sz = 0x100000;
   /* Distinct aligned non-overlapping slices pass. */
   EXPECT(hk_g17p_linear_tiled_slice_addrs_valid(
             0x200000, 0, sz, 0x100000, 0, sz, sz),
          "valid non-overlapping copy rejected");
   /* Adjacent slices pass. */
   EXPECT(hk_g17p_linear_tiled_slice_addrs_valid(
             0x100000, 0, sz, 0x200000, 0, sz, sz),
          "adjacent copy rejected");
   /* Same address fails. */
   EXPECT(!hk_g17p_linear_tiled_slice_addrs_valid(
             0x100000, 0, sz, 0x100000, 0, sz, sz),
          "self-copy accepted");
   /* Partial VA overlap (aliased VkImages) fails. */
   EXPECT(!hk_g17p_linear_tiled_slice_addrs_valid(
             0x180000, 0, sz, 0x100000, 0, sz, sz),
          "overlapping aliased images accepted");
   /* Unaligned dst or src address fails. */
   EXPECT(!hk_g17p_linear_tiled_slice_addrs_valid(
             0x200004, 0, sz, 0x100000, 0, sz, sz),
          "unaligned dst accepted");
   EXPECT(!hk_g17p_linear_tiled_slice_addrs_valid(
             0x200000, 0, sz, 0x100004, 0, sz, sz),
          "unaligned src accepted");
   /* Layout bounds check fails on dst and src. */
   EXPECT(!hk_g17p_linear_tiled_slice_addrs_valid(
             0x200000, 0x10, sz, 0x100000, 0, sz, sz),
          "out-of-bounds dst accepted");
   EXPECT(!hk_g17p_linear_tiled_slice_addrs_valid(
             0x200000, 0, sz, 0x100000, 0x10, sz, sz),
          "out-of-bounds src accepted");
}

int
main(void)
{
   test_cross_format_and_pot_morton();
   test_rejections();
   test_address_validation();
   return failures ? 1 : 0;
}
