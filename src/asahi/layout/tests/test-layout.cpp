/*
 * Copyright 2022 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include <gtest/gtest.h>
#include "layout.h"

TEST(Layout, M5DepthStencilArrayPageStride)
{
   for (auto format : {PIPE_FORMAT_Z16_UNORM, PIPE_FORMAT_Z32_FLOAT,
                       PIPE_FORMAT_S8_UINT}) {
      struct ail_layout layout = {};
      layout.policy = AIL_LAYOUT_POLICY_M5;
      layout.tiling = AIL_TILING_TWIDDLED;
      layout.format = format;
      layout.width_px = 16;
      layout.height_px = 16;
      layout.depth_px = 4;
      layout.levels = 1;
      layout.sample_count_sa = 1;
      layout.renderable = true;
      ail_make_miptree(&layout);
      EXPECT_TRUE(layout.page_aligned_layers);
      EXPECT_EQ(layout.layer_stride_B, AIL_PAGESIZE);
      EXPECT_EQ(layout.size_B, 4u * AIL_PAGESIZE);
   }
}

TEST(Layout, M5SparseMipTailPageStride)
{
   struct ail_layout ordinary = {};
   ordinary.policy = AIL_LAYOUT_POLICY_M5;
   ordinary.tiling = AIL_TILING_GPU;
   ordinary.format = PIPE_FORMAT_R32_UINT;
   ordinary.width_px = 128;
   ordinary.height_px = 64;
   ordinary.depth_px = 1;
   ordinary.levels = 8;
   ordinary.sample_count_sa = 1;
   struct ail_layout sparse = ordinary;
   sparse.sparse_residency = true;
   ail_make_miptree(&ordinary);
   ail_make_miptree(&sparse);

   EXPECT_NE(ordinary.layer_stride_B % AIL_PAGESIZE, 0u);
   EXPECT_EQ(sparse.layer_stride_B % AIL_PAGESIZE, 0u);
   EXPECT_TRUE(sparse.page_aligned_layers);
   const unsigned tail = sparse.mip_tail_first_lod;
   EXPECT_LT(sparse.level_offsets_B[tail] / AIL_PAGESIZE,
             sparse.layer_stride_B / AIL_PAGESIZE);
   EXPECT_EQ(sparse.size_B, sparse.layer_stride_B);
   for (unsigned l = 0; l <= sparse.levels; ++l)
      EXPECT_EQ(ordinary.level_offsets_B[l], sparse.level_offsets_B[l]);
}

static struct ail_layout
m5_array(enum pipe_format format, unsigned layers, unsigned levels,
         bool writeable)
{
   struct ail_layout layout = {};
   layout.policy = AIL_LAYOUT_POLICY_M5;
   layout.tiling = AIL_TILING_GPU;
   layout.format = format;
   layout.width_px = 16;
   layout.height_px = 16;
   layout.depth_px = layers;
   layout.levels = levels;
   layout.sample_count_sa = 1;
   layout.writeable_image = writeable;
   return layout;
}

/* A layered store to a mipmapped array is a 3D store whose slice stride
 * follows a deeper mip chain; page-aligned layers hold it (layout.c). */
TEST(Layout, M5WriteableMipmappedArrayPageStride)
{
   struct ail_layout layout = m5_array(PIPE_FORMAT_R32_UINT, 6, 5, true);
   ail_make_miptree(&layout);
   EXPECT_TRUE(layout.page_aligned_layers);
   EXPECT_EQ(layout.layer_stride_B, AIL_PAGESIZE);
   EXPECT_EQ(layout.size_B, 6u * AIL_PAGESIZE);
}

TEST(Layout, M5CompactArrayStrides)
{
   /* Not writeable, or a single level: the compact stride stays. */
   struct ail_layout ro = m5_array(PIPE_FORMAT_R32_UINT, 6, 5, false);
   struct ail_layout one = m5_array(PIPE_FORMAT_R32_UINT, 6, 1, true);
   ail_make_miptree(&ro);
   ail_make_miptree(&one);
   EXPECT_FALSE(ro.page_aligned_layers);
   EXPECT_LT(ro.layer_stride_B, AIL_PAGESIZE);
   EXPECT_FALSE(one.page_aligned_layers);
   EXPECT_LT(one.layer_stride_B, AIL_PAGESIZE);
}

TEST(Layout, M5G18GDepthArrayStrides)
{
   /* G18G keeps compact depth layers, except for writeable mipmapped
    * arrays. */
   struct ail_layout flat = m5_array(PIPE_FORMAT_Z32_FLOAT, 4, 1, true);
   flat.renderable = true;
   flat.g18g = true;
   struct ail_layout mip = flat;
   mip.levels = 5;
   ail_make_miptree(&flat);
   ail_make_miptree(&mip);
   EXPECT_FALSE(flat.page_aligned_layers);
   EXPECT_LT(flat.layer_stride_B, AIL_PAGESIZE);
   EXPECT_TRUE(mip.page_aligned_layers);
   EXPECT_EQ(mip.layer_stride_B, AIL_PAGESIZE);
}

TEST(Layout, M5EightSampleGeometry)
{
   EXPECT_EQ(ail_effective_width_sa(37, 8), 74u);
   EXPECT_EQ(ail_effective_height_sa(19, 8), 76u);

   struct ail_tile rgba16f = ail_get_max_tile_size(128);
   EXPECT_EQ(rgba16f.width_el, 16u);
   EXPECT_EQ(rgba16f.height_el, 8u);

   struct ail_tile rgba32f = ail_get_max_tile_size(256);
   EXPECT_EQ(rgba32f.width_el, 8u);
   EXPECT_EQ(rgba32f.height_el, 8u);
}

TEST(Cubemap, Nonmipmapped)
{
   struct ail_layout layout = {
      .width_px = 512,
      .height_px = 512,
      .depth_px = 6,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.layer_stride_B, ALIGN_POT(512 * 512 * 4, 0x4000));
   EXPECT_EQ(layout.size_B, ALIGN_POT(512 * 512 * 4 * 6, 0x4000));
}

TEST(Cubemap, RoundsToOnePage)
{
   struct ail_layout layout = {
      .width_px = 63,
      .height_px = 63,
      .depth_px = 6,
      .sample_count_sa = 1,
      .levels = 6,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R32_FLOAT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.level_offsets_B[0], 0);
   EXPECT_EQ(layout.level_offsets_B[1], 0x4000);
   EXPECT_EQ(layout.level_offsets_B[2], 0x5000);
   EXPECT_EQ(layout.level_offsets_B[3], 0x5400);
   EXPECT_EQ(layout.level_offsets_B[4], 0x5500);
   EXPECT_TRUE(layout.page_aligned_layers);
   EXPECT_EQ(layout.layer_stride_B, 0x8000);
   EXPECT_EQ(layout.size_B, 0x30000);
}

TEST(Linear, SmokeTestBuffer)
{
   struct ail_layout layout = {
      .width_px = 81946,
      .height_px = 1,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_LINEAR,
      .format = PIPE_FORMAT_R8_UINT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.size_B, ALIGN_POT(81946, AIL_CACHELINE));
}

TEST(Miptree, AllMipLevels)
{
   struct ail_layout layout = {
      .width_px = 1024,
      .height_px = 1024,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 11,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UINT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.size_B, 0x555680);
   EXPECT_EQ(layout.sparse_folios_per_layer, 2);
   EXPECT_EQ(layout.sparse_table_size_B, 0x1000);
}

TEST(Miptree, SomeMipLevels)
{
   struct ail_layout layout = {
      .width_px = 1024,
      .height_px = 1024,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 4,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R8G8B8A8_UINT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.size_B, 0x555680);
   EXPECT_EQ(layout.sparse_folios_per_layer, 2);
   EXPECT_EQ(layout.sparse_table_size_B, 0x1000);
}

TEST(Miptree, SmallPartialMiptree2DArray)
{
   struct ail_layout layout = {
      .width_px = 32,
      .height_px = 16,
      .depth_px = 64,
      .sample_count_sa = 1,
      .levels = 4,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R32_FLOAT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.layer_stride_B, 0xc00);
   EXPECT_EQ(layout.size_B, 0x30000);
   EXPECT_EQ(layout.sparse_folios_per_layer, 1);
   EXPECT_EQ(layout.sparse_table_size_B, 0x20000);
}

TEST(Miptree, SmallPartialMiptree3D)
{
   struct ail_layout layout = {
      .width_px = 32,
      .height_px = 16,
      .depth_px = 64,
      .sample_count_sa = 1,
      .levels = 4,
      .mipmapped_z = true,
      .tiling = AIL_TILING_GPU,
      .format = PIPE_FORMAT_R32_FLOAT,
   };

   ail_make_miptree(&layout);

   EXPECT_EQ(layout.layer_stride_B, 0xc80);
   EXPECT_EQ(layout.size_B, 0x32000);
   EXPECT_EQ(layout.sparse_folios_per_layer, 1);
   EXPECT_EQ(layout.sparse_table_size_B, 0x20000);
}

TEST(Linear, Packed422Rows)
{
   struct ail_layout layout = {
      .width_px = 130,
      .height_px = 3,
      .depth_px = 2,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_LINEAR,
      .format = PIPE_FORMAT_R16G16_R16B16_422_UNORM,
   };
   ail_make_miptree(&layout);
   /* 65 eight-byte words per row, not 130 full RGBA texels. */
   EXPECT_EQ(layout.linear_stride_B, ALIGN_POT(520, AIL_CACHELINE));
   EXPECT_EQ(layout.layer_stride_B, layout.linear_stride_B * 3);
   EXPECT_EQ(layout.size_B, layout.layer_stride_B * 2);

   layout.linear_stride_B = 1024;
   ail_make_miptree(&layout);
   EXPECT_EQ(layout.layer_stride_B, 3072u);
   EXPECT_EQ(layout.size_B, 6144u);
}

/*
 * Compressed Interchange (DRM_FORMAT_MOD_APPLE_INTERCHANGE; gated per chip by
 * AGX_CAP_GEN9_2D_COMPRESSION).  The macroblock row pitch, data size
 * and header offset are the contract with appledrm, which lays out
 * pitch == ceil(w/16)*1024, header offset == data size and a buffer
 * of at least data + (8 << (log2ceil(tiles_x) + log2ceil(tiles_y))) bytes.
 * appledrm derives that layout itself and ignores the framebuffer pitch, but
 * DRM core checks (height - 1) * pitch + width * cpp against the buffer, so
 * the pitch handed to the window system is per pixel row: a sixteenth of the
 * macroblock row.
 */
static void
check_interchange(unsigned w, unsigned h)
{
   struct ail_layout layout = {
      .width_px = w,
      .height_px = h,
      .depth_px = 1,
      .sample_count_sa = 1,
      .levels = 1,
      .tiling = AIL_TILING_INTERCHANGE,
      .compressed = true,
      .format = PIPE_FORMAT_B8G8R8A8_UNORM,
      .renderable = true,
   };

   ail_make_miptree(&layout);

   const unsigned tx = DIV_ROUND_UP(w, 16), ty = DIV_ROUND_UP(h, 16);
   const uint64_t pitch = tx * 1024;
   const uint64_t data = ty * pitch;
   const uint64_t kernel_header =
      8ull << (util_logbase2_ceil(tx) + util_logbase2_ceil(ty));

   EXPECT_EQ(ail_get_interchange_stride_B(&layout), pitch);
   EXPECT_EQ(ail_get_wsi_stride_B(&layout, 0), pitch / 16);
   EXPECT_LE((h - 1) * (pitch / 16) + w * 4, layout.size_B);
   EXPECT_EQ(layout.level_offsets_B[0], 0u);
   EXPECT_EQ(ail_get_level_size_B(&layout, 0), data);
   EXPECT_EQ(layout.layer_stride_B, data);
   EXPECT_EQ(layout.metadata_offset_B, data);
   EXPECT_EQ(layout.tilesize_el[0].width_el, 16u);
   EXPECT_EQ(layout.tilesize_el[0].height_el, 16u);
   EXPECT_EQ(layout.stride_el[0], tx * 16);
   EXPECT_FALSE(layout.page_aligned_layers);
   EXPECT_TRUE(ail_is_level_logically_compressed(&layout, 0));
   EXPECT_FALSE(ail_is_level_twiddled_uncompressed(&layout, 0));
   EXPECT_EQ(layout.metadata_offset_B % 128, 0u);
   /* Mesa's metadata covers what DCP reads. */
   EXPECT_GE(layout.compression_layer_stride_B, kernel_header);
   EXPECT_EQ(layout.size_B, ALIGN_POT(data + layout.compression_layer_stride_B, 128));
   EXPECT_GE(layout.size_B, data + kernel_header);
}

TEST(Interchange, Neo2408x1506)
{
   check_interchange(2408, 1506);
}

TEST(Interchange, OddSizes)
{
   check_interchange(256, 128);
   check_interchange(33, 17);
   check_interchange(1920, 1080);
   check_interchange(16384, 16);
}

TEST(Interchange, Modifier)
{
   EXPECT_EQ(DRM_FORMAT_MOD_APPLE_GPU_TILED, 0x0c00000000000001ull);
   EXPECT_EQ(DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED, 0x0c00000000000002ull);
   EXPECT_EQ(DRM_FORMAT_MOD_APPLE_INTERCHANGE, 0x0c00000000000003ull);
   EXPECT_EQ(ail_drm_modifier_to_tiling(DRM_FORMAT_MOD_APPLE_INTERCHANGE),
             AIL_TILING_INTERCHANGE);
   EXPECT_TRUE(ail_is_drm_modifier_compressed(DRM_FORMAT_MOD_APPLE_INTERCHANGE));
}

TEST(G17PLayout, CompressedMipmapped2DArray)
{
   struct ail_layout layout = {
      .width_px = 128,
      .height_px = 128,
      .depth_px = 4,
      .sample_count_sa = 1,
      .policy = AIL_LAYOUT_POLICY_LEGACY,
      .levels = 8,
      .tiling = AIL_TILING_GPU,
      .compressed = true,
      .format = PIPE_FORMAT_R8G8B8A8_UNORM,
      .renderable = true,
   };
   ail_make_miptree(&layout);

   EXPECT_TRUE(layout.page_aligned_layers);
   EXPECT_EQ(layout.layer_stride_B % AIL_PAGESIZE, 0u);
   EXPECT_EQ(layout.metadata_offset_B,
             (uint64_t)layout.layer_stride_B * layout.depth_px);
   EXPECT_GT(layout.compression_layer_stride_B, 0u);
   EXPECT_EQ(layout.compression_layer_stride_B % AIL_CACHELINE, 0u);
   EXPECT_EQ(layout.size_B,
             layout.metadata_offset_B +
                (uint64_t)layout.compression_layer_stride_B * layout.depth_px);

   /* Levels 0..3 (128x128 down to 16x16) are compressed; tail levels 4..7
    * (< 16x16) are logically uncompressed. */
   for (unsigned l = 0; l < layout.levels; ++l) {
      const bool expect_comp = l < 4;
      EXPECT_EQ(ail_is_level_logically_compressed(&layout, l), expect_comp);
      if (expect_comp && l > 0) {
         EXPECT_GT(layout.level_offsets_compressed_B[l],
                   layout.level_offsets_compressed_B[l - 1]);
      }
      if (expect_comp) {
         EXPECT_EQ(layout.level_offsets_compressed_B[l] % AIL_CACHELINE, 0u);
         for (unsigned layer = 0; layer < layout.depth_px; ++layer) {
            const uint64_t meta =
               layout.metadata_offset_B +
               (uint64_t)layer * layout.compression_layer_stride_B +
               layout.level_offsets_compressed_B[l];
            EXPECT_LT(meta, layout.size_B);
         }
      }
      for (unsigned layer = 0; layer < layout.depth_px; ++layer) {
         const uint64_t body = ail_get_layer_level_B(&layout, layer, l);
         EXPECT_LT(body, layout.metadata_offset_B);
      }
   }
}
