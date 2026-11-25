/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Verify the AGX3 Gen10 compressed buffer-to-image compute upload format gate,
 * 16x16-aligned 32x16 dispatch grid, and XOR-swizzled shared-memory Morton
 * mapping (coverage, bit-exact payload transfer, single-block-per-warp
 * ownership, and zero bank conflicts).
 */
#include "hk_g17p_meta_copy_core.h"

#include <stdbool.h>
#include <stdint.h>
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

static uint32_t
compact_bits_8(uint32_t v)
{
   v &= 0x55u;
   v = (v | (v >> 1)) & 0x33u;
   return (v | (v >> 2)) & 0x0fu;
}

static void
check_formats(void)
{
   static const VkFormat admitted[] = {
      VK_FORMAT_R16_UINT,
      VK_FORMAT_R8G8B8A8_UINT,
      VK_FORMAT_R16G16_UINT,
      VK_FORMAT_R32_UINT,
      VK_FORMAT_R16G16B16A16_UINT,
      VK_FORMAT_R32G32_UINT,
      VK_FORMAT_R32G32B32A32_UINT,
   };
   for (unsigned i = 0; i < sizeof(admitted) / sizeof(admitted[0]); ++i) {
      EXPECT(hk_agx3_compressed_buf2img_view_format_supported(admitted[i]),
             "admitted view format %u rejected", admitted[i]);
   }

   static const VkFormat rejected[] = {
      VK_FORMAT_UNDEFINED,
      VK_FORMAT_R8_UINT,
      VK_FORMAT_R8G8_UINT,
      VK_FORMAT_R5G6B5_UNORM_PACK16,
      VK_FORMAT_A2B10G10R10_UINT_PACK32,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_FORMAT_R32_SFLOAT,
   };
   for (unsigned i = 0; i < sizeof(rejected) / sizeof(rejected[0]); ++i) {
      EXPECT(!hk_agx3_compressed_buf2img_view_format_supported(rejected[i]),
             "unsupported view format %u admitted", rejected[i]);
   }
}

static void
check_grid_bounds(void)
{
   struct hk_agx3_compressed_buf2img_grid grid;

   EXPECT(!hk_agx3_compressed_buf2img_grid_init(0, 0, 0, 16, 65535, 65535, &grid),
          "zero width must fail");
   EXPECT(!hk_agx3_compressed_buf2img_grid_init(0, 0, 16, 0, 65535, 65535, &grid),
          "zero height must fail");
   EXPECT(!hk_agx3_compressed_buf2img_grid_init(UINT32_MAX, 0, 2, 1, 65535, 65535,
                                                &grid),
          "x overflow must fail");
   EXPECT(!hk_agx3_compressed_buf2img_grid_init(0, UINT32_MAX, 1, 2, 65535, 65535,
                                                &grid),
          "y overflow must fail");
   EXPECT(!hk_agx3_compressed_buf2img_grid_init(0, 0, 4096, 4096, 127, 256,
                                                &grid),
          "exceeding max_groups_x must fail");
   EXPECT(!hk_agx3_compressed_buf2img_grid_init(0, 0, 4096, 4096, 128, 255,
                                                &grid),
          "exceeding max_groups_y must fail");

   EXPECT(hk_agx3_compressed_buf2img_grid_init(0, 0, 4096, 4096, 65535, 65535,
                                               &grid),
          "4096x4096 failed");
   EXPECT(grid.tile_x0 == 0 && grid.tile_y0 == 0 && grid.groups_x == 128 &&
             grid.groups_y == 256,
          "4096x4096 grid mismatch: (%u,%u) %ux%u", grid.tile_x0, grid.tile_y0,
          grid.groups_x, grid.groups_y);
}

static void
simulate_rect(uint32_t img_w, uint32_t img_h, uint32_t x0, uint32_t y0,
              uint32_t width, uint32_t height, uint32_t row_stride_px)
{
   struct hk_agx3_compressed_buf2img_grid grid;
   EXPECT(hk_agx3_compressed_buf2img_grid_init(x0, y0, width, height, 65535,
                                               65535, &grid),
          "rect (%u,%u)+%ux%u failed grid init", x0, y0, width, height);
   EXPECT((grid.tile_x0 & 15u) == 0 && (grid.tile_y0 & 15u) == 0,
          "tile origin (%u,%u) not 16-aligned", grid.tile_x0, grid.tile_y0);

   uint32_t *src = calloc((size_t)row_stride_px * height, sizeof(uint32_t));
   uint32_t *dst = calloc((size_t)img_w * img_h, sizeof(uint32_t));
   uint8_t *hits = calloc((size_t)img_w * img_h, 1);

   for (uint32_t ry = 0; ry < height; ++ry) {
      for (uint32_t rx = 0; rx < width; ++rx)
         src[ry * row_stride_px + rx] = 0xA5000000u | (ry << 12) | rx;
   }

   for (uint32_t wy = 0; wy < grid.groups_y; ++wy) {
      for (uint32_t wx = 0; wx < grid.groups_x; ++wx) {
         const uint32_t wg_x0 = grid.tile_x0 + (wx << 5);
         const uint32_t wg_y0 = grid.tile_y0 + (wy << 4);
         const uint32_t rx0 = wg_x0 - x0;
         const uint32_t ry0 = wg_y0 - y0;
         const bool full_wg = (rx0 <= width && (width - rx0) >= 32u) &&
                              (ry0 <= height && (height - ry0) >= 16u);

         if (full_wg) {
            uint32_t tile[128][4];
            memset(tile, 0, sizeof(tile));

            /* Store phase: each 128-bit vector store across 32 lanes writes
             * 512 B (4 x 128 B across 32 x 4 B banks). Verify each 8-lane
             * quarter-warp hits all 32 banks once with zero conflicts. */
            for (unsigned pass = 0; pass < 2; ++pass) {
               for (unsigned warp = 0; warp < 2; ++warp) {
                  for (unsigned quarter = 0; quarter < 4; ++quarter) {
                     uint8_t bank_used[32] = {0};
                     for (unsigned q_lane = 0; q_lane < 8; ++q_lane) {
                        const unsigned lane = (quarter << 3) | q_lane;
                        const unsigned lid = (warp << 5) | lane;
                        const unsigned c = lid & 7u;
                        const unsigned r = lid >> 3;
                        const unsigned swiz =
                           ((r << 3) | (c ^ ((r & 3u) << 1))) + pass * 64u;
                        for (unsigned comp = 0; comp < 4; ++comp) {
                           const unsigned bank = ((swiz << 2) | comp) & 31u;
                           if (bank_used[bank]) {
                              EXPECT(false,
                                     "store bank conflict pass=%u warp=%u q=%u",
                                     pass, warp, quarter);
                              goto done;
                           }
                           bank_used[bank] = 1;
                        }
                     }
                  }
               }
               for (unsigned lid = 0; lid < 64; ++lid) {
                  const unsigned c = lid & 7u;
                  const unsigned r = lid >> 3;
                  const unsigned swiz =
                     ((r << 3) | (c ^ ((r & 3u) << 1))) + pass * 64u;
                  const uint32_t src_row = ry0 + r + pass * 8u;
                  const uint32_t src_col = rx0 + (c << 2);
                  for (unsigned comp = 0; comp < 4; ++comp)
                     tile[swiz][comp] =
                        src[src_row * row_stride_px + src_col + comp];
               }
            }

            /* Load + Morton store phase: verify 0 bank conflicts and single
             * 16x16 compression block per warp. */
            for (unsigned t = 0; t < 8; ++t) {
               const unsigned tx = ((t >> 1) & 1u) * 8u;
               const unsigned ty = ((t & 1u) * 4u) + ((t >> 2) * 8u);
               for (unsigned warp = 0; warp < 2; ++warp) {
                  uint8_t bank_used[32] = {0};
                  uint32_t block_x = UINT32_MAX, block_y = UINT32_MAX;
                  for (unsigned lane = 0; lane < 32; ++lane) {
                     const unsigned mx = compact_bits_8(lane);
                     const unsigned my = compact_bits_8(lane >> 1);
                     const unsigned lx = (warp << 4) + tx + mx;
                     const unsigned ly = ty + my;
                     const unsigned swiz_lx = lx ^ (my << 3);
                     const unsigned vec_idx = (ly << 3) | (swiz_lx >> 2);
                     const unsigned comp_idx = mx & 3u;
                     const unsigned bank = ((vec_idx << 2) | comp_idx) & 31u;
                     if (bank_used[bank]) {
                        EXPECT(false, "load bank conflict t=%u warp=%u", t,
                               warp);
                        goto done;
                     }
                     bank_used[bank] = 1;

                     const uint32_t px = wg_x0 + lx;
                     const uint32_t py = wg_y0 + ly;
                     if (block_x == UINT32_MAX) {
                        block_x = px >> 4;
                        block_y = py >> 4;
                     } else {
                        EXPECT((px >> 4) == block_x && (py >> 4) == block_y,
                               "warp crossed 16x16 compression block");
                     }
                     dst[py * img_w + px] = tile[vec_idx][comp_idx];
                     hits[py * img_w + px]++;
                  }
               }
            }
         } else {
            for (unsigned t = 0; t < 8; ++t) {
               const unsigned tx = ((t >> 1) & 1u) * 8u;
               const unsigned ty = ((t & 1u) * 4u) + ((t >> 2) * 8u);
               for (unsigned warp = 0; warp < 2; ++warp) {
                  uint32_t block_x = UINT32_MAX, block_y = UINT32_MAX;
                  for (unsigned lane = 0; lane < 32; ++lane) {
                     const unsigned mx = compact_bits_8(lane);
                     const unsigned my = compact_bits_8(lane >> 1);
                     const uint32_t px = wg_x0 + (warp << 4) + tx + mx;
                     const uint32_t py = wg_y0 + ty + my;
                     const uint32_t rx = px - x0;
                     const uint32_t ry = py - y0;
                     if (rx < width && ry < height) {
                        if (block_x == UINT32_MAX) {
                           block_x = px >> 4;
                           block_y = py >> 4;
                        } else {
                           EXPECT((px >> 4) == block_x && (py >> 4) == block_y,
                                  "edge warp crossed 16x16 compression block");
                        }
                        dst[py * img_w + px] = src[ry * row_stride_px + rx];
                        hits[py * img_w + px]++;
                     }
                  }
               }
            }
         }
      }
   }

   for (uint32_t py = 0; py < img_h; ++py) {
      for (uint32_t px = 0; px < img_w; ++px) {
         const bool inside =
            px >= x0 && px < x0 + width && py >= y0 && py < y0 + height;
         const uint8_t want_hits = inside ? 1 : 0;
         if (hits[py * img_w + px] != want_hits) {
            EXPECT(false,
                   "rect (%u,%u)+%ux%u pixel (%u,%u): hits=%u want=%u", x0, y0,
                   width, height, px, py, hits[py * img_w + px], want_hits);
            goto done;
         }
         if (inside) {
            const uint32_t want =
               src[(py - y0) * row_stride_px + (px - x0)];
            if (dst[py * img_w + px] != want) {
               EXPECT(false,
                      "rect (%u,%u)+%ux%u pixel (%u,%u): got=0x%x want=0x%x",
                      x0, y0, width, height, px, py, dst[py * img_w + px],
                      want);
               goto done;
            }
         }
      }
   }

done:
   free(src);
   free(dst);
   free(hits);
}

int
main(void)
{
   check_formats();
   check_grid_bounds();

   simulate_rect(128, 128, 0, 0, 64, 64, 64);
   simulate_rect(128, 128, 0, 0, 48, 32, 53);
   simulate_rect(128, 128, 3, 5, 45, 37, 45);
   simulate_rect(128, 128, 15, 17, 1, 1, 1);
   simulate_rect(128, 128, 1, 1, 63, 63, 71);
   simulate_rect(128, 128, 31, 15, 33, 33, 40);
   simulate_rect(128, 128, 16, 0, 32, 16, 32);
   simulate_rect(128, 128, 3, 7, 100, 80, 107);

   if (failures) {
      fprintf(stderr, "%u failure(s)\n", failures);
      return 1;
   }
   printf("hk_agx3_compressed_buf2img: OK\n");
   return 0;
}
