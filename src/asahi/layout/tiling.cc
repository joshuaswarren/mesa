/*
 * Copyright 2021 Alyssa Rosenzweig
 * Copyright 2019 Collabora, Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "util/macros.h"
#include "layout.h"

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

/* Z-order with rectangular (NxN or 2NxN) tiles, at most 128x128:
 *
 * 	[y6][x6][y5][x5][y4][x4]y3][x3][y2][x2][y1][x1][y0][x0]
 *
 * Efficient tiling algorithm described in
 * https://fgiesen.wordpress.com/2011/01/17/texture-tiling-and-swizzling/ but
 * for posterity, we split into X and Y parts, and are faced with the problem
 * of incrementing:
 *
 * 	0 [x6] 0 [x5] 0 [x4] 0 [x3] 0 [x2] 0 [x1] 0 [x0]
 *
 * To do so, we fill in the "holes" with 1's by adding the bitwise inverse of
 * the mask of bits we care about
 *
 * 	0 [x6] 0 [x5] 0 [x4] 0 [x3] 0 [x2] 0 [x1] 0 [x0]
 *    + 1  0   1  0   1  0   1  0   1  0   1  0   1  0
 *    ------------------------------------------------
 * 	1 [x6] 1 [x5] 1 [x4] 1 [x3] 1 [x2] 1 [x1] 1 [x0]
 *
 * Then when we add one, the holes are passed over by forcing carry bits high.
 * Finally, we need to zero out the holes, by ANDing with the mask of bits we
 * care about. In total, we get the expression (X + ~mask + 1) & mask, and
 * applying the two's complement identity, we are left with (X - mask) & mask
 */

typedef struct {
   uint64_t lo;
   uint64_t hi;
} __attribute__((packed)) ail_uint128_t;

/*
 * Given a power-of-two block width/height, construct the mask of "X" bits. This
 * is found by restricting the full mask of alternating 0s and 1s to only cover
 * the bottom 2 * log2(dim) bits. That's the same as modding by dim^2.
 */
static uint32_t
ail_space_mask(unsigned x)
{
   assert(util_is_power_of_two_nonzero(x));

   return MOD_POT(0x55555555, x * x);
}

/* memcpy_small on a box in elements (blocks), not pixels */
template <typename T, bool is_store>
static void
memcpy_small_el(void *_tiled, void *_linear,
                const struct ail_layout *tiled_layout, unsigned level,
                unsigned linear_pitch_B, unsigned sx_el, unsigned sy_el,
                unsigned swidth_el, unsigned sheight_el)
{
   unsigned linear_pitch_el = linear_pitch_B / sizeof(T);
   unsigned stride_el = tiled_layout->stride_el[level];
   unsigned sx_end_el = sx_el + swidth_el;
   unsigned sy_end_el = sy_el + sheight_el;

   struct ail_tile tile_size = tiled_layout->tilesize_el[level];
   unsigned tile_area_el = tile_size.width_el * tile_size.height_el;
   unsigned tiles_per_row = DIV_ROUND_UP(stride_el, tile_size.width_el);
   unsigned y_offs_el = ail_space_bits(MOD_POT(sy_el, tile_size.height_el))
                        << 1;
   unsigned x_offs_start_el =
      ail_space_bits(MOD_POT(sx_el, tile_size.width_el));
   unsigned space_mask_x = ail_space_mask(tile_size.width_el);
   unsigned space_mask_y = ail_space_mask(tile_size.height_el) << 1;
   unsigned log2_tile_width_el = util_logbase2(tile_size.width_el);
   unsigned log2_tile_height_el = util_logbase2(tile_size.height_el);

   T *linear = (T *)_linear;
   T *tiled = (T *)_tiled;

   for (unsigned y_el = sy_el; y_el < sy_end_el; ++y_el) {
      unsigned y_rowtile = y_el >> log2_tile_height_el;
      unsigned y_tile = y_rowtile * tiles_per_row;
      unsigned x_offs_el = x_offs_start_el;

      T *linear_row = linear;

      for (unsigned x_el = sx_el; x_el < sx_end_el; ++x_el) {
         unsigned tile_idx = (y_tile + (x_el >> log2_tile_width_el));
         unsigned tile_offset_el = tile_idx * tile_area_el;

         T *ptiled = &tiled[tile_offset_el + y_offs_el + x_offs_el];
         T *plinear = (linear_row++);

         if (is_store)
            *ptiled = *plinear;
         else
            *plinear = *ptiled;

         x_offs_el = (x_offs_el - space_mask_x) & space_mask_x;
      }

      y_offs_el = (y_offs_el - space_mask_y) & space_mask_y;
      linear += linear_pitch_el;
   }
}

template <typename T, bool is_store>
void
memcpy_small(void *_tiled, void *_linear, const struct ail_layout *tiled_layout,
             unsigned level, unsigned linear_pitch_B, unsigned sx_px,
             unsigned sy_px, unsigned swidth_px, unsigned sheight_px)
{
   enum pipe_format format = tiled_layout->format;

   memcpy_small_el<T, is_store>(_tiled, _linear, tiled_layout, level,
                                linear_pitch_B,
                                util_format_get_nblocksx(format, sx_px),
                                util_format_get_nblocksy(format, sy_px),
                                util_format_get_nblocksx(format, swidth_px),
                                util_format_get_nblocksy(format, sheight_px));
}

/* Gather the bits of v at the even positions (Morton decode) */
static inline unsigned
ail_compact_bits(unsigned v)
{
   v &= 0x55555555;
   v = (v | (v >> 1)) & 0x33333333;
   v = (v | (v >> 2)) & 0x0f0f0f0f;
   v = (v | (v >> 4)) & 0x00ff00ff;
   v = (v | (v >> 8)) & 0x0000ffff;
   return v;
}

/*
 * Detile 32-bit elements tile by tile. The Z-order puts every 4x4 block of a
 * tile in 16 consecutive elements (one 64-byte line), rows two pixel pairs at
 * offsets {0,1,4,5} + {0,2,8,10}[row], so whole tiles are read sequentially
 * and written as 8-byte pairs. Reading the tiled memory in order and in whole
 * lines matters: it is usually write-combined, where small reads are slow. Tiles
 * the box only partially covers fall back to memcpy_small.
 */
static void
ail_detile_32bpp(void *_tiled, void *_linear,
                 const struct ail_layout *tiled_layout, unsigned level,
                 unsigned linear_pitch_B, unsigned sx, unsigned sy,
                 unsigned width, unsigned height)
{
   static const unsigned row_offs[4] = {0, 2, 8, 10};
   struct ail_tile tile_size = tiled_layout->tilesize_el[level];
   unsigned tw = tile_size.width_el, th = tile_size.height_el;
   unsigned area = tw * th;
   unsigned tiles_per_row = DIV_ROUND_UP(tiled_layout->stride_el[level], tw);
   unsigned space_mask_x = ail_space_mask(tw);
   unsigned space_mask_y = ail_space_mask(th) << 1;
   unsigned ex = sx + width, ey = sy + height;
   const uint32_t *tiled = (const uint32_t *)_tiled;
   uint8_t *linear = (uint8_t *)_linear;

   for (unsigned ty = sy / th; ty <= (ey - 1) / th; ++ty) {
      unsigned y0 = ty * th;

      for (unsigned tx = sx / tw; tx <= (ex - 1) / tw; ++tx) {
         unsigned x0 = tx * tw;

         if (x0 < sx || x0 + tw > ex || y0 < sy || y0 + th > ey) {
            unsigned px = MAX2(x0, sx), py = MAX2(y0, sy);

            memcpy_small<uint32_t, false>(
               _tiled, linear + (py - sy) * linear_pitch_B + (px - sx) * 4,
               tiled_layout, level, linear_pitch_B, px, py,
               MIN2(x0 + tw, ex) - px, MIN2(y0 + th, ey) - py);
            continue;
         }

         const uint32_t *tile = tiled + (ty * tiles_per_row + tx) * area;
         uint8_t *dst = linear + (y0 - sy) * linear_pitch_B + (x0 - sx) * 4;

         for (unsigned i = 0; i < area; i += 16) {
            unsigned bx = ail_compact_bits(i & space_mask_x);
            unsigned by = ail_compact_bits((i & space_mask_y) >> 1);
            uint32_t src[16];

            /* One 64-byte read: wide loads, which uncached memory needs */
            memcpy(src, tile + i, sizeof(src));

            for (unsigned r = 0; r < 4; ++r) {
               uint8_t *d = dst + (by + r) * linear_pitch_B + bx * 4;

               memcpy(d, src + row_offs[r], 8);
               memcpy(d + 8, src + row_offs[r] + 4, 8);
            }
         }
      }
   }
}

/*
 * The store direction of ail_detile_32bpp: gather each 4x4 block of a fully
 * covered tile from four linear rows and write it as one sequential 64-byte
 * line. The tiled side is usually write-combined, where whole in-order lines
 * are fast and scattered 4-byte stores are not. Tiles the box only partially
 * covers fall back to memcpy_small.
 */
static void
ail_tile_32bpp(void *_tiled, void *_linear,
               const struct ail_layout *tiled_layout, unsigned level,
               unsigned linear_pitch_B, unsigned sx, unsigned sy,
               unsigned width, unsigned height)
{
   static const unsigned row_offs[4] = {0, 2, 8, 10};
   struct ail_tile tile_size = tiled_layout->tilesize_el[level];
   unsigned tw = tile_size.width_el, th = tile_size.height_el;
   unsigned area = tw * th;
   unsigned tiles_per_row = DIV_ROUND_UP(tiled_layout->stride_el[level], tw);
   unsigned space_mask_x = ail_space_mask(tw);
   unsigned space_mask_y = ail_space_mask(th) << 1;
   unsigned ex = sx + width, ey = sy + height;
   uint32_t *tiled = (uint32_t *)_tiled;
   const uint8_t *linear = (const uint8_t *)_linear;

   for (unsigned ty = sy / th; ty <= (ey - 1) / th; ++ty) {
      unsigned y0 = ty * th;

      for (unsigned tx = sx / tw; tx <= (ex - 1) / tw; ++tx) {
         unsigned x0 = tx * tw;

         if (x0 < sx || x0 + tw > ex || y0 < sy || y0 + th > ey) {
            unsigned px = MAX2(x0, sx), py = MAX2(y0, sy);

            memcpy_small<uint32_t, true>(
               _tiled,
               (void *)(linear + (py - sy) * linear_pitch_B + (px - sx) * 4),
               tiled_layout, level, linear_pitch_B, px, py,
               MIN2(x0 + tw, ex) - px, MIN2(y0 + th, ey) - py);
            continue;
         }

         uint32_t *tile = tiled + (ty * tiles_per_row + tx) * area;
         const uint8_t *src =
            linear + (y0 - sy) * linear_pitch_B + (x0 - sx) * 4;

         for (unsigned i = 0; i < area; i += 16) {
            unsigned bx = ail_compact_bits(i & space_mask_x);
            unsigned by = ail_compact_bits((i & space_mask_y) >> 1);
            uint32_t blk[16];

            for (unsigned r = 0; r < 4; ++r) {
               const uint8_t *s = src + (by + r) * linear_pitch_B + bx * 4;

               memcpy(blk + row_offs[r], s, 8);
               memcpy(blk + row_offs[r] + 4, s + 8, 8);
            }

            /* One 64-byte write: whole lines, which WC memory needs */
            memcpy(tile + i, blk, sizeof(blk));
         }
      }
   }
}

#if defined(__aarch64__)
/*
 * The permutation of a line that is an 8-element-wide block (8 rows of 8
 * bytes, 8-bit, or 4 rows of 16 bytes, 16-bit) into Z-order (storing) or back
 * (loading). In such a line the Z-order puts x in bits 0, 2 and 4 of the
 * element index and y in bits 1, 3 (and 5), so it is the same in every tile.
 */
template <typename T, bool is_store> struct ail_line_perm {
   uint8_t p[64];

   constexpr ail_line_perm() : p()
   {
      for (unsigned k = 0; k < 64; ++k) {
         unsigned e = k / sizeof(T);
         unsigned dx = (e & 1) | ((e >> 1) & 2) | ((e >> 2) & 4);
         unsigned dy = ((e >> 1) & 1) | ((e >> 2) & 2) | ((e >> 3) & 4);
         unsigned q = dy * 8 * sizeof(T) + dx * sizeof(T) + k % sizeof(T);

         if (is_store)
            p[k] = q;
         else
            p[q] = k;
      }
   }
};

template <typename T, bool is_store>
static constexpr ail_line_perm<T, is_store> ail_line_perms{};
#endif

/* What every line of a box needs, built for its first line */
struct ail_line_tables {
   /* Whether a line is an 8-element-wide block (always, with the NxN and
    * 2NxN tiles): then ail_line_small permutes it with NEON on arm64.
    */
   bool neon;
   /* If not, or not on arm64: the linear offset of each element of a line */
   unsigned offs[64];
};

/* The tables of a box's lines, for its pitch and the tile's Z-order */
template <typename T, bool is_store>
static void
ail_line_tables_init(struct ail_line_tables *tab, unsigned linear_pitch_B,
                     unsigned space_mask_x, unsigned space_mask_y)
{
   const unsigned n = 64 / sizeof(T);

   tab->neon = ((n - 1) & space_mask_x) == (0x15 & (n - 1)) &&
               ((n - 1) & space_mask_y) == (0x2a & (n - 1));
#if defined(__aarch64__)
   if (tab->neon)
      return;
#endif

   for (unsigned e = 0; e < n; ++e)
      tab->offs[e] = ail_compact_bits(e & space_mask_x) * sizeof(T) +
                     ail_compact_bits((e & space_mask_y) >> 1) * linear_pitch_B;
}

/*
 * Move one line of the tile at t from or to its block of linear rows at l.
 * On arm64 an 8-element-wide block is loaded into four q registers and
 * permuted with one 64-byte table lookup.
 */
template <typename T, bool is_store>
static inline void
ail_line_small(T *t, uint8_t *l, unsigned linear_pitch_B, const unsigned *offs,
               bool neon
#if defined(__aarch64__)
               ,
               uint8x16x4_t perm
#endif
)
{
#if defined(__aarch64__)
   if (neon) {
      const unsigned rows = 64 / (8 * sizeof(T));
      uint8x16x4_t q;

      if (is_store) {
         for (unsigned r = 0; r < 4; ++r) {
            if (rows == 8)
               q.val[r] =
                  vcombine_u8(vld1_u8(l + 2 * r * linear_pitch_B),
                              vld1_u8(l + (2 * r + 1) * linear_pitch_B));
            else
               q.val[r] = vld1q_u8(l + r * linear_pitch_B);
         }
         uint8x16x4_t v = {{vqtbl4q_u8(q, perm.val[0]),
                            vqtbl4q_u8(q, perm.val[1]),
                            vqtbl4q_u8(q, perm.val[2]),
                            vqtbl4q_u8(q, perm.val[3])}};
         vst1q_u8_x4((uint8_t *)t, v);
      } else {
         q = vld1q_u8_x4((const uint8_t *)t);
         for (unsigned r = 0; r < 4; ++r) {
            uint8x16_t v = vqtbl4q_u8(q, perm.val[r]);

            if (rows == 8) {
               vst1_u8(l + 2 * r * linear_pitch_B, vget_low_u8(v));
               vst1_u8(l + (2 * r + 1) * linear_pitch_B, vget_high_u8(v));
            } else {
               vst1q_u8(l + r * linear_pitch_B, v);
            }
         }
      }
      return;
   }
#endif
   const unsigned n = 64 / sizeof(T);
   T blk[64 / sizeof(T)];

   if (is_store) {
      for (unsigned e = 0; e < n; ++e)
         memcpy(&blk[e], l + offs[e], sizeof(T));
      memcpy(t, blk, sizeof(blk));
   } else {
      memcpy(blk, t, sizeof(blk));
      for (unsigned e = 0; e < n; ++e)
         memcpy(l + offs[e], &blk[e], sizeof(T));
   }
}

/*
 * The lines of a whole tile, in order: the tiled side is one sequential run.
 * linear points at the tile's first element. Not inlined, like
 * ail_lines_interior: in a function that calls nothing, the permutation stays
 * in registers. The NEON flag is copied out of the tables first, as a store
 * through the tile may alias them.
 */
template <typename T, bool is_store>
static ATTRIBUTE_NOINLINE void
ail_lines_tile(T *tile, uint8_t *linear, unsigned linear_pitch_B, unsigned area,
               unsigned space_mask_x, unsigned space_mask_y,
               const struct ail_line_tables *tab)
{
   const unsigned *offs = tab->offs;
   bool neon = tab->neon;
#if defined(__aarch64__)
   uint8x16x4_t perm = vld1q_u8_x4(ail_line_perms<T, is_store>.p);
#endif

   for (unsigned i = 0; i < area; i += 64 / sizeof(T)) {
      unsigned bx = ail_compact_bits(i & space_mask_x);
      unsigned by = ail_compact_bits((i & space_mask_y) >> 1);

      ail_line_small<T, is_store>(tile + i,
                                  linear + by * linear_pitch_B + bx * sizeof(T),
                                  linear_pitch_B, offs, neon
#if defined(__aarch64__)
                                  ,
                                  perm
#endif
      );
   }
}

/*
 * The whole lines of a partially covered tile: the blocks from dx0,dy0 to
 * dx1,dy1 of the tile (exclusive, on line boundaries), linear pointing at
 * dx0,dy0. The Z-order offsets step a line at a time within their masks, as
 * in memcpy_small.
 */
template <typename T, bool is_store>
static ATTRIBUTE_NOINLINE void
ail_lines_interior(T *tile, uint8_t *linear, unsigned linear_pitch_B,
                   unsigned dx0, unsigned dy0, unsigned dx1, unsigned dy1,
                   unsigned line_w, unsigned line_h, unsigned space_mask_x,
                   unsigned space_mask_y, const struct ail_line_tables *tab)
{
   const unsigned *offs = tab->offs;
   bool neon = tab->neon;
   unsigned step_x = ail_space_bits(line_w);
   unsigned step_y = ail_space_bits(line_h) << 1;
   unsigned x_start = ail_space_bits(dx0), y_offs = ail_space_bits(dy0) << 1;
#if defined(__aarch64__)
   uint8x16x4_t perm = vld1q_u8_x4(ail_line_perms<T, is_store>.p);
#endif

   for (unsigned dy = dy0; dy < dy1; dy += line_h) {
      uint8_t *l = linear + (dy - dy0) * linear_pitch_B;
      unsigned x_offs = x_start;

      for (unsigned dx = dx0; dx < dx1; dx += line_w) {
         ail_line_small<T, is_store>(tile + y_offs + x_offs,
                                     l + (dx - dx0) * sizeof(T), linear_pitch_B,
                                     offs, neon
#if defined(__aarch64__)
                                     ,
                                     perm
#endif
         );
         x_offs = ((x_offs | ~space_mask_x) + step_x) & space_mask_x;
      }
      y_offs = ((y_offs | ~space_mask_y) + step_y) & space_mask_y;
   }
}

/*
 * ail_tile_32bpp/ail_detile_32bpp for 8- and 16-bit elements: each 64-byte
 * line of a fully covered tile holds an 8x8 (8-bit) or 8x4 (16-bit) Z-order
 * block, gathered from or scattered to the linear rows through a table of
 * offsets, while the tiled side is written or read as one sequential line.
 * Video planes (R8, R8G8) take this path; one store per element to
 * write-combined memory made a 4K frame upload take ~15 ms. In a tile the box
 * only partially covers, the whole lines inside the box take the same path
 * and at most four strips around them fall back to memcpy_small: the bottom
 * tiles of a 2160-row plane hold 112 of their 128 rows. A box that holds no
 * whole line of the tile goes to memcpy_small as one rectangle, as before.
 */
template <typename T, bool is_store>
static void
ail_lines_small(void *_tiled, void *_linear,
                const struct ail_layout *tiled_layout, unsigned level,
                unsigned linear_pitch_B, unsigned sx, unsigned sy,
                unsigned width, unsigned height)
{
   const unsigned n = 64 / sizeof(T);
   struct ail_tile tile_size = tiled_layout->tilesize_el[level];
   unsigned tw = tile_size.width_el, th = tile_size.height_el;
   unsigned area = tw * th;
   unsigned tiles_per_row = DIV_ROUND_UP(tiled_layout->stride_el[level], tw);
   unsigned space_mask_x = ail_space_mask(tw);
   unsigned space_mask_y = ail_space_mask(th) << 1;
   unsigned ex = sx + width, ey = sy + height;
   T *tiled = (T *)_tiled;
   uint8_t *linear = (uint8_t *)_linear;
   struct ail_line_tables tab;
   bool have_tab = false;
   /* Built for the box's first line: a box of no whole line needs none */
   auto tables = [&]() {
      if (!have_tab)
         ail_line_tables_init<T, is_store>(&tab, linear_pitch_B, space_mask_x,
                                           space_mask_y);
      have_tab = true;
      return &tab;
   };
   /* The block a line holds: line_w x line_h elements */
   unsigned line_w = ail_compact_bits((n - 1) & space_mask_x) + 1;
   unsigned line_h = ail_compact_bits(((n - 1) & space_mask_y) >> 1) + 1;

   assert(util_is_power_of_two_nonzero(line_w) &&
          util_is_power_of_two_nonzero(line_h));

   for (unsigned ty = sy / th; ty <= (ey - 1) / th; ++ty) {
      unsigned y0 = ty * th;

      for (unsigned tx = sx / tw; tx <= (ex - 1) / tw; ++tx) {
         unsigned x0 = tx * tw;
         T *tile = tiled + (ty * tiles_per_row + tx) * area;

         if (x0 >= sx && x0 + tw <= ex && y0 >= sy && y0 + th <= ey) {
            uint8_t *l =
               linear + (y0 - sy) * linear_pitch_B + (x0 - sx) * sizeof(T);

            ail_lines_tile<T, is_store>(tile, l, linear_pitch_B, area,
                                        space_mask_x, space_mask_y, tables());
            continue;
         }

         /* The box in this tile (p to q), and the whole lines in it (i to j) */
         unsigned px = MAX2(x0, sx), py = MAX2(y0, sy);
         unsigned qx = MIN2(x0 + tw, ex), qy = MIN2(y0 + th, ey);
         unsigned ix = x0 + ((px - x0 + line_w - 1) & ~(line_w - 1));
         unsigned iy = y0 + ((py - y0 + line_h - 1) & ~(line_h - 1));
         unsigned jx = x0 + ((qx - x0) & ~(line_w - 1));
         unsigned jy = y0 + ((qy - y0) & ~(line_h - 1));
         unsigned rect[5][4] = {
            {px, py, qx, qy}, /* no whole line: the box in the tile */
            {px, py, qx, iy}, /* or the strips above, */
            {px, jy, qx, qy}, /* below, */
            {px, iy, ix, jy}, /* left of */
            {jx, iy, qx, jy}, /* and right of the whole lines */
         };
         bool lines = ix < jx && iy < jy;

         for (unsigned r = lines ? 1 : 0; r < (lines ? 5 : 1); ++r) {
            unsigned *b = rect[r];

            /* ail_lines_small_ok: an element is a pixel */
            if (b[0] < b[2] && b[1] < b[3]) {
               uint8_t *l = linear + (b[1] - sy) * linear_pitch_B +
                            (b[0] - sx) * sizeof(T);

               memcpy_small_el<T, is_store>(_tiled, l, tiled_layout, level,
                                            linear_pitch_B, b[0], b[1],
                                            b[2] - b[0], b[3] - b[1]);
            }
         }

         if (lines) {
            uint8_t *l =
               linear + (iy - sy) * linear_pitch_B + (ix - sx) * sizeof(T);

            ail_lines_interior<T, is_store>(
               tile, l, linear_pitch_B, ix - x0, iy - y0, jx - x0, jy - y0,
               line_w, line_h, space_mask_x, space_mask_y, tables());
         }
      }
   }
}

/* Whether ail_lines_small handles the box: 8/16-bit, a tile of whole lines */
static bool
ail_lines_small_ok(const struct ail_layout *tiled_layout, unsigned level,
                   unsigned blocksize_B, unsigned swidth_px,
                   unsigned sheight_px)
{
   struct ail_tile tile_size = tiled_layout->tilesize_el[level];

   return (blocksize_B == 1 || blocksize_B == 2) && swidth_px && sheight_px &&
          util_format_get_blockwidth(tiled_layout->format) == 1 &&
          util_format_get_blockheight(tiled_layout->format) == 1 &&
          tile_size.width_el * tile_size.height_el >= 64 / blocksize_B;
}

#define TILED_UNALIGNED_TYPES(blocksize_B, store)                              \
   if (blocksize_B == 1) {                                                     \
      memcpy_small<uint8_t, store>(_tiled, _linear, tiled_layout, level,       \
                                   linear_pitch_B, sx_px, sy_px, swidth_px,    \
                                   sheight_px);                                \
   } else if (blocksize_B == 2) {                                              \
      memcpy_small<uint16_t, store>(_tiled, _linear, tiled_layout, level,      \
                                    linear_pitch_B, sx_px, sy_px, swidth_px,   \
                                    sheight_px);                               \
   } else if (blocksize_B == 4) {                                              \
      memcpy_small<uint32_t, store>(_tiled, _linear, tiled_layout, level,      \
                                    linear_pitch_B, sx_px, sy_px, swidth_px,   \
                                    sheight_px);                               \
   } else if (blocksize_B == 8) {                                              \
      memcpy_small<uint64_t, store>(_tiled, _linear, tiled_layout, level,      \
                                    linear_pitch_B, sx_px, sy_px, swidth_px,   \
                                    sheight_px);                               \
   } else if (blocksize_B == 16) {                                             \
      memcpy_small<ail_uint128_t, store>(_tiled, _linear, tiled_layout, level, \
                                         linear_pitch_B, sx_px, sy_px,         \
                                         swidth_px, sheight_px);               \
   } else {                                                                    \
      UNREACHABLE("Invalid block size");                                       \
   }

void
ail_detile(void *_tiled, void *_linear, const struct ail_layout *tiled_layout,
           unsigned level, unsigned linear_pitch_B, unsigned sx_px,
           unsigned sy_px, unsigned swidth_px, unsigned sheight_px)
{
   unsigned width_px = u_minify(tiled_layout->width_px, level);
   unsigned height_px = u_minify(tiled_layout->height_px, level);
   unsigned blocksize_B = util_format_get_blocksize(tiled_layout->format);

   assert(level < tiled_layout->levels && "Mip level out of bounds");
   assert(ail_is_level_twiddled_uncompressed(tiled_layout, level) &&
          "Invalid usage");
   assert((sx_px + swidth_px) <= width_px && "Invalid usage");
   assert((sy_px + sheight_px) <= height_px && "Invalid usage");

   struct ail_tile tile_size = tiled_layout->tilesize_el[level];
   if (blocksize_B == 4 && swidth_px && sheight_px &&
       util_format_get_blockwidth(tiled_layout->format) == 1 &&
       util_format_get_blockheight(tiled_layout->format) == 1 &&
       tile_size.width_el >= 4 && tile_size.height_el >= 4) {
      ail_detile_32bpp(_tiled, _linear, tiled_layout, level, linear_pitch_B,
                       sx_px, sy_px, swidth_px, sheight_px);
      return;
   }

   if (ail_lines_small_ok(tiled_layout, level, blocksize_B, swidth_px,
                          sheight_px)) {
      if (blocksize_B == 1)
         ail_lines_small<uint8_t, false>(_tiled, _linear, tiled_layout, level,
                                         linear_pitch_B, sx_px, sy_px,
                                         swidth_px, sheight_px);
      else
         ail_lines_small<uint16_t, false>(_tiled, _linear, tiled_layout,
                                          level, linear_pitch_B, sx_px, sy_px,
                                          swidth_px, sheight_px);
      return;
   }

   TILED_UNALIGNED_TYPES(blocksize_B, false);
}

void
ail_tile(void *_tiled, void *_linear, const struct ail_layout *tiled_layout,
         unsigned level, unsigned linear_pitch_B, unsigned sx_px,
         unsigned sy_px, unsigned swidth_px, unsigned sheight_px)
{
   unsigned width_px = u_minify(tiled_layout->width_px, level);
   unsigned height_px = u_minify(tiled_layout->height_px, level);
   unsigned blocksize_B = ail_get_blocksize_B(tiled_layout);

   assert(level < tiled_layout->levels && "Mip level out of bounds");
   assert(ail_is_level_twiddled_uncompressed(tiled_layout, level) &&
          "Invalid usage");
   assert((sx_px + swidth_px) <= width_px && "Invalid usage");
   assert((sy_px + sheight_px) <= height_px && "Invalid usage");

   struct ail_tile tile_size = tiled_layout->tilesize_el[level];
   if (blocksize_B == 4 && swidth_px && sheight_px &&
       util_format_get_blockwidth(tiled_layout->format) == 1 &&
       util_format_get_blockheight(tiled_layout->format) == 1 &&
       tile_size.width_el >= 4 && tile_size.height_el >= 4) {
      ail_tile_32bpp(_tiled, _linear, tiled_layout, level, linear_pitch_B,
                     sx_px, sy_px, swidth_px, sheight_px);
      return;
   }

   if (ail_lines_small_ok(tiled_layout, level, blocksize_B, swidth_px,
                          sheight_px)) {
      if (blocksize_B == 1)
         ail_lines_small<uint8_t, true>(_tiled, _linear, tiled_layout, level,
                                        linear_pitch_B, sx_px, sy_px,
                                        swidth_px, sheight_px);
      else
         ail_lines_small<uint16_t, true>(_tiled, _linear, tiled_layout, level,
                                         linear_pitch_B, sx_px, sy_px,
                                         swidth_px, sheight_px);
      return;
   }

   TILED_UNALIGNED_TYPES(blocksize_B, true);
}
