/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"

KERNEL(32)
libagx_fill(global uint32_t *address, uint32_t value)
{
   assert((((uintptr_t)address) & 0x3) == 0);

   address[cl_global_id.x] = value;
}

KERNEL(32)
libagx_copy_uint4(global uint4 *dest, global uint4 *src)
{
   assert((((uintptr_t)dest) & 0xf) == 0);
   assert((((uintptr_t)src) & 0xf) == 0);

   dest[cl_global_id.x] = src[cl_global_id.x];
}

KERNEL(32)
libagx_copy_uint(global uint32_t *dest, global uint32_t *src)
{
   assert((((uintptr_t)dest) & 0x3) == 0);
   assert((((uintptr_t)src) & 0x3) == 0);

   dest[cl_global_id.x] = src[cl_global_id.x];
}

KERNEL(16)
libagx_copy_uchar(global uint8_t *dest, global uint8_t *src)
{
   dest[cl_global_id.x] = src[cl_global_id.x];
}

static uint
libagx_space_bits(uint x)
{
   x &= 0x3fffu;
   x = (x | (x << 8)) & 0x00ff00ffu;
   x = (x | (x << 4)) & 0x0f0f0f0fu;
   x = (x | (x << 2)) & 0x33333333u;
   x = (x | (x << 1)) & 0x55555555u;
   return x;
}

static uint
libagx_space_bits_7(uint x)
{
   x &= 0x7fu;
   x = (x | (x << 4)) & 0x0f0fu;
   x = (x | (x << 2)) & 0x3333u;
   x = (x | (x << 1)) & 0x5555u;
   return x;
}

/*
 * Raw texture copies operate on format elements, not interpreted
 * colours.  An element is a pixel for ordinary formats and a whole block for
 * BC formats.  Keeping both sides in element coordinates also implements the
 * specified BC<->RGBA32 copy compatibility without format conversion: one
 * 16-byte BC block is copied to or from one 16-byte RGBA32 pixel.
 *
 * CPU command recording validates every rectangle against the logical image
 * or placed footprint before this kernel is dispatched.  The coordinate
 * guard is a second line of defence against an over-wide launch.
 */
static uint
libagx_copy_image_offset(uint x, uint y, uint pitch_B, uint stride_el,
                         uint tile_width_el, uint tile_height_el,
                         uint blocksize_B, uint tiled)
{
   if (!tiled)
      return y * pitch_B + x * blocksize_B;

   uint tiles_per_row = (stride_el + tile_width_el - 1) / tile_width_el;
   uint tile_idx = (y / tile_height_el) * tiles_per_row +
                   (x / tile_width_el);
   uint tile_area_el = tile_width_el * tile_height_el;
   uint morton_el = libagx_space_bits(x & (tile_width_el - 1)) |
                    (libagx_space_bits(y & (tile_height_el - 1)) << 1);

   return (tile_idx * tile_area_el + morton_el) * blocksize_B;
}

/* Reduce power-of-two tile geometry on the CPU. Small tiles can use the
 * seven-bit encoder, but legacy AIL mip tails may use larger square tiles.
 * Keep their high coordinate bits just as the generic helper does. */
static uint
libagx_copy_image_offset_pow2(uint x, uint y, uint pitch_B,
                              uint tiles_per_row, uint tile_width_log2,
                              uint tile_height_log2, uint blocksize_B,
                              uint tiled)
{
   if (!tiled)
      return y * pitch_B + x * blocksize_B;

   uint tile_width_el = 1u << tile_width_log2;
   uint tile_height_el = 1u << tile_height_log2;
   uint tile_idx = (y >> tile_height_log2) * tiles_per_row +
                   (x >> tile_width_log2);
   uint tile_area_el = 1u << (tile_width_log2 + tile_height_log2);
   uint tile_x = x & (tile_width_el - 1);
   uint tile_y = y & (tile_height_el - 1);
   uint morton_el = tile_width_log2 <= 7 && tile_height_log2 <= 7
      ? libagx_space_bits_7(tile_x) | (libagx_space_bits_7(tile_y) << 1)
      : libagx_space_bits(tile_x) | (libagx_space_bits(tile_y) << 1);

   return (tile_idx * tile_area_el + morton_el) * blocksize_B;
}

KERNEL(32)
libagx_copy_image(global uint8_t *dest, global uint8_t *src,
                  uint width_el, uint height_el,
                  uint dest_x_el, uint dest_y_el,
                  uint src_x_el, uint src_y_el,
                  uint dest_pitch_B, uint src_pitch_B,
                  uint dest_tiles_per_row, uint src_tiles_per_row,
                  uint dest_tile_width_log2, uint dest_tile_height_log2,
                  uint src_tile_width_log2, uint src_tile_height_log2,
                  uint blocksize_B, uint dest_tiled, uint src_tiled)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= width_el || y >= height_el)
      return;

   uint dest_offset_B = libagx_copy_image_offset_pow2(
      dest_x_el + x, dest_y_el + y, dest_pitch_B, dest_tiles_per_row,
      dest_tile_width_log2, dest_tile_height_log2, blocksize_B, dest_tiled);
   uint src_offset_B = libagx_copy_image_offset_pow2(
      src_x_el + x, src_y_el + y, src_pitch_B, src_tiles_per_row,
      src_tile_width_log2, src_tile_height_log2, blocksize_B, src_tiled);

   global uint8_t *d = dest + dest_offset_B;
   global uint8_t *s = src + src_offset_B;
   uintptr_t align = (uintptr_t)d | (uintptr_t)s;

   if (blocksize_B == 16 && (align & 15) == 0) {
      *(global uint4 *)d = *(global uint4 *)s;
   } else if (blocksize_B == 8 && (align & 7) == 0) {
      *(global uint2 *)d = *(global uint2 *)s;
   } else if (blocksize_B == 4 && (align & 3) == 0) {
      *(global uint32_t *)d = *(global uint32_t *)s;
   } else if (blocksize_B == 2 && (align & 1) == 0) {
      *(global uint16_t *)d = *(global uint16_t *)s;
   } else if (blocksize_B == 1) {
      *d = *s;
   } else if ((blocksize_B & 15) == 0 && (align & 15) == 0) {
      global uint4 *d4 = (global uint4 *)d;
      global uint4 *s4 = (global uint4 *)s;
      for (uint i = 0; i < (blocksize_B >> 4); ++i)
         d4[i] = s4[i];
   } else if ((blocksize_B & 3) == 0 && (align & 3) == 0) {
      global uint32_t *d1 = (global uint32_t *)d;
      global uint32_t *s1 = (global uint32_t *)s;
      for (uint i = 0; i < (blocksize_B >> 2); ++i)
         d1[i] = s1[i];
   } else if ((blocksize_B & 1) == 0 && (align & 1) == 0) {
      global uint16_t *d2 = (global uint16_t *)d;
      global uint16_t *s2 = (global uint16_t *)s;
      for (uint i = 0; i < (blocksize_B >> 1); ++i)
         d2[i] = s2[i];
   } else {
      for (uint b = 0; b < blocksize_B; ++b)
         d[b] = s[b];
   }
}

/* Round an unsigned four-sample average like G17P's linear sampler followed
 * by a normalized colour write. Exact output ties move toward code 128. This
 * avoids the still-unqualified G17P float-to-integer convert encoding.
 *
 * Each 16-bit SWAR lane holds the sum of four bytes in [0, 1020] (bit 9 is 1
 * iff quotient >= 128), so adding 2 - ((sum >> 9) & 1) before >> 2 rounds
 * exact half-ties upward below 128 and downward at or above 128, bit-exact
 * with the scalar rule. */
static uint
libagx_average4_unorm8x4(uint p00, uint p10, uint p01, uint p11)
{
   uint rb = (p00 & 0x00ff00ffu) + (p10 & 0x00ff00ffu) +
             (p01 & 0x00ff00ffu) + (p11 & 0x00ff00ffu);
   uint ga = ((p00 >> 8) & 0x00ff00ffu) + ((p10 >> 8) & 0x00ff00ffu) +
             ((p01 >> 8) & 0x00ff00ffu) + ((p11 >> 8) & 0x00ff00ffu);
   rb = ((rb + 0x00020002u - ((rb >> 9) & 0x00010001u)) >> 2) & 0x00ff00ffu;
   ga = ((ga + 0x00020002u - ((ga >> 9) & 0x00010001u)) >> 2) & 0x00ff00ffu;
   return rb | (ga << 8);
}

/* Compact the even bits of a 6-bit Morton index (0..63) to 0..7. */
static uint
libagx_compact_bits_6(uint v)
{
   v &= 0x55u;
   v = (v | (v >> 1)) & 0x33u;
   return (v | (v >> 2)) & 0x0fu;
}

/* Byte offset of a 4-byte GPU-tiled pixel with tile_width/height_log2 <= 7. */
static uint
libagx_tiled_offset_4b(uint x, uint y, uint tiles_per_row,
                       uint tile_width_log2, uint tile_height_log2)
{
   uint tile_idx = (y >> tile_height_log2) * tiles_per_row +
                   (x >> tile_width_log2);
   uint tile_x = x & ((1u << tile_width_log2) - 1u);
   uint tile_y = y & ((1u << tile_height_log2) - 1u);
   uint morton = libagx_space_bits_7(tile_x) |
                 (libagx_space_bits_7(tile_y) << 1);
   return ((tile_idx << (tile_width_log2 + tile_height_log2)) + morton) << 2;
}

/*
 * Exact full-level 2x downsample for ordinary four-channel UNORM8 images.
 * RGBA and BGRA are both channel-order agnostic here: each stored byte is
 * filtered independently and written back in the same order.
 */
KERNEL2D(8, 8)
libagx_mipmap_unorm8x4(global uint8_t *dest, global uint8_t *src,
                       uint dest_width_px, uint dest_height_px,
                       uint src_scale_x, uint src_scale_y,
                       uint dest_tiles_per_row, uint src_tiles_per_row,
                       uint dest_tile_width_log2, uint dest_tile_height_log2,
                       uint src_tile_width_log2, uint src_tile_height_log2)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= dest_width_px || y >= dest_height_px)
      return;

   uint dest_offset = libagx_tiled_offset_4b(
      x, y, dest_tiles_per_row, dest_tile_width_log2, dest_tile_height_log2);
   uint src_x = x * src_scale_x;
   uint src_y = y * src_scale_y;
   uint src00 = libagx_tiled_offset_4b(
      src_x, src_y, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2);

   uint p00, p10, p01, p11;
   if (src_scale_x == 2u && src_scale_y == 2u &&
       src_tile_width_log2 >= 1u && src_tile_height_log2 >= 1u &&
       (((uintptr_t)src) & 0xfu) == 0u) {
      /* In AIL Morton order, the 2x2 texel block at even (src_x, src_y)
       * occupies four contiguous 32-bit pixels (16 bytes, 16-byte aligned). */
      uint4 block = *((const global uint4 *)(src + src00));
      p00 = block.x;
      p10 = block.y;
      p01 = block.z;
      p11 = block.w;
   } else {
      uint src10 = libagx_tiled_offset_4b(
         src_x + src_scale_x - 1u, src_y, src_tiles_per_row,
         src_tile_width_log2, src_tile_height_log2);
      uint src01 = libagx_tiled_offset_4b(
         src_x, src_y + src_scale_y - 1u, src_tiles_per_row,
         src_tile_width_log2, src_tile_height_log2);
      uint src11 = libagx_tiled_offset_4b(
         src_x + src_scale_x - 1u, src_y + src_scale_y - 1u,
         src_tiles_per_row, src_tile_width_log2, src_tile_height_log2);
      p00 = *((const global uint *)(src + src00));
      p10 = *((const global uint *)(src + src10));
      p01 = *((const global uint *)(src + src01));
      p11 = *((const global uint *)(src + src11));
   }

   *((global uint *)(dest + dest_offset)) =
      libagx_average4_unorm8x4(p00, p10, p01, p11);
}

/*
 * Fused two-level exact 2x+2x downsample for GPU-tiled UNORM8x4 images:
 * each invocation reads one 4x4 Morton-contiguous block (4 x uint4) from src,
 * writes the resulting 2x2 Morton-contiguous block (1 x uint4) to dest1, and
 * averages those four results in registers to write one pixel to dest2.
 *
 * Callers must dispatch full 16x8 workgroups (aligned to 16 in X, 8 in Y)
 * with src_tile_log2 >= 2, dest1_tile_log2 >= 1, all tile_log2 <= 7, and
 * 16-byte aligned src and dest1 pointers.
 */
KERNEL2D(16, 8)
libagx_mipmap_unorm8x4_2level(global uint8_t *dest2, global uint8_t *dest1,
                              global uint8_t *src,
                              uint dest2_width_px, uint dest2_height_px,
                              uint dest2_tiles_per_row,
                              uint dest1_tiles_per_row,
                              uint src_tiles_per_row,
                              uint tile_log2_packed)
{
   /* Order each 32-lane subgroup in Morton order (8x4 blocks in dest2) so
    * reads from src and writes to dest1 and dest2 coalesce into contiguous
    * cachelines. */
   uint lid = cl_local_id.y * 16u + cl_local_id.x;
   uint lx = libagx_compact_bits_6(lid & 63u) | ((lid >> 3) & 8u);
   uint ly = libagx_compact_bits_6((lid & 63u) >> 1);
   uint x = cl_group_id.x * 16u + lx;
   uint y = cl_group_id.y * 8u + ly;
   if (x >= dest2_width_px || y >= dest2_height_px)
      return;

   uint d2_tw = tile_log2_packed & 0xfu;
   uint d2_th = (tile_log2_packed >> 4) & 0xfu;
   uint d1_tw = (tile_log2_packed >> 8) & 0xfu;
   uint d1_th = (tile_log2_packed >> 12) & 0xfu;
   uint s_tw = (tile_log2_packed >> 16) & 0xfu;
   uint s_th = (tile_log2_packed >> 20) & 0xfu;

   uint src_off = libagx_tiled_offset_4b(x << 2, y << 2, src_tiles_per_row,
                                         s_tw, s_th);
   const global uint4 *src4 = (const global uint4 *)(src + src_off);
   uint4 b00 = src4[0];
   uint4 b10 = src4[1];
   uint4 b01 = src4[2];
   uint4 b11 = src4[3];

   uint o00 = libagx_average4_unorm8x4(b00.x, b00.y, b00.z, b00.w);
   uint o10 = libagx_average4_unorm8x4(b10.x, b10.y, b10.z, b10.w);
   uint o01 = libagx_average4_unorm8x4(b01.x, b01.y, b01.z, b01.w);
   uint o11 = libagx_average4_unorm8x4(b11.x, b11.y, b11.z, b11.w);

   uint d1_off = libagx_tiled_offset_4b(x << 1, y << 1, dest1_tiles_per_row,
                                        d1_tw, d1_th);
   *((global uint4 *)(dest1 + d1_off)) = (uint4)(o00, o10, o01, o11);

   uint d2_off = libagx_tiled_offset_4b(x, y, dest2_tiles_per_row, d2_tw,
                                        d2_th);
   *((global uint *)(dest2 + d2_off)) =
      libagx_average4_unorm8x4(o00, o10, o01, o11);
}

/*
 * Exact odd-size full-level downsample for four-channel UNORM8 images.  The
 * CPU builds one packed Q8 coordinate entry per destination X and Y, avoiding
 * every per-pixel divide and all 64-bit arithmetic here.  G17P's sampler
 * rounds the four-tap result half-up to four fractional colour bits; its
 * normalized write rounds exact ties toward the midpoint code 128.
 */
KERNEL2D(8, 8)
libagx_mipmap_unorm8x4_odd(global uint8_t *dest, global uint8_t *src,
                           global uint *axis,
                           uint dest_width_px, uint dest_height_px,
                           uint dest_tiles_per_row, uint src_tiles_per_row,
                           uint dest_tile_width_log2,
                           uint dest_tile_height_log2,
                           uint src_tile_width_log2,
                           uint src_tile_height_log2)
{
   uint x = cl_global_id.x;
   uint y = cl_global_id.y;
   if (x >= dest_width_px || y >= dest_height_px)
      return;

   uint x_entry = axis[x];
   uint y_entry = axis[dest_width_px + y];
   uint src_x0 = x_entry & 0xffffu;
   uint src_y0 = y_entry & 0xffffu;
   uint wx = (x_entry >> 16) & 0xffu;
   uint wy = (y_entry >> 16) & 0xffu;
   uint src_x1 = src_x0 + (wx != 0u);
   uint src_y1 = src_y0 + (wy != 0u);

   uint dest_offset = libagx_tiled_offset_4b(
      x, y, dest_tiles_per_row, dest_tile_width_log2, dest_tile_height_log2);
   uint src00 = libagx_tiled_offset_4b(
      src_x0, src_y0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2);
   uint src10 = libagx_tiled_offset_4b(
      src_x1, src_y0, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2);
   uint src01 = libagx_tiled_offset_4b(
      src_x0, src_y1, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2);
   uint src11 = libagx_tiled_offset_4b(
      src_x1, src_y1, src_tiles_per_row, src_tile_width_log2,
      src_tile_height_log2);

   uint p00 = *((const global uint *)(src + src00));
   uint p10 = *((const global uint *)(src + src10));
   uint p01 = *((const global uint *)(src + src01));
   uint p11 = *((const global uint *)(src + src11));

   uint ix = 256u - wx;
   uint iy = 256u - wy;
   uint w00 = ix * iy;
   uint w10 = wx * iy;
   uint w01 = ix * wy;
   uint w11 = wx * wy;

   uint packed = 0u;
   for (uint channel = 0; channel < 4; ++channel) {
      uint shift = channel * 8u;
      uint value = ((p00 >> shift) & 0xffu) * w00 +
                   ((p10 >> shift) & 0xffu) * w10 +
                   ((p01 >> shift) & 0xffu) * w01 +
                   ((p11 >> shift) & 0xffu) * w11;
      uint fixed4 = (value + 2048u) >> 12;
      uint out = fixed4 >> 4;
      uint remainder = fixed4 & 15u;
      out += remainder > 8u || (remainder == 8u && out < 128u);
      packed |= (out & 0xffu) << shift;
   }
   *((global uint *)(dest + dest_offset)) = packed;
}

/*
 * Resolve an unorm8x4 multisample surface into a single-sample surface.
 * AIL tiles one logical pixel as a block containing every sample. Averaging
 * bytes is format-order agnostic for the four 8-bit UNORM channel formats
 * this entry point accepts on the CPU side.
 */
KERNEL(32)
libagx_resolve_unorm8x4(global uint8_t *dest, global uint8_t *src,
                        uint width_px, uint height_px,
                        uint dest_x_px, uint dest_y_px,
                        uint src_x_px, uint src_y_px,
                        uint dest_pitch_B, uint dest_stride_el,
                        uint src_stride_el,
                        uint dest_tile_width_el, uint dest_tile_height_el,
                        uint src_tile_width_el, uint src_tile_height_el,
                        uint sample_count, uint dest_tiled, uint src_tiled,
                        uint resolve_mode)
{
   uint id = cl_global_id.x;
   if (id >= width_px * height_px)
      return;

   uint x = id % width_px;
   uint y = id / width_px;
   uint dest_offset_B = libagx_copy_image_offset(
      dest_x_px + x, dest_y_px + y, dest_pitch_B, dest_stride_el,
      dest_tile_width_el, dest_tile_height_el, 4, dest_tiled);

   for (uint b = 0; b < 4; ++b) {
      uint sum = 0;
      uint extrema = resolve_mode == 1 ? 255 : 0;
      for (uint s = 0; s < sample_count; ++s) {
         uint src_offset_B = libagx_copy_image_offset(
            src_x_px + x, src_y_px + y, 0, src_stride_el,
            src_tile_width_el, src_tile_height_el,
            4 * sample_count, src_tiled);
         uint value = src[src_offset_B + s * 4 + b];
         sum += value;
         extrema = resolve_mode == 1 ? min(extrema, value)
                                      : max(extrema, value);
      }

      dest[dest_offset_B + b] = resolve_mode
                                   ? extrema
                                   : (sum + sample_count / 2) / sample_count;
   }
}

KERNEL(256)
libagx_fill_uint4(global uint4 *address, uint layer_stride_uint4, uint a,
                  uint b, uint c, uint d)
{
   uint offs = cl_global_id.x + cl_global_id.y * layer_stride_uint4;
   address[offs] = (uint4)(a, b, c, d);
}

/*
 * Native M5 acceleration structures (vulkan/bvh/agx3_envelope.comp). An
 * image starts at a 1KiB-aligned base. Slot 0 holds the root and the header
 * starts at byte 64: byte 72 is the image size from the base, 112 the
 * descriptor offset, 128 the instance pointer-table offset, 144 the absolute
 * base address and 176 the device-wide structure ID. Driver-only empty
 * metadata has a zero size and occupies bytes 0..191. Every other offset is
 * relative to the base and instance records reference bottom-level images by
 * address, so a word copy that rewrites the base and ID words relocates an
 * image.
 *
 * The serialized form is the Vulkan header (driver UUID twice, serialized
 * size, deserialized size, handle count), one bottom-level handle per
 * instance in instance-index order, then the image.
 */
#define AGX3_AS_EMPTY_BYTES      192
#define AGX3_AS_SERIAL_WORDS     7
#define AGX3_AS_SLACK            1023
#define AGX3_AS_INSTANCE_RECORD  26

static uint64_t
agx3_as_bytes(global uint64_t *image)
{
   return image[9] ? image[9] : AGX3_AS_EMPTY_BYTES;
}

static uint32_t
agx3_as_instance_slots(global uint64_t *image)
{
   global uint32_t *words = (global uint32_t *)image;
   if (!image[9] || (words[16] >> 24) != 0x83)
      return 0;
   return (image[14] - image[16]) / 16;
}

/* Instance pointer-table entries are written only for encoded instances, so
 * an entry counts only when it names a record carrying its own index. */
static global uint64_t *
agx3_as_instance_record(global uint64_t *image, uint32_t index)
{
   global uint64_t *table =
      (global uint64_t *)((global uint8_t *)image + image[16]);
   uint64_t offset = table[2 * index];
   uint64_t bytes = agx3_as_bytes(image);
   if (table[2 * index + 1] != 1 || offset < 4 || offset - 4 > bytes ||
       bytes - (offset - 4) < 128 || ((offset - 4) & 63))
      return NULL;

   global uint64_t *record =
      (global uint64_t *)((global uint8_t *)image + offset - 4);
   global uint32_t *words = (global uint32_t *)record;
   if (words[0] != AGX3_AS_INSTANCE_RECORD || words[25] != index)
      return NULL;
   return record;
}

static void
agx3_as_copy_words(global uint64_t *dst, global uint64_t *src, uint32_t chunk,
                   uint64_t bytes, uint32_t dst_id, bool relocate)
{
   for (uint32_t w = chunk * 2; w < chunk * 2 + 2; ++w) {
      if ((uint64_t)w * 8 >= bytes)
         return;
      uint64_t value = src[w];
      if (relocate && w == 18)
         value = (uint64_t)dst;
      else if (relocate && w == 22)
         value = (value & 0xffffffff00000000ul) | dst_id;
      dst[w] = value;
   }
}

KERNEL(32)
libagx_agx3_as_copy(global uint64_t *dst, global uint64_t *src,
                    uint32_t dst_id)
{
   agx3_as_copy_words(dst, src, cl_global_id.x, agx3_as_bytes(src), dst_id,
                      src[9] != 0);
}

KERNEL(32)
libagx_agx3_as_serialize(global uint64_t *dst, global uint64_t *src,
                         uint32_t uuid0, uint32_t uuid1, uint32_t uuid2,
                         uint32_t uuid3)
{
   uint32_t i = cl_global_id.x;
   uint64_t bytes = agx3_as_bytes(src);
   uint32_t handles = agx3_as_instance_slots(src);

   if (i == 0) {
      dst[0] = dst[2] = uuid0 | ((uint64_t)uuid1 << 32);
      dst[1] = dst[3] = uuid2 | ((uint64_t)uuid3 << 32);
      dst[4] = AGX3_AS_SERIAL_WORDS * 8 + (uint64_t)handles * 8 + bytes;
      dst[5] = bytes + AGX3_AS_SLACK;
      dst[6] = handles;
   }

   if (i < handles) {
      global uint64_t *record = agx3_as_instance_record(src, i);
      dst[AGX3_AS_SERIAL_WORDS + i] = record ? record[11] : 0;
   }

   agx3_as_copy_words(dst + AGX3_AS_SERIAL_WORDS + handles, src, i, bytes, 0,
                      false);
}

KERNEL(32)
libagx_agx3_as_deserialize(global uint64_t *dst, global uint64_t *src,
                           uint32_t dst_id)
{
   uint64_t handles = src[6];
   global uint64_t *image = src + AGX3_AS_SERIAL_WORDS + handles;
   uint64_t bytes = src[4] - AGX3_AS_SERIAL_WORDS * 8 - handles * 8;
   agx3_as_copy_words(dst, image, cl_global_id.x, bytes, dst_id,
                      image[9] != 0);
}

/* After deserialization, point each instance at the bottom-level handle the
 * application left in the serialized header. IDs come from the driver's
 * created-object table: replacement BLAS need not have been built yet. Bounds
 * remain those of the serialized geometry, which must be compatible. */
KERNEL(32)
libagx_agx3_as_relink(global uint64_t *dst, global uint64_t *src,
                      global uint64_t *table, uint32_t table_count)
{
   uint32_t i = cl_global_id.x;
   if (i >= src[6] || i >= agx3_as_instance_slots(dst))
      return;

   global uint64_t *record = agx3_as_instance_record(dst, i);
   uint64_t blas = src[AGX3_AS_SERIAL_WORDS + i];
   if (!record)
      return;

   record[11] = blas;
   if (blas) {
      global uint32_t *words = (global uint32_t *)record;
      for (uint32_t id = 1; id <= table_count; ++id) {
         if (table[2 * id] == blas) {
            words[1] = id;
            break;
         }
      }
   }
}

/* kind: 0 compacted/structure size, 1 serialization size, 2 pointer count. */
KERNEL(1)
libagx_agx3_as_property(global uint64_t *image, global uint64_t *report,
                        uint32_t kind)
{
   uint64_t bytes = agx3_as_bytes(image);
   uint64_t handles = agx3_as_instance_slots(image);
   if (kind == 2)
      *report = handles;
   else if (kind == 1)
      *report = AGX3_AS_SERIAL_WORDS * 8 + handles * 8 + bytes;
   else
      *report = bytes + AGX3_AS_SLACK;
}
