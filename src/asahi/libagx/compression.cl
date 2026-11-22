/*
 * Copyright 2024 Valve Corporation
 * SPDX-License-Identifier: MIT
 */
#include "compiler/libcl/libcl.h"
#include "compiler/nir/nir_defines.h"
#include "compiler/shader_enums.h"
#include "asahi/compiler/agx_nir_texture.h"
#include "agx_pack.h"
#include "compression.h"
#include "libagx_intrinsics.h"

/*
 * Decompress in place. The metadata is updated, so other processes can read the
 * image with a compressed texture descriptor.
 *
 * Each workgroup processes one 16x16 tile, avoiding races. We use 32x1
 * workgroups, matching the warp size, meaning each work-item must process
 * (16*16)/(32*1) = 8 sampels. Matching the warp size eliminates cross-warp
 * barriers. It also minimizes launched threads, accelerating the early exit.
 */

/* Our compiler represents a bindless handle as a uint2 of a uniform base and an
 * offset in bytes. Since the descriptors are all in the u0_u1 push, the former
 * is hardcoded and the latter is an offsetof.
 */
#define HANDLE(field)                                                          \
   nir_bindless_image_agx(offsetof(struct libagx_decompress_images, field), 0)

/*
 * The metadata buffer is fully twiddled, so interleave the X/Y coordinate bits.
 * While dimensions are padded to powers-of-two, they are not padded to a
 * square. If the width is more than 2x the height or vice versa, the additional
 * bits are linear. So we interleave as much as possible, and then add what's
 * remaining. Finally, layers are strided linear and added at the end.
 */
static uint
index_metadata(uint3 c, uint width, uint height, uint layer_stride)
{
   uint major_coord = width > height ? c.x : c.y;
   uint minor_dim = min(width, height);

   uint intl_bits = util_logbase2_ceil(minor_dim);
   uint intl_mask = (1 << intl_bits) - 1;
   uint2 intl_coords = c.xy & intl_mask;

   return nir_interleave(intl_coords.x, intl_coords.y) +
          ((major_coord & ~intl_mask) << intl_bits) + (layer_stride * c.z);
}

/*
 * For multisampled images, a 2x2 or 1x2 group of samples form a single pixel.
 * The following two helpers convert a coordinate in samples into a coordinate
 * in pixels and a sample ID, respectively. They each assume that samples > 1.
 */
static int4
decompose_px(int4 c, uint samples)
{
   if (samples == 4)
      c.xy >>= 1;
   else
      c.y >>= 1;

   return c;
}

static uint
sample_id(int4 c, uint samples)
{
   if (samples == 4)
      return (c.x & 1) | ((c.y & 1) << 1);
   else
      return c.y & 1;
}

/*
 * AGX2-only in-place decompression kernel. For multisampled attachments it
 * calls libagx_image_texel_address() directly, which unpacks the AGX2 PBE
 * descriptor and its software sideband layout. Callers must guard dispatch
 * with hk_descriptor_generation_is_agx2().
 */
KERNEL(32)
libagx_decompress(constant struct libagx_decompress_images *images,
                  global uint64_t *metadata, uint64_t tile_uncompressed,
                  uint32_t metadata_layer_stride_tl, uint32_t layer_stride_sa,
                  uint16_t metadata_width_tl, uint16_t metadata_height_tl,
                  uint log2_samples__3 /* 1x, 2x, 4x */)
{
   uint samples = 1 << log2_samples__3;

   /* Index into the metadata buffer */
   uint index_tl = index_metadata(cl_group_id, metadata_width_tl,
                                  metadata_height_tl, metadata_layer_stride_tl);

   /* If the tile is already uncompressed, there's nothing to do. */
   if (metadata[index_tl] == tile_uncompressed)
      return;

   /* Tiles are 16x16 */
   uint2 coord_sa = (cl_group_id.xy * 16);
   uint layer = cl_group_id.z;

   /* Since we use a 32x1 workgroup, each work-item handles half of a row. */
   uint offs_y_sa = cl_local_id.x >> 1;
   uint offs_x_sa = (cl_local_id.x & 1) ? 8 : 0;

   int2 img_coord_sa_2d = convert_int2(coord_sa) + (int2)(offs_x_sa, offs_y_sa);
   int4 img_coord_sa = (int4)(img_coord_sa_2d.x, img_coord_sa_2d.y, layer, 0);

   /* Read our half-row into registers. */
   uint4 texels[8];
   for (uint i = 0; i < 8; ++i) {
      int4 c_sa = img_coord_sa + (int4)(i, 0, 0, 0);
      if (samples == 1) {
         texels[i] = nir_bindless_image_load(
            HANDLE(compressed), c_sa, 0, 0, GLSL_SAMPLER_DIM_2D, true, 0,
            ACCESS_IN_BOUNDS, nir_type_uint32);
      } else {
         int4 dec_px = decompose_px(c_sa, samples);
         texels[i] = nir_bindless_image_load(
            HANDLE(compressed), dec_px, sample_id(c_sa, samples), 0,
            GLSL_SAMPLER_DIM_MS, true, 0, ACCESS_IN_BOUNDS,
            nir_type_uint32);
      }
   }

   /* Wait for all compressed fetches across the tile before overwriting it
    * in place. A subgroup barrier lowers to nothing on AGX2, allowing the
    * first store to issue before pending texture scoreboards complete.
    */
   barrier(CLK_LOCAL_MEM_FENCE);

   /* Now that the whole tile is read, we write without racing. */
   for (uint i = 0; i < 8; ++i) {
      int4 c_sa = img_coord_sa + (int4)(i, 0, 0, 0);
      if (samples == 1) {
         nir_bindless_image_store(HANDLE(uncompressed), c_sa, 0, texels[i], 0,
                                  GLSL_SAMPLER_DIM_2D, true, 0,
                                  ACCESS_NON_READABLE, nir_type_uint32);
      } else {
         int4 dec_px = decompose_px(c_sa, samples);
         /* Multisampled stores address the linear buffer PBE by sample index.
          * Lowering a formatless layered MS store would divide the byte layer
          * stride by PIPE_FORMAT_NONE's block size (1), striding layers by
          * blocksize_B too far; compute the in-layer index and add the exact
          * sample layer stride here.
          */
         uint index_sa =
            (uint)libagx_image_texel_address(&images->uncompressed,
                                             convert_uint4(dec_px),
                                             sample_id(c_sa, samples), 1,
                                             false, true, false, true) +
            (layer * layer_stride_sa);
         int4 coord2d = (int4)(index_sa % AGX_TEXTURE_BUFFER_WIDTH,
                               index_sa / AGX_TEXTURE_BUFFER_WIDTH, 0, 0);

         nir_bindless_image_store(HANDLE(uncompressed), coord2d, 0, texels[i],
                                  0, GLSL_SAMPLER_DIM_2D, false, 0,
                                  ACCESS_NON_READABLE, nir_type_uint32);
      }
   }

   /* Drain the uncompressed PBE body stores before marking the tile
    * uncompressed. On G13S, the dispatch-level AGX_BARRIER_ALL | AGX_PREGFX
    * barrier alone did not order the final decompress dispatch's PBE body
    * stores before the following render pass without an in-kernel PBE drain.
    */
   nir_fence_pbe_to_tex_agx();

   /* We've replaced the body buffer. Mark the tile as uncompressed. */
   if (cl_local_id.x == 0) {
      metadata[index_tl] = tile_uncompressed;
   }

   /* Flush the metadata update and invalidate texture caches before the
    * dispatch completes (the dispatch barrier orders visibility to the
    * subsequent render pass).
    */
   atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE,
                          memory_order_acq_rel, memory_scope_device);
}

/*
 * Mark a compressed image's metadata tiles as uncompressed without reading or
 * rewriting the body buffer. Used before a spilled render pass when the
 * attachment's previous contents are discarded, avoiding compressed texture
 * reads of stale or uninitialized metadata and redundant in-place body writes.
 */
KERNEL(32)
libagx_reset_compression_metadata(global uint64_t *metadata,
                                  uint64_t tile_uncompressed,
                                  uint32_t metadata_layer_stride_tl,
                                  uint16_t metadata_width_tl,
                                  uint16_t metadata_height_tl)
{
   if (cl_global_id.x < metadata_width_tl &&
       cl_global_id.y < metadata_height_tl) {
      uint index_tl = index_metadata(cl_global_id, metadata_width_tl,
                                     metadata_height_tl,
                                     metadata_layer_stride_tl);
      metadata[index_tl] = tile_uncompressed;
   }

   atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE,
                          memory_order_acq_rel, memory_scope_device);
}

static ushort2
stretched_sa_to_px(ushort2 px, uint samples)
{
   /* clang-format off */
   switch (samples) {
   case  4: return (ushort2)(px.x / 2, px.y / 2);
   case  2: return (ushort2)(px.x, px.y / 2);
   case  1: return px;
   default: UNREACHABLE("invalid sample count");
   }
   /* clang-format on */
}

/* A half promoted exactly to float needs no rounding to recover its bits. */
static uint
libagx_m5_recover_half(uint value)
{
   const uint sign = (value >> 16) & 0x8000u;
   const uint exponent = (value >> 23) & 0xffu;
   const uint fraction = value & 0x7fffffu;
   if (exponent == 255)
      return sign | 0x7c00u | (fraction >> 13);
   if (exponent < 103)
      return sign;
   if (exponent < 113)
      return sign | ((fraction | 0x800000u) >> (126u - exponent));
   return sign | ((exponent - 112u) << 10) | (fraction >> 13);
}

/* Native RGBA16F spill conversion. The texture heap index names the existing
 * compressed view; the native fetch promotes its half lanes exactly to float.
 * Each workgroup reads a complete metadata tile before overwriting its body.
 */
KERNEL(32)
libagx_m5_decompress_rgba16f(global void *texture_heap,
                            global void *sampler_heap,
                            global uint64_t *metadata, global uint2 *body,
                            uint read_index, uint metadata_width_tl,
                            uint metadata_height_tl, uint aligned_width_px,
                            ushort tile_width_px, ushort tile_height_px,
                            uint samples_log2__2)
{
   const uint samples = 2u << samples_log2__2;
   const uint index_tl = index_metadata(cl_group_id, metadata_width_tl,
                                        metadata_height_tl, 0);
   if (metadata[index_tl] == ~(uint64_t)0)
      return;

   const uint handle = nir_bindless_image_agx(read_index, 0);
   const uint2 origin = cl_group_id.xy * 16u;
   const uint2 lane = (uint2)((cl_local_id.x & 1u) * 8u,
                              cl_local_id.x >> 1);
   uint2 values[8];
   for (uint i = 0; i < 8; i++) {
      const int4 sample_coord = (int4)(origin.x + lane.x + i,
                                       origin.y + lane.y, 0, 0);
      const uint4 v = nir_bindless_image_load(
         handle, decompose_px(sample_coord, samples),
         sample_id(sample_coord, samples), 0, GLSL_SAMPLER_DIM_MS, false,
         0, ACCESS_IN_BOUNDS, nir_type_uint32);
      values[i] = (uint2)(libagx_m5_recover_half(v.x) |
                            (libagx_m5_recover_half(v.y) << 16),
                          libagx_m5_recover_half(v.z) |
                            (libagx_m5_recover_half(v.w) << 16));
   }

   barrier(CLK_LOCAL_MEM_FENCE);
   for (uint i = 0; i < 8; i++) {
      const int4 sample_coord = (int4)(origin.x + lane.x + i,
                                       origin.y + lane.y, 0, 0);
      const int4 pixel = decompose_px(sample_coord, samples);
      const uint offset = libagx_twiddle_coordinates(
         convert_ushort2(pixel.xy), tile_width_px, tile_height_px,
         aligned_width_px);
      body[offset * samples + sample_id(sample_coord, samples)] =
         values[i];
   }
   if (cl_local_id.x == 0)
      metadata[index_tl] = ~(uint64_t)0;
}

/* Native G17P single-sample 16-byte/texel (R32G32B32A32) in-place spill
 * decompression. Each 32-thread workgroup covers one 16x16 metadata tile
 * (8 texels per thread), reading through the view's compressed 2D-array
 * eMRT texture descriptor before overwriting the twiddled body and marking
 * the metadata tile uncompressed.
 */
KERNEL(32)
libagx_g17p_decompress_16b(global void *texture_heap,
                           global void *sampler_heap,
                           global uint64_t *metadata,
                           global uint4 *body,
                           uint64_t tile_uncompressed,
                           uint read_index,
                           uint metadata_layer_stride_tl,
                           uint body_layer_stride_el,
                           uint metadata_width_tl,
                           uint metadata_height_tl,
                           uint level_width_px,
                           uint level_height_px,
                           uint aligned_width_px,
                           ushort tile_width_px,
                           ushort tile_height_px)
{
   const uint index_tl = index_metadata(cl_group_id, metadata_width_tl,
                                        metadata_height_tl,
                                        metadata_layer_stride_tl);
   if (metadata[index_tl] == tile_uncompressed)
      return;

   const uint handle = nir_bindless_image_agx(read_index, 0);
   const uint2 origin = cl_group_id.xy * 16u;
   const uint layer = cl_group_id.z;
   const uint2 lane = (uint2)((cl_local_id.x & 1u) * 8u,
                              cl_local_id.x >> 1);
   const uint y = origin.y + lane.y;
   uint4 values[8];
   for (uint i = 0; i < 8; i++) {
      const uint x = origin.x + lane.x + i;
      if (x < level_width_px && y < level_height_px) {
         const int4 coord = (int4)(x, y, layer, 0);
         values[i] = nir_bindless_image_load(
            handle, coord, 0, 0, GLSL_SAMPLER_DIM_2D, true, 0,
            ACCESS_IN_BOUNDS, nir_type_uint32);
      }
   }

   barrier(CLK_LOCAL_MEM_FENCE);

   for (uint i = 0; i < 8; i++) {
      const uint x = origin.x + lane.x + i;
      if (x < level_width_px && y < level_height_px) {
         const uint offset = libagx_twiddle_coordinates(
            (ushort2)(x, y), tile_width_px, tile_height_px,
            aligned_width_px);
         body[layer * body_layer_stride_el + offset] = values[i];
      }
   }

   atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE,
                          memory_order_acq_rel, memory_scope_device);
   barrier(CLK_LOCAL_MEM_FENCE);
   if (cl_local_id.x == 0)
      metadata[index_tl] = tile_uncompressed;
   atomic_work_item_fence(CLK_GLOBAL_MEM_FENCE | CLK_IMAGE_MEM_FENCE,
                          memory_order_acq_rel, memory_scope_device);
}

/*
 * Compute kernel to clear a compressed image to a solid colour by filling the
 * compression metadata to all solid and writing the colour to each tile.
 *
 * Invoked as <width in compression tiles, height in compression tiles,
 * layers>. Each invocation clears a single compression tile: 16x16 samples,
 * or 16x8 for M5 16-byte formats (ail_metadata_tile, given here as the log2
 * of its height).
 *
 * Despite the ALU complexity, this is twice as fast as clearing with the
 * 3D pipe for a few reasons.
 *
 * 1. At a kernel/firmware level, CDM submission is much faster than VDM.
 * 2. Far fewer threads are dispatched (1 per 16x16 tile, rather than 1 per
 *    pixel like the background program used to implement 3D clears).
 * 3. No useless tilebuffer load/stores.
 * 4. Many CDM dispatches can share a single control stream, whereas a VDM
 *    control stream covers only 1 render pass.
 *
 * AGX compression is not fully understood, but we know enough to clear solid
 * colours:
 *
 * - Metadata buffer has one byte per 8x4 subtile, grouped into 16x16 tiles with
 *   8 bytes of metadata.
 * - The 8x4 "solid mode" will read its colour from the beginning of the tile in
 *   the body buffer, addressed as-if twiddled. To clear the whole tile, we need
 *   (16x16) / (8x4) = 8 copies of the colour per 16x16 tile.
 *
 * Also, multisampling requires stretching the coordinates. This corresponds to
 * ail_effective_width_sa and ail_effective_height_sa. With multisampling, we
 * are invoked on tiles of 16x16 samples, so we need to squish back to tiles of
 * 16x8 or 8x8 pixels.
 */
KERNEL(32)
libagx_fast_clear(global uint2 *meta, global uint4 *body, uint width_tl,
                  uint height_tl, uint tile_w_el, uint tile_h_el,
                  uint meta_layer_stride_tl, uint body_layer_stride_uint4,
                  uint meta_tile_h_log2, uint blocksize_B_log2__5,
                  uint samples_log2__3, uint a,
                  uint b, uint c, uint d, uint solid_mode)
{
   uint samples = 1 << samples_log2__3;
   uint blocksize_B = 1 << blocksize_B_log2__5;

   /* Set the tile to be solid */
   meta[index_metadata(cl_global_id, width_tl, height_tl, meta_layer_stride_tl)] =
      solid_mode;

   ushort2 coord_el = (ushort2)(cl_global_id.x << 4,
                                cl_global_id.y << meta_tile_h_log2);

   /* Calculate the offset of the tile in the body buffer */
   ushort2 dim_sa = (ushort2)(width_tl << 4, height_tl << meta_tile_h_log2);
   uint aligned_width_px =
      align(stretched_sa_to_px(dim_sa, samples).x, tile_w_el);

   ushort2 coord_px = stretched_sa_to_px(coord_el, samples);
   uint twiddled = libagx_twiddle_coordinates(coord_px, tile_w_el, tile_h_el,
                                              aligned_width_px);

   /* We scale blocksize_B by # of samples to make addressing work properly and
    * to splat the clear colour across the samples. See the corresponding scale
    * in libagx_image_texel_address.
    *
    * TODO: Check that this is actually right? Passes CTS at least.
    */
   blocksize_B *= samples;
   uint body_offs = (twiddled * blocksize_B) / sizeof(uint4);
   body_offs += cl_global_id.z * body_layer_stride_uint4;

   /* Write 8 copies of the clear colour. We vectorize with 128-bit writes,
    * unless 8 copies would be less than 128-bit which only occurs for 8-bit
    * formats where we write only 64-bit.
    */
   if (blocksize_B == 1) {
      *(global uint2 *)(body + body_offs) = (uint2)(a, b);
   } else {
      for (uint i = 0; i < (8 * blocksize_B) / 16; ++i) {
         body[body_offs + i] = (uint4)(a, b, c, d);
      }
   }
}
