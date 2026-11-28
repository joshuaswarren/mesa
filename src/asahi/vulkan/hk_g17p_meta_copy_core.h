/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "asahi/lib/agx_chip.h"
#include "asahi/lib/agx_chip_caps.h"
#include "vulkan/vulkan_core.h"

#include <stdbool.h>
#include <stdint.h>

struct hk_g17p_buffer_to_image_copy {
   enum agx_chip chip;
   VkFormat format;
   /* Whether the destination plane's copy-view format is renderable, i.e.
    * whether the driver advertises COLOR_ATTACHMENT for it. D32/D16 use their
    * bit-compatible R32_UINT/R16_UINT copy views. The path below the gate is
    * generic vk_meta, which renders the staged texels through a native EOT
    * view, so that -- not membership of a hardcoded list -- is what decides
    * whether the copy can be serviced. The caller supplies it because the
    * answer lives in ail_pixel_format[], which needs the device. */
   bool plane_format_renderable;
   VkImageLayout layout;
   VkImageType image_type;
   VkSampleCountFlagBits samples;
   uint8_t plane_count;
   uint32_t mip_levels;
   uint32_t array_layers;
   VkExtent3D image_extent;

   VkImageAspectFlags aspect;
   uint32_t mip_level;
   uint32_t base_array_layer;
   uint32_t layer_count;
   VkOffset3D image_offset;
   VkExtent3D copy_extent;
   uint32_t buffer_row_length;
   uint32_t buffer_image_height;
};

struct hk_g17p_compressed_copy_plan {
   uint32_t block_width;
   uint32_t block_height;
   uint32_t block_size_B;
   uint32_t base_x_el;
   uint32_t base_y_el;
   uint32_t blocks_x;
   uint32_t blocks_y;
   uint32_t first_slice;
   uint32_t slice_count;
   uint64_t buffer_row_stride_B;
   uint64_t buffer_slice_stride_B;
   uint64_t buffer_span_B;
};

/* A raw linear-buffer to AIL-tiled upload.  This is intentionally narrower
 * than the generic render-based copy path: it covers the byte-exact R8 and
 * four-channel 8-bit texture shapes used by STK without interpreting their
 * colour space or channel order. */
struct hk_g17p_uncompressed_copy_plan {
   uint32_t block_size_B;
   uint32_t base_x_el;
   uint32_t base_y_el;
   uint32_t width_el;
   uint32_t height_el;
   uint32_t first_slice;
   uint32_t slice_count;
   uint64_t buffer_row_stride_B;
   uint64_t buffer_slice_stride_B;
   uint64_t buffer_span_B;
};

struct hk_g17p_tile_address_config {
   uint32_t tiles_per_row;
   uint32_t tile_width_log2;
   uint32_t tile_height_log2;
};

struct hk_g17p_image_copy_element_plan {
   uint32_t block_size_B;
   uint32_t src_x_el;
   uint32_t src_y_el;
   uint32_t dst_x_el;
   uint32_t dst_y_el;
   uint32_t width_el;
   uint32_t height_el;
};

struct hk_g17p_copy_aspect_plan {
   unsigned count;
   VkImageAspectFlags src[2];
   VkImageAspectFlags dst[2];
};

/* Keep combined depth/stencil copies plane-wise; maintenance8 also permits
 * a single depth or stencil aspect to copy raw bits to/from a color aspect. */
bool hk_g17p_copy_aspect_plan(VkImageAspectFlags src, VkImageAspectFlags dst,
                              struct hk_g17p_copy_aspect_plan *plan);

/* Reduce AIL's power-of-two tile geometry once on the CPU. G17P transfer tiles
 * are at most 128 elements per axis. Raw copy and mip kernels consume this
 * form so every element uses shifts and a seven-bit Morton encoder instead of
 * integer division and the generic fourteen-bit encoder. */
bool hk_g17p_tile_address_config_init(
   uint32_t stride_el, uint32_t tile_width_el, uint32_t tile_height_el,
   struct hk_g17p_tile_address_config *config);

/* Convert one Vulkan image-copy region to raw source-format elements. Vulkan
 * defines compressed/uncompressed compatible copies in the source format's
 * block geometry, while each endpoint keeps its own offset geometry. */
bool hk_g17p_image_copy_element_plan(
   VkFormat src_format, VkFormat dst_format, const VkImageCopy2 *region,
   struct hk_g17p_image_copy_element_plan *plan);

/* A raw copy moves storage, so its element is the physical one.  Three-
 * component images are stored four-component on the native path, so the API
 * block size (3, 6 or 12 bytes) is smaller than the element the tiled
 * addressing indexes; using it copies only three quarters of the surface.
 * Compressed pairs and every non-padded format already agree.  Fails when the
 * two endpoints disagree or either is narrower than the API block, so a new
 * format cannot silently reintroduce the mismatch.
 */
bool hk_g17p_image_copy_storage_block(uint32_t api_block_B,
                                      uint32_t src_storage_B,
                                      uint32_t dst_storage_B,
                                      uint32_t *storage_B);

/* The three layouts vkCmdCopyBufferToImage accepts as dstImageLayout:
 * VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL and
 * VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR.  Honeykrisp never acts on a layout, so
 * this is an API-validity check rather than a capability check: it exists only
 * so the G17P fast paths reject a destination the caller could not have meant.
 * Keep the whole legal set, because a client that only ever passed
 * VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL starts passing VK_IMAGE_LAYOUT_GENERAL
 * for every copy the moment it enables VK_KHR_unified_image_layouts. */
bool hk_g17p_transfer_dst_layout_supported(VkImageLayout layout);

/* The corresponding legal set for vkCmdCopyImageToBuffer srcImageLayout. */
bool hk_g17p_transfer_src_layout_supported(VkImageLayout layout);

/* The complete BC family has captured G17P sampled descriptors and shares the
 * hardware-validated Morton-of-4x4-blocks layout. */
bool hk_g17p_bc_format_supported(VkFormat format);

enum hk_g17p_block_format_family {
   HK_G17P_BLOCK_FORMAT_NONE,
   HK_G17P_BLOCK_FORMAT_BC,
   HK_G17P_BLOCK_FORMAT_ETC2_EAC,
   HK_G17P_BLOCK_FORMAT_ASTC_LDR,
   HK_G17P_BLOCK_FORMAT_ASTC_HDR,
};

/* Opaque block copies support both ASTC format classes. Keep HDR separate
 * from LDR: copy eligibility does not establish sampling/feature support. */
enum hk_g17p_block_format_family
hk_g17p_block_format_family(VkFormat format);

/* Native color uploads preserve rectangles and mip/layer/Z
 * subresources. Array and 3D regions are split into native single-slice
 * passes without changing source layout or copying index/pixel data. 8-bit
 * multi-planar YCbCr uploads use the same per-plane R8/RG8 render path.
 * Single-level 2D D32/D16 uploads (arrays on native AGX3, one layer per pass)
 * use an R32_UINT/R16_UINT color view so the PBE maintains compression
 * metadata for compressed destinations. */
bool hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy);

/* Resolve one block-compressed buffer upload into source strides and AIL
 * layer/slice coordinates. This is shared by BC, ETC2/EAC and ASTC LDR: only
 * their Vulkan block geometry differs. The returned byte span reaches the
 * final source byte read and therefore covers padding gaps between rows and
 * slices. */
bool hk_g17p_compressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan);

/* Resolve the same block geometry for a raw compressed image readback. */
bool hk_g17p_compressed_image_to_buffer_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan);

/* Resolve an R8/RGBA8/BGRA8 upload into one raw compute copy per 2D array
 * layer.  The returned byte span includes row and layer padding and is checked
 * against the source buffer before any command is emitted. */
bool hk_g17p_uncompressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_uncompressed_copy_plan *plan);

/* HAL200's qualified tile programs are single-layer.  Convert one logical
 * slice of a valid 2D array, 3D volume, or 2D-array<->3D image copy without
 * changing its rectangle or mip levels.  The resolved source/destination
 * slice counts must describe the same copy. */
bool hk_g17p_image_copy_slice(enum agx_chip chip, VkImageType src_type,
                              VkImageType dst_type, uint32_t src_layers,
                              uint32_t dst_layers,
                              const VkImageCopy2 *region, uint32_t slice,
                              VkImageCopy2 *out);

bool hk_g17p_image_blit_slice(enum agx_chip chip, VkImageType src_type,
                              VkImageType dst_type, uint32_t src_layers,
                              uint32_t dst_layers,
                              const VkImageBlit2 *region, uint32_t slice,
                              VkImageBlit2 *out);

/* Convert one 2D array slice of a valid resolve region into a single-layer
 * region. HAL200 writes compressed arrays one layer at a time (the BG/EOT
 * contract is single-layer), and the generic vk_meta_resolve_image path
 * otherwise reaches a compressed array through a layered render pass. The
 * resolved source/destination layer counts must match, and the layer count
 * must be at least one. */
bool hk_g17p_image_resolve_slice(enum agx_chip chip, VkImageType src_type,
                                 VkImageType dst_type, uint32_t src_layers,
                                 uint32_t dst_layers,
                                 const VkImageResolve2 *region, uint32_t slice,
                                 VkImageResolve2 *out);

/* One destination coordinate's exact G17P linear-filter footprint.  The low
 * half is the lower source coordinate and bits 16..23 are the Q8 upper-texel
 * weight.  A carried weight of 256 is normalized to the following source
 * coordinate with weight zero, keeping every table entry in one word. */
bool hk_g17p_build_mip_axis(uint32_t src_extent, uint32_t dst_extent,
                            uint32_t *entries);

/* Compressed G17P mip levels of at least this many texels are generated in
 * a render pass, which stores compressed tiles; smaller ones in compute,
 * which writes them raw. Measured per level on G17P
 * (g17p-vulkan-compressed-mip-blit --grid), a render pass in a chain costs
 * about 0.04 ms at any size up to 640x360, and compute about 0.26-0.33 ns
 * per texel: they cross between 480x270 (129,600 texels, compute faster
 * for smooth content) and 640x360 (230,400, the render pass faster for any
 * content), near 2^17.
 */
#define HK_G17P_MIP_PBE_MIN_TEXELS (1u << 17)

/* Classic-submission (G16G) crossover measured per level on Apple M4
 * (G16G B0, g17p-vulkan-compressed-mip-blit --grid, 42/42 bit-exact PASS
 * across gradient, texture and noise): a PBE render pass in a chain costs
 * about 0.13-0.24 ms up to 960x540 (vs ~0.04 ms on dual-submission G17P),
 * while compute takes about 0.13-0.24 ns per texel. Compute is 24-37x
 * faster at 32x32-64x64, 3.7-5.0x at 512x512, 1.9-3.0x at 960x540
 * (518,400 texels) and 1.18-1.53x at 768x768 (589,824 texels); they cross
 * between 1280x720 (921,600 texels: 1.012x gradient, 0.853x texture,
 * 1.030x noise) and 1024x1024 (1,048,576 texels = 2^20), above which PBE
 * wins across all content classes (whole-chain 4096x4096 1.23-3.41x vs
 * plain storage).
 */
#define HK_CLASSIC_MIP_PBE_MIN_TEXELS (1u << 20)

static inline bool
hk_g17p_mip_level_through_pbe(uint32_t width, uint32_t height)
{
   return (uint64_t)width * height >= HK_G17P_MIP_PBE_MIN_TEXELS;
}

/* The crossover for the chip's submission model. */
static inline bool
hk_mip_level_through_pbe(enum agx_chip chip, uint32_t width, uint32_t height)
{
   const uint64_t min_texels =
      agx_chip_submission_model(chip) == AGX_SUBMISSION_MODEL_CLASSIC
         ? HK_CLASSIC_MIP_PBE_MIN_TEXELS
         : HK_G17P_MIP_PBE_MIN_TEXELS;
   return (uint64_t)width * height >= min_texels;
}

/* Whether the chip supports generating exact 2x2 compressed mip levels in
 * compute via an imageblock store and compressed PBE block store. */
static inline bool
hk_g17p_block_mip_supported(enum agx_chip chip)
{
   return agx_chip_has_cap(chip, AGX_CAP_HAL200_LOSSLESS_COMPRESSION) &&
          agx_chip_isa_encoding(chip) == AGX_ISA_ENCODING_AGX3B;
}

/* Whether a compressed mip blit level should fall back to a render pass
 * instead of compute. Exact 2x2 levels on chips with compressed block-mip
 * store compressed tiles directly through the PBE in compute, avoiding
 * render-pass overhead and render/compute queue handoffs at all sizes;
 * odd levels and chips without block-mip fall back to a render pass at
 * the raw-compute crossover so large levels stay compressed. */
static inline bool
hk_compressed_mip_level_through_pbe(enum agx_chip chip, uint32_t src_width,
                                    uint32_t src_height, uint32_t width,
                                    uint32_t height)
{
   if (hk_g17p_block_mip_supported(chip) &&
       src_width == (uint64_t)width * 2 &&
       src_height == (uint64_t)height * 2)
      return false;
   return hk_mip_level_through_pbe(chip, width, height);
}

/* The 16x16 metadata tiles of one compressed level, as libagx_fast_clear
 * indexes them: the low bits of both tile coordinates interleave (X in the
 * even bits) up to the smaller grid dimension rounded to a power of two, and
 * the remaining bits of the longer axis follow linearly. */
struct hk_g17p_metadata_grid {
   uint32_t width_tl;
   uint32_t height_tl;
   uint32_t interleave_bits;
   bool major_is_x;
};

bool hk_g17p_metadata_grid_init(uint32_t width_tl, uint32_t height_tl,
                                struct hk_g17p_metadata_grid *grid);

/* The 64-bit metadata word index of tile (x_tl, y_tl). */
uint32_t hk_g17p_metadata_tile_index(const struct hk_g17p_metadata_grid *grid,
                                     uint32_t x_tl, uint32_t y_tl);

/* Dispatch grid for the AGX3 Gen10 compressed buffer-to-image compute upload.
 * Each 64-thread workgroup covers 32x16 texels (two adjacent 16x16 compression
 * blocks aligned to 16-texel boundaries, one block per 32-lane warp). */
struct hk_agx3_compressed_buf2img_grid {
   uint32_t tile_x0;
   uint32_t tile_y0;
   uint32_t groups_x;
   uint32_t groups_y;
};

/* Whether a canonical integer storage-PBE view format is supported by the
 * AGX3 Gen10 compressed buffer-to-image compute upload. */
bool hk_agx3_compressed_buf2img_view_format_supported(VkFormat view_format);

bool hk_agx3_compressed_buf2img_grid_init(
   uint32_t x0, uint32_t y0, uint32_t width, uint32_t height,
   uint32_t max_groups_x, uint32_t max_groups_y,
   struct hk_agx3_compressed_buf2img_grid *grid);

struct hk_g17p_linear_tiled_slice_copy {
   bool src_tiled;
   bool dst_tiled;
   bool src_compressed;
   bool dst_compressed;
   bool src_sparse;
   bool dst_sparse;
   VkImageType src_type;
   VkImageType dst_type;
   VkSampleCountFlagBits src_samples;
   VkSampleCountFlagBits dst_samples;
   uint32_t src_w_el;
   uint32_t src_h_el;
   uint32_t dst_w_el;
   uint32_t dst_h_el;
   uint32_t src_tile_w_el;
   uint32_t src_tile_h_el;
   uint32_t dst_tile_w_el;
   uint32_t dst_tile_h_el;
   uint32_t src_tiles_per_row;
   uint32_t dst_tiles_per_row;
   uint32_t raw_block_size_B;
   uint32_t src_level;
   uint32_t src_levels;
   uint32_t dst_level;
   uint32_t dst_levels;
   uint64_t src_level_size_B;
   uint64_t dst_level_size_B;
   uint32_t max_groups_x;
   struct hk_g17p_image_copy_element_plan plan;
};

/* Return the 16-byte-aligned byte span for a full-slice GPU-tiled copy whose
 * source and destination share identical tile geometry, or 0 when the copy
 * must use element-wise Morton addressing. */
uint64_t hk_g17p_linear_tiled_slice_bytes(
   const struct hk_g17p_linear_tiled_slice_copy *desc);

/* Whether one slice's GPU virtual address ranges are 16-byte aligned, within
 * their owning layouts, and non-overlapping for a linear uint4 copy. */
bool hk_g17p_linear_tiled_slice_addrs_valid(uint64_t dst_addr, uint64_t dst_off,
                                            uint64_t dst_layout_size_B,
                                            uint64_t src_addr, uint64_t src_off,
                                            uint64_t src_layout_size_B,
                                            uint64_t slice_B);
