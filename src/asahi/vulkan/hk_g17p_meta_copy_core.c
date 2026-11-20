/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */

#include "hk_g17p_meta_copy_core.h"
#include "util/compiler.h"
#include "util/u_math.h"
#include "vk_format.h"

#include <limits.h>
#include <string.h>

bool
hk_g17p_build_mip_axis(uint32_t src_extent, uint32_t dst_extent,
                       uint32_t *entries)
{
   if (!entries || !src_extent || !dst_extent || src_extent > UINT16_MAX + 1u ||
       !(src_extent == dst_extent ||
         src_extent == (uint64_t)dst_extent * 2u ||
         src_extent == (uint64_t)dst_extent * 2u + 1u))
      return false;

   if (src_extent == dst_extent) {
      for (uint32_t i = 0; i < dst_extent; ++i)
         entries[i] = i;
      return true;
   }

   /* vk_meta computes the normalized scale in double on the CPU, casts it to
    * f32 push data, then applies it to FragCoord with an FFMA. */
   const float scale = (float)(1.0 / (double)dst_extent);
   for (uint32_t i = 0; i < dst_extent; ++i) {
      /* This work happens once while recording the dispatch, not per pixel. */
      const float uv = ((float)i + 0.5f) * scale;
      const float source = uv * (float)src_extent - 0.5f;
      uint32_t lower = (uint32_t)source;
      uint32_t weight = (uint32_t)((source - (float)lower) * 256.0f + 0.5f);

      if (weight == 256) {
         ++lower;
         weight = 0;
      }

      if (lower >= src_extent)
         lower = src_extent - 1;

      entries[i] = lower | (weight << 16);
   }

   return true;
}

bool
hk_g17p_tile_address_config_init(
   uint32_t stride_el, uint32_t tile_width_el, uint32_t tile_height_el,
   struct hk_g17p_tile_address_config *config)
{
   if (config)
      memset(config, 0, sizeof(*config));

   if (!config || !stride_el || !tile_width_el || !tile_height_el ||
       (tile_width_el & (tile_width_el - 1)) ||
       (tile_height_el & (tile_height_el - 1)))
      return false;

   const uint32_t width_log2 = util_logbase2(tile_width_el);
   const uint32_t height_log2 = util_logbase2(tile_height_el);
   if (width_log2 > 7 || height_log2 > 7)
      return false;

   *config = (struct hk_g17p_tile_address_config){
      .tiles_per_row =
         ((uint64_t)stride_el + tile_width_el - 1) >> width_log2,
      .tile_width_log2 = width_log2,
      .tile_height_log2 = height_log2,
   };
   return true;
}

bool
hk_g17p_copy_aspect_plan(VkImageAspectFlags src, VkImageAspectFlags dst,
                         struct hk_g17p_copy_aspect_plan *plan)
{
   if (plan)
      memset(plan, 0, sizeof(*plan));
   if (!plan)
      return false;

   const VkImageAspectFlags color = VK_IMAGE_ASPECT_COLOR_BIT;
   const VkImageAspectFlags depth = VK_IMAGE_ASPECT_DEPTH_BIT;
   const VkImageAspectFlags stencil = VK_IMAGE_ASPECT_STENCIL_BIT;
   const bool src_single = src == color || src == depth || src == stencil;
   const bool dst_single = dst == color || dst == depth || dst == stencil;
   if (src == (depth | stencil) && dst == src) {
      *plan = (struct hk_g17p_copy_aspect_plan){
         .count = 2, .src = {depth, stencil}, .dst = {depth, stencil},
      };
   } else if (src_single && dst_single &&
              (src == dst || src == color || dst == color)) {
      *plan = (struct hk_g17p_copy_aspect_plan){
         .count = 1, .src = {src}, .dst = {dst},
      };
   } else {
      return false;
   }
   return true;
}

bool
hk_g17p_image_copy_element_plan(
   VkFormat src_format, VkFormat dst_format, const VkImageCopy2 *region,
   struct hk_g17p_image_copy_element_plan *plan)
{
   if (plan)
      memset(plan, 0, sizeof(*plan));

   const uint32_t src_block_B = vk_format_get_blocksize(src_format);
   const uint32_t dst_block_B = vk_format_get_blocksize(dst_format);
   const uint32_t src_block_w = vk_format_get_blockwidth(src_format);
   const uint32_t src_block_h = vk_format_get_blockheight(src_format);
   const uint32_t dst_block_w = vk_format_get_blockwidth(dst_format);
   const uint32_t dst_block_h = vk_format_get_blockheight(dst_format);
   if (!plan || !region || !src_block_B || src_block_B != dst_block_B ||
       !src_block_w || !src_block_h || !dst_block_w || !dst_block_h ||
       region->srcOffset.x < 0 || region->srcOffset.y < 0 ||
       region->dstOffset.x < 0 || region->dstOffset.y < 0 ||
       !region->extent.width || !region->extent.height)
      return false;

   *plan = (struct hk_g17p_image_copy_element_plan){
      .block_size_B = src_block_B,
      .src_x_el = (uint32_t)region->srcOffset.x / src_block_w,
      .src_y_el = (uint32_t)region->srcOffset.y / src_block_h,
      .dst_x_el = (uint32_t)region->dstOffset.x / dst_block_w,
      .dst_y_el = (uint32_t)region->dstOffset.y / dst_block_h,
      .width_el = DIV_ROUND_UP(region->extent.width, src_block_w),
      .height_el = DIV_ROUND_UP(region->extent.height, src_block_h),
   };
   return true;
}

bool
hk_g17p_image_copy_storage_block(uint32_t api_block_B, uint32_t src_storage_B,
                                 uint32_t dst_storage_B, uint32_t *storage_B)
{
   if (storage_B)
      *storage_B = 0;
   if (!storage_B || !api_block_B || src_storage_B != dst_storage_B ||
       src_storage_B < api_block_B)
      return false;

   *storage_B = src_storage_B;
   return true;
}

bool
hk_g17p_transfer_dst_layout_supported(VkImageLayout layout)
{
   return layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL ||
          layout == VK_IMAGE_LAYOUT_GENERAL ||
          layout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
}

bool
hk_g17p_transfer_src_layout_supported(VkImageLayout layout)
{
   return layout == VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL ||
          layout == VK_IMAGE_LAYOUT_GENERAL ||
          layout == VK_IMAGE_LAYOUT_SHARED_PRESENT_KHR;
}

bool
hk_g17p_bc_format_supported(VkFormat format)
{
   return hk_g17p_block_format_family(format) == HK_G17P_BLOCK_FORMAT_BC;
}

enum hk_g17p_block_format_family
hk_g17p_block_format_family(VkFormat format)
{
   if (format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK &&
       format <= VK_FORMAT_BC7_SRGB_BLOCK)
      return HK_G17P_BLOCK_FORMAT_BC;

   if (format >= VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK &&
       format <= VK_FORMAT_EAC_R11G11_SNORM_BLOCK)
      return HK_G17P_BLOCK_FORMAT_ETC2_EAC;

   if (format >= VK_FORMAT_ASTC_4x4_UNORM_BLOCK &&
       format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK)
      return HK_G17P_BLOCK_FORMAT_ASTC_LDR;

   if (format >= VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK &&
       format <= VK_FORMAT_ASTC_12x12_SFLOAT_BLOCK)
      return HK_G17P_BLOCK_FORMAT_ASTC_HDR;

   return HK_G17P_BLOCK_FORMAT_NONE;
}

static bool
hk_u64_mul(uint64_t a, uint64_t b, uint64_t *out)
{
   if (a && b > UINT64_MAX / a)
      return false;

   *out = a * b;
   return true;
}

static bool
hk_u64_add(uint64_t a, uint64_t b, uint64_t *out)
{
   if (b > UINT64_MAX - a)
      return false;

   *out = a + b;
   return true;
}

static uint32_t
hk_minify(uint32_t value, uint32_t level)
{
   const uint32_t minified = value >> level;
   return minified ? minified : 1;
}

static bool
hk_g17p_compressed_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   bool image_is_source, struct hk_g17p_compressed_copy_plan *plan)
{
   if (plan)
      memset(plan, 0, sizeof(*plan));

   if (!copy || !plan ||
       !agx_chip_native_or_m5(copy->chip) ||
       hk_g17p_block_format_family(copy->format) ==
          HK_G17P_BLOCK_FORMAT_NONE ||
       !(image_is_source
            ? hk_g17p_transfer_src_layout_supported(copy->layout)
            : hk_g17p_transfer_dst_layout_supported(copy->layout)) ||
       copy->samples != VK_SAMPLE_COUNT_1_BIT || copy->plane_count != 1 ||
       copy->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
       copy->mip_levels == 0 || copy->mip_level >= copy->mip_levels ||
       copy->mip_level >= 32 || copy->array_layers == 0 ||
       copy->image_extent.width == 0 || copy->image_extent.height == 0 ||
       copy->image_extent.depth == 0 || copy->copy_extent.width == 0 ||
       copy->copy_extent.height == 0 || copy->copy_extent.depth == 0 ||
       copy->image_offset.x < 0 || copy->image_offset.y < 0 ||
       copy->image_offset.z < 0)
      return false;

   const uint32_t block_w = vk_format_get_blockwidth(copy->format);
   const uint32_t block_h = vk_format_get_blockheight(copy->format);
   const uint32_t block_B = vk_format_get_blocksize(copy->format);
   if (!block_w || !block_h || (block_B != 8 && block_B != 16))
      return false;

   const uint32_t mip_w =
      hk_minify(copy->image_extent.width, copy->mip_level);
   const uint32_t mip_h =
      hk_minify(copy->image_extent.height, copy->mip_level);
   const uint32_t mip_d =
      hk_minify(copy->image_extent.depth, copy->mip_level);
   const uint32_t x = copy->image_offset.x;
   const uint32_t y = copy->image_offset.y;
   const uint32_t z = copy->image_offset.z;
   const uint32_t width = copy->copy_extent.width;
   const uint32_t height = copy->copy_extent.height;

   uint32_t first_slice, slice_count;
   switch (copy->image_type) {
   case VK_IMAGE_TYPE_1D:
      if (copy->image_extent.height != 1 || copy->image_extent.depth != 1 ||
          y != 0 || z != 0 || height != 1 || copy->copy_extent.depth != 1)
         return false;
      FALLTHROUGH;
   case VK_IMAGE_TYPE_2D:
      if (copy->image_extent.depth != 1 || z != 0 ||
          copy->copy_extent.depth != 1 ||
          copy->base_array_layer >= copy->array_layers ||
          copy->layer_count == 0)
         return false;

      first_slice = copy->base_array_layer;
      slice_count = copy->layer_count == VK_REMAINING_ARRAY_LAYERS
                       ? copy->array_layers - copy->base_array_layer
                       : copy->layer_count;
      if (slice_count > copy->array_layers - copy->base_array_layer)
         return false;
      break;

   case VK_IMAGE_TYPE_3D:
      if (copy->array_layers != 1 || copy->base_array_layer != 0 ||
          copy->layer_count != 1 || z >= mip_d ||
          copy->copy_extent.depth > mip_d - z)
         return false;

      first_slice = z;
      slice_count = copy->copy_extent.depth;
      break;

   default:
      return false;
   }

   /* A copy covers whole blocks. As on AGX2, whose compute copy converts
    * the region with util_format_get_nblocks{x,y}, an extent that is not a
    * multiple of the block and does not reach the level edge still copies
    * every block it touches. The offset is block-aligned and the extent stays
    * inside the level, so the rounded-up blocks stay inside it too. */
   if (x >= mip_w || y >= mip_h || width > mip_w - x ||
       height > mip_h - y || (x % block_w) || (y % block_h) ||
       (copy->buffer_row_length &&
        (copy->buffer_row_length < width ||
         copy->buffer_row_length % block_w)) ||
       (copy->buffer_image_height &&
        (copy->buffer_image_height < height ||
         copy->buffer_image_height % block_h)) ||
       buffer_offset % block_B)
      return false;

   const uint32_t row_px =
      copy->buffer_row_length ? copy->buffer_row_length : width;
   const uint32_t image_h_px =
      copy->buffer_image_height ? copy->buffer_image_height : height;
   const uint32_t blocks_x =
      ((uint64_t)width + block_w - 1) / block_w;
   const uint32_t blocks_y =
      ((uint64_t)height + block_h - 1) / block_h;
   if (blocks_x > UINT32_MAX / blocks_y)
      return false;

   const uint64_t buffer_row_blocks =
      ((uint64_t)row_px + block_w - 1) / block_w;
   const uint64_t buffer_image_rows =
      ((uint64_t)image_h_px + block_h - 1) / block_h;
   uint64_t buffer_row_B, buffer_slice_B;
   if (!hk_u64_mul(buffer_row_blocks, block_B, &buffer_row_B) ||
       buffer_row_B > INT32_MAX ||
       !hk_u64_mul(buffer_image_rows, buffer_row_B, &buffer_slice_B))
      return false;

   uint64_t span = 0, term;
   if (!hk_u64_mul(slice_count - 1, buffer_slice_B, &span) ||
       !hk_u64_mul(blocks_y - 1, buffer_row_B, &term) ||
       !hk_u64_add(span, term, &span) ||
       !hk_u64_mul(blocks_x, block_B, &term) ||
       !hk_u64_add(span, term, &span) || buffer_offset > buffer_size ||
       span > buffer_size - buffer_offset)
      return false;

   *plan = (struct hk_g17p_compressed_copy_plan){
      .block_width = block_w,
      .block_height = block_h,
      .block_size_B = block_B,
      .base_x_el = x / block_w,
      .base_y_el = y / block_h,
      .blocks_x = blocks_x,
      .blocks_y = blocks_y,
      .first_slice = first_slice,
      .slice_count = slice_count,
      .buffer_row_stride_B = buffer_row_B,
      .buffer_slice_stride_B = buffer_slice_B,
      .buffer_span_B = span,
   };
   return true;
}

bool
hk_g17p_compressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan)
{
   return hk_g17p_compressed_copy_plan(copy, buffer_offset, buffer_size,
                                       false, plan);
}

bool
hk_g17p_compressed_image_to_buffer_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_compressed_copy_plan *plan)
{
   return hk_g17p_compressed_copy_plan(copy, buffer_offset, buffer_size,
                                       true, plan);
}

static bool
hk_g17p_uncompressed_upload_format(VkFormat format)
{
   const uint32_t block_B = vk_format_get_blocksize(format);

   /* This path copies raw format elements and never interprets their colour
    * channels.  Admit every ordinary power-of-two format through 16 bytes;
    * the caller separately excludes depth/stencil and multi-plane images.
    *
    * A single-plane 4:2:2 element is an ordinary element too -- one 2x1 pixel
    * block of eight bytes -- and ail lays those images out on that same grid
    * (ail_initialize_linear/miptree size every level with
    * util_format_get_nblocksx, and the linear assert admits
    * UTIL_FORMAT_LAYOUT_SUBSAMPLED), so the raw copy walks it unchanged once
    * the region is converted to elements below.  Block-compressed formats are
    * excluded: hk_CmdCopyBufferToImage2 routes them to their own planner
    * before reaching this one.
    */
   return !vk_format_is_block_compressed(format) &&
          (block_B == 1 || block_B == 2 || block_B == 4 || block_B == 8 ||
           block_B == 16);
}

bool
hk_g17p_uncompressed_buffer_to_image_copy_plan(
   const struct hk_g17p_buffer_to_image_copy *copy,
   VkDeviceSize buffer_offset, VkDeviceSize buffer_size,
   struct hk_g17p_uncompressed_copy_plan *plan)
{
   if (plan)
      memset(plan, 0, sizeof(*plan));

   if (!copy || !plan || !hk_g17p_uncompressed_upload_format(copy->format) ||
       (copy->image_type != VK_IMAGE_TYPE_1D &&
        copy->image_type != VK_IMAGE_TYPE_2D &&
        copy->image_type != VK_IMAGE_TYPE_3D) ||
       copy->plane_count != 1 ||
       copy->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
       !hk_g17p_buffer_to_image_copy_supported(copy))
      return false;

   const uint32_t block_B = vk_format_get_blocksize(copy->format);
   if ((block_B != 1 && block_B != 2 && block_B != 4 && block_B != 8 &&
        block_B != 16) ||
       buffer_offset % block_B)
      return false;

   const uint32_t width = copy->copy_extent.width;
   const uint32_t height = copy->copy_extent.height;
   if (width > UINT32_MAX / height)
      return false;

   /* Everything below this point counts elements, not pixels.  For every plain
    * format the block is 1x1 and the two are the same; for a single-plane
    * 4:2:2 format one element spans two pixels horizontally.  Vulkan requires
    * the image offset to be block-aligned, and bufferRowLength/
    * bufferImageHeight -- which are in pixels -- to be block multiples too, so
    * reject anything that is not rather than silently rounding it. */
   const uint32_t block_w = vk_format_get_blockwidth(copy->format);
   const uint32_t block_h = vk_format_get_blockheight(copy->format);
   if (!block_w || !block_h ||
       (copy->image_offset.x % block_w) || (copy->image_offset.y % block_h) ||
       (copy->buffer_row_length && copy->buffer_row_length % block_w) ||
       (copy->buffer_image_height && copy->buffer_image_height % block_h))
      return false;

   const uint32_t row_px =
      copy->buffer_row_length ? copy->buffer_row_length : width;
   const uint32_t image_height_px =
      copy->buffer_image_height ? copy->buffer_image_height : height;
   /* A copy covers whole elements: an extent that stops mid-element still
    * transfers the element it touches, exactly as the compressed planner
    * above rounds its block counts up. */
   const uint32_t width_el = ((uint64_t)width + block_w - 1) / block_w;
   const uint32_t height_el = ((uint64_t)height + block_h - 1) / block_h;
   const uint32_t row_el = ((uint64_t)row_px + block_w - 1) / block_w;
   const uint32_t image_height_el =
      ((uint64_t)image_height_px + block_h - 1) / block_h;
   const bool volume = copy->image_type == VK_IMAGE_TYPE_3D;
   const uint32_t first_slice =
      volume ? copy->image_offset.z : copy->base_array_layer;
   const uint32_t slice_count = volume
      ? copy->copy_extent.depth
      : (copy->layer_count == VK_REMAINING_ARRAY_LAYERS
            ? copy->array_layers - copy->base_array_layer
            : copy->layer_count);

   uint64_t row_B, slice_B, span = 0, term;
   if (!hk_u64_mul(row_el, block_B, &row_B) || row_B > UINT32_MAX ||
       !hk_u64_mul(image_height_el, row_B, &slice_B) ||
       !hk_u64_mul(slice_count - 1, slice_B, &span) ||
       !hk_u64_mul(height_el - 1, row_B, &term) ||
       !hk_u64_add(span, term, &span) ||
       !hk_u64_mul(width_el, block_B, &term) ||
       !hk_u64_add(span, term, &span) || buffer_offset > buffer_size ||
       span > buffer_size - buffer_offset)
      return false;

   *plan = (struct hk_g17p_uncompressed_copy_plan){
      .block_size_B = block_B,
      .base_x_el = (uint32_t)copy->image_offset.x / block_w,
      .base_y_el = (uint32_t)copy->image_offset.y / block_h,
      .width_el = width_el,
      .height_el = height_el,
      .first_slice = first_slice,
      .slice_count = slice_count,
      .buffer_row_stride_B = row_B,
      .buffer_slice_stride_B = slice_B,
      .buffer_span_B = span,
   };
   return true;
}

bool
hk_g17p_buffer_to_image_copy_supported(
   const struct hk_g17p_buffer_to_image_copy *copy)
{
   const bool one_dimensional =
      copy && copy->image_type == VK_IMAGE_TYPE_1D;
   const bool volume = copy && copy->image_type == VK_IMAGE_TYPE_3D;
   if (!copy || !agx_chip_native_or_m5(copy->chip) ||
       !hk_g17p_transfer_dst_layout_supported(copy->layout) ||
       (!one_dimensional && copy->image_type != VK_IMAGE_TYPE_2D && !volume) ||
       copy->samples != VK_SAMPLE_COUNT_1_BIT ||
       copy->mip_levels == 0 || copy->array_layers == 0 ||
       copy->mip_level >= copy->mip_levels || copy->mip_level >= 32 ||
       copy->base_array_layer >= copy->array_layers ||
       copy->layer_count == 0 ||
       (copy->layer_count != VK_REMAINING_ARRAY_LAYERS &&
        copy->layer_count > copy->array_layers - copy->base_array_layer) ||
       (volume && (copy->array_layers != 1 || copy->base_array_layer != 0 ||
                   copy->layer_count != 1)) ||
       copy->image_offset.x < 0 || copy->image_offset.y < 0 ||
       (volume ? copy->image_offset.z < 0 : copy->image_offset.z != 0) ||
       (!volume && copy->image_extent.depth != 1) ||
       (!volume && copy->copy_extent.depth != 1) ||
       (one_dimensional &&
        (copy->image_extent.height != 1 || copy->image_offset.y != 0 ||
         copy->copy_extent.height != 1)) ||
       copy->image_extent.width == 0 || copy->image_extent.height == 0 ||
       copy->image_extent.depth == 0 || copy->copy_extent.width == 0 ||
       copy->copy_extent.height == 0 || copy->copy_extent.depth == 0 ||
       (volume && copy->copy_extent.depth - 1 >
                     (uint32_t)(INT32_MAX - copy->image_offset.z)))
      return false;

   VkFormat plane_format = copy->format;
   uint32_t width_scale = 1, height_scale = 1;
   const struct vk_format_ycbcr_info *ycbcr =
      vk_format_get_ycbcr_info(copy->format);

   if (ycbcr && ycbcr->n_planes > 1) {
      if (one_dimensional || volume)
         return false;
      uint32_t plane;
      switch (copy->aspect) {
      case VK_IMAGE_ASPECT_PLANE_0_BIT:
         plane = 0;
         break;
      case VK_IMAGE_ASPECT_PLANE_1_BIT:
         plane = 1;
         break;
      case VK_IMAGE_ASPECT_PLANE_2_BIT:
         plane = 2;
         break;
      default:
         return false;
      }

      if (copy->plane_count != ycbcr->n_planes || plane >= ycbcr->n_planes ||
          copy->mip_levels != 1)
         return false;

      plane_format = ycbcr->planes[plane].format;
      width_scale = ycbcr->planes[plane].denominator_scales[0];
      height_scale = ycbcr->planes[plane].denominator_scales[1];
      if (!width_scale || !height_scale ||
          copy->image_extent.width % width_scale ||
          copy->image_extent.height % height_scale)
         return false;
   } else if ((copy->format == VK_FORMAT_D32_SFLOAT ||
               copy->format == VK_FORMAT_D16_UNORM ||
               copy->format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
               copy->format == VK_FORMAT_D16_UNORM_S8_UINT) &&
              copy->aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      /* Compressed D32/D16 cannot use the raw AIL upload kernel. The graphics
       * copy writes its bit-compatible R32_UINT/R16_UINT view through the
       * PBE, which also updates compression metadata. Keep this fallback
       * bounded to the single-level 2D shapes G17P depth compression admits
       * (hk_g17p_compression.h): arrays only where the caller uploads them
       * one layer per pass (native AGX3). A combined depth/stencil image is
       * split into a depth plane and a stencil plane; the stencil aspect is
       * the branch below. */
      const bool split = copy->format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
                         copy->format == VK_FORMAT_D16_UNORM_S8_UINT;
      if (copy->plane_count != (split ? 2 : 1) ||
          copy->image_type != VK_IMAGE_TYPE_2D || copy->mip_levels != 1 ||
          (copy->array_layers != 1 && !agx_chip_native_agx3(copy->chip)))
         return false;
      plane_format = (copy->format == VK_FORMAT_D32_SFLOAT ||
                      copy->format == VK_FORMAT_D32_SFLOAT_S8_UINT)
                        ? VK_FORMAT_R32_UINT
                        : VK_FORMAT_R16_UINT;
   } else if ((copy->format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
               copy->format == VK_FORMAT_D16_UNORM_S8_UINT) &&
              copy->aspect == VK_IMAGE_ASPECT_STENCIL_BIT) {
      /* A compressed stencil plane takes the same graphics copy through its
       * bit-compatible R8_UINT view, in the same single-level 2D shapes. */
      if (copy->plane_count != 2 || copy->image_type != VK_IMAGE_TYPE_2D ||
          copy->mip_levels != 1 ||
          (copy->array_layers != 1 && !agx_chip_native_agx3(copy->chip)))
         return false;
      plane_format = VK_FORMAT_R8_UINT;
   } else if (copy->plane_count != 1 ||
              copy->aspect != VK_IMAGE_ASPECT_COLOR_BIT) {
      return false;
   }

   /* A hardcoded allowlist here refused copies the driver advertises as
    * supported, and the consequence is far worse than one lost image: the
    * refusal fails vkEndCommandBuffer, so Zink skips vkQueueSubmit for the
    * WHOLE frame and latches device_lost, after which nothing that process
    * draws ever reaches the GPU again -- with no GL error at any point.
    * SuperTuxKart hangs that way, and vkQuake's black world was one refused
    * R32_UINT lightmap upload invalidating the staging batch it shares with
    * every other texture.
    *
    * What the generic vk_meta path below actually needs is colour-attachment
    * support, which is exactly ail_pixel_format[].renderable -- the same bit
    * hk_get_format_features() reports as COLOR_ATTACHMENT. Ask that instead.
    * Formats that are not renderable still fail closed and fall back to the
    * ordinary copy path. */
   if (!copy->plane_format_renderable)
      return false;
   /* Multi-plane destinations select their plane format above; a renderable
    * plane is still required for each one. */
   (void)plane_format;

   const uint32_t width =
      (copy->image_extent.width / width_scale) >> copy->mip_level;
   const uint32_t height =
      (copy->image_extent.height / height_scale) >> copy->mip_level;
   const uint32_t depth = copy->image_extent.depth >> copy->mip_level;
   const uint32_t mip_width = width ? width : 1;
   const uint32_t mip_height = height ? height : 1;
   const uint32_t mip_depth = depth ? depth : 1;
   const uint32_t x = copy->image_offset.x, y = copy->image_offset.y,
                  z = copy->image_offset.z;
   if (x >= mip_width || y >= mip_height ||
       copy->copy_extent.width > mip_width - x ||
       copy->copy_extent.height > mip_height - y ||
       (volume &&
        (z >= mip_depth || copy->copy_extent.depth > mip_depth - z)))
      return false;

   return (copy->buffer_row_length == 0 ||
           copy->buffer_row_length >= copy->copy_extent.width) &&
          (copy->buffer_image_height == 0 ||
           copy->buffer_image_height >= copy->copy_extent.height);
}

bool
hk_g17p_image_copy_slice(enum agx_chip chip, VkImageType src_type,
                         VkImageType dst_type, uint32_t src_layers,
                         uint32_t dst_layers, const VkImageCopy2 *region,
                         uint32_t slice, VkImageCopy2 *out)
{
   if (out)
      *out = (VkImageCopy2){0};

   const bool src_3d = src_type == VK_IMAGE_TYPE_3D;
   const bool dst_3d = dst_type == VK_IMAGE_TYPE_3D;
   const bool src_1d_or_2d = src_type == VK_IMAGE_TYPE_1D ||
                             src_type == VK_IMAGE_TYPE_2D;
   const bool dst_1d_or_2d = dst_type == VK_IMAGE_TYPE_1D ||
                             dst_type == VK_IMAGE_TYPE_2D;
   const bool valid_types = (src_1d_or_2d || src_3d) &&
                            (dst_1d_or_2d || dst_3d);
   const bool same_type = src_type == dst_type;
   const bool one_d_two_d =
      ((src_type == VK_IMAGE_TYPE_1D && dst_type == VK_IMAGE_TYPE_2D) ||
       (dst_type == VK_IMAGE_TYPE_1D && src_type == VK_IMAGE_TYPE_2D)) &&
      src_layers == 1 && dst_layers == 1;
   const bool two_d_three_d = (src_1d_or_2d && dst_3d) ||
                              (dst_1d_or_2d && src_3d);
   const bool valid_pair = same_type || one_d_two_d || two_d_three_d;
   if (!agx_chip_native_or_m5(chip) ||
       !region || !out || !valid_types ||
       !valid_pair || src_layers == 0 ||
       src_layers != dst_layers || slice >= src_layers ||
       ((src_3d || dst_3d) && region->extent.depth != src_layers) ||
       (!src_3d && !dst_3d && region->extent.depth != 1) ||
       (!src_3d && region->srcOffset.z != 0) ||
       (!dst_3d && region->dstOffset.z != 0) ||
       (src_3d &&
        (region->srcOffset.z < 0 ||
         slice > (uint32_t)(INT32_MAX - region->srcOffset.z))) ||
       (!src_3d &&
        slice > UINT32_MAX - region->srcSubresource.baseArrayLayer) ||
       (dst_3d &&
        (region->dstOffset.z < 0 ||
         slice > (uint32_t)(INT32_MAX - region->dstOffset.z))) ||
       (!dst_3d &&
        slice > UINT32_MAX - region->dstSubresource.baseArrayLayer))
      return false;

   *out = *region;
   out->extent.depth = 1;
   if (src_3d)
      out->srcOffset.z += slice;
   else
      out->srcSubresource.baseArrayLayer += slice;
   out->srcSubresource.layerCount = 1;
   if (dst_3d)
      out->dstOffset.z += slice;
   else
      out->dstSubresource.baseArrayLayer += slice;
   out->dstSubresource.layerCount = 1;
   return true;
}

bool
hk_g17p_image_blit_slice(enum agx_chip chip, VkImageType src_type,
                         VkImageType dst_type, uint32_t src_layers,
                         uint32_t dst_layers, const VkImageBlit2 *region,
                         uint32_t slice, VkImageBlit2 *out)
{
   if (out)
      *out = (VkImageBlit2){0};

   if (!agx_chip_native_agx3(chip) || !region || !out ||
       src_type != VK_IMAGE_TYPE_2D || dst_type != VK_IMAGE_TYPE_2D ||
       src_layers == 0 || src_layers != dst_layers || slice >= src_layers ||
       region->srcOffsets[0].z != 0 || region->srcOffsets[1].z != 1 ||
       region->dstOffsets[0].z != 0 || region->dstOffsets[1].z != 1 ||
       slice > UINT32_MAX - region->srcSubresource.baseArrayLayer ||
       slice > UINT32_MAX - region->dstSubresource.baseArrayLayer)
      return false;

   *out = *region;
   out->srcSubresource.baseArrayLayer += slice;
   out->srcSubresource.layerCount = 1;
   out->dstSubresource.baseArrayLayer += slice;
   out->dstSubresource.layerCount = 1;
   return true;
}

bool
hk_g17p_metadata_grid_init(uint32_t width_tl, uint32_t height_tl,
                           struct hk_g17p_metadata_grid *grid)
{
   if (grid)
      memset(grid, 0, sizeof(*grid));
   if (!grid || !width_tl || !height_tl || width_tl > UINT16_MAX ||
       height_tl > UINT16_MAX)
      return false;

   *grid = (struct hk_g17p_metadata_grid){
      .width_tl = width_tl,
      .height_tl = height_tl,
      .interleave_bits = util_logbase2_ceil(MIN2(width_tl, height_tl)),
      .major_is_x = width_tl > height_tl,
   };
   return true;
}

static uint32_t
hk_g17p_spread_bits(uint32_t v)
{
   v &= 0xffff;
   v = (v | (v << 8)) & 0x00ff00ff;
   v = (v | (v << 4)) & 0x0f0f0f0f;
   v = (v | (v << 2)) & 0x33333333;
   return (v | (v << 1)) & 0x55555555;
}

uint32_t
hk_g17p_metadata_tile_index(const struct hk_g17p_metadata_grid *grid,
                            uint32_t x_tl, uint32_t y_tl)
{
   const uint32_t mask = (1u << grid->interleave_bits) - 1;
   const uint32_t major = grid->major_is_x ? x_tl : y_tl;

   return (hk_g17p_spread_bits(x_tl & mask) |
           (hk_g17p_spread_bits(y_tl & mask) << 1)) +
          ((major & ~mask) << grid->interleave_bits);
}
