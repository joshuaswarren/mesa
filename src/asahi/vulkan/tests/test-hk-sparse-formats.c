/* SPDX-License-Identifier: MIT */
#undef NDEBUG
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"
#include "hk_sparse.h"

static void
check_queries(struct hk_physical_device *pdev)
{
   VkPhysicalDevice handle = hk_physical_device_to_handle(pdev);
   const VkFormat formats[] = {
      VK_FORMAT_R8_UNORM,
      VK_FORMAT_R16_SFLOAT,
      VK_FORMAT_R32_SFLOAT,
      VK_FORMAT_R32G32_SFLOAT,
      VK_FORMAT_R32G32B32A32_SFLOAT,
      VK_FORMAT_ASTC_10x6_UNORM_BLOCK,
      VK_FORMAT_ASTC_12x10_UNORM_BLOCK,
      VK_FORMAT_ASTC_6x5_UNORM_BLOCK,
      VK_FORMAT_ASTC_8x8_UNORM_BLOCK,
      VK_FORMAT_BC6H_SFLOAT_BLOCK,
      VK_FORMAT_EAC_R11_SNORM_BLOCK,
      VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK,
      VK_FORMAT_R8G8B8_UNORM,
      VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM,
   };
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (VkImageType type = VK_IMAGE_TYPE_1D; type <= VK_IMAGE_TYPE_3D;
           ++type) {
         for (VkImageTiling tiling = VK_IMAGE_TILING_OPTIMAL;
              tiling <= VK_IMAGE_TILING_LINEAR; ++tiling) {
            bool supported = f < 12 && type != VK_IMAGE_TYPE_1D &&
                             tiling == VK_IMAGE_TILING_OPTIMAL;
            VkPhysicalDeviceImageFormatInfo2 info = {
               .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
               .format = formats[f],
               .type = type,
               .tiling = tiling,
               .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
               .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                        VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
            };
            VkImageFormatProperties2 props = {
               .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
            };
            VkResult result = hk_GetPhysicalDeviceImageFormatProperties2(
               handle, &info, &props);
            assert(result ==
                   (supported ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED));
            if (supported)
               assert(props.imageFormatProperties.sampleCounts ==
                      VK_SAMPLE_COUNT_1_BIT);
            else
               assert(props.imageFormatProperties.maxExtent.width == 0);

            for (unsigned sample = 1; sample <= 16; sample *= 2) {
               VkPhysicalDeviceSparseImageFormatInfo2 sparse = {
                  .sType =
                     VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
                  .format = formats[f],
                  .type = type,
                  .tiling = tiling,
                  .samples = sample,
                  .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
               };
               uint32_t count = 0;
               hk_GetPhysicalDeviceSparseImageFormatProperties2(handle, &sparse,
                                                                &count, NULL);
               assert(count == (supported && sample == 1));
               VkSparseImageFormatProperties2 page = {
                  .sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_FORMAT_PROPERTIES_2,
               };
               count = 1;
               hk_GetPhysicalDeviceSparseImageFormatProperties2(handle, &sparse,
                                                                &count, &page);
               assert(count == (supported && sample == 1));
               if (count) {
                  assert(page.properties.imageGranularity.depth == 1);
                  assert(
                     !!(page.properties.flags &
                        VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT) ==
                     (type == VK_IMAGE_TYPE_3D));
                  const VkExtent3D g = page.properties.imageGranularity;
                  assert((g.width / vk_format_get_blockwidth(formats[f])) *
                            (g.height / vk_format_get_blockheight(formats[f])) *
                            vk_format_get_blocksize(formats[f]) ==
                         65536);
               }
            }
         }
      }
   }

   /* Withdrawing an optional feature must close both query paths, while
    * ordinary images and opaque sparse binding remain available. */
   pdev->vk.supported_features.sparseResidencyImage3D = false;
   VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = VK_FORMAT_R8_UNORM,
      .type = VK_IMAGE_TYPE_3D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
               VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
   };
   VkImageFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
   assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
          VK_ERROR_FORMAT_NOT_SUPPORTED);
   VkPhysicalDeviceSparseImageFormatInfo2 sparse = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SPARSE_IMAGE_FORMAT_INFO_2,
      .format = info.format,
      .type = info.type,
      .tiling = info.tiling,
      .usage = info.usage,
      .samples = VK_SAMPLE_COUNT_1_BIT,
   };
   uint32_t count = 7;
   hk_GetPhysicalDeviceSparseImageFormatProperties2(handle, &sparse, &count,
                                                    NULL);
   assert(count == 0);
   info.flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT;
   assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
          VK_SUCCESS);
   info.flags = 0;
   assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
          VK_SUCCESS);
}

static void
check_tiles_and_tails(void)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   assert(dev);
   dev->vk.base.type = VK_OBJECT_TYPE_DEVICE;
   dev->dev.chip = AGX_CHIP_G17P;
   const enum pipe_format formats[] = {
      PIPE_FORMAT_R8_UNORM,           PIPE_FORMAT_R16_FLOAT,
      PIPE_FORMAT_R32_FLOAT,          PIPE_FORMAT_R32G32_FLOAT,
      PIPE_FORMAT_R32G32B32A32_FLOAT, PIPE_FORMAT_ASTC_10x6,
      PIPE_FORMAT_ASTC_12x10,         PIPE_FORMAT_ASTC_6x5,
      PIPE_FORMAT_ASTC_8x8,           PIPE_FORMAT_BPTC_RGB_FLOAT,
      PIPE_FORMAT_ETC2_R11_SNORM,     PIPE_FORMAT_ETC2_SRGBA8,
   };
   for (unsigned f = 0; f < ARRAY_SIZE(formats); ++f) {
      for (unsigned mipmapped = 0; mipmapped < 2; ++mipmapped) {
         struct hk_image image = {0};
         image.vk.base.type = VK_OBJECT_TYPE_IMAGE;
         image.vk.create_flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
                                 VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT;
         image.vk.image_type = VK_IMAGE_TYPE_3D;
         image.vk.aspects = VK_IMAGE_ASPECT_COLOR_BIT;
         image.plane_count = 1;
         struct ail_layout *layout = &image.planes[0].layout;
         const struct ail_tile tile =
            ail_get_max_tile_size(util_format_get_blocksize(formats[f]));
         *layout = (struct ail_layout){
            .tiling = AIL_TILING_GPU,
            .format = formats[f],
            .width_px =
               (4 * tile.width_el - 1) * util_format_get_blockwidth(formats[f]),
            .height_px = (4 * tile.height_el - 1) *
                         util_format_get_blockheight(formats[f]),
            .depth_px = 5,
            .levels = mipmapped ? 4 : 1,
            .sample_count_sa = 1,
            .mipmapped_z = true,
            .writeable_image = true,
         };
         ail_make_miptree(layout);
         for (unsigned l = 0; l < layout->mip_tail_first_lod; ++l) {
            const unsigned w =
               DIV_ROUND_UP(layout->stride_el[l], tile.width_el);
            /* Different nonzero X/Y tile coordinates in several Z slices;
             * compressed coordinates used to get divided by the block twice. */
            for (unsigned z = 0; z < u_minify(5, l); ++z) {
               uint64_t actual = hk_sparse_image_tile_offset_B(
                  AGX_CHIP_G17P, layout, l, tile.width_el, tile.height_el, z);
               uint64_t expected = z * layout->layer_stride_B +
                                   layout->level_offsets_B[l] +
                                   (w + 1) * AIL_PAGESIZE;
               assert(actual == expected);
               assert(hk_sparse_image_tile_offset_B(
                         AGX_CHIP_G13G, layout, l, tile.width_el,
                         tile.height_el, z) == expected);
            }
         }
         VkImageSparseMemoryRequirementsInfo2 info = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_SPARSE_MEMORY_REQUIREMENTS_INFO_2,
            .image = hk_image_to_handle(&image),
         };
         VkSparseImageMemoryRequirements2 req = {
            .sType = VK_STRUCTURE_TYPE_SPARSE_IMAGE_MEMORY_REQUIREMENTS_2,
         };
         uint32_t count = 1;
         hk_GetImageSparseMemoryRequirements2(hk_device_to_handle(dev), &info,
                                              &count, &req);
         assert(count == 1);
         const VkSparseImageMemoryRequirements *r = &req.memoryRequirements;
         assert(r->formatProperties.flags ==
                (VK_SPARSE_IMAGE_FORMAT_SINGLE_MIPTAIL_BIT |
                 VK_SPARSE_IMAGE_FORMAT_NONSTANDARD_BLOCK_SIZE_BIT));
         assert(r->formatProperties.imageGranularity.depth == 1);
         assert(r->imageMipTailFirstLod == layout->mip_tail_first_lod);
         assert(r->imageMipTailSize % 65536 == 0);
         assert(r->imageMipTailStride == 0);
         if (layout->mip_tail_first_lod < layout->levels) {
            assert(r->imageMipTailSize >=
                   layout->depth_px * layout->mip_tail_stride);
            assert(r->imageMipTailOffset == HK_MIP_TAIL_START_OFFSET);
         } else {
            assert(r->imageMipTailSize == 0);
         }
         /* A small image lives wholly in the opaque tail. */
         layout->width_px = layout->height_px = 1;
         ail_make_miptree(layout);
         count = 1;
         hk_GetImageSparseMemoryRequirements2(hk_device_to_handle(dev), &info,
                                              &count, &req);
         assert(count == 1);
         assert(req.memoryRequirements.imageMipTailFirstLod == 0);
         assert(req.memoryRequirements.imageMipTailOffset == 0);
         assert(req.memoryRequirements.imageMipTailSize >= layout->size_B);
         assert(req.memoryRequirements.imageMipTailSize % 65536 == 0);
      }
   }
   free(dev);
}

int
main(void)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(pdev);
   pdev->vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   pdev->dev.chip = AGX_CHIP_G17P;
   pdev->vk.supported_features.sparseBinding = true;
   pdev->vk.supported_features.sparseResidencyImage2D = true;
   pdev->vk.supported_features.sparseResidencyImage3D = true;
   pdev->vk.properties.maxImageDimension3D = 2048;
   pdev->vk.properties.framebufferColorSampleCounts = 1 | 2 | 4;
   check_queries(pdev);
   check_tiles_and_tails();
   free(pdev);
   puts("G17P sparse query, block addressing and mip-tail checks passed");
   return 0;
}
