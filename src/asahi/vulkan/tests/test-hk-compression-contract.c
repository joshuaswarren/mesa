/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>

#include "hk_entrypoints.h"
#include "hk_physical_device.h"

static bool
query_feedback_disables_color_compression(struct hk_physical_device *pdev)
{
   VkPhysicalDevice handle = hk_physical_device_to_handle(pdev);
   const VkImageUsageFlags base_usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
      VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT;

   VkHostImageCopyDevicePerformanceQueryEXT plain_hic = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
   };
   VkImageFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &plain_hic,
   };
   VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = base_usage,
   };
   assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
          VK_SUCCESS);

   VkHostImageCopyDevicePerformanceQueryEXT feedback_hic = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
   };
   props.pNext = &feedback_hic;
   info.usage = base_usage | VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
          VK_SUCCESS);

   /* Colour images with ATTACHMENT_FEEDBACK_LOOP_BIT_EXT always stay plain. */
   assert(feedback_hic.identicalMemoryLayout == VK_TRUE);
   assert(feedback_hic.optimalDeviceAccess == VK_TRUE);

   return !plain_hic.identicalMemoryLayout &&
          feedback_hic.identicalMemoryLayout;
}

int
main(void)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(pdev);
   pdev->vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   pdev->vk.properties.maxImageDimension3D = 2048;
   pdev->vk.properties.framebufferColorSampleCounts = 1 | 2 | 4;

   /* G17P and every M5 SKU compress normal 2D colour render targets and drop
    * compression when ATTACHMENT_FEEDBACK_LOOP_BIT_EXT is present. G15/G16/G18G
    * keep 2D colour render targets plain in both queries. */
   const struct {
      enum agx_chip chip;
      bool expect_feedback_disables;
   } cases[] = {
      { AGX_CHIP_G15G, false },
      { AGX_CHIP_G16G, false },
      { AGX_CHIP_G17P, true },
      { AGX_CHIP_G17G, true },
      { AGX_CHIP_G17S, true },
      { AGX_CHIP_G17C, true },
      { AGX_CHIP_G18G, false },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      pdev->dev.chip = cases[i].chip;
      pdev->dev.debug = 0;
      assert(query_feedback_disables_color_compression(pdev) ==
             cases[i].expect_feedback_disables);
   }

   /* Disabling compression disables the contract query too. */
   pdev->dev.chip = AGX_CHIP_G17S;
   pdev->dev.debug = AGX_DBG_NOCOMPRESS;
   assert(!query_feedback_disables_color_compression(pdev));

   free(pdev);
   return 0;
}
