/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>

#include "asahi/lib/agx_chip_caps.h"
#include "drm-uapi/drm_fourcc.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

static bool
format_list_has_interchange(struct hk_physical_device *pdev, VkFormat format,
                            VkFormatFeatureFlags2 *features_out)
{
   VkPhysicalDevice handle = hk_physical_device_to_handle(pdev);
   VkDrmFormatModifierPropertiesEXT mods1[8];
   VkDrmFormatModifierPropertiesListEXT list1 = {
      .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT,
      .drmFormatModifierCount = ARRAY_SIZE(mods1),
      .pDrmFormatModifierProperties = mods1,
   };
   VkDrmFormatModifierProperties2EXT mods2[8];
   VkDrmFormatModifierPropertiesList2EXT list2 = {
      .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
      .pNext = &list1,
      .drmFormatModifierCount = ARRAY_SIZE(mods2),
      .pDrmFormatModifierProperties = mods2,
   };
   VkFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
      .pNext = &list2,
   };
   hk_GetPhysicalDeviceFormatProperties2(handle, format, &props);
   assert(list1.drmFormatModifierCount == list2.drmFormatModifierCount);

   bool found = false;
   for (uint32_t i = 0; i < list2.drmFormatModifierCount; ++i) {
      assert(mods1[i].drmFormatModifier == mods2[i].drmFormatModifier);
      assert(mods1[i].drmFormatModifierPlaneCount ==
             mods2[i].drmFormatModifierPlaneCount);
      if (mods2[i].drmFormatModifier == DRM_FORMAT_MOD_APPLE_INTERCHANGE) {
         assert(!found);
         assert(mods2[i].drmFormatModifierPlaneCount == 1);
         found = true;
         if (features_out)
            *features_out = mods2[i].drmFormatModifierTilingFeatures;
      }
   }
   return found;
}

static void
check_interchange_modifiers(struct hk_physical_device *pdev)
{
   static const VkFormat supported[] = {
      VK_FORMAT_B8G8R8A8_UNORM,
      VK_FORMAT_B8G8R8A8_SRGB,
      VK_FORMAT_R8G8B8A8_UNORM,
      VK_FORMAT_R8G8B8A8_SRGB,
      VK_FORMAT_A2R10G10B10_UNORM_PACK32,
      VK_FORMAT_A2B10G10R10_UNORM_PACK32,
   };
   static const VkFormat unsupported[] = {
      VK_FORMAT_R16G16B16A16_SFLOAT,
      VK_FORMAT_A2R10G10B10_UINT_PACK32,
      VK_FORMAT_A2B10G10R10_UINT_PACK32,
      VK_FORMAT_B10G11R11_UFLOAT_PACK32,
      VK_FORMAT_D32_SFLOAT,
   };

   const bool want_ix =
      agx_chip_has_cap(pdev->dev.chip, AGX_CAP_GEN9_2D_COMPRESSION) &&
      !(pdev->dev.debug & AGX_DBG_NOCOMPRESS);
   VkPhysicalDevice handle = hk_physical_device_to_handle(pdev);

   for (unsigned i = 0; i < ARRAY_SIZE(supported); ++i) {
      assert(hk_interchange_format(&pdev->dev, supported[i]) == want_ix);

      VkFormatFeatureFlags2 feats = 0;
      assert(format_list_has_interchange(pdev, supported[i], &feats) ==
             want_ix);
      if (want_ix) {
         assert(feats & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
         assert(feats & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT);
         assert(feats & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
         assert(feats & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
         assert(feats & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT);
         assert(feats & VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT);
         assert(!(feats & (VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                           VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT)));
      }

      VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info = {
         .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
         .drmFormatModifier = DRM_FORMAT_MOD_APPLE_INTERCHANGE,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .pNext = &mod_info,
         .format = supported[i],
         .type = VK_IMAGE_TYPE_2D,
         .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                  VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      };
      VkImageFormatProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      };
      VkResult res =
         hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props);
      if (want_ix) {
         assert(res == VK_SUCCESS);
         assert(props.imageFormatProperties.maxMipLevels == 1);
         assert(props.imageFormatProperties.maxArrayLayers == 1);
         assert(props.imageFormatProperties.sampleCounts ==
                VK_SAMPLE_COUNT_1_BIT);
      } else {
         assert(res == VK_ERROR_FORMAT_NOT_SUPPORTED);
      }
   }

   for (unsigned i = 0; i < ARRAY_SIZE(unsupported); ++i) {
      assert(!hk_interchange_format(&pdev->dev, unsupported[i]));
      assert(!format_list_has_interchange(pdev, unsupported[i], NULL));

      VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info = {
         .sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
         .drmFormatModifier = DRM_FORMAT_MOD_APPLE_INTERCHANGE,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      VkPhysicalDeviceImageFormatInfo2 info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
         .pNext = &mod_info,
         .format = unsupported[i],
         .type = VK_IMAGE_TYPE_2D,
         .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
         .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      };
      VkImageFormatProperties2 props = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      };
      assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
             VK_ERROR_FORMAT_NOT_SUPPORTED);
   }
}

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

static void
check_r11g11b10_eot_class(const struct agx_device *dev)
{
   const bool own_class = agx_chip_native_hal200(dev->chip);
   assert(hk_eot_r11g11b10_own_class(dev) == own_class);

   const enum pipe_format pbe =
      hk_eot_pbe_format(dev, PIPE_FORMAT_R11G11B10_FLOAT);
   const enum pipe_format key =
      hk_eot_key_format(dev, PIPE_FORMAT_R11G11B10_FLOAT);
   if (own_class) {
      assert(pbe == PIPE_FORMAT_R11G11B10_FLOAT);
      assert(key == PIPE_FORMAT_R10G10B10A2_UNORM);
   } else if (agx_chip_g17_m5(dev->chip)) {
      assert(pbe == PIPE_FORMAT_R11G11B10_FLOAT);
      assert(key == PIPE_FORMAT_R11G11B10_FLOAT);
   } else {
      assert(pbe == PIPE_FORMAT_R32_UINT);
      assert(key == PIPE_FORMAT_R11G11B10_FLOAT);
   }
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
      check_interchange_modifiers(pdev);
      check_r11g11b10_eot_class(&pdev->dev);
   }

   /* Disabling compression disables the contract query and Interchange too. */
   pdev->dev.chip = AGX_CHIP_G17S;
   pdev->dev.debug = AGX_DBG_NOCOMPRESS;
   assert(!query_feedback_disables_color_compression(pdev));
   pdev->dev.chip = AGX_CHIP_G17P;
   pdev->dev.debug = AGX_DBG_NOCOMPRESS;
   check_interchange_modifiers(pdev);

   free(pdev);
   return 0;
}
