/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 */
#undef NDEBUG
#include <assert.h>
#include <stdlib.h>

#include "asahi/lib/agx_chip_caps.h"
#include "drm-uapi/drm_fourcc.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_instance.h"
#include "vk_alloc.h"
#include "hk_g17p_compression.h"
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

static bool
query_hic_detects_compression(struct hk_physical_device *pdev, VkFormat format,
                              VkImageCreateFlags flags,
                              VkImageUsageFlags usage)
{
   VkPhysicalDevice handle = hk_physical_device_to_handle(pdev);
   VkHostImageCopyDevicePerformanceQueryEXT hic = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
   };
   VkImageFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &hic,
   };
   VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .flags = flags,
      .format = format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
   };
   assert(hk_GetPhysicalDeviceImageFormatProperties2(handle, &info, &props) ==
          VK_SUCCESS);
   assert(hic.identicalMemoryLayout == hic.optimalDeviceAccess);
   return !hic.identicalMemoryLayout;
}

static bool
format_list_has_modifier(struct hk_physical_device *pdev, VkFormat format,
                         uint64_t modifier,
                         VkFormatFeatureFlags2 *features_out)
{
   VkPhysicalDevice handle = hk_physical_device_to_handle(pdev);
   VkDrmFormatModifierProperties2EXT mods[8];
   VkDrmFormatModifierPropertiesList2EXT list = {
      .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
      .drmFormatModifierCount = ARRAY_SIZE(mods),
      .pDrmFormatModifierProperties = mods,
   };
   VkFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
      .pNext = &list,
   };
   hk_GetPhysicalDeviceFormatProperties2(handle, format, &props);
   for (uint32_t i = 0; i < list.drmFormatModifierCount; ++i) {
      if (mods[i].drmFormatModifier == modifier) {
         if (features_out)
            *features_out = mods[i].drmFormatModifierTilingFeatures;
         return true;
      }
   }
   return false;
}

static void
check_hal200_format_and_modifier_contracts(struct hk_physical_device *pdev)
{
   const enum agx_chip chip = pdev->dev.chip;
   const bool nocompress = (pdev->dev.debug & AGX_DBG_NOCOMPRESS) != 0;
   const bool hal200 =
      !nocompress &&
      agx_chip_has_cap(chip, AGX_CAP_HAL200_LOSSLESS_COMPRESSION);
   const bool m5 = !nocompress && agx_chip_g17_m5(chip);
   const VkImageUsageFlags color_rt =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   const VkImageUsageFlags depth_rt =
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

   /* 16-byte rgba32f colour compresses on native HAL200 (G16G and G17P) and
    * on single-layer 2D M5 (hk_m5_color_compression_eligible); its cube form
    * compresses on HAL200 alone (M5 rejects 16B cube-compat). 1-8B
    * cube-compatible colour compresses on HAL200 and M5. All three stay plain
    * on G15*, G18G, AGX2 and unadmitted parts. */
   assert(query_hic_detects_compression(pdev, VK_FORMAT_R32G32B32A32_SFLOAT, 0,
                                        color_rt) == (hal200 || m5));
   assert(query_hic_detects_compression(
             pdev, VK_FORMAT_R32G32B32A32_SFLOAT,
             VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, color_rt) == hal200);
   assert(query_hic_detects_compression(
             pdev, VK_FORMAT_R8G8B8A8_UNORM,
             VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT, color_rt) ==
          (hal200 || m5));

   assert(query_hic_detects_compression(pdev, VK_FORMAT_R8G8B8A8_UNORM, 0,
                                        VK_IMAGE_USAGE_SAMPLED_BIT) ==
          (hal200 || m5));

   /* Single-sample depth (d32f, d32s8) compresses across HAL200 and M5. */
   assert(query_hic_detects_compression(pdev, VK_FORMAT_D32_SFLOAT, 0,
                                        depth_rt) == (hal200 || m5));
   assert(query_hic_detects_compression(pdev, VK_FORMAT_D32_SFLOAT_S8_UINT, 0,
                                        depth_rt) == (hal200 || m5));

   /* DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED advertisement and feature
    * stripping (no storage/host-transfer) follow
    * hk_chip_shared_compression_supported() across HAL200 (G16G/G17P) and M5. */
   const bool want_shared =
      !nocompress && hk_chip_shared_compression_supported(chip);
   VkFormatFeatureFlags2 feats = 0;
   assert(format_list_has_modifier(pdev, VK_FORMAT_R8G8B8A8_UNORM,
                                   DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
                                   &feats) == want_shared);
   if (want_shared) {
      assert(feats & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT);
      assert(feats & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT);
      assert(!(feats & (VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                        VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT)));
   }
   assert(!format_list_has_modifier(pdev, VK_FORMAT_R32G32B32A32_SFLOAT,
                                    DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
                                    NULL));
}

static void
check_image_creation(struct hk_physical_device *pdev)
{
   struct hk_device *dev = calloc(1, sizeof(*dev));
   struct hk_instance instance = {0};
   assert(dev);
   vk_object_base_init(NULL, &instance.vk.base, VK_OBJECT_TYPE_INSTANCE);
   instance.vk.base.client_visible = true;
   list_inithead(&instance.vk.debug_utils.callbacks);
   list_inithead(&instance.vk.debug_report.callbacks);
   pdev->vk.instance = &instance.vk;
   dev->vk.physical = &pdev->vk;
   dev->vk.alloc = *vk_default_allocator();
   dev->dev.chip = pdev->dev.chip;
   vk_object_base_init(&dev->vk, &dev->vk.base, VK_OBJECT_TYPE_DEVICE);
   VkDevice device = hk_device_to_handle(dev);

   VkImageStencilUsageCreateInfo stencil = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_STENCIL_USAGE_CREATE_INFO,
      .stencilUsage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                      VK_IMAGE_USAGE_HOST_TRANSFER_BIT,
   };
   VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = &stencil,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_D32_SFLOAT_S8_UINT,
      .extent = {64, 64, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
   };
   VkImage image;
   assert(hk_CreateImage(device, &info, NULL, &image) == VK_SUCCESS);
   assert(hk_image_from_handle(image)->planes[0].layout.compressed);
   assert(!hk_image_from_handle(image)->planes[1].layout.compressed);
   hk_DestroyImage(device, image, NULL);

   const bool expect_msaa_color =
      agx_chip_has_cap(pdev->dev.chip, AGX_CAP_HAL200_MSAA_COLOR_COMPRESSION);
   VkImageCreateInfo msaa_color = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {64, 64, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_2_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   assert(hk_CreateImage(device, &msaa_color, NULL, &image) == VK_SUCCESS);
   assert(hk_image_from_handle(image)->planes[0].layout.compressed ==
          expect_msaa_color);
   hk_DestroyImage(device, image, NULL);
   msaa_color.samples = VK_SAMPLE_COUNT_4_BIT;
   msaa_color.arrayLayers = 4;
   assert(hk_CreateImage(device, &msaa_color, NULL, &image) == VK_SUCCESS);
   assert(hk_image_from_handle(image)->planes[0].layout.compressed ==
          expect_msaa_color);
   hk_DestroyImage(device, image, NULL);
   msaa_color.format = VK_FORMAT_R32G32B32A32_SFLOAT;
   assert(hk_CreateImage(device, &msaa_color, NULL, &image) == VK_SUCCESS);
   assert(!hk_image_from_handle(image)->planes[0].layout.compressed);
   hk_DestroyImage(device, image, NULL);

   const uint64_t modifier = DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED;
   VkImageDrmFormatModifierListCreateInfoEXT list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .drmFormatModifierCount = 1,
      .pDrmFormatModifiers = &modifier,
   };
   VkExternalMemoryImageCreateInfo external = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .pNext = &list,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   info.pNext = &external;
   info.format = VK_FORMAT_R8G8B8A8_UNORM;
   info.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
   info.extent = (VkExtent3D){8, 8, 1};
   assert(hk_CreateImage(device, &info, NULL, &image) == VK_ERROR_UNKNOWN);
   info.extent = (VkExtent3D){64, 64, 1};
   assert(hk_CreateImage(device, &info, NULL, &image) == VK_SUCCESS);
   hk_DestroyImage(device, image, NULL);
   info.usage |= VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   assert(hk_CreateImage(device, &info, NULL, &image) == VK_ERROR_UNKNOWN);

   VkPhysicalDeviceImageDrmFormatModifierInfoEXT query_mod = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
      .drmFormatModifier = modifier,
   };
   VkPhysicalDeviceExternalImageFormatInfo query_external = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
      .pNext = &query_mod,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkPhysicalDeviceImageFormatInfo2 query = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext = &query_external,
      .format = info.format,
      .type = info.imageType,
      .tiling = info.tiling,
      .usage = info.usage,
   };
   VkExternalImageFormatProperties external_props = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES,
   };
   VkImageFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &external_props,
   };
   assert(hk_GetPhysicalDeviceImageFormatProperties2(
             hk_physical_device_to_handle(pdev), &query, &props) ==
          VK_SUCCESS);
   assert(external_props.externalMemoryProperties.externalMemoryFeatures &
          VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT);

   VkSubresourceLayout layout = {.rowPitch = 64 * 4};
   VkImageDrmFormatModifierExplicitCreateInfoEXT import = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = modifier,
      .drmFormatModifierPlaneCount = 1,
      .pPlaneLayouts = &layout,
   };
   external.pNext = &import;
   assert(hk_CreateImage(device, &info, NULL, &image) == VK_SUCCESS);
   assert(hk_image_from_handle(image)->planes[0].layout.compressed);
   hk_DestroyImage(device, image, NULL);
   layout.rowPitch += 64;
   assert(hk_CreateImage(device, &info, NULL, &image) ==
          VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);

   vk_object_base_finish(&dev->vk.base);
   vk_object_base_finish(&instance.vk.base);
   pdev->vk.instance = NULL;
   free(dev);
}

int
main(void)
{
   struct hk_physical_device *pdev = calloc(1, sizeof(*pdev));
   assert(pdev);
   pdev->vk.base.type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
   pdev->vk.properties.maxImageDimension3D = 2048;
   pdev->vk.properties.framebufferColorSampleCounts = 1 | 2 | 4;
   pdev->vk.properties.framebufferDepthSampleCounts = 1 | 2 | 4;
   pdev->vk.properties.framebufferStencilSampleCounts = 1 | 2 | 4;

   /* Native HAL200 (G16G, G17P) and every M5 SKU compress normal 2D colour
    * render targets and drop compression when
    * ATTACHMENT_FEEDBACK_LOOP_BIT_EXT is present. G15* (M3) and G18G keep 2D
    * colour render targets plain in both queries. */
   const struct {
      enum agx_chip chip;
      bool expect_feedback_disables;
   } cases[] = {
      { AGX_CHIP_G15G, false },
      { AGX_CHIP_G15S, false },
      { AGX_CHIP_G15C, false },
      { AGX_CHIP_G16G, true },
      { AGX_CHIP_G17P, true },
      { AGX_CHIP_G17G, true },
      { AGX_CHIP_G17S, true },
      { AGX_CHIP_G17C, true },
      { AGX_CHIP_G17D, true },
      { AGX_CHIP_G18G, false },
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      pdev->dev.chip = cases[i].chip;
      pdev->dev.debug = 0;
      assert(query_feedback_disables_color_compression(pdev) ==
             cases[i].expect_feedback_disables);
      check_interchange_modifiers(pdev);
      check_r11g11b10_eot_class(&pdev->dev);
      if (agx_chip_has_cap(pdev->dev.chip, AGX_CAP_HAL200_LOSSLESS_COMPRESSION))
         check_image_creation(pdev);
      check_hal200_format_and_modifier_contracts(pdev);
   }

   /* Disabling compression disables the contract query, shared modifier, and
    * Interchange on HAL200 and M5 alike. */
   const enum agx_chip nocompress_chips[] = {
      AGX_CHIP_G16G,
      AGX_CHIP_G17P,
      AGX_CHIP_G17S,
   };
   for (unsigned i = 0; i < ARRAY_SIZE(nocompress_chips); ++i) {
      pdev->dev.chip = nocompress_chips[i];
      pdev->dev.debug = AGX_DBG_NOCOMPRESS;
      assert(!query_feedback_disables_color_compression(pdev));
      check_interchange_modifiers(pdev);
      check_hal200_format_and_modifier_contracts(pdev);
   }

   free(pdev);
   return 0;
}
