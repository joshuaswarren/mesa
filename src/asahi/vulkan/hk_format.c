/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "drm-uapi/drm_fourcc.h"
#include "vulkan/vulkan_core.h"

#include "hk_buffer_view.h"
#include "hk_entrypoints.h"
#include "hk_image.h"
#include "hk_physical_device.h"

#include "vk_enum_defines.h"
#include "vk_format.h"

/* GPU import/export capabilities, not a scanout modifier list.  Display WSI
 * intersects these with the selected plane's IN_FORMATS; Wayland/X11 use the
 * modifiers supplied by the compositor/display server.
 */
uint64_t agx_best_modifiers[] = {
   DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED,
   DRM_FORMAT_MOD_APPLE_GPU_TILED,
   DRM_FORMAT_MOD_APPLE_INTERCHANGE,
   DRM_FORMAT_MOD_LINEAR,
};

static VkFormatFeatureFlags2
hk_modifier_features(const struct agx_device *dev, uint64_t mod,
                     VkFormat vk_format, const VkFormatProperties3 *props)
{
   /* There's no corresponding fourcc, so don't advertise modifiers */
   if (vk_format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
       vk_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) {
      return 0;
   }

   /* Don't advertise compression for the uncompressable. A compressed image
    * has no host-copy layout (images with host-transfer usage are never
    * compressed). */
   if (mod == DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED) {
      if (!hk_can_compress_format(dev, vk_format))
         return 0;

      /* G17P and M5 share compressed images without storage or host-copy
       * access (hk_g17p_shared_compression_shape()). */
      if (agx_chip_g17p_or_m5(dev->chip))
         return props->optimalTilingFeatures &
                ~(VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT |
                  VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
      return props->optimalTilingFeatures &
             ~VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT;
   }

   /* Interchange: the 32-bit RGBA formats on G17P; never storage or host
    * copies (compressed). */
   if (mod == DRM_FORMAT_MOD_APPLE_INTERCHANGE) {
      if (!hk_interchange_format(dev, vk_format))
         return 0;
      return props->optimalTilingFeatures &
             ~(VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
               VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT |
               VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
               VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT |
               VK_FORMAT_FEATURE_2_HOST_IMAGE_TRANSFER_BIT_EXT);
   }

   if (mod == DRM_FORMAT_MOD_LINEAR)
      return props->linearTilingFeatures;
   else
      return props->optimalTilingFeatures;
}

static void
get_drm_format_modifier_properties_list(
   const struct hk_physical_device *physical_device, VkFormat vk_format,
   VkDrmFormatModifierPropertiesListEXT *list, const VkFormatProperties3 *props)
{
   VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierPropertiesEXT, out,
                          list->pDrmFormatModifierProperties,
                          &list->drmFormatModifierCount);

   for (unsigned i = 0; i < ARRAY_SIZE(agx_best_modifiers); ++i) {
      uint64_t mod = agx_best_modifiers[i];
      VkFormatFeatureFlags2 flags =
         hk_modifier_features(&physical_device->dev, mod, vk_format, props);

      if (!flags)
         continue;

      vk_outarray_append_typed(VkDrmFormatModifierPropertiesEXT, &out,
                               out_props)
      {
         *out_props = (VkDrmFormatModifierPropertiesEXT){
            .drmFormatModifier = mod,
            .drmFormatModifierPlaneCount = vk_format_get_plane_count(vk_format),
            .drmFormatModifierTilingFeatures =
               vk_format_features2_to_features(flags),
         };
      };
   }
}

static void
get_drm_format_modifier_properties_list_2(
   const struct hk_physical_device *physical_device, VkFormat vk_format,
   VkDrmFormatModifierPropertiesList2EXT *list, const VkFormatProperties3 *props)
{
   VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierProperties2EXT, out,
                          list->pDrmFormatModifierProperties,
                          &list->drmFormatModifierCount);

   for (unsigned i = 0; i < ARRAY_SIZE(agx_best_modifiers); ++i) {
      uint64_t mod = agx_best_modifiers[i];
      VkFormatFeatureFlags2 flags =
         hk_modifier_features(&physical_device->dev, mod, vk_format, props);

      if (!flags)
         continue;

      vk_outarray_append_typed(VkDrmFormatModifierProperties2EXT, &out,
                               out_props)
      {
         *out_props = (VkDrmFormatModifierProperties2EXT){
            .drmFormatModifier = mod,
            .drmFormatModifierPlaneCount = vk_format_get_plane_count(vk_format),
            .drmFormatModifierTilingFeatures = flags,
         };
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
hk_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physicalDevice,
                                      VkFormat format,
                                      VkFormatProperties2 *pFormatProperties)
{
   VK_FROM_HANDLE(hk_physical_device, pdevice, physicalDevice);

   VkFormatFeatureFlags2 linear2, optimal2, buffer2;
   linear2 =
      hk_get_image_format_features(pdevice, format, VK_IMAGE_TILING_LINEAR);
   optimal2 =
      hk_get_image_format_features(pdevice, format, VK_IMAGE_TILING_OPTIMAL);
   buffer2 = hk_get_buffer_format_features(pdevice, format);

   pFormatProperties->formatProperties = (VkFormatProperties){
      .linearTilingFeatures = vk_format_features2_to_features(linear2),
      .optimalTilingFeatures = vk_format_features2_to_features(optimal2),
      .bufferFeatures = vk_format_features2_to_features(buffer2),
   };

   /* Modifier lists report the 64-bit feature sets (host image transfer,
    * formatless storage, ...), which VkFormatProperties cannot carry. */
   const VkFormatProperties3 props3 = {
      .linearTilingFeatures = linear2,
      .optimalTilingFeatures = optimal2,
      .bufferFeatures = buffer2,
   };

   vk_foreach_struct(sType, ext, pFormatProperties->pNext) {
      switch (sType) {
      case VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3: {
         VkFormatProperties3 *p = ext;
         p->linearTilingFeatures = linear2;
         p->optimalTilingFeatures = optimal2;
         p->bufferFeatures = buffer2;
         break;
      }

      case VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT:
         get_drm_format_modifier_properties_list(pdevice, format, ext,
                                                 &props3);
         break;

      case VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT:
         get_drm_format_modifier_properties_list_2(pdevice, format, ext,
                                                   &props3);
         break;

      default:
         vk_debug_ignored_stype(sType);
         break;
      }
   }
}
