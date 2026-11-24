/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_wsi.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_memory_budget.h"
#include "hk_instance.h"
#include "wsi_common.h"

#include <xf86drm.h>

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
hk_wsi_proc_addr(VkPhysicalDevice physicalDevice, const char *pName)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   return vk_instance_get_proc_addr_unchecked(pdev->vk.instance, pName);
}

static bool
hk_wsi_can_present_on_device(VkPhysicalDevice physicalDevice, int fd)
{
   VK_FROM_HANDLE(hk_physical_device, pdev, physicalDevice);
   if (pdev->dev.is_virtio) {
      return false;
   }

   drmDevicePtr device;
   if (drmGetDevice2(fd, 0, &device) != 0) {
      return false;
   }

   /* Allow on-device presentation for all non-virtio devices with bus type
    * PLATFORM */
   bool match = device->bustype == DRM_BUS_PLATFORM;

   drmFreeDevice(&device);

   return match;
}

/* WSI allocates all images during swapchain creation. On non-virtio UMA
 * devices each VM_BIND pins UMA backing, so even a single maximum-extent
 * attempt can exceed a small process limit before the kernel can return OOM.
 * This WSI callback bounds that creation peak only; application/AS allocations
 * never consult a cgroup snapshot. Imports and virtio keep the ordinary
 * allocator.
 */
VkResult
hk_wsi_allocate_memory(VkDevice device, const VkMemoryAllocateInfo *info,
                       const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   const VkImportMemoryFdInfoKHR *import =
      vk_find_struct_const(info->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   if (!dev->dev.is_virtio &&
       !(import && import->handleType)) {
      uint64_t available;
      if (hk_get_cgroup_memory_available("/proc/self/cgroup",
                                         "/proc/self/mountinfo", &available) &&
          info->allocationSize > available - available / 16) {
         simple_mtx_lock(&dev->dev.bo_map_lock);
         agx_bo_cache_evict_all(&dev->dev);
         simple_mtx_unlock(&dev->dev.bo_map_lock);
         if (hk_get_cgroup_memory_available("/proc/self/cgroup",
                                            "/proc/self/mountinfo", &available) &&
             info->allocationSize > available - available / 16)
            return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      }
   }
   /* WSI's dma-buf sync-file capability probe does not supply the private
    * allocation info. Tag it here as well, before BO allocation/binding, so
    * failures at any point bypass the BO cache. Preserve an existing info's
    * synchronization flags and every other allocation extension. */
   struct wsi_memory_allocate_info wsi_info = {
      .sType = VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA,
      .pNext = info->pNext,
   };
   VkMemoryAllocateInfo tagged_info = *info;
   if (!dev->dev.is_virtio &&
       !vk_find_struct_const(info->pNext, WSI_MEMORY_ALLOCATE_INFO_MESA))
      tagged_info.pNext = &wsi_info;
   return hk_AllocateMemory(device, &tagged_info, alloc, out);
}

VkResult
hk_init_wsi(struct hk_physical_device *pdev)
{
   VkResult result;

   struct wsi_device_options wsi_options = {.sw_device = false};
   result = wsi_device_init(
      &pdev->wsi_device, hk_physical_device_to_handle(pdev), hk_wsi_proc_addr,
      &pdev->vk.instance->alloc, pdev->master_fd,
      &hk_physical_device_instance(pdev)->drirc.options, &wsi_options);
   if (result != VK_SUCCESS)
      return result;

   pdev->wsi_device.AllocateMemory = hk_wsi_allocate_memory;
   pdev->wsi_device.supports_scanout = false;
   pdev->wsi_device.supports_modifiers = true;
   pdev->wsi_device.can_present_on_device = hk_wsi_can_present_on_device;

   pdev->vk.wsi_device = &pdev->wsi_device;

   return result;
}

void
hk_finish_wsi(struct hk_physical_device *pdev)
{
   pdev->vk.wsi_device = NULL;
   wsi_device_finish(&pdev->wsi_device, &pdev->vk.instance->alloc);
}
