/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A queue's first submission is a vkQueueBindSparse of one tile of a
 * sparse-residency image, with no command buffer, as vkd3d-proton issues for
 * a tiled resource. On G17P the bind also flushes the image's sparse table,
 * which is a one-command GPU submission of its own. It must succeed: the
 * device must not be lost, and the fence must signal. Nothing executes on
 * the GPU, so the test also runs under the no-op DRM shim.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "FAIL: %s: %d\n", #call, r_);                         \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

int
main(void)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   VkPhysicalDevice pdev;
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count == 0) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }

   VkPhysicalDeviceFeatures features;
   vkGetPhysicalDeviceFeatures(pdev, &features);
   if (!features.sparseBinding || !features.sparseResidencyImage2D) {
      fprintf(stderr, "sparseBinding or sparseResidencyImage2D missing\n");
      return 1;
   }

   VkQueueFamilyProperties families[8];
   uint32_t family_count = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families);
   uint32_t family = UINT32_MAX;
   for (uint32_t i = 0; i < family_count && family == UINT32_MAX; i++) {
      if (families[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT)
         family = i;
   }
   if (family == UINT32_MAX) {
      fprintf(stderr, "no sparse binding queue\n");
      return 1;
   }

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceFeatures enabled = {
      .sparseBinding = VK_TRUE,
      .sparseResidencyImage2D = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .pEnabledFeatures = &enabled,
   };
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, family, 0, &queue);

   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_SPARSE_BINDING_BIT |
               VK_IMAGE_CREATE_SPARSE_RESIDENCY_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {512, 512, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &image_info, NULL, &image));

   VkMemoryRequirements reqs;
   vkGetImageMemoryRequirements(dev, image, &reqs);
   VkSparseImageMemoryRequirements sparse_reqs;
   uint32_t sparse_count = 1;
   vkGetImageSparseMemoryRequirements(dev, image, &sparse_count, &sparse_reqs);
   if (sparse_count == 0) {
      fprintf(stderr, "FAIL: no sparse memory requirements\n");
      return 1;
   }

   uint32_t type = 0;
   while (type < 32 && !(reqs.memoryTypeBits & (1u << type)))
      type++;
   const VkMemoryAllocateInfo alloc_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = reqs.alignment,
      .memoryTypeIndex = type,
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc_info, NULL, &memory));

   /* One tile, the first one of level 0 */
   const VkExtent3D tile = sparse_reqs.formatProperties.imageGranularity;
   const VkSparseImageMemoryBind bind = {
      .subresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0},
      .offset = {0, 0, 0},
      .extent = tile,
      .memory = memory,
      .memoryOffset = 0,
   };
   const VkSparseImageMemoryBindInfo image_bind = {
      .image = image,
      .bindCount = 1,
      .pBinds = &bind,
   };
   const VkBindSparseInfo bind_info = {
      .sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO,
      .imageBindCount = 1,
      .pImageBinds = &image_bind,
   };
   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   CHECK(vkCreateFence(dev, &fence_info, NULL, &fence));

   CHECK(vkQueueBindSparse(queue, 1, &bind_info, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull));
   CHECK(vkDeviceWaitIdle(dev));

   vkDestroyFence(dev, fence, NULL);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, memory, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("sparse bind as the first submission: ok\n");
   return 0;
}
