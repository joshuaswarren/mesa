/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 *
 * hk-vulkan-bg-eot-threads [THREADS ROUNDS]: record dynamic-rendering passes
 * from THREADS threads at once. Each pass uses a different subset of eight
 * colour attachments with mixed load and store ops, so the device's
 * background/end-of-tile shader cache (dev->bg_eot) misses and inserts from
 * several threads at once (the shape of upstream Mesa issue 16812). Each of
 * ROUNDS starts from a fresh device, so its cache is empty again. Exits 0
 * and prints "ok", or crashes. Nothing is submitted, so it runs under the
 * no-op DRM shim.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>
#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "%s = %d\n", #x, r_); exit(1); } } while (0)
static VkPhysicalDevice pd; static VkDevice dev; static unsigned mt;
static const VkFormat fmts[8] = { VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R8_UNORM,
   VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R16_UINT, VK_FORMAT_A2B10G10R10_UNORM_PACK32 };
static void *worker(void *arg) {
   unsigned t = (unsigned)(uintptr_t)arg; VkImageView views[8]; VkImage imgs[8]; VkDeviceMemory mems[8];
   for (int i = 0; i < 8; ++i) {
      VkImage img; VkDeviceMemory mem; VkMemoryRequirements mr;
      CK(vkCreateImage(dev, &(VkImageCreateInfo){ .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmts[i],
         .extent = {64, 64, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = 1, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT }, NULL, &img));
      vkGetImageMemoryRequirements(dev, img, &mr);
      CK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size, .memoryTypeIndex = mt }, NULL, &mem));
      CK(vkBindImageMemory(dev, img, mem, 0));
      CK(vkCreateImageView(dev, &(VkImageViewCreateInfo){ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = fmts[i], .subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 } }, NULL, &views[i]));
      imgs[i] = img; mems[i] = mem;
   }
   VkCommandPool pool; VkCommandBuffer cb;
   CK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT }, NULL, &pool));
   CK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1 }, &cb));
   for (unsigned m = 1; m < 256; ++m) {
      unsigned mask = (m * 37u + t * 11u) & 255u; if (!mask) mask = 1;
      CK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }));
      VkRenderingAttachmentInfo att[8] = {0};
      for (int i = 0; i < 8; ++i) {
         att[i].sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
         if (mask & (1u << i)) {
            att[i].imageView = views[i]; att[i].imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            att[i].loadOp = ((m + i) & 1) ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            att[i].storeOp = ((m >> 1) + i) & 1 ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE;
            att[i].clearValue.color.uint32[0] = m;
         }
      }
      vkCmdBeginRendering(cb, &(VkRenderingInfo){ .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {64, 64}}, .layerCount = 1,
         .colorAttachmentCount = 8, .pColorAttachments = att });
      vkCmdEndRendering(cb);
      CK(vkEndCommandBuffer(cb));
      CK(vkResetCommandBuffer(cb, 0));
   }
   vkDestroyCommandPool(dev, pool, NULL);
   for (int i = 0; i < 8; ++i) {
      vkDestroyImageView(dev, views[i], NULL);
      vkDestroyImage(dev, imgs[i], NULL);
      vkFreeMemory(dev, mems[i], NULL);
   }
   return NULL;
}
int main(int argc, char **argv) {
   unsigned nt = argc > 1 ? atoi(argv[1]) : 16, rounds = argc > 2 ? atoi(argv[2]) : 20;
   VkInstance inst;
   CK(vkCreateInstance(&(VkInstanceCreateInfo){ .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){ .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3 } }, NULL, &inst));
   uint32_t n = 1; vkEnumeratePhysicalDevices(inst, &n, &pd);
   VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   for (mt = 0; !(mp.memoryTypes[mt].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT); ++mt);
   for (unsigned r = 0; r < rounds; ++r) {
      float prio = 1;
      CK(vkCreateDevice(pd, &(VkDeviceCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
         .pNext = &(VkPhysicalDeviceVulkan13Features){ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES, .dynamicRendering = VK_TRUE },
         .queueCreateInfoCount = 1, .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueCount = 1, .pQueuePriorities = &prio } }, NULL, &dev));
      pthread_t th[64];
      for (unsigned t = 0; t < nt; ++t) pthread_create(&th[t], NULL, worker, (void *)(uintptr_t)t);
      for (unsigned t = 0; t < nt; ++t) pthread_join(th[t], NULL);
      vkDestroyDevice(dev, NULL);
   }
   puts("ok");
   return 0;
}
