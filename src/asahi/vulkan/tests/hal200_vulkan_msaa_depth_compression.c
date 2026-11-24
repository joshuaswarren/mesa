/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * HAL200 (G16G, G17P) multisampled depth compression through the real image
 * creation and format-query paths, with the usages and flags zink gives every
 * depth attachment: ATTACHMENT_FEEDBACK_LOOP usage, the sample-location
 * compatible flag, and (until the host-copy query says otherwise)
 * HOST_TRANSFER usage.
 *
 * Compression is observed through the memory size: a compressed image carries
 * a metadata plane, so it is larger than the plain w * h * samples * 4 bytes
 * of a D32_SFLOAT image. The input-attachment shape stays plain on every
 * HAL200 part and checks that baseline.
 *
 * usage: hal200-vulkan-msaa-depth-compression [--expect-1x-compressed]
 *   --expect-1x-compressed  single-sample depth compresses too (G17P); without
 *                           it, single-sample depth must stay plain (G16G)
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define DIM 256u

static VkPhysicalDevice pdev;
static VkDevice dev;
static unsigned failures;

static void
expect(bool cond, const char *what)
{
   printf("%s: %s\n", cond ? "ok  " : "FAIL", what);
   if (!cond)
      failures++;
}

static VkDeviceSize
image_size(VkSampleCountFlagBits samples, VkImageUsageFlags usage,
           VkImageCreateFlags flags)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_D32_SFLOAT,
      .extent = {DIM, DIM, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &info, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   vkDestroyImage(dev, image, NULL);
   return req.size;
}

/* VkHostImageCopyDevicePerformanceQueryEXT::optimalDeviceAccess for depth
 * with zink's usage and flags. */
static VkBool32
host_copy_optimal(VkImageUsageFlags usage, VkImageCreateFlags flags)
{
   const VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = VK_FORMAT_D32_SFLOAT,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage | VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT,
      .flags = flags,
   };
   VkHostImageCopyDevicePerformanceQueryEXT hic = {
      .sType = VK_STRUCTURE_TYPE_HOST_IMAGE_COPY_DEVICE_PERFORMANCE_QUERY_EXT,
   };
   VkImageFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &hic,
   };
   CHECK(vkGetPhysicalDeviceImageFormatProperties2(pdev, &info, &props));
   expect(hic.optimalDeviceAccess == hic.identicalMemoryLayout,
          "host-copy query: optimalDeviceAccess == identicalMemoryLayout");
   return hic.optimalDeviceAccess;
}

int
main(int argc, char **argv)
{
   bool expect_1x = argc > 1 && !strcmp(argv[1], "--expect-1x-compressed");

   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&ici, NULL, &instance));
   uint32_t count = 1;
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count == 0) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }

   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   const VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
   };
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   const VkImageUsageFlags ds = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                VK_IMAGE_USAGE_SAMPLED_BIT |
                                VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   const VkImageUsageFlags fb =
      VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
   const VkImageCreateFlags sl =
      VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT;

   for (unsigned s = 2; s <= 4; s <<= 1) {
      const VkSampleCountFlagBits samples = (VkSampleCountFlagBits)s;
      const VkDeviceSize plain = (VkDeviceSize)DIM * DIM * s * 4;
      char what[160];

      snprintf(what, sizeof(what), "%ux input-attachment depth is plain", s);
      expect(image_size(samples, ds | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT,
                        0) == plain, what);
      snprintf(what, sizeof(what), "%ux depth compresses", s);
      expect(image_size(samples, ds, 0) > plain, what);
      snprintf(what, sizeof(what), "%ux depth + sample locations compresses",
               s);
      expect(image_size(samples, ds, sl) > plain, what);
      snprintf(what, sizeof(what), "%ux depth + feedback loop compresses", s);
      expect(image_size(samples, ds | fb, 0) > plain, what);
      snprintf(what, sizeof(what),
               "%ux depth + feedback loop + sample locations (zink) compresses",
               s);
      expect(image_size(samples, ds | fb, sl) > plain, what);
   }

   const VkDeviceSize plain_1x = (VkDeviceSize)DIM * DIM * 4;
   const VkDeviceSize zink_1x = image_size(VK_SAMPLE_COUNT_1_BIT, ds | fb, sl);
   expect(expect_1x ? zink_1x > plain_1x : zink_1x == plain_1x,
          expect_1x ? "1x zink depth compresses"
                    : "1x zink depth stays plain");

   /* Some depth image of this usage compresses, so host-transfer images
    * cannot promise the device layout: zink must drop HOST_TRANSFER. */
   expect(!host_copy_optimal(ds | fb, sl),
          "host-copy query reports a non-optimal device layout for zink depth");
   expect(!host_copy_optimal(ds, 0),
          "host-copy query reports a non-optimal device layout for depth");

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%s\n", failures ? "FAILED" : "PASSED");
   return failures ? 1 : 0;
}
