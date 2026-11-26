/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Images created with VkImageDrmFormatModifierExplicitCreateInfoEXT (the
 * LINEAR modifier) whose planes do not start at offset 0: memory
 * requirements must cover offset + size, vkGetImageSubresourceLayout must
 * report the explicit offset, binding must place each plane at the binding
 * offset plus its explicit offset (for a disjoint image, its own binding),
 * and memory that ends before a plane does must be refused, as must
 * invalid layouts. The same holds for the GPU-tiled modifiers, compressed
 * ones included, whose metadata follows the plane's texels in its memory
 * plane. Nothing executes on the GPU, so the test also runs under the
 * no-op DRM shim.
 */
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#ifndef DRM_FORMAT_MOD_LINEAR
#define DRM_FORMAT_MOD_LINEAR 0
#endif
#define APPLE_MODIFIER(n) ((UINT64_C(0x0c) << 56) | (n))
#define DRM_FORMAT_MOD_APPLE_GPU_TILED            APPLE_MODIFIER(1)
#define DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED APPLE_MODIFIER(2)

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static VkPhysicalDevice pdev;
static VkDevice dev;
static unsigned failures, checks;

#define EXPECT(cond, ...)                                                      \
   do {                                                                        \
      checks++;                                                                \
      if (!(cond)) {                                                           \
         fprintf(stderr, "FAIL: " __VA_ARGS__);                                \
         fprintf(stderr, "\n");                                                \
         failures++;                                                           \
      }                                                                        \
   } while (0)

/* Whether a modifier supports format with these flags and usage */
static bool
modifier_supported(uint64_t modifier, VkFormat format,
                   VkImageCreateFlags flags, VkImageUsageFlags usage,
                   uint32_t *plane_count)
{
   VkDrmFormatModifierProperties2EXT mods[16];
   VkDrmFormatModifierPropertiesList2EXT list = {
      .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
      .drmFormatModifierCount = 16,
      .pDrmFormatModifierProperties = mods,
   };
   VkFormatProperties2 props = {
      .sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2,
      .pNext = &list,
   };
   vkGetPhysicalDeviceFormatProperties2(pdev, format, &props);

   bool found = false;
   for (uint32_t i = 0; i < list.drmFormatModifierCount; ++i) {
      if (mods[i].drmFormatModifier == modifier) {
         *plane_count = mods[i].drmFormatModifierPlaneCount;
         found = true;
      }
   }
   if (!found)
      return false;

   const VkPhysicalDeviceImageDrmFormatModifierInfoEXT mod_info = {
      .sType =
         VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
      .drmFormatModifier = modifier,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   const VkPhysicalDeviceImageFormatInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .pNext = &mod_info,
      .format = format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = usage,
      .flags = flags,
   };
   VkImageFormatProperties2 out = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
   };
   return vkGetPhysicalDeviceImageFormatProperties2(pdev, &info, &out) ==
          VK_SUCCESS;
}

static bool
linear_supported(VkFormat format, VkImageCreateFlags flags,
                 VkImageUsageFlags usage, uint32_t *plane_count)
{
   return modifier_supported(DRM_FORMAT_MOD_LINEAR, format, flags, usage,
                             plane_count);
}

static VkResult
create_modifier_image(uint64_t modifier, VkFormat format, uint32_t w,
                      uint32_t h, VkImageCreateFlags flags,
                      VkImageUsageFlags usage, uint32_t planes,
                      const VkSubresourceLayout *layouts, VkImage *image)
{
   const VkImageDrmFormatModifierExplicitCreateInfoEXT explicit_info = {
      .sType =
         VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = modifier,
      .drmFormatModifierPlaneCount = planes,
      .pPlaneLayouts = layouts,
   };
   const VkImageDrmFormatModifierListCreateInfoEXT list_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .drmFormatModifierCount = 1,
      .pDrmFormatModifiers = &modifier,
   };
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = layouts ? (const void *)&explicit_info : (const void *)&list_info,
      .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = usage,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   return vkCreateImage(dev, &info, NULL, image);
}

static VkResult
create_image(VkFormat format, uint32_t w, uint32_t h, VkImageCreateFlags flags,
             VkImageUsageFlags usage, uint32_t planes,
             const VkSubresourceLayout *layouts, VkImage *image)
{
   return create_modifier_image(DRM_FORMAT_MOD_LINEAR, format, w, h, flags,
                                usage, planes, layouts, image);
}

static VkMemoryRequirements
requirements(VkImage image, VkImageAspectFlagBits plane)
{
   const VkImagePlaneMemoryRequirementsInfo plane_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_PLANE_MEMORY_REQUIREMENTS_INFO,
      .planeAspect = plane,
   };
   const VkImageMemoryRequirementsInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2,
      .pNext = plane ? &plane_info : NULL,
      .image = image,
   };
   VkMemoryRequirements2 req = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2,
   };
   vkGetImageMemoryRequirements2(dev, &info, &req);
   return req.memoryRequirements;
}

static VkSubresourceLayout
subresource_layout(VkImage image, VkImageAspectFlags aspect)
{
   const VkImageSubresource subresource = {.aspectMask = aspect};
   VkSubresourceLayout layout;
   vkGetImageSubresourceLayout(dev, image, &subresource, &layout);
   return layout;
}

static VkDeviceMemory
allocate(VkDeviceSize size, uint32_t type_bits, uint8_t **map)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   uint32_t type = UINT32_MAX;
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i) {
      if ((type_bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want)
         type = i;
   }
   if (type == UINT32_MAX) {
      fprintf(stderr, "no host-visible memory type\n");
      exit(2);
   }
   const VkMemoryAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = size,
      .memoryTypeIndex = type,
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &info, NULL, &memory));
   CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)map));
   memset(*map, 0, size);
   return memory;
}

static VkResult
bind(VkImage image, VkDeviceMemory memory, VkDeviceSize offset,
     VkImageAspectFlagBits plane)
{
   const VkBindImagePlaneMemoryInfo plane_info = {
      .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_PLANE_MEMORY_INFO,
      .planeAspect = plane,
   };
   const VkBindImageMemoryInfo info = {
      .sType = VK_STRUCTURE_TYPE_BIND_IMAGE_MEMORY_INFO,
      .pNext = plane ? &plane_info : NULL,
      .image = image,
      .memory = memory,
      .memoryOffset = offset,
   };
   return vkBindImageMemory2(dev, 1, &info);
}

static const VkImageUsageFlags usage =
   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

/* The largest whole number of 16 KiB pages below size: memory objects are
 * page granular, so memory any closer to size could still hold the plane */
static VkDeviceSize
short_of(VkDeviceSize size)
{
   assert(size > 0x4000);
   return (size - 1) & ~(VkDeviceSize)0x3fff;
}

/* One RGBA8 plane at 0x1000, pitch 512 */
static void
check_single_plane(void)
{
   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   uint32_t planes;
   if (!linear_supported(format, 0, VK_IMAGE_USAGE_SAMPLED_BIT, &planes)) {
      EXPECT(false, "RGBA8 LINEAR modifier unsupported");
      return;
   }

   const VkSubresourceLayout layout = {.offset = 0x1000, .rowPitch = 512};
   VkImage image;
   CHECK(create_image(format, 64, 32, 0, usage, 1, &layout, &image));

   const VkMemoryRequirements req = requirements(image, 0);
   EXPECT(req.size >= 0x1000 + 512 * 32,
          "RGBA8: size 0x%llx does not cover offset + plane",
          (unsigned long long)req.size);

   const VkSubresourceLayout color =
      subresource_layout(image, VK_IMAGE_ASPECT_COLOR_BIT);
   const VkSubresourceLayout mem0 =
      subresource_layout(image, VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT);
   EXPECT(color.offset == 0x1000 && mem0.offset == 0x1000,
          "RGBA8: subresource offsets 0x%llx / 0x%llx, want 0x1000",
          (unsigned long long)color.offset, (unsigned long long)mem0.offset);
   EXPECT(color.rowPitch == 512, "RGBA8: row pitch %llu, want 512",
          (unsigned long long)color.rowPitch);

   /* Bound at a nonzero memory offset: the plane is at that plus 0x1000 */
   const VkDeviceSize bind_offset = req.alignment > 0x2000 ? req.alignment
                                                           : 0x2000;
   uint8_t *map;
   VkDeviceMemory memory =
      allocate(bind_offset + req.size, req.memoryTypeBits, &map);
   EXPECT(bind(image, memory, bind_offset, 0) == VK_SUCCESS,
          "RGBA8: bind failed");
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, memory, NULL);

   /* Memory ending before the plane does is refused */
   CHECK(create_image(format, 64, 32, 0, usage, 1, &layout, &image));
   memory = allocate(short_of(req.size), req.memoryTypeBits, &map);
   EXPECT(bind(image, memory, 0, 0) != VK_SUCCESS,
          "RGBA8: binding memory smaller than offset + plane succeeded");
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, memory, NULL);

   /* An offset the texture and PBE paths cannot address is refused */
   const VkSubresourceLayout misaligned = {.offset = 0x1040, .rowPitch = 512};
   EXPECT(create_image(format, 64, 32, 0, usage, 1, &misaligned, &image) ==
             VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
          "RGBA8: misaligned offset accepted");
}

/* NV12 in one binding: the CbCr plane at 0x1000, past a gap */
static void
check_nv12(bool disjoint)
{
   const VkFormat format = VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
   const VkImageCreateFlags flags = disjoint ? VK_IMAGE_CREATE_DISJOINT_BIT : 0;
   const char *name = disjoint ? "NV12 disjoint" : "NV12";
   uint32_t planes = 0;
   if (!linear_supported(format, flags, VK_IMAGE_USAGE_SAMPLED_BIT,
                         &planes)) {
      printf("%s: LINEAR modifier unsupported, skipped\n", name);
      return;
   }
   EXPECT(planes == 2, "%s: %u memory planes", name, planes);

   VkImage image;

   if (!disjoint) {
      const VkSubresourceLayout layouts[2] = {
         {.offset = 0, .rowPitch = 64},
         {.offset = 0x1000, .rowPitch = 64},
      };
      CHECK(create_image(format, 64, 32, flags, usage, 2, layouts, &image));
      const VkMemoryRequirements req = requirements(image, 0);
      EXPECT(req.size >= 0x1000 + 64 * 16,
             "%s: size 0x%llx does not cover the CbCr plane", name,
             (unsigned long long)req.size);
      EXPECT(subresource_layout(image, VK_IMAGE_ASPECT_PLANE_1_BIT).offset ==
                0x1000 &&
             subresource_layout(image, VK_IMAGE_ASPECT_MEMORY_PLANE_1_BIT_EXT)
                   .offset == 0x1000,
             "%s: CbCr subresource offset is not 0x1000", name);

      uint8_t *map;
      const VkDeviceSize bind_offset = req.alignment > 0x4000 ? req.alignment
                                                              : 0x4000;
      VkDeviceMemory memory =
         allocate(bind_offset + req.size, req.memoryTypeBits, &map);
      EXPECT(bind(image, memory, bind_offset, 0) == VK_SUCCESS,
             "%s: bind failed", name);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, memory, NULL);

      /* Planes sharing bytes of one binding are refused */
      const VkSubresourceLayout overlapping[2] = {
         {.offset = 0, .rowPitch = 64},
         {.offset = 0x400, .rowPitch = 64},
      };
      EXPECT(create_image(format, 64, 32, flags, usage, 2, overlapping,
                          &image) ==
                VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT,
             "%s: overlapping planes accepted", name);
      return;
   }

   /* Disjoint: each offset is relative to the plane's own binding, so two
    * dma-bufs whose planes both start at 0 are valid, as is a CbCr plane
    * at 0x80 into its buffer. */
   static const uint64_t cbcr_offsets[] = {0, 0x80};
   for (unsigned i = 0; i < 2; ++i) {
      const VkSubresourceLayout layouts[2] = {
         {.offset = 0, .rowPitch = 64},
         {.offset = cbcr_offsets[i], .rowPitch = 64},
      };
      VkResult result =
         create_image(format, 64, 32, flags, usage, 2, layouts, &image);
      EXPECT(result == VK_SUCCESS,
             "%s: planes at 0 and 0x%llx refused (%d)", name,
             (unsigned long long)cbcr_offsets[i], result);
      if (result != VK_SUCCESS)
         continue;

      const VkMemoryRequirements req0 =
         requirements(image, VK_IMAGE_ASPECT_PLANE_0_BIT);
      const VkMemoryRequirements req1 =
         requirements(image, VK_IMAGE_ASPECT_PLANE_1_BIT);
      EXPECT(req0.size >= 64 * 32 && req1.size >= cbcr_offsets[i] + 64 * 16,
             "%s: plane sizes 0x%llx / 0x%llx too small", name,
             (unsigned long long)req0.size, (unsigned long long)req1.size);
      EXPECT(subresource_layout(image, VK_IMAGE_ASPECT_PLANE_1_BIT).offset ==
                cbcr_offsets[i],
             "%s: CbCr subresource offset is not 0x%llx", name,
             (unsigned long long)cbcr_offsets[i]);

      uint8_t *map0, *map1;
      VkDeviceMemory mem0 = allocate(req0.size, req0.memoryTypeBits, &map0);
      const VkDeviceSize bind1 = req1.alignment > 0x1000 ? req1.alignment
                                                         : 0x1000;
      VkDeviceMemory mem1 =
         allocate(bind1 + req1.size, req1.memoryTypeBits, &map1);
      EXPECT(bind(image, mem0, 0, VK_IMAGE_ASPECT_PLANE_0_BIT) == VK_SUCCESS &&
                bind(image, mem1, bind1, VK_IMAGE_ASPECT_PLANE_1_BIT) ==
                   VK_SUCCESS,
             "%s: plane binds failed", name);
      vkDestroyImage(dev, image, NULL);
      vkFreeMemory(dev, mem0, NULL);
      vkFreeMemory(dev, mem1, NULL);
   }
}

/* GPU-tiled modifiers, compressed ones with their metadata: an explicit
 * offset moves the whole memory plane, so the requirements grow by exactly
 * the offset and binding still needs all of it, metadata included. */
static void
check_tiled_modifiers(void)
{
   static const struct {
      uint64_t modifier;
      const char *name;
   } mods[] = {
      {DRM_FORMAT_MOD_APPLE_GPU_TILED, "GPU_TILED"},
      {DRM_FORMAT_MOD_APPLE_GPU_TILED_COMPRESSED, "GPU_TILED_COMPRESSED"},
   };
   const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
   const VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                   VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   VkDeviceSize sizes[2] = {0, 0};
   for (unsigned m = 0; m < 2; ++m) {
      uint32_t planes;
      if (!modifier_supported(mods[m].modifier, format, 0, usage, &planes)) {
         printf("%s: unsupported for RGBA8, skipped\n", mods[m].name);
         continue;
      }
      EXPECT(planes == 1, "%s: %u memory planes", mods[m].name, planes);

      /* An exporter's layout at offset 0, then the same at 0x4000 */
      VkSubresourceLayout layout = {0};
      VkImage image0, image1;
      CHECK(create_modifier_image(mods[m].modifier, format, 256, 256, 0,
                                  usage, 0, NULL, &image0));
      layout.rowPitch =
         subresource_layout(image0, VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT)
            .rowPitch;
      layout.offset = 0x4000;
      CHECK(create_modifier_image(mods[m].modifier, format, 256, 256, 0,
                                  usage, 1, &layout, &image1));

      const VkMemoryRequirements req0 = requirements(image0, 0);
      const VkMemoryRequirements req1 = requirements(image1, 0);
      sizes[m] = req0.size;
      EXPECT(req1.size == req0.size + 0x4000,
             "%s: size 0x%llx at offset 0x4000, 0x%llx at 0", mods[m].name,
             (unsigned long long)req1.size, (unsigned long long)req0.size);
      EXPECT(subresource_layout(image1, VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT)
                   .offset == 0x4000,
             "%s: subresource offset is not 0x4000", mods[m].name);

      uint8_t *map;
      VkDeviceMemory memory =
         allocate(req1.alignment + req1.size, req1.memoryTypeBits, &map);
      EXPECT(bind(image1, memory, req1.alignment, 0) == VK_SUCCESS,
             "%s: bind failed", mods[m].name);
      vkFreeMemory(dev, memory, NULL);

      /* Memory ending before the plane does is refused */
      VkImage image2;
      CHECK(create_modifier_image(mods[m].modifier, format, 256, 256, 0,
                                  usage, 1, &layout, &image2));
      memory = allocate(short_of(req1.size), req1.memoryTypeBits, &map);
      EXPECT(bind(image2, memory, 0, 0) != VK_SUCCESS,
             "%s: binding memory short of the plane succeeded", mods[m].name);
      vkFreeMemory(dev, memory, NULL);

      vkDestroyImage(dev, image0, NULL);
      vkDestroyImage(dev, image1, NULL);
      vkDestroyImage(dev, image2, NULL);
   }

   /* The compressed memory plane holds the metadata too */
   if (sizes[0] && sizes[1])
      EXPECT(sizes[1] > sizes[0],
             "GPU_TILED_COMPRESSED: size 0x%llx not above GPU_TILED 0x%llx",
             (unsigned long long)sizes[1], (unsigned long long)sizes[0]);
}

int
main(void)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_4,
   };
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&instance_info, NULL, &instance));
   uint32_t count = 1;
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));

   VkPhysicalDeviceVulkan11Features features11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features11,
   };
   vkGetPhysicalDeviceFeatures2(pdev, &features);

   VkPhysicalDeviceVulkan11Features enable11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .samplerYcbcrConversion = features11.samplerYcbcrConversion,
   };
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *extensions[] = {VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enable11,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));

   check_single_plane();
   check_nv12(false);
   check_nv12(true);
   check_tiled_modifiers();

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%u checks: %s\n", checks, failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
