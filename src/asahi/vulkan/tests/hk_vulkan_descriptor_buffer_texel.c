/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * G17P texel buffer descriptors written by vkGetDescriptorEXT carry the same
 * software metadata (agx_abi.h) as those of a VkBufferView, which the texel
 * fetch, store and atomic lowerings read: every view's metadata names its
 * exact address (the texture's tail offset counts only texture coordinates),
 * and a view of a sparse-residency buffer names the buffer's residency map,
 * from which the shader reads non-resident texels as zero. The views lie in
 * a sparse-residency buffer, which needs no memory. On M5 (argument "m5")
 * a storage buffer descriptor of an address in that buffer names the same
 * map in its sparse map page word; elsewhere that word stays zero.
 * Nothing executes on the GPU, so the test also runs under the no-op DRM
 * shim.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "asahi/lib/agx_abi.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

static VkDevice dev;
static PFN_vkGetDescriptorEXT get_descriptor;
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

static uint32_t
word32(const uint8_t *desc, unsigned offset)
{
   uint32_t v;
   memcpy(&v, desc + offset, sizeof(v));
   return v;
}

static uint64_t
word64(const uint8_t *desc, unsigned offset)
{
   uint64_t v;
   memcpy(&v, desc + offset, sizeof(v));
   return v;
}

/* A texel buffer descriptor of `elements` R32_UINT texels at address */
static void
texel_descriptor(VkDescriptorType type, VkDeviceAddress address,
                 VkDeviceSize elements, uint8_t *desc, size_t size)
{
   const VkDescriptorAddressInfoEXT info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
      .address = address,
      .range = elements * 4,
      .format = VK_FORMAT_R32_UINT,
   };
   const VkDescriptorGetInfoEXT get = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
      .type = type,
      .data.pStorageTexelBuffer = &info,
   };
   memset(desc, 0xcd, size);
   get_descriptor(dev, &get, size, desc);
}

int
main(int argc, char **argv)
{
   const bool m5 = argc > 1 && !strcmp(argv[1], "m5");

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

   VkPhysicalDeviceDescriptorBufferFeaturesEXT buffer_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &buffer_features,
   };
   vkGetPhysicalDeviceFeatures2(pdev, &features);
   if (!buffer_features.descriptorBuffer || !features.features.sparseBinding ||
       !features.features.sparseResidencyBuffer) {
      fprintf(stderr, "descriptorBuffer or sparse buffers missing\n");
      return 1;
   }

   VkPhysicalDeviceDescriptorBufferPropertiesEXT props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &props,
   };
   vkGetPhysicalDeviceProperties2(pdev, &props2);

   VkQueueFamilyProperties families[8];
   uint32_t family_count = 8;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families);
   uint32_t family = 0;
   while (family < family_count &&
          !(families[family].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT))
      family++;

   VkPhysicalDeviceVulkan12Features enabled12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .bufferDeviceAddress = VK_TRUE,
   };
   VkPhysicalDeviceDescriptorBufferFeaturesEXT enabled_buffer = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .pNext = &enabled12,
      .descriptorBuffer = VK_TRUE,
   };
   const VkPhysicalDeviceFeatures2 enabled = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &enabled_buffer,
      .features = {
         .sparseBinding = VK_TRUE,
         .sparseResidencyBuffer = VK_TRUE,
      },
   };
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = family,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *extensions[] = {VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME};
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = extensions,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   get_descriptor =
      (PFN_vkGetDescriptorEXT)vkGetDeviceProcAddr(dev, "vkGetDescriptorEXT");

   const VkDeviceSize elements = 1ull << 24;
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .flags = VK_BUFFER_CREATE_SPARSE_BINDING_BIT |
               VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT,
      .size = elements * 4,
      .usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
               VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
   };
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &buffer_info, NULL, &buffer));
   const VkBufferDeviceAddressInfo address_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
      .buffer = buffer,
   };
   const VkDeviceAddress address = vkGetBufferDeviceAddress(dev, &address_info);

   uint8_t desc[256];
   uint32_t map_page = 0;
   const struct {
      const char *name;
      VkDescriptorType type;
      size_t size;
   } types[] = {
      {"uniform", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
       props.robustUniformTexelBufferDescriptorSize},
      {"storage", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
       props.robustStorageTexelBufferDescriptorSize},
   };
   for (unsigned t = 0; t < 2; t++) {
      const char *name = types[t].name;
      if (types[t].size > sizeof(desc) ||
          types[t].size < AGX_G17P_BUFFER_TEXTURE_DESCRIPTOR_SIZE) {
         EXPECT(0, "%s texel buffer descriptor size %zu", name, types[t].size);
         continue;
      }

      const VkDeviceAddress view = address + 4 * 1000;

      /* A view in a texel window */
      texel_descriptor(types[t].type, view, 4096, desc, types[t].size);
      EXPECT(word32(desc, AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET) == 4096,
             "%s view: %u elements", name,
             word32(desc, AGX_G17P_BUFFER_TEXTURE_SIZE_OFFSET));
      EXPECT(word64(desc, AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET) == view,
             "%s view: address 0x%llx, view at 0x%llx", name,
             (unsigned long long)word64(desc, AGX_G17P_BUFFER_TEXTURE_ADDRESS_OFFSET),
             (unsigned long long)view);
      const uint32_t sparse = word32(desc, AGX_G17P_BUFFER_TEXTURE_SPARSE_OFFSET);
      EXPECT((sparse >> AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT) != 0 &&
                (sparse & AGX_G17P_BUFFER_TEXTURE_SPARSE_ELSIZE_MASK) == 2,
             "%s view: sparse word 0x%x, not a residency map with "
             "4-byte texels", name, sparse);
      map_page = sparse >> AGX_G17P_BUFFER_TEXTURE_SPARSE_MAP_SHIFT;
   }

   /* A storage buffer descriptor inside the buffer: base address, size,
    * then the residency map page (hk_buffer_address). */
   const VkDescriptorAddressInfoEXT ssbo_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
      .address = address + 4 * 1000,
      .range = 4096,
   };
   const VkDescriptorGetInfoEXT ssbo_get = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .data.pStorageBuffer = &ssbo_info,
   };
   if (props.robustStorageBufferDescriptorSize < 16 ||
       props.robustStorageBufferDescriptorSize > sizeof(desc)) {
      EXPECT(0, "storage buffer descriptor size %zu",
             props.robustStorageBufferDescriptorSize);
   } else {
      memset(desc, 0xcd, sizeof(desc));
      get_descriptor(dev, &ssbo_get, props.robustStorageBufferDescriptorSize,
                     desc);
      EXPECT(word64(desc, 0) == ssbo_info.address,
             "storage buffer: address 0x%llx",
             (unsigned long long)word64(desc, 0));
      const uint32_t expected = m5 ? map_page : 0;
      EXPECT(map_page != 0 && word32(desc, 12) == expected,
             "storage buffer: map page 0x%x, expected 0x%x", word32(desc, 12),
             expected);
   }

   vkDestroyBuffer(dev, buffer, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
