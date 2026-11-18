/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * VK_EXT_descriptor_buffer sizes as vkd3d-proton uses them. It lays out its
 * heaps as descriptor buffer bindings of one type or of a mutable type list,
 * places element i at i times the largest reported size of the binding's
 * types, and copies descriptors only of 4, 8, 16, 32, 48, 64 or 128 bytes
 * (vkd3d_bindless_find_copy_template); any other size leaves its heap
 * unmapped. So every size it reads must be one of those, and each binding's
 * stride must be exactly that size: the offset of a binding that follows 64
 * elements is 64 strides. A multi-plane sampler binding takes one combined
 * image sampler descriptor per plane. vkGetDescriptorEXT must define every
 * byte of the reported size, padding included, since applications copy it
 * whole. Nothing executes on the GPU, so the test also runs under the no-op
 * DRM shim.
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

#define COUNT 64

static VkPhysicalDevice pdev;
static VkDevice dev;
static PFN_vkGetDescriptorSetLayoutBindingOffsetEXT get_binding_offset;
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

static bool
vkd3d_copies(VkDeviceSize size)
{
   switch (size) {
   case 4: case 8: case 16: case 32: case 48: case 64: case 128:
      return true;
   default:
      return false;
   }
}

/* The stride of binding 0 (COUNT descriptors of type, or of the mutable
 * types), from the offset of a uniform buffer binding after it. */
static VkDeviceSize
binding_stride(VkDescriptorType type, const VkDescriptorType *mutable_types,
               uint32_t mutable_count, const VkSampler *immutable_samplers)
{
   const VkDescriptorSetLayoutBinding bindings[] = {
      {0, type, COUNT, VK_SHADER_STAGE_ALL, immutable_samplers},
      {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL, NULL},
   };
   const VkMutableDescriptorTypeListEXT lists[] = {
      {mutable_count, mutable_types},
      {0, NULL},
   };
   const VkMutableDescriptorTypeCreateInfoEXT mutable_info = {
      .sType = VK_STRUCTURE_TYPE_MUTABLE_DESCRIPTOR_TYPE_CREATE_INFO_EXT,
      .mutableDescriptorTypeListCount = 2,
      .pMutableDescriptorTypeLists = lists,
   };
   const VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = mutable_count ? &mutable_info : NULL,
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &info, NULL, &layout));
   VkDeviceSize offset0, offset1;
   get_binding_offset(dev, layout, 0, &offset0);
   get_binding_offset(dev, layout, 1, &offset1);
   vkDestroyDescriptorSetLayout(dev, layout, NULL);
   return (offset1 - offset0) / COUNT;
}

static void
check_type(const char *name, VkDescriptorType type, VkDeviceSize size)
{
   EXPECT(vkd3d_copies(size), "%s descriptor size %llu is not one vkd3d-proton copies",
          name, (unsigned long long)size);
   VkDeviceSize stride = binding_stride(type, NULL, 0, NULL);
   EXPECT(stride == size, "%s binding stride %llu, descriptor size %llu", name,
          (unsigned long long)stride, (unsigned long long)size);
}

static void
check_mutable(const char *name, const VkDescriptorType *types, uint32_t count,
              const VkDeviceSize *sizes)
{
   VkDeviceSize max = 0;
   for (uint32_t i = 0; i < count; i++)
      max = sizes[i] > max ? sizes[i] : max;
   VkDeviceSize stride =
      binding_stride(VK_DESCRIPTOR_TYPE_MUTABLE_EXT, types, count, NULL);
   EXPECT(vkd3d_copies(stride), "%s mutable stride %llu is not one vkd3d-proton copies",
          name, (unsigned long long)stride);
   EXPECT(stride == max, "%s mutable stride %llu, largest descriptor size %llu",
          name, (unsigned long long)stride, (unsigned long long)max);
}

/* The descriptor written over memory of 0x00 and of 0xff must be the same:
 * no byte of the reported size may be left as it was. */
static void
check_defined(const char *name, const VkDescriptorGetInfoEXT *info,
              VkDeviceSize size)
{
   PFN_vkGetDescriptorEXT get_descriptor = (PFN_vkGetDescriptorEXT)
      vkGetDeviceProcAddr(dev, "vkGetDescriptorEXT");
   uint8_t a[256], b[256];
   if (size > sizeof(a))
      abort();
   memset(a, 0x00, sizeof(a));
   memset(b, 0xff, sizeof(b));
   get_descriptor(dev, info, size, a);
   get_descriptor(dev, info, size, b);
   EXPECT(memcmp(a, b, size) == 0,
          "%s descriptor leaves bytes of its %llu undefined", name,
          (unsigned long long)size);
}

/* A three-plane YCbCr sampler: one combined image sampler descriptor per
 * plane, each at the reported size. */
static void
check_multiplane(const VkPhysicalDeviceDescriptorBufferPropertiesEXT *props)
{
   const VkFormat format = VK_FORMAT_G8_B8_R8_3PLANE_420_UNORM;
   VkFormatProperties fmt;
   vkGetPhysicalDeviceFormatProperties(pdev, format, &fmt);
   if (!(fmt.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) {
      printf("no sampled %d, multi-plane check skipped\n", format);
      return;
   }

   VkSamplerYcbcrConversionImageFormatProperties ycbcr = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_IMAGE_FORMAT_PROPERTIES,
   };
   VkImageFormatProperties2 image_props = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
      .pNext = &ycbcr,
   };
   const VkPhysicalDeviceImageFormatInfo2 image_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = format,
      .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
   };
   CHECK(vkGetPhysicalDeviceImageFormatProperties2(pdev, &image_info, &image_props));

   const VkSamplerYcbcrConversionCreateInfo conversion_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
      .format = format,
      .ycbcrModel = VK_SAMPLER_YCBCR_MODEL_CONVERSION_YCBCR_709,
      .ycbcrRange = VK_SAMPLER_YCBCR_RANGE_ITU_NARROW,
      .xChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
      .yChromaOffset = VK_CHROMA_LOCATION_MIDPOINT,
      .chromaFilter = VK_FILTER_NEAREST,
   };
   VkSamplerYcbcrConversion conversion;
   CHECK(vkCreateSamplerYcbcrConversion(dev, &conversion_info, NULL, &conversion));
   const VkSamplerYcbcrConversionInfo sampler_conversion = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
      .conversion = conversion,
   };
   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = &sampler_conversion,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sampler_info, NULL, &sampler));
   VkSampler samplers[COUNT];
   for (unsigned i = 0; i < COUNT; i++)
      samplers[i] = sampler;

   const VkDeviceSize expected = ycbcr.combinedImageSamplerDescriptorCount *
                                 props->combinedImageSamplerDescriptorSize;
   const VkDeviceSize stride = binding_stride(
      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NULL, 0, samplers);
   EXPECT(stride == expected,
          "3-plane sampler binding stride %llu, %u descriptors of %llu",
          (unsigned long long)stride, ycbcr.combinedImageSamplerDescriptorCount,
          (unsigned long long)props->combinedImageSamplerDescriptorSize);

   vkDestroySampler(dev, sampler, NULL);
   vkDestroySamplerYcbcrConversion(dev, conversion, NULL);
}

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
   VkResult r = vkEnumeratePhysicalDevices(instance, &count, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || count == 0) {
      fprintf(stderr, "no physical device\n");
      return 2;
   }

   VkPhysicalDeviceRobustness2FeaturesEXT robustness2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
   };
   VkPhysicalDeviceMutableDescriptorTypeFeaturesEXT mutable_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MUTABLE_DESCRIPTOR_TYPE_FEATURES_EXT,
      .pNext = &robustness2,
   };
   VkPhysicalDeviceDescriptorBufferFeaturesEXT buffer_features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .pNext = &mutable_features,
   };
   VkPhysicalDeviceVulkan11Features features11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &buffer_features,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
      .pNext = &features11,
   };
   vkGetPhysicalDeviceFeatures2(pdev, &features);
   if (!buffer_features.descriptorBuffer ||
       !mutable_features.mutableDescriptorType) {
      fprintf(stderr, "descriptorBuffer or mutableDescriptorType missing\n");
      return 1;
   }
   /* Enable what the chain reports, without the core 1.0 features and of
    * robustness2 only null descriptors (for check_defined()). */
   VkPhysicalDeviceFeatures2 enabled_features = features;
   enabled_features.features = (VkPhysicalDeviceFeatures){0};
   robustness2.robustBufferAccess2 = robustness2.robustImageAccess2 = false;
   if (!robustness2.nullDescriptor)
      mutable_features.pNext = NULL;

   VkPhysicalDeviceDescriptorBufferPropertiesEXT props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 props2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
      .pNext = &props,
   };
   vkGetPhysicalDeviceProperties2(pdev, &props2);

   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const char *extensions[] = {
      VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME,
      VK_EXT_MUTABLE_DESCRIPTOR_TYPE_EXTENSION_NAME,
      VK_EXT_ROBUSTNESS_2_EXTENSION_NAME,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &enabled_features,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
      .enabledExtensionCount = robustness2.nullDescriptor ? 3 : 2,
      .ppEnabledExtensionNames = extensions,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   get_binding_offset = (PFN_vkGetDescriptorSetLayoutBindingOffsetEXT)
      vkGetDeviceProcAddr(dev, "vkGetDescriptorSetLayoutBindingOffsetEXT");

   /* The sizes vkd3d_get_descriptor_size_for_type() reads */
   check_type("sampler", VK_DESCRIPTOR_TYPE_SAMPLER, props.samplerDescriptorSize);
   check_type("sampled image", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
              props.sampledImageDescriptorSize);
   check_type("storage image", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
              props.storageImageDescriptorSize);
   check_type("uniform texel buffer", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
              props.robustUniformTexelBufferDescriptorSize);
   check_type("storage texel buffer", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
              props.robustStorageTexelBufferDescriptorSize);
   check_type("uniform buffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
              props.robustUniformBufferDescriptorSize);
   check_type("storage buffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
              props.robustStorageBufferDescriptorSize);
   EXPECT(binding_stride(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, NULL, 0,
                         NULL) == props.combinedImageSamplerDescriptorSize,
          "combined image sampler binding stride is not its descriptor size");

   /* vkd3d_bindless_build_mutable_type_list(): typed and raw resource heaps */
   const VkDescriptorType typed[] = {
      VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
      VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
      VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
   };
   const VkDeviceSize typed_sizes[] = {
      props.sampledImageDescriptorSize,
      props.robustUniformTexelBufferDescriptorSize,
      props.storageImageDescriptorSize,
      props.robustStorageTexelBufferDescriptorSize,
   };
   check_mutable("typed", typed, 4, typed_sizes);

   const VkDescriptorType raw[] = {
      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
   };
   const VkDeviceSize raw_sizes[] = {
      props.robustUniformBufferDescriptorSize,
      props.robustStorageBufferDescriptorSize,
   };
   check_mutable("raw", raw, 2, raw_sizes);

   if (features11.samplerYcbcrConversion)
      check_multiplane(&props);

   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sampler_info, NULL, &sampler));
   check_defined("sampler",
                 &(VkDescriptorGetInfoEXT){
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                    .type = VK_DESCRIPTOR_TYPE_SAMPLER,
                    .data.pSampler = &sampler,
                 },
                 props.samplerDescriptorSize);
   if (robustness2.nullDescriptor) {
      /* Null resources, which need no memory under the shim */
      const struct {
         const char *name;
         VkDescriptorType type;
         VkDeviceSize size;
      } nulls[] = {
         {"null sampled image", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
          props.sampledImageDescriptorSize},
         {"null storage image", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
          props.storageImageDescriptorSize},
         {"null uniform texel buffer", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER,
          props.robustUniformTexelBufferDescriptorSize},
         {"null storage texel buffer", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER,
          props.robustStorageTexelBufferDescriptorSize},
         {"null uniform buffer", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
          props.robustUniformBufferDescriptorSize},
         {"null storage buffer", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
          props.robustStorageBufferDescriptorSize},
      };
      for (unsigned i = 0; i < sizeof(nulls) / sizeof(nulls[0]); i++) {
         /* Every data member is a pointer; NULL is the null descriptor. */
         check_defined(nulls[i].name,
                       &(VkDescriptorGetInfoEXT){
                          .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT,
                          .type = nulls[i].type,
                       },
                       nulls[i].size);
      }
   } else {
      printf("no nullDescriptor, null descriptor checks skipped\n");
   }
   vkDestroySampler(dev, sampler, NULL);

   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);
   printf("%u checks, %u failures\n", checks, failures);
   return failures ? 1 : 0;
}
