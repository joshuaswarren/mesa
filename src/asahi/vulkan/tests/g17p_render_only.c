/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Render-only Vulkan submit for T8140/G17P: clear, draw one full-screen
 * triangle into a one-tile RGBA8 image, store at end of tile. No readback.
 *
 * This exists to exercise the 3D/TA launch path on its own. The image->buffer
 * copy of g17p-render-only is a *compute* dispatch, which is blocked
 * on its own separate problem, so removing it isolates whether a render
 * submit reaches the GPU and completes.
 *
 * Every wait is bounded: a launch that faults never signals its fence, and a
 * hung wait would cost a reboot. VK_TIMEOUT is reported as its own outcome
 * rather than blocking.
 */

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <dlfcn.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "g17p_wsi_triangle_frag.h"
#include "g17p_wsi_triangle_vert.h"
#define G17P_VERTEX_SPV g17p_wsi_triangle_vert_spv
#define G17P_FRAGMENT_SPV g17p_wsi_triangle_frag_spv

#ifndef AGX_TEST_DEVICE_ID
#define AGX_TEST_DEVICE_ID 0x8140
#endif
#ifndef IMAGE_WIDTH
#define IMAGE_WIDTH 32
#endif
#ifndef IMAGE_HEIGHT
#define IMAGE_HEIGHT 32
#endif
#define IMAGE_BYTES (IMAGE_WIDTH * IMAGE_HEIGHT * 4)
#define COMMAND_BUFFER_COUNT 1

#define VK_FN(name) static PFN_##name name

VK_FN(vkGetInstanceProcAddr);
VK_FN(vkGetDeviceProcAddr);
VK_FN(vkCreateInstance);
VK_FN(vkCreateDebugUtilsMessengerEXT);
VK_FN(vkEnumeratePhysicalDevices);
VK_FN(vkGetPhysicalDeviceProperties);
VK_FN(vkGetPhysicalDeviceMemoryProperties);
VK_FN(vkGetPhysicalDeviceQueueFamilyProperties);
VK_FN(vkGetPhysicalDeviceFormatProperties);
VK_FN(vkCreateDevice);
VK_FN(vkGetDeviceQueue);
VK_FN(vkCreateBuffer);
VK_FN(vkCmdCopyImageToBuffer);
VK_FN(vkGetBufferMemoryRequirements);
VK_FN(vkCreateImage);
VK_FN(vkGetImageMemoryRequirements);
VK_FN(vkAllocateMemory);
VK_FN(vkBindBufferMemory);
VK_FN(vkBindImageMemory);
VK_FN(vkMapMemory);
VK_FN(vkCreateImageView);
VK_FN(vkCreateRenderPass);
VK_FN(vkCreateShaderModule);
VK_FN(vkCreatePipelineLayout);
VK_FN(vkCreateGraphicsPipelines);
VK_FN(vkCreateFramebuffer);
VK_FN(vkCreateCommandPool);
VK_FN(vkAllocateCommandBuffers);
VK_FN(vkBeginCommandBuffer);
VK_FN(vkCmdBeginRenderPass);
VK_FN(vkCmdBindPipeline);
VK_FN(vkCmdBindVertexBuffers);
VK_FN(vkCmdBindIndexBuffer);
VK_FN(vkCmdDraw);
VK_FN(vkCmdDrawIndirect);
VK_FN(vkCmdDrawIndexed);
VK_FN(vkCmdDrawIndexedIndirect);
VK_FN(vkCmdEndRenderPass);
VK_FN(vkCmdPipelineBarrier);
VK_FN(vkCmdBlitImage);
VK_FN(vkEndCommandBuffer);
VK_FN(vkCreateFence);
VK_FN(vkQueueSubmit);
VK_FN(vkWaitForFences);
VK_FN(vkUnmapMemory);

static void
die(const char *what)
{
   fprintf(stderr, "g17p-render-only: %s\n", what);
   exit(EXIT_FAILURE);
}

static void
check(VkResult result, const char *what)
{
   if (result != VK_SUCCESS) {
      fprintf(stderr, "g17p-render-only: %s returned %d\n", what,
              result);
      exit(EXIT_FAILURE);
   }
}

#define LOAD_GLOBAL(name)                                                      \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(VK_NULL_HANDLE, #name);         \
      if (!name)                                                               \
         die("loader is missing " #name);                                     \
   } while (0)

#define LOAD_INSTANCE(instance, name)                                          \
   do {                                                                        \
      name = (PFN_##name)vkGetInstanceProcAddr(instance, #name);               \
      if (!name)                                                               \
         die("instance is missing " #name);                                   \
   } while (0)

#define LOAD_DEVICE(device, name)                                              \
   do {                                                                        \
      name = (PFN_##name)vkGetDeviceProcAddr(device, #name);                   \
      if (!name)                                                               \
         die("device is missing " #name);                                     \
   } while (0)

/* Non-fatal variant: returns UINT32_MAX instead of dying, so an optional
 * memory class can be probed. */
static uint32_t
try_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                uint32_t allowed, VkMemoryPropertyFlags required)
{
   for (uint32_t i = 0; i < properties->memoryTypeCount; ++i) {
      if (getenv("AGX_RENDER_UNCACHED") &&
          (properties->memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))
         continue;
      if ((allowed & (1u << i)) &&
          (properties->memoryTypes[i].propertyFlags & required) == required)
         return i;
   }
   return UINT32_MAX;
}

static uint32_t
find_memory_type(const VkPhysicalDeviceMemoryProperties *properties,
                 uint32_t allowed, VkMemoryPropertyFlags required)
{
   uint32_t type = try_memory_type(properties, allowed, required);
   if (type != UINT32_MAX)
      return type;

   die("required Vulkan memory type is unavailable");
   return UINT32_MAX;
}

static VKAPI_ATTR VkBool32 VKAPI_CALL
render_debug_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                     VkDebugUtilsMessageTypeFlagsEXT types,
                     const VkDebugUtilsMessengerCallbackDataEXT *data,
                     void *user_data)
{
   (void)severity;
   (void)types;
   (void)user_data;
   fprintf(stderr, "render debug: %s\n", data->pMessage);
   return VK_FALSE;
}

int
main(
   void
)
{
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   if (!loader)
      die(dlerror());

   vkGetInstanceProcAddr =
      (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   if (!vkGetInstanceProcAddr)
      die("libvulkan.so.1 has no vkGetInstanceProcAddr");
   LOAD_GLOBAL(vkCreateInstance);

   const VkApplicationInfo app_info = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p-render-only",
      .applicationVersion = 1,
      .pEngineName = "none",
      .engineVersion = 1,
      .apiVersion = VK_API_VERSION_1_2,
   };
   const bool debug_messages = getenv("AGX_RENDER_DEBUG") != NULL;
   const char *debug_extension = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app_info,
      .enabledExtensionCount = debug_messages ? 1 : 0,
      .ppEnabledExtensionNames = &debug_extension,
   };
   VkInstance instance;
   check(vkCreateInstance(&instance_info, NULL, &instance),
         "vkCreateInstance");

   if (debug_messages) {
      LOAD_INSTANCE(instance, vkCreateDebugUtilsMessengerEXT);
      const VkDebugUtilsMessengerCreateInfoEXT debug_info = {
         .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
         .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                            VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
         .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
         .pfnUserCallback = render_debug_message,
      };
      VkDebugUtilsMessengerEXT messenger;
      check(vkCreateDebugUtilsMessengerEXT(instance, &debug_info, NULL,
                                            &messenger),
            "vkCreateDebugUtilsMessengerEXT");
   }

   LOAD_INSTANCE(instance, vkGetDeviceProcAddr);
   LOAD_INSTANCE(instance, vkEnumeratePhysicalDevices);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceMemoryProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceQueueFamilyProperties);
   LOAD_INSTANCE(instance, vkGetPhysicalDeviceFormatProperties);
   LOAD_INSTANCE(instance, vkCreateDevice);

   uint32_t physical_count = 0;
   check(vkEnumeratePhysicalDevices(instance, &physical_count, NULL),
         "vkEnumeratePhysicalDevices(count)");
   if (!physical_count)
      die("Vulkan loader found no physical device");

   VkPhysicalDevice *physical_devices =
      calloc(physical_count, sizeof(*physical_devices));
   if (!physical_devices)
      die("out of host memory");
   check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                    physical_devices),
         "vkEnumeratePhysicalDevices(list)");

   VkPhysicalDevice physical = VK_NULL_HANDLE;
   uint32_t queue_family = UINT32_MAX;
   VkPhysicalDeviceProperties properties = {0};
   for (uint32_t p = 0; p < physical_count && !physical; ++p) {
      VkPhysicalDeviceProperties candidate;
      vkGetPhysicalDeviceProperties(physical_devices[p], &candidate);
      if (candidate.deviceID != AGX_TEST_DEVICE_ID)
         continue;

      uint32_t queue_count = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, NULL);
      VkQueueFamilyProperties *queues = calloc(queue_count, sizeof(*queues));
      if (!queues)
         die("out of host memory");
      vkGetPhysicalDeviceQueueFamilyProperties(physical_devices[p],
                                                &queue_count, queues);
      for (uint32_t q = 0; q < queue_count; ++q) {
         const VkQueueFlags required =
            VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT;
         if ((queues[q].queueFlags & required) == required) {
            physical = physical_devices[p];
            queue_family = q;
            properties = candidate;
            break;
         }
      }
      free(queues);
   }
   free(physical_devices);
   if (!physical)
      die("requested Apple device has no graphics-and-transfer queue");

   VkFormatProperties format_properties;
   vkGetPhysicalDeviceFormatProperties(physical, VK_FORMAT_R8G8B8A8_UNORM,
                                       &format_properties);
   const VkFormatFeatureFlags required_format =
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT;
   if ((format_properties.optimalTilingFeatures & required_format) !=
       required_format)
      die("RGBA8 optimal images lack color-attachment or transfer-src support");

   fprintf(stderr, "g17p-render-only: device %s, queue %u\n",
           properties.deviceName, queue_family);





   const float queue_priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = queue_family,
      .queueCount = 1,
      .pQueuePriorities = &queue_priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   VkDevice device;
   check(vkCreateDevice(physical, &device_info, NULL, &device),
         "vkCreateDevice");

   LOAD_DEVICE(device, vkGetDeviceQueue);
   LOAD_DEVICE(device, vkCreateBuffer);
   LOAD_DEVICE(device, vkCmdCopyImageToBuffer);
   LOAD_DEVICE(device, vkGetBufferMemoryRequirements);
   LOAD_DEVICE(device, vkCreateImage);
   LOAD_DEVICE(device, vkGetImageMemoryRequirements);
   LOAD_DEVICE(device, vkAllocateMemory);
   LOAD_DEVICE(device, vkBindBufferMemory);
   LOAD_DEVICE(device, vkBindImageMemory);
   LOAD_DEVICE(device, vkMapMemory);
   LOAD_DEVICE(device, vkCreateImageView);
   LOAD_DEVICE(device, vkCreateRenderPass);
   LOAD_DEVICE(device, vkCreateShaderModule);
   LOAD_DEVICE(device, vkCreatePipelineLayout);
   LOAD_DEVICE(device, vkCreateGraphicsPipelines);
   LOAD_DEVICE(device, vkCreateFramebuffer);
   LOAD_DEVICE(device, vkCreateCommandPool);
   LOAD_DEVICE(device, vkAllocateCommandBuffers);
   LOAD_DEVICE(device, vkBeginCommandBuffer);
   LOAD_DEVICE(device, vkCmdBeginRenderPass);
   LOAD_DEVICE(device, vkCmdBindPipeline);
   LOAD_DEVICE(device, vkCmdBindVertexBuffers);
   LOAD_DEVICE(device, vkCmdBindIndexBuffer);
   LOAD_DEVICE(device, vkCmdDraw);
   LOAD_DEVICE(device, vkCmdDrawIndirect);
   LOAD_DEVICE(device, vkCmdDrawIndexed);
   LOAD_DEVICE(device, vkCmdDrawIndexedIndirect);
    LOAD_DEVICE(device, vkCmdEndRenderPass);
    LOAD_DEVICE(device, vkCmdPipelineBarrier);
    LOAD_DEVICE(device, vkCmdBlitImage);
   LOAD_DEVICE(device, vkEndCommandBuffer);
   LOAD_DEVICE(device, vkCreateFence);
   LOAD_DEVICE(device, vkQueueSubmit);
   LOAD_DEVICE(device, vkWaitForFences);
   LOAD_DEVICE(device, vkUnmapMemory);

   VkQueue queue;
   vkGetDeviceQueue(device, queue_family, 0, &queue);

   VkPhysicalDeviceMemoryProperties memory_properties;
   vkGetPhysicalDeviceMemoryProperties(physical, &memory_properties);

   static const float vertices[][2] = {
      {-1.0f, -1.0f},
      {3.0f, -1.0f},
      {-1.0f, 3.0f},
   };
   const bool alternate_geometry = getenv("AGX_RENDER_ALTERNATE") != NULL;
   static const float clipped_vertices[][2] = {
      {4.0f, 4.0f}, {6.0f, 4.0f}, {4.0f, 6.0f},
   };
   const size_t vertex_bytes = sizeof(vertices) * (alternate_geometry ? 2 : 1);
   /* The first four words also form an arrays-indirect draw of three
    * vertices. Keep the index data separate from both draw records. */
   const VkDrawIndexedIndirectCommand repeat_indirect = {3, 1, 0, 0, 0};
   const uint32_t repeat_indices[] = {0, 1, 2};
   const VkDeviceSize repeat_index_offset =
      vertex_bytes + sizeof(repeat_indirect);
   const VkDeviceSize vertex_allocation_size =
      repeat_index_offset + sizeof(repeat_indices);
   const VkBufferCreateInfo vertex_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = vertex_allocation_size,
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
               VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   VkBuffer vertex_buffer;
   check(vkCreateBuffer(device, &vertex_info, NULL, &vertex_buffer),
         "vkCreateBuffer(vertex)");
   VkMemoryRequirements vertex_requirements;
   vkGetBufferMemoryRequirements(device, vertex_buffer, &vertex_requirements);
   const uint32_t vertex_memory_type = find_memory_type(
      &memory_properties, vertex_requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   const VkMemoryAllocateInfo vertex_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = vertex_requirements.size,
      .memoryTypeIndex = vertex_memory_type,
   };
   VkDeviceMemory vertex_memory;
   check(vkAllocateMemory(device, &vertex_memory_info, NULL, &vertex_memory),
         "vkAllocateMemory(vertex)");
   check(vkBindBufferMemory(device, vertex_buffer, vertex_memory, 0),
         "vkBindBufferMemory(vertex)");
   void *vertex_map = NULL;
   check(vkMapMemory(device, vertex_memory, 0, vertex_allocation_size, 0,
                     &vertex_map),
         "vkMapMemory(vertex)");
   memcpy(vertex_map, vertices, sizeof(vertices));
   memcpy((uint8_t *)vertex_map + vertex_bytes, &repeat_indirect,
          sizeof(repeat_indirect));
   memcpy((uint8_t *)vertex_map + repeat_index_offset, repeat_indices,
          sizeof(repeat_indices));
   if (alternate_geometry)
      memcpy((uint8_t *)vertex_map + sizeof(vertices), clipped_vertices,
             sizeof(clipped_vertices));




   const bool blit_test = getenv("AGX_RENDER_BLIT") != NULL;
   const bool readback_test = getenv("AGX_RENDER_READBACK") != NULL;
   if (blit_test && readback_test)
      die("select one of AGX_RENDER_BLIT and AGX_RENDER_READBACK");
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .extent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage image;
   check(vkCreateImage(device, &image_info, NULL, &image), "vkCreateImage");
   VkMemoryRequirements image_requirements;
   vkGetImageMemoryRequirements(device, image, &image_requirements);
   /* Inspect the source independently before testing an optional transfer.
    * This prevents a successful copy of an undrawn image from passing. */
   uint32_t image_memory_type =
      try_memory_type(&memory_properties, image_requirements.memoryTypeBits,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   bool image_host_visible = image_memory_type != UINT32_MAX;
   if (!image_host_visible)
      image_memory_type =
         find_memory_type(&memory_properties, image_requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   fprintf(stderr, "g17p-render-only: image memory host-visible=%d type=%u\n",
           (int)image_host_visible, image_memory_type);
   const VkMemoryAllocateInfo image_memory_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = image_requirements.size,
      .memoryTypeIndex = image_memory_type,
   };
   VkDeviceMemory image_memory;
   check(vkAllocateMemory(device, &image_memory_info, NULL, &image_memory),
         "vkAllocateMemory(image)");
   check(vkBindImageMemory(device, image, image_memory, 0),
         "vkBindImageMemory");

   VkImage blit_image = VK_NULL_HANDLE;
   VkDeviceMemory blit_memory = VK_NULL_HANDLE;
   uint8_t *blit_pixels = NULL;
   VkBuffer readback_buffer = VK_NULL_HANDLE;
   VkDeviceMemory readback_memory = VK_NULL_HANDLE;
   uint8_t *readback_map = NULL;
   if (blit_test) {
      VkImageCreateInfo copy_info = image_info;
      copy_info.usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
      check(vkCreateImage(device, &copy_info, NULL, &blit_image),
            "vkCreateImage(blit)");
      VkMemoryRequirements requirements;
      vkGetImageMemoryRequirements(device, blit_image, &requirements);
      const VkMemoryAllocateInfo allocation = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = requirements.size,
         .memoryTypeIndex = find_memory_type(&memory_properties,
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      check(vkAllocateMemory(device, &allocation, NULL, &blit_memory),
            "vkAllocateMemory(blit)");
      check(vkBindImageMemory(device, blit_image, blit_memory, 0),
            "vkBindImageMemory(blit)");
      check(vkMapMemory(device, blit_memory, 0, IMAGE_BYTES, 0,
                        (void **)&blit_pixels), "vkMapMemory(blit)");
      memset(blit_pixels, 0xa5, IMAGE_BYTES);
   }
   if (readback_test) {
      const VkBufferCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = IMAGE_BYTES + 32,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
         .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      };
      check(vkCreateBuffer(device, &info, NULL, &readback_buffer), "vkCreateBuffer(readback)");
      VkMemoryRequirements requirements;
      vkGetBufferMemoryRequirements(device, readback_buffer, &requirements);
      const VkMemoryAllocateInfo allocation = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = requirements.size,
         .memoryTypeIndex = find_memory_type(&memory_properties,
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      check(vkAllocateMemory(device, &allocation, NULL, &readback_memory),
            "vkAllocateMemory(readback)");
      check(vkBindBufferMemory(device, readback_buffer, readback_memory, 0),
            "vkBindBufferMemory(readback)");
      check(vkMapMemory(device, readback_memory, 0, IMAGE_BYTES + 32, 0,
                        (void **)&readback_map), "vkMapMemory(readback)");
      memset(readback_map, 0xa5, IMAGE_BYTES + 32);
      blit_pixels = readback_map + 16;
   }

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
         .levelCount = 1,
         .layerCount = 1,
      },
   };
   VkImageView view;
   check(vkCreateImageView(device, &view_info, NULL, &view),
         "vkCreateImageView");

   const VkAttachmentDescription attachment = {
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
   };
   const VkAttachmentReference color_ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   const VkSubpassDescription subpass = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &color_ref,
   };
   const VkSubpassDependency dependencies[] = {
      {
         .srcSubpass = VK_SUBPASS_EXTERNAL,
         .dstSubpass = 0,
         .srcStageMask = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      },
      {
         .srcSubpass = 0,
         .dstSubpass = VK_SUBPASS_EXTERNAL,
         .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      },
   };
   const VkRenderPassCreateInfo render_pass_info = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &attachment,
      .subpassCount = 1,
      .pSubpasses = &subpass,
      .dependencyCount = 2,
      .pDependencies = dependencies,
   };
   VkRenderPass render_pass;
   check(vkCreateRenderPass(device, &render_pass_info, NULL, &render_pass),
         "vkCreateRenderPass");

   const VkShaderModuleCreateInfo vert_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(G17P_VERTEX_SPV),
      .pCode = G17P_VERTEX_SPV,
   };
   const VkShaderModuleCreateInfo frag_module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(G17P_FRAGMENT_SPV),
      .pCode = G17P_FRAGMENT_SPV,
   };
   VkShaderModule vert_module, frag_module;
   check(vkCreateShaderModule(device, &vert_module_info, NULL, &vert_module),
         "vkCreateShaderModule(vertex)");
   check(vkCreateShaderModule(device, &frag_module_info, NULL, &frag_module),
         "vkCreateShaderModule(fragment)");


   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pipeline_layout;
   check(vkCreatePipelineLayout(device, &layout_info, NULL, &pipeline_layout),
         "vkCreatePipelineLayout");

   const VkPipelineShaderStageCreateInfo stages[] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vert_module,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = frag_module,
         .pName = "main",
      },
   };
   const VkVertexInputBindingDescription vertex_binding = {
      .binding = 0,
      .stride = sizeof(vertices[0]),
      .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
   };
   const VkVertexInputAttributeDescription vertex_attribute = {
      .location = 0,
      .binding = 0,
      .format = VK_FORMAT_R32G32_SFLOAT,
   };
   const VkPipelineVertexInputStateCreateInfo vertex_input = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &vertex_binding,
      .vertexAttributeDescriptionCount = 1,
      .pVertexAttributeDescriptions = &vertex_attribute,
   };
   const VkPipelineInputAssemblyStateCreateInfo input_assembly = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   const VkRect2D scissor = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}};
   const VkPipelineViewportStateCreateInfo viewport_state = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo raster = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo multisample = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState blend_attachment = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   const VkPipelineColorBlendStateCreateInfo blend = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &blend_attachment,
   };
   const VkGraphicsPipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vertex_input,
      .pInputAssemblyState = &input_assembly,
      .pViewportState = &viewport_state,
      .pRasterizationState = &raster,
      .pMultisampleState = &multisample,
      .pColorBlendState = &blend,
      .layout = pipeline_layout,
      .renderPass = render_pass,
   };
   VkPipeline pipeline;
   check(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                   NULL, &pipeline),
         "vkCreateGraphicsPipelines");
   fprintf(stderr, "g17p-render-only: graphics pipeline created\n");

   const VkFramebufferCreateInfo framebuffer_info = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = render_pass,
      .attachmentCount = 1,
      .pAttachments = &view,
      .width = IMAGE_WIDTH,
      .height = IMAGE_HEIGHT,
      .layers = 1,
   };
   VkFramebuffer framebuffer;
   check(vkCreateFramebuffer(device, &framebuffer_info, NULL, &framebuffer),
         "vkCreateFramebuffer");

   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = queue_family,
   };
   VkCommandPool command_pools[COMMAND_BUFFER_COUNT];
   check(vkCreateCommandPool(device, &pool_info, NULL, &command_pools[0]),
         "vkCreateCommandPool");
   unsigned frame_index = 0;
   unsigned frame_count = 1;
   const char *repeat_env = getenv("AGX_RENDER_REPEAT");
   if (repeat_env) {
      char *end = NULL;
      const unsigned long requested = strtoul(repeat_env, &end, 10);
      if (!*repeat_env || *end || requested < 1 || requested > 64)
         die("AGX_RENDER_REPEAT must be between 1 and 64");
      frame_count = requested;
   }
   if ((blit_test || readback_test) && (frame_count != 1 || alternate_geometry))
      die("copy tests require one full-frame draw");
   /* Allocate and record a fresh command buffer on every iteration. Reusing
    * old pixels or a fixed native descriptor cannot satisfy the sentinel
    * check below after the application has erased the previous result.
    */
next_render_frame:;
   VkCommandBufferAllocateInfo command_allocate = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer command_buffers[COMMAND_BUFFER_COUNT];
   for (unsigned command = 0; command < COMMAND_BUFFER_COUNT; ++command) {
      command_allocate.commandPool = command_pools[command];
      check(vkAllocateCommandBuffers(device, &command_allocate,
                                     &command_buffers[command]),
            "vkAllocateCommandBuffers");
   }
   VkCommandBuffer command_buffer = command_buffers[0];
   VkCommandBuffer blit_command = VK_NULL_HANDLE;
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   check(vkBeginCommandBuffer(command_buffer, &begin_info),
         "vkBeginCommandBuffer");
   const VkClearValue clear = {.color = {{0.0f, 0.0f, 0.0f, 1.0f}}};
   const VkRenderPassBeginInfo render_begin = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = render_pass,
      .framebuffer = framebuffer,
      .renderArea = {.extent = {IMAGE_WIDTH, IMAGE_HEIGHT}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(command_buffer, &render_begin,
                        VK_SUBPASS_CONTENTS_INLINE);
   vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                     pipeline);
   const VkDeviceSize vertex_offset =
      alternate_geometry && (frame_index & 1) ? sizeof(vertices) : 0;
   vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffer,
                          &vertex_offset);
   vkCmdDraw(command_buffer, 3, 1, 0, 0);
   vkCmdEndRenderPass(command_buffer);



   check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");
   if (blit_test || readback_test) {
      command_allocate.commandPool = command_pools[0];
      check(vkAllocateCommandBuffers(device, &command_allocate, &blit_command),
            "vkAllocateCommandBuffers(blit)");
      check(vkBeginCommandBuffer(blit_command, &begin_info),
            "vkBeginCommandBuffer(blit)");
      const VkImageMemoryBarrier barriers[] = {
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
         },
         {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = blit_image,
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
         },
      };
      const VkMemoryBarrier host_write = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      };
      vkCmdPipelineBarrier(blit_command,
         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_HOST_BIT,
         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &host_write, 0, NULL,
         blit_test ? 2 : 1, barriers);
      const VkImageBlit region = {
         .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .srcOffsets = {{0,0,0}, {IMAGE_WIDTH,IMAGE_HEIGHT,1}},
         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
         .dstOffsets = {{0,0,0}, {IMAGE_WIDTH,IMAGE_HEIGHT,1}},
      };
      if (blit_test) {
         vkCmdBlitImage(blit_command, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        blit_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        1, &region, VK_FILTER_NEAREST);
      } else {
         const VkBufferImageCopy copy = {
            .bufferOffset = 16,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {IMAGE_WIDTH, IMAGE_HEIGHT, 1},
         };
         vkCmdCopyImageToBuffer(blit_command, image,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback_buffer, 1, &copy);
      }
      const VkMemoryBarrier host_read = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
         .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier(blit_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
         VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host_read, 0, NULL, 0, NULL);
      check(vkEndCommandBuffer(blit_command), "vkEndCommandBuffer(blit)");
      fprintf(stderr, "g17p-render-only: separate %s command recorded; no submission yet\n",
              blit_test ? "blit" : "readback");
   }
   if (getenv("AGX_RENDER_RECORD_ONLY")) {
      printf("Apple %#x: %ux%u Vulkan draw recorded; no submission\n",
             properties.deviceID, IMAGE_WIDTH, IMAGE_HEIGHT);
      return EXIT_SUCCESS;
   }
   fprintf(stderr, "g17p-render-only: command buffer recorded\n");


   const VkFenceCreateInfo fence_info = {
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
   };
   VkFence fence;
   check(vkCreateFence(device, &fence_info, NULL, &fence), "vkCreateFence");
   const VkSubmitInfo submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &command_buffer,
   };
   check(vkQueueSubmit(queue, 1, &submit, fence), "vkQueueSubmit");

   fprintf(stderr, "g17p-render-only: submitted\n");

   /* Five seconds: long enough for any real 3D kick, short enough that a
    * launch which never signals is reported rather than hanging the box. */
   const VkResult waited =
      vkWaitForFences(device, 1, &fence, true,
                      UINT64_C(5000000000));
   if (waited == VK_TIMEOUT) {
      fprintf(stderr, "g17p-render-only: TIMEOUT - fence never signalled\n");
      return EXIT_FAILURE;
   }
   if (waited != VK_SUCCESS) {
      fprintf(stderr, "g17p-render-only: vkWaitForFences returned %d\n",
              waited);
      return EXIT_FAILURE;
   }






   /* Verify PIXELS, not just the fence. The clear is opaque black and the
    * vertex data is a full-viewport triangle, so a correct render leaves NO
    * cleared pixel behind. */
   if (!image_host_visible) {
      printf("g17p-render-only: INCONCLUSIVE - fence signalled but the render "
             "target is not host-visible, so no pixel was verified\n");
      return EXIT_FAILURE;
   }
   uint8_t *pixels = NULL;
   check(vkMapMemory(device, image_memory, 0, IMAGE_BYTES, 0,
                     (void **)&pixels),
         "vkMapMemory(image)");
   /* Classify every pixel into exactly three buckets. The expected colour is a
    * shader immediate and the clear is opaque black, so "still the clear
    * value" (never stored) and "some third value" (stored wrong) are different
    * failures with different causes, and a bare "is it non-black" test reports
    * the second one as a pass. */
   static const uint8_t cleared_pixel[4] = {0x00, 0x00, 0x00, 0xff};
   uint8_t expected_pixel[4] = {0xff, 0x47, 0x0d, 0xff};
   if (alternate_geometry && (frame_index & 1))
      memcpy(expected_pixel, cleared_pixel, sizeof(expected_pixel));

   char *classes = malloc(IMAGE_WIDTH * IMAGE_HEIGHT);
   if (!classes)
      die("out of memory classifying the render target");

   unsigned exact = 0, still_cleared = 0, wrong = 0;
   uint32_t first = 0;
   unsigned first_bad = UINT32_MAX;
   uint32_t first_bad_value = 0;
   for (unsigned i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; i++) {
      const uint8_t *p = &pixels[i * 4];
      const uint32_t value = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                             ((uint32_t)p[2] << 8) | p[3];
      if (i == 0)
         first = value;

      if (memcmp(p, expected_pixel, sizeof(expected_pixel)) == 0) {
         classes[i] = '#';
         exact++;
         continue;
      }

      if (memcmp(p, cleared_pixel, sizeof(cleared_pixel)) == 0) {
         classes[i] = '.';
         still_cleared++;
      } else {
         classes[i] = '?';
         wrong++;
      }

      if (first_bad == UINT32_MAX) {
         first_bad = i;
         first_bad_value = value;
      }
   }

   const uint32_t expected_value = ((uint32_t)expected_pixel[0] << 24) |
                                   ((uint32_t)expected_pixel[1] << 16) |
                                   ((uint32_t)expected_pixel[2] << 8) |
                                   expected_pixel[3];
   fprintf(stderr,
           "g17p-render-only: pixels exact=%u/%u cleared=%u wrong=%u "
           "first=0x%08x expected=0x%08x\n",
           exact, IMAGE_WIDTH * IMAGE_HEIGHT, still_cleared, wrong, first,
           expected_value);
   if (first_bad != UINT32_MAX)
      fprintf(stderr,
              "g17p-render-only: first non-exact pixel (%u,%u)=0x%08x\n",
              first_bad % IMAGE_WIDTH, first_bad / IMAGE_WIDTH,
              first_bad_value);

   /* Report the granularity of any loss. A fragment SIMD group covers an 8x4
    * pixel footprint and a derivative quad covers 2x2, so counting how many of
    * each are uniform decides, from a single run, whether pixels are lost per
    * pixel, per quad, or per SIMD group. The previous map sampled one row in
    * four and one column in two, which cannot tell those apart. */
   struct {
      unsigned w, h;
      const char *name;
   } shapes[2] = {{2, 2, "quads(2x2)"}, {8, 4, "simd-groups(8x4)"}};
   for (unsigned s = 0; s < 2; ++s) {
      unsigned total = 0, mixed = 0;
      for (unsigned by = 0; by < IMAGE_HEIGHT; by += shapes[s].h) {
         for (unsigned bx = 0; bx < IMAGE_WIDTH; bx += shapes[s].w) {
            const char first_class = classes[by * IMAGE_WIDTH + bx];
            bool uniform = true;
            for (unsigned y = by; y < by + shapes[s].h && y < IMAGE_HEIGHT; ++y)
               for (unsigned x = bx; x < bx + shapes[s].w && x < IMAGE_WIDTH;
                    ++x)
                  uniform &= classes[y * IMAGE_WIDTH + x] == first_class;
            total++;
            mixed += !uniform;
         }
      }
      fprintf(stderr, "g17p-render-only: %s mixed=%u/%u\n", shapes[s].name,
              mixed, total);
   }

   /* Every row, every column: the map is the evidence, so it must not
    * subsample. '#' is exact, '.' is the untouched clear, '?' is a third
    * value. */
   if (IMAGE_WIDTH <= 128 && IMAGE_HEIGHT <= 128) {
      char *row = malloc(IMAGE_WIDTH + 1);
      if (!row)
         die("out of memory printing the render target");
      row[IMAGE_WIDTH] = 0;
      for (unsigned y = 0; y < IMAGE_HEIGHT; y++) {
         memcpy(row, classes + (size_t)y * IMAGE_WIDTH, IMAGE_WIDTH);
         fprintf(stderr, "g17p-render-only: %3u |%s|\n", y, row);
      }
      free(row);
   }
   free(classes);

   fprintf(stderr, "g17p-render-only: frame=%u/%u pixels_exact=%u/%u\n",
           frame_index + 1, frame_count, exact, IMAGE_WIDTH * IMAGE_HEIGHT);
   if ((blit_test || readback_test) && exact == IMAGE_WIDTH * IMAGE_HEIGHT) {
      VkFence blit_fence;
      check(vkCreateFence(device, &fence_info, NULL, &blit_fence),
            "vkCreateFence(blit)");
      const VkSubmitInfo copy_submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &blit_command,
      };
      check(vkQueueSubmit(queue, 1, &copy_submit, blit_fence), "vkQueueSubmit(blit)");
      check(vkWaitForFences(device, 1, &blit_fence, true, UINT64_C(5000000000)),
            "vkWaitForFences(blit)");
      unsigned copied = 0, sentinel = 0;
      for (unsigned i = 0; i < IMAGE_WIDTH * IMAGE_HEIGHT; ++i) {
         copied += !memcmp(blit_pixels + 4*i, expected_pixel, 4);
         sentinel += !memcmp(blit_pixels + 4*i, "\xa5\xa5\xa5\xa5", 4);
      }
      fprintf(stderr, "g17p-render-only: %s pixels exact=%u/%u sentinel=%u first=%02x%02x%02x%02x\n",
         blit_test ? "blit" : "readback", copied, IMAGE_WIDTH * IMAGE_HEIGHT, sentinel,
         blit_pixels[0], blit_pixels[1], blit_pixels[2], blit_pixels[3]);
      if (readback_test) {
         for (unsigned i = 0; i < 16; i++) {
            if (readback_map[i] != 0xa5 || readback_map[16 + IMAGE_BYTES + i] != 0xa5)
               die("readback changed guard bytes outside its destination");
         }
         fprintf(stderr, "g17p-render-only: readback guard bytes intact\n");
      }
      /* Retain and optionally export the actual destination for inspection. */
      pixels = blit_pixels;
      exact = copied;
   }

   const char *output_path = getenv("AGX_RENDER_OUTPUT");
   if (output_path) {
      FILE *output = fopen(output_path, "wb");
      if (!output)
         die("cannot open render output");
      const size_t written = fwrite(pixels, 1, IMAGE_BYTES, output);
      const int closed = fclose(output);
      if (written != IMAGE_BYTES || closed != 0)
         die("cannot write complete render output");
   }
   if (getenv("AGX_RENDER_HOLD") && frame_index + 1 == frame_count) {
      fprintf(stderr, "g17p-render-only: HOLD pid=%ld pixels_exact=%u/%u; "
                      "resources retained for inspection\n",
              (long)getpid(), exact, IMAGE_WIDTH * IMAGE_HEIGHT);
      fflush(stderr);
      for (;;)
         sleep(60);
   }
   if (exact != IMAGE_WIDTH * IMAGE_HEIGHT)
      die("render target differs from the exact expected colour");
   if (memcmp(vertex_map, vertices, sizeof(vertices)) != 0)
      die("vertex source changed during query render");

   if (++frame_index < frame_count) {
      memset(pixels, 0xa5, IMAGE_BYTES);
      vkUnmapMemory(device, image_memory);
      goto next_render_frame;
   }


   printf("g17p-render-only: PASS - %ux%u RGBA8, %u/%u exact pixels%s\n",
          IMAGE_WIDTH, IMAGE_HEIGHT, exact, IMAGE_WIDTH * IMAGE_HEIGHT,
          ""
   );
   return EXIT_SUCCESS;
}
