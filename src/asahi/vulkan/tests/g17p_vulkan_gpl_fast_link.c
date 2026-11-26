/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A graphics pipeline fast-linked from four pipeline libraries, as DXVK binds
 * one at the first draw while it compiles the optimized pipeline in the
 * background. The fragment library has no output interface, so the driver
 * links its fragment shader with the colour epilog at draw time. That link
 * must not compile: on G17P an inline-blend program is a full backend compile
 * of the fragment shader, and DXVK records on one thread, so every new
 * fragment shader stalled the frame (Call of Duty: Advanced Warfare spent
 * minutes of its first level load on them).
 *
 * The test records a blended draw with the fast-linked pipeline and with one
 * linked from the same libraries with link-time optimization, counting the
 * shaders compiled while recording (vulkan_compile_count.h). Both counts must
 * be zero: the fast link uses the split epilog, and the optimized pipeline
 * built its inline-blend program when it was created. Nothing is submitted.
 *
 * Usage: g17p-vulkan-gpl-fast-link   Exit status 0 = pass.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "g17p_vulkan_gpl_fast_link_shaders.h"
#include "vulkan_compile_count.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define W 64
#define H 64

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkPipelineLayout layout;
static const VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;

static uint32_t memory_type(uint32_t bits) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if (bits & (1u << i))
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

static VkShaderModule module(const uint32_t *code, size_t size) {
  VkShaderModule m;
  CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size,
      .pCode = code}, NULL, &m));
  return m;
}

/* One library holding the given parts of a pipeline's state. */
static VkPipeline library(VkGraphicsPipelineLibraryFlagsEXT parts,
                          VkGraphicsPipelineCreateInfo info) {
  VkGraphicsPipelineLibraryCreateInfoEXT lib = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
      .flags = parts};
  VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .pNext = &lib, .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &format};
  info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  info.pNext = &rendering;
  info.flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR |
               VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT;
  info.layout = layout;
  VkPipeline p;
  CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &info, NULL, &p));
  return p;
}

static VkPipeline link_libraries(const VkPipeline *libs, VkPipelineCreateFlags flags) {
  VkPipeline p;
  CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineLibraryCreateInfoKHR){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR,
          .libraryCount = 4, .pLibraries = libs},
      .flags = flags, .layout = layout}, NULL, &p));
  return p;
}

/* Records one blended draw with the pipeline and returns the number of
 * shaders compiled while recording. */
static unsigned record(VkCommandPool pool, VkImageView view, VkPipeline p) {
  VkCommandBuffer cmd;
  CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1}, &cmd));
  struct compile_count count;
  compile_count_begin(&count);
  CHECK(vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
  vkCmdBeginRendering(cmd, &(VkRenderingInfo){
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {W, H}}, .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = view,
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
  vkCmdSetViewport(cmd, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
  vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){{0, 0}, {W, H}});
  vkCmdDraw(cmd, 3, 1, 0, 0);
  vkCmdEndRendering(cmd);
  CHECK(vkEndCommandBuffer(cmd));
  unsigned n = compile_count_end(&count);
  vkFreeCommandBuffers(dev, pool, 1, &cmd);
  return n;
}

int main(void) {
  /* Before the first instance: the flag is part of the cache identity. */
  compile_count_enable();

  VkInstance inst;
  CHECK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
          .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
          .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
  uint32_t n = 1;
  vkEnumeratePhysicalDevices(inst, &n, &pdev);
  if (n == 0) {
    fprintf(stderr, "no Vulkan device\n");
    return 2;
  }
  CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &(VkPhysicalDeviceVulkan13Features){
          .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
          .dynamicRendering = VK_TRUE,
          .pNext = &(VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT){
              .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT,
              .graphicsPipelineLibrary = VK_TRUE}},
      .enabledExtensionCount = 2,
      .ppEnabledExtensionNames = (const char *[]){
          VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
          VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME},
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
          .pQueuePriorities = (float[]){1}}}, NULL, &dev));

  VkImage img;
  CHECK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = format, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT}, NULL, &img));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, img, &mr);
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = memory_type(mr.memoryTypeBits)}, NULL, &mem));
  CHECK(vkBindImageMemory(dev, img, mem, 0));
  VkImageView view;
  CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &view));

  CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}, NULL, &layout));
  VkShaderModule vs = module(gpl_fast_link_vert, sizeof(gpl_fast_link_vert));
  VkShaderModule fs = module(gpl_fast_link_frag, sizeof(gpl_fast_link_frag));
  const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};

  VkPipeline libs[4];
  libs[0] = library(VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT,
                    (VkGraphicsPipelineCreateInfo){
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
          .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST}});
  libs[1] = library(VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
                    (VkGraphicsPipelineCreateInfo){
      .stageCount = 1,
      .pStages = &(VkPipelineShaderStageCreateInfo){
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_VERTEX_BIT, vs, "main", NULL},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
          .viewportCount = 1, .scissorCount = 1},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
          .lineWidth = 1},
      .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
          .dynamicStateCount = 2,
          .pDynamicStates = (VkDynamicState[]){VK_DYNAMIC_STATE_VIEWPORT,
                                               VK_DYNAMIC_STATE_SCISSOR}}});
  libs[2] = library(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT,
                    (VkGraphicsPipelineCreateInfo){
      .stageCount = 1,
      .pStages = &(VkPipelineShaderStageCreateInfo){
          VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
          VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL},
      .pMultisampleState = &ms,
      .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO}});
  libs[3] = library(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
                    (VkGraphicsPipelineCreateInfo){
      .pMultisampleState = &ms,
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
          .attachmentCount = 1,
          .pAttachments = &(VkPipelineColorBlendAttachmentState){
              .blendEnable = VK_TRUE,
              .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
              .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
              .colorBlendOp = VK_BLEND_OP_ADD,
              .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
              .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
              .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf}}});

  VkPipeline fast = link_libraries(libs, 0);
  VkPipeline optimized =
      link_libraries(libs, VK_PIPELINE_CREATE_LINK_TIME_OPTIMIZATION_BIT_EXT);

  VkCommandPool pool;
  CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &pool));
  const unsigned fast_compiles = record(pool, view, fast);
  const unsigned optimized_compiles = record(pool, view, optimized);
  printf("shaders compiled while recording: fast link %u, optimized link %u\n",
         fast_compiles, optimized_compiles);

  vkDestroyCommandPool(dev, pool, NULL);
  vkDestroyPipeline(dev, fast, NULL);
  vkDestroyPipeline(dev, optimized, NULL);
  for (unsigned i = 0; i < 4; i++)
    vkDestroyPipeline(dev, libs[i], NULL);
  vkDestroyShaderModule(dev, vs, NULL);
  vkDestroyShaderModule(dev, fs, NULL);
  vkDestroyPipelineLayout(dev, layout, NULL);
  vkDestroyImageView(dev, view, NULL);
  vkDestroyImage(dev, img, NULL);
  vkFreeMemory(dev, mem, NULL);
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);

  if (fast_compiles || optimized_compiles) {
    fprintf(stderr, "FAIL: a draw compiled a shader while recording\n");
    return 1;
  }
  printf("PASS\n");
  return 0;
}
