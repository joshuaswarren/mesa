/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A program that creates its pipelines before its first image view with a
 * minLod above its base level (VK_EXT_image_view_min_lod), as games that build
 * pipelines on worker threads during loading do. Its shaders are compiled
 * without the view minLod emulation and get an emulating twin built at the
 * first draw that binds them. The driver must remember that the program needed
 * the emulation, so that its next run compiles the shaders with it from the
 * start and draws without building twins.
 *
 * The test starts from an empty cache directory and runs the same sequence
 * from two Vulkan instances in turn: create the pipeline, then a three-level
 * texture whose levels are cleared to different colours and a view of all
 * three levels with minLod 1.0, then record a draw that samples the texture,
 * counting the shaders compiled while recording (vulkan_compile_count.h) and
 * reading back the image. The draw magnifies the texture, so without the
 * view's minLod it would read level 0. Both runs must read level 1 only; the
 * first recording must build a twin, and the second must compile nothing.
 *
 * The extension is only exposed with the hk_image_view_min_lod option, which
 * the test sets for itself.
 *
 * Usage: vulkan-min-lod-emulation-cache   Exit status 0 = pass, 77 = skip.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "vulkan_border_emulation_cache_shaders.h"
#include "vulkan_compile_count.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define W 256
#define H 256
#define LEVELS 3

static const float level_colour[LEVELS][3] = {
  {0.2f, 0.6f, 0.3f},
  {0.9f, 0.1f, 0.8f},
  {0.1f, 0.3f, 0.9f},
};

static VkPhysicalDevice pdev;
static VkDevice dev;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

static VkDeviceMemory bind_image(VkImage img) {
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, img, &mr);
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = memory_type(mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
      NULL, &mem));
  CHECK(vkBindImageMemory(dev, img, mem, 0));
  return mem;
}

static VkImageView view_of(VkImage img, uint32_t levels, const void *pnext) {
  VkImageView view;
  CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .pNext = pnext,
      .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1}}, NULL,
      &view));
  return view;
}

static VkShaderModule module(const uint32_t *code, size_t size) {
  VkShaderModule m;
  CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size,
      .pCode = code}, NULL, &m));
  return m;
}

static void barrier(VkCommandBuffer cb, VkImage img, uint32_t levels,
                    VkImageLayout from, VkImageLayout to,
                    VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
                    VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access) {
  vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &(VkImageMemoryBarrier2){
          .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
          .srcStageMask = src, .srcAccessMask = src_access,
          .dstStageMask = dst, .dstAccessMask = dst_access,
          .oldLayout = from, .newLayout = to, .image = img,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels, 0, 1}}});
}

static bool has_extension(const char *name) {
  uint32_t n = 0;
  vkEnumerateDeviceExtensionProperties(pdev, NULL, &n, NULL);
  VkExtensionProperties *props = calloc(n, sizeof(*props));
  if (!props)
    exit(2);
  vkEnumerateDeviceExtensionProperties(pdev, NULL, &n, props);
  bool found = false;
  for (uint32_t i = 0; i < n; i++)
    found |= !strcmp(props[i].extensionName, name);
  free(props);
  return found;
}

/* One instance: pipeline first, then the minLod view, then the draw. Returns
 * false if the device does not expose the extension; otherwise the recording
 * time, the number of shaders compiled while recording, a hash of the image
 * and the centre and corner pixels. */
static bool run(double *record_ms, unsigned *compiles, uint64_t *hash,
                uint32_t *centre, uint32_t *corner) {
  VkInstance inst;
  CHECK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
          .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
          .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
  uint32_t n = 1;
  vkEnumeratePhysicalDevices(inst, &n, &pdev);
  if (!has_extension(VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME)) {
    vkDestroyInstance(inst, NULL);
    return false;
  }
  CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &(VkPhysicalDeviceVulkan13Features){
          .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
          .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE,
          .pNext = &(VkPhysicalDeviceImageViewMinLodFeaturesEXT){
              .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_VIEW_MIN_LOD_FEATURES_EXT,
              .minLod = VK_TRUE}},
      .enabledExtensionCount = 1,
      .ppEnabledExtensionNames = (const char *[]){
          VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME},
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
          .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
          .pQueuePriorities = (float[]){1}}}, NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, 0, 0, &q);

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){
          0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
          VK_SHADER_STAGE_FRAGMENT_BIT, NULL}}, NULL, &dsl));
  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl}, NULL, &pl));
  VkShaderModule vs = module(border_emulation_cache_vert,
                             sizeof(border_emulation_cache_vert));
  VkShaderModule fs = module(border_emulation_cache_frag,
                             sizeof(border_emulation_cache_frag));
  const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
  VkPipeline pipe;
  CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineRenderingCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
          .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt},
      .stageCount = 2,
      .pStages = (VkPipelineShaderStageCreateInfo[]){
          {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
           VK_SHADER_STAGE_VERTEX_BIT, vs, "main", NULL},
          {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
           VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL}},
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
          .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
          .viewportCount = 1, .pViewports = &(VkViewport){0, 0, W, H, 0, 1},
          .scissorCount = 1, .pScissors = &(VkRect2D){{0, 0}, {W, H}}},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
          .lineWidth = 1},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
          .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
          .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
          .attachmentCount = 1,
          .pAttachments = &(VkPipelineColorBlendAttachmentState){
              .colorWriteMask = 0xf}},
      .layout = pl}, NULL, &pipe));

  VkSampler sampler;
  CHECK(vkCreateSampler(dev, &(VkSamplerCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST, .minFilter = VK_FILTER_NEAREST,
      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      .maxLod = VK_LOD_CLAMP_NONE}, NULL, &sampler));

  VkImage tex, target;
  CHECK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt, .extent = {4, 4, 1}, .mipLevels = LEVELS,
      .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT},
      NULL, &tex));
  CHECK(vkCreateImage(dev, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt, .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT}, NULL, &target));
  VkDeviceMemory tex_mem = bind_image(tex), target_mem = bind_image(target);
  /* Only now the program's first view with a minLod above its base level. */
  VkImageView tex_view = view_of(tex, LEVELS, &(VkImageViewMinLodCreateInfoEXT){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_MIN_LOD_CREATE_INFO_EXT,
      .minLod = 1.0f});
  VkImageView target_view = view_of(target, 1, NULL);

  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){
          VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}, NULL, &pool));
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl},
      &set));
  vkUpdateDescriptorSets(dev, 1, &(VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &(VkDescriptorImageInfo){
          sampler, tex_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
      0, NULL);

  VkBuffer rb;
  CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = W * H * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT}, NULL, &rb));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, rb, &mr);
  VkDeviceMemory rmem;
  CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
      .memoryTypeIndex = memory_type(mr.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)},
      NULL, &rmem));
  CHECK(vkBindBufferMemory(dev, rb, rmem, 0));

  VkCommandPool cp;
  CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &cp));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cp,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cb));

  struct compile_count count;
  struct timespec t0, t1;
  compile_count_begin(&count);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
  barrier(cb, tex, LEVELS, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
          VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
  for (uint32_t l = 0; l < LEVELS; l++) {
    vkCmdClearColorImage(cb, tex, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                         &(VkClearColorValue){.float32 = {
                             level_colour[l][0], level_colour[l][1],
                             level_colour[l][2], 1}}, 1,
                         &(VkImageSubresourceRange){
                             VK_IMAGE_ASPECT_COLOR_BIT, l, 1, 0, 1});
  }
  barrier(cb, tex, LEVELS, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_CLEAR_BIT,
          VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  barrier(cb, target, 1, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  vkCmdBeginRendering(cb, &(VkRenderingInfo){
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {W, H}},
      .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &(VkRenderingAttachmentInfo){
          .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
          .imageView = target_view,
          .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
  vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
  vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1, &set, 0,
                          NULL);
  vkCmdDraw(cb, 3, 1, 0, 0);
  vkCmdEndRendering(cb);
  barrier(cb, target, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
          VK_ACCESS_2_TRANSFER_READ_BIT);
  vkCmdCopyImageToBuffer(cb, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb, 1,
                         &(VkBufferImageCopy){
                             .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                             .imageExtent = {W, H, 1}});
  CHECK(vkEndCommandBuffer(cb));
  clock_gettime(CLOCK_MONOTONIC, &t1);
  *compiles = compile_count_end(&count);
  *record_ms = (t1.tv_sec - t0.tv_sec) * 1e3 + (t1.tv_nsec - t0.tv_nsec) / 1e6;

  VkFence fence;
  CHECK(vkCreateFence(dev, &(VkFenceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}, NULL, &fence));
  CHECK(vkQueueSubmit(q, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1,
      .pCommandBuffers = &cb}, fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));

  const uint32_t *px;
  CHECK(vkMapMemory(dev, rmem, 0, VK_WHOLE_SIZE, 0, (void **)&px));
  uint64_t h = 1469598103934665603ull;
  for (unsigned i = 0; i < W * H; i++)
    h = (h ^ px[i]) * 1099511628211ull;
  *hash = h;
  *centre = px[(H / 2) * W + W / 2];
  *corner = px[4 * W + 4];
  vkUnmapMemory(dev, rmem);

  vkDestroyFence(dev, fence, NULL);
  vkDestroyCommandPool(dev, cp, NULL);
  vkDestroyBuffer(dev, rb, NULL);
  vkFreeMemory(dev, rmem, NULL);
  vkDestroyDescriptorPool(dev, pool, NULL);
  vkDestroyImageView(dev, tex_view, NULL);
  vkDestroyImageView(dev, target_view, NULL);
  vkDestroyImage(dev, tex, NULL);
  vkDestroyImage(dev, target, NULL);
  vkFreeMemory(dev, tex_mem, NULL);
  vkFreeMemory(dev, target_mem, NULL);
  vkDestroySampler(dev, sampler, NULL);
  vkDestroyPipeline(dev, pipe, NULL);
  vkDestroyShaderModule(dev, vs, NULL);
  vkDestroyShaderModule(dev, fs, NULL);
  vkDestroyPipelineLayout(dev, pl, NULL);
  vkDestroyDescriptorSetLayout(dev, dsl, NULL);
  /* Destroying the device and instance flushes the disk cache writes. */
  vkDestroyDevice(dev, NULL);
  vkDestroyInstance(inst, NULL);
  return true;
}

/* RGBA8 little-endian: red in the low byte. */
static bool near(uint32_t px, const float c[3]) {
  for (int i = 0; i < 3; i++) {
    const int want = (int)(c[i] * 255.0f + 0.5f);
    const int got = (px >> (8 * i)) & 0xff;
    if (abs(got - want) > 2)
      return false;
  }
  return true;
}

int main(void) {
  char dir[] = "/tmp/hk-min-lod-emulation-cache-XXXXXX";
  if (!mkdtemp(dir)) {
    perror("mkdtemp");
    return 2;
  }
  setenv("MESA_SHADER_CACHE_DIR", dir, 1);
  unsetenv("MESA_SHADER_CACHE_DISABLE");
  setenv("hk_image_view_min_lod", "true", 1);
  compile_count_enable();

  double first, second;
  unsigned first_compiles, second_compiles;
  uint64_t h0, h1;
  uint32_t centre0, corner0, centre1, corner1;
  const bool supported = run(&first, &first_compiles, &h0, &centre0, &corner0);
  if (supported)
    run(&second, &second_compiles, &h1, &centre1, &corner1);

  char cmd[96];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", dir);
  if (system(cmd) != 0)
    fprintf(stderr, "could not remove %s\n", dir);

  if (!supported) {
    printf("SKIP: no %s\n", VK_EXT_IMAGE_VIEW_MIN_LOD_EXTENSION_NAME);
    return 77;
  }
  printf("recording with the emulating twin built: %u compiles, %.2f ms; "
         "emulating from the start: %u compiles, %.2f ms\n",
         first_compiles, first, second_compiles, second);

  const uint32_t px[4] = {centre0, corner0, centre1, corner1};
  for (unsigned i = 0; i < 4; i++) {
    if (!near(px[i], level_colour[1])) {
      fprintf(stderr, "FAIL: run %u %s pixel %08x is not level 1 (%s)\n",
              i / 2 + 1, i % 2 ? "corner" : "centre", px[i],
              near(px[i], level_colour[0]) ? "level 0: minLod ignored"
                                           : "neither level");
      return 1;
    }
  }
  if (h0 != h1) {
    fprintf(stderr, "FAIL: the second run renders differently\n");
    return 1;
  }
  if (first_compiles == 0) {
    fprintf(stderr, "FAIL: the first recording compiled nothing, so the draw "
            "no longer builds a twin and this test checks nothing\n");
    return 1;
  }
  if (second_compiles != 0) {
    fprintf(stderr, "FAIL: the second recording compiled %u shaders: it "
            "built an emulating twin again\n", second_compiles);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
