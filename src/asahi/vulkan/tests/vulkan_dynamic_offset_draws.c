/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 *
 * 1024 small quads, one draw each, the way D3D translation layers draw: a
 * new dynamic uniform buffer offset per draw. Every new offset uploads a new
 * root and republishes the stage programs that follow it, so this measures
 * what per-draw rebinding costs against the same draws from one binding.
 *
 * Modes, each checked pixel by pixel against the image computed on the CPU:
 *   dyn    new vertex-stage dynamic offset per draw, fragment shader reads a
 *          plain uniform buffer.
 *   fixed  the same quads from one binding, by instance index: the cost of
 *          the draws without any root change.
 *   fsdyn  as dyn, but the fragment shader reads a dynamic uniform buffer
 *          too.
 *   push   as dyn, plus a fragment-visible push constant changed every draw:
 *          a fragment program kept across roots would draw wrong colours.
 *
 * On the M4 Pro (2026-11-05), dyn costs ~195 ns/draw of GPU time against
 * ~158 for fixed, and ~80 ns/draw more recording time. Keeping the fragment
 * program across roots it cannot read (M5's hk_m5_fragment_root()) changed
 * neither there: the HAL200 bind-tail memo already rebuilds only the fragment
 * record, and the comparison cost what the rebuild saved. The recording cost
 * is led by cleaning each write-back root to the point of coherency.
 *
 * Usage: vulkan-dynamic-offset-draws [rounds=15] [record-only mode]
 * Exit status 0 = all exact. With a mode name (dyn, fixed, fsdyn, push) it
 * only records that mode 50 * rounds times without submitting: the CPU cost
 * of the recording path.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#include "vulkan_dynamic_offset_draws_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

enum { W = 256, H = 256, GRID = 32, DRAWS = GRID * GRID, STRIDE = 64,
       PASSES = 8 };
enum mode { DYN, FIXED, FSDYN, PUSH, MODES };
static const char *const mode_name[MODES] = {"dyn", "fixed", "fsdyn", "push"};

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue q;
static VkCommandBuffer cb;
static VkFence fence;
static VkImage target;
static VkImageView target_view;
static VkBuffer staging;
static VkPipelineLayout layout;
static VkPipeline pipe[MODES];
static VkDescriptorSet set_one, set_all;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
}

static VkDeviceMemory bind_memory(VkMemoryRequirements mr,
                                  VkMemoryPropertyFlags flags) {
  VkDeviceMemory mem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                             .allocationSize = mr.size,
                             .memoryTypeIndex =
                                memory_type(mr.memoryTypeBits, flags)},
     NULL, &mem));
  return mem;
}

static VkBuffer host_buffer(VkDeviceSize size, VkBufferUsageFlags usage,
                            void **map) {
  VkBuffer buf;
  CHECK(vkCreateBuffer(dev,
                       &(VkBufferCreateInfo){
                          .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                          .size = size, .usage = usage},
                       NULL, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkDeviceMemory mem =
     bind_memory(mr, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  CHECK(vkBindBufferMemory(dev, buf, mem, 0));
  CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, map));
  return buf;
}

static VkShaderModule module(const uint32_t *code, size_t size) {
  VkShaderModule m;
  CHECK(vkCreateShaderModule(
     dev,
     &(VkShaderModuleCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = size, .pCode = code},
     NULL, &m));
  return m;
}

static void barrier(VkImageLayout from, VkImageLayout to) {
  vkCmdPipelineBarrier2(
     cb, &(VkDependencyInfo){
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &(VkImageMemoryBarrier2){
               .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
               .dstAccessMask =
                  VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
               .oldLayout = from, .newLayout = to, .image = target,
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}});
}

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return x < y ? -1 : x > y;
}

/* Record `passes` renders of the 1024 quads in mode m, submit and wait.
 * Returns the GPU wall time (submit to fence) in ms; *record_ms gets the
 * recording time. */
static double run(enum mode m, unsigned passes, bool readback,
                  double *record_ms) {
  const double r0 = now_ms();
  CHECK(vkBeginCommandBuffer(
     cb, &(VkCommandBufferBeginInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
  barrier(VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
  for (unsigned p = 0; p < passes; p++) {
    vkCmdBeginRendering(
       cb, &(VkRenderingInfo){
              .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
              .renderArea = {{0, 0}, {W, H}}, .layerCount = 1,
              .colorAttachmentCount = 1,
              .pColorAttachments = &(VkRenderingAttachmentInfo){
                 .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                 .imageView = target_view,
                 .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                 .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe[m]);
    vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
    vkCmdSetScissor(cb, 0, 1, &(VkRect2D){{0, 0}, {W, H}});
    if (m == FIXED) {
      const uint32_t offsets[2] = {0, 0};
      vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0,
                              1, &set_all, 2, offsets);
      for (uint32_t i = 0; i < DRAWS; i++)
        vkCmdDraw(cb, 4, 1, 0, i);
    } else {
      for (uint32_t i = 0; i < DRAWS; i++) {
        const uint32_t offsets[2] = {i * STRIDE, 0};
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                                0, 1, &set_one, 2, offsets);
        if (m == PUSH) {
          const uint32_t x = i & 7;
          vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4,
                             &x);
        }
        vkCmdDraw(cb, 4, 1, 0, 0);
      }
    }
    vkCmdEndRendering(cb);
  }
  if (readback) {
    barrier(VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(
       cb, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging, 1,
       &(VkBufferImageCopy){.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                                                 0, 1},
                            .imageExtent = {W, H, 1}});
  }
  CHECK(vkEndCommandBuffer(cb));
  const double t0 = now_ms();
  *record_ms = t0 - r0;
  CHECK(vkQueueSubmit(q, 1,
                      &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                      .commandBufferCount = 1,
                                      .pCommandBuffers = &cb},
                      fence));
  CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
  const double t1 = now_ms();
  CHECK(vkResetFences(dev, 1, &fence));
  CHECK(vkResetCommandBuffer(cb, 0));
  return t1 - t0;
}

static uint32_t colour(unsigned k) {
  return ((k * 37u) & 255) | ((k * 91u) & 255) << 8 | ((k * 53u) & 255) << 16 |
         0xffu << 24;
}

int main(int argc, char **argv) {
  unsigned rounds = argc > 1 ? strtoul(argv[1], NULL, 10) : 15;
  if (!rounds)
    rounds = 15;
  VkInstance inst;
  CHECK(vkCreateInstance(
     &(VkInstanceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &(VkApplicationInfo){
           .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
           .apiVersion = VK_API_VERSION_1_3}},
     NULL, &inst));
  uint32_t n = 1;
  CHECK(vkEnumeratePhysicalDevices(inst, &n, &pdev));
  if (!n) {
    fprintf(stderr, "no device\n");
    return 2;
  }
  VkPhysicalDeviceProperties prop;
  vkGetPhysicalDeviceProperties(pdev, &prop);
  if (prop.limits.minUniformBufferOffsetAlignment > STRIDE ||
      prop.limits.maxUniformBufferRange < DRAWS * STRIDE) {
    fprintf(stderr, "limits: offset alignment %u, range %u\n",
            (unsigned)prop.limits.minUniformBufferOffsetAlignment,
            prop.limits.maxUniformBufferRange);
    return 2;
  }
  float prio = 1;
  CHECK(vkCreateDevice(
     pdev,
     &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &(VkPhysicalDeviceVulkan13Features){
           .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
           .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE},
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
           .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
           .queueCount = 1, .pQueuePriorities = &prio}},
     NULL, &dev));
  vkGetDeviceQueue(dev, 0, 0, &q);

  CHECK(vkCreateImage(
     dev,
     &(VkImageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                          .imageType = VK_IMAGE_TYPE_2D,
                          .format = VK_FORMAT_R8G8B8A8_UNORM,
                          .extent = {W, H, 1}, .mipLevels = 1,
                          .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                          .tiling = VK_IMAGE_TILING_OPTIMAL,
                          .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT},
     NULL, &target));
  VkMemoryRequirements mr;
  vkGetImageMemoryRequirements(dev, target, &mr);
  CHECK(vkBindImageMemory(
     dev, target, bind_memory(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
  CHECK(vkCreateImageView(
     dev,
     &(VkImageViewCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = target,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
     NULL, &target_view));
  uint32_t *pixels;
  staging = host_buffer((VkDeviceSize)W * H * 4,
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT, (void **)&pixels);

  /* Quad i covers cell (i % GRID, i / GRID), 8x8 pixels on pixel edges, in
   * colour (i * 7) % 64. */
  struct rec {
    float rect[4];
    uint32_t ci[4];
    float pad[8];
  } *recs;
  _Static_assert(sizeof(struct rec) == STRIDE, "std140 record stride");
  VkBuffer rec_buf = host_buffer(DRAWS * STRIDE,
                                 VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                 (void **)&recs);
  for (uint32_t i = 0; i < DRAWS; i++)
    recs[i] = (struct rec){{(float)(i % GRID) / GRID, (float)(i / GRID) / GRID,
                            1.0f / GRID, 1.0f / GRID},
                           {(i * 7) % 64}};
  float(*colours)[4];
  VkBuffer colour_buf = host_buffer(64 * 16, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    (void **)&colours);
  for (unsigned k = 0; k < 64; k++) {
    const uint32_t c = colour(k);
    for (unsigned j = 0; j < 4; j++)
      colours[k][j] = ((c >> (8 * j)) & 255) / 255.0f;
  }

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(
     dev,
     &(VkDescriptorSetLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = (VkDescriptorSetLayoutBinding[]){
           {0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
            VK_SHADER_STAGE_VERTEX_BIT},
           {1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT},
           {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1,
            VK_SHADER_STAGE_FRAGMENT_BIT}}},
     NULL, &dsl));
  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(
     dev,
     &(VkDescriptorPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 2,
        .poolSizeCount = 2,
        .pPoolSizes = (VkDescriptorPoolSize[]){
           {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4},
           {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2}}},
     NULL, &pool));
  VkDescriptorSet sets[2];
  CHECK(vkAllocateDescriptorSets(
     dev,
     &(VkDescriptorSetAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 2,
        .pSetLayouts = (VkDescriptorSetLayout[]){dsl, dsl}},
     sets));
  set_one = sets[0];
  set_all = sets[1];
  for (unsigned s = 0; s < 2; s++) {
    const VkDeviceSize range = s ? DRAWS * STRIDE : STRIDE;
    vkUpdateDescriptorSets(
       dev, 3,
       (VkWriteDescriptorSet[]){
          {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[s],
           .dstBinding = 0, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
           .pBufferInfo = &(VkDescriptorBufferInfo){rec_buf, 0, range}},
          {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[s],
           .dstBinding = 1, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
           .pBufferInfo = &(VkDescriptorBufferInfo){colour_buf, 0,
                                                    VK_WHOLE_SIZE}},
          {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[s],
           .dstBinding = 2, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
           .pBufferInfo = &(VkDescriptorBufferInfo){colour_buf, 0, 64 * 16}}},
       0, NULL);
  }
  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
           VK_SHADER_STAGE_FRAGMENT_BIT, 0, 4}},
     NULL, &layout));

  const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
  const VkShaderModule rec_vs = module(dod_rec_vert, sizeof(dod_rec_vert));
  const VkShaderModule inst_vs = module(dod_inst_vert, sizeof(dod_inst_vert));
  const VkShaderModule static_fs =
     module(dod_static_frag, sizeof(dod_static_frag));
  const VkShaderModule stages[MODES][2] = {
     [DYN] = {rec_vs, static_fs},
     [FIXED] = {inst_vs, static_fs},
     [FSDYN] = {rec_vs, module(dod_dyn_frag, sizeof(dod_dyn_frag))},
     [PUSH] = {rec_vs, module(dod_push_frag, sizeof(dod_push_frag))},
  };
  const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                VK_DYNAMIC_STATE_SCISSOR};
  for (unsigned m = 0; m < MODES; m++) {
    CHECK(vkCreateGraphicsPipelines(
       dev, VK_NULL_HANDLE, 1,
       &(VkGraphicsPipelineCreateInfo){
          .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
          .pNext = &(VkPipelineRenderingCreateInfo){
             .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
             .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt},
          .stageCount = 2,
          .pStages = (VkPipelineShaderStageCreateInfo[]){
             {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = stages[m][0],
              .pName = "main"},
             {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
              .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = stages[m][1],
              .pName = "main"}},
          .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
             .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
          .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
             .sType =
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
             .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP},
          .pViewportState = &(VkPipelineViewportStateCreateInfo){
             .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
             .viewportCount = 1, .scissorCount = 1},
          .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
             .sType =
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
             .lineWidth = 1},
          .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
             .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
             .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
          .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
             .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
             .attachmentCount = 1,
             .pAttachments = &(VkPipelineColorBlendAttachmentState){
                .colorWriteMask = 0xf}},
          .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
             .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
             .dynamicStateCount = 2, .pDynamicStates = dyn},
          .layout = layout},
       NULL, &pipe[m]));
  }

  VkCommandPool cmd_pool;
  CHECK(vkCreateCommandPool(
     dev,
     &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT},
     NULL, &cmd_pool));
  CHECK(vkAllocateCommandBuffers(
     dev,
     &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1},
     &cb));
  CHECK(vkCreateFence(
     dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
     NULL, &fence));

  if (argc > 2) {
    for (unsigned m = 0; m < MODES; m++) {
      if (strcmp(argv[2], mode_name[m]))
        continue;
      for (unsigned r = 0; r < 50 * rounds; r++) {
        CHECK(vkBeginCommandBuffer(
           cb, &(VkCommandBufferBeginInfo){
                  .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
        vkCmdBeginRendering(
           cb, &(VkRenderingInfo){
                  .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                  .renderArea = {{0, 0}, {W, H}}, .layerCount = 1,
                  .colorAttachmentCount = 1,
                  .pColorAttachments = &(VkRenderingAttachmentInfo){
                     .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                     .imageView = target_view,
                     .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                     .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe[m]);
        vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
        vkCmdSetScissor(cb, 0, 1, &(VkRect2D){{0, 0}, {W, H}});
        for (uint32_t i = 0; i < DRAWS; i++) {
          const uint32_t offsets[2] = {m == FIXED ? 0 : i * STRIDE, 0};
          vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                                  0, 1, m == FIXED ? &set_all : &set_one, 2,
                                  offsets);
          vkCmdDraw(cb, 4, 1, 0, m == FIXED ? i : 0);
        }
        vkCmdEndRendering(cb);
        CHECK(vkEndCommandBuffer(cb));
        CHECK(vkResetCommandBuffer(cb, 0));
      }
      return 0;
    }
    fprintf(stderr, "unknown mode %s\n", argv[2]);
    return 2;
  }

  unsigned bad_total = 0;
  for (unsigned m = 0; m < MODES; m++) {
    double rec;
    memset(pixels, 0, (size_t)W * H * 4);
    run(m, 1, true, &rec);
    unsigned bad = 0;
    for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
        const unsigned i = (y / (H / GRID)) * GRID + x / (W / GRID);
        const unsigned ci = (i * 7) % 64;
        const uint32_t want = colour(m == PUSH ? (ci ^ (i & 7)) & 63 : ci);
        if (pixels[y * W + x] != want && bad++ < 4)
          printf("%s: pixel %u,%u = %08x, want %08x\n", mode_name[m], x, y,
                 pixels[y * W + x], want);
      }
    }
    if (bad)
      printf("%s: FAIL, %u pixels wrong\n", mode_name[m], bad);
    bad_total += bad;
  }

  /* Interleave the modes so that all see the same clocks. */
  double *gpu = calloc((size_t)MODES * rounds, sizeof(double));
  double *cpu = calloc((size_t)MODES * rounds, sizeof(double));
  for (unsigned r = 0; r < rounds; r++)
    for (unsigned m = 0; m < MODES; m++)
      gpu[m * rounds + r] =
         run(m, PASSES, false, &cpu[m * rounds + r]) / PASSES;
  printf("%u draws/pass, median of %u rounds of %u passes:\n", DRAWS, rounds,
         PASSES);
  for (unsigned m = 0; m < MODES; m++) {
    qsort(gpu + m * rounds, rounds, sizeof(double), compare_double);
    qsort(cpu + m * rounds, rounds, sizeof(double), compare_double);
    printf("  %-5s gpu %7.1f us/pass (%5.0f ns/draw)  record %7.1f us/pass\n",
           mode_name[m], gpu[m * rounds + rounds / 2] * 1e3,
           gpu[m * rounds + rounds / 2] * 1e6 / DRAWS,
           cpu[m * rounds + rounds / 2] * 1e3 / PASSES);
  }

  if (bad_total)
    return 1;
  printf("PASS\n");
  return 0;
}
