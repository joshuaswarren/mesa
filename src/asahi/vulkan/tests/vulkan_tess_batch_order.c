/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Ordering of the software tessellation stages that G17P batches across the
 * draws of a render (hk_tess_batch.h): each stage's launches run back to back
 * with no barrier between them, and one cache flush follows the stage. That
 * flush must wait for every launch of the stage, not only the last one.
 *
 * Each of the 256 draws of a render pass is one tessellated triangle. Its
 * vertex stage spins for a while in one draw and writes vs_out[draw]; its
 * control stage reads what the vertex stage wrote for all draws up to its own
 * (or, with prev, only for the draw before it), spins in another draw, writes
 * the result to tcs_out[draw] and a patch output, and takes its tessellation
 * level from it; the evaluation stage writes the patch output to
 * tes_out[draw] and the finest denominator on the patch's outer edges, which
 * is the level the count, prefix sum and tessellation stages used, to
 * tes_level[draw]. A flush that let a slow launch run on would leave stale
 * values (the buffer is cleared before every pass) or a wrong level behind.
 *
 * The shaders' storage writes and reads across draws have no API barrier
 * between them: the expected values are what Honeykrisp's in-order stages
 * give, batched or not, and are not a Vulkan guarantee.
 *
 * Usage: vulkan-tess-batch-order [passes] [--shim [--indirect]]
 *   --shim      no-op DRM shim: record and submit one pass, skip the checks.
 *   --indirect  with --shim: record the pass as one vkCmdDrawIndirect, which
 *               is not batched and takes the indirect tessellation launches.
 * Exit status 0 = correct output.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_tess_batch_order_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s: %d\n", #call, r);                                   \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

#define DRAWS 256
/* About a millisecond of one GPU thread: much longer than issuing the rest
 * of the stage's launches. */
#define SPIN 400000

struct pc {
  uint32_t draw, iter, spin_vs, spin_tcs, prev;
};

struct out {
  uint32_t vs[DRAWS], tcs[DRAWS], tes[DRAWS], level[DRAWS];
};

static VkDevice dev;
static VkPhysicalDevice pdev;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties m;
  vkGetPhysicalDeviceMemoryProperties(pdev, &m);
  for (uint32_t i = 0; i < m.memoryTypeCount; i++)
    if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  fprintf(stderr, "no memory type\n");
  exit(2);
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

static uint32_t lcg(uint32_t x, uint32_t rounds) {
  for (uint32_t i = 0; i < rounds; i++)
    x = x * 1664525u + 1013904223u;
  return x;
}

/* The pass's draws: which one's vertex and which one's control stage spin. */
static struct pc params(unsigned d, unsigned iter, bool prev, unsigned slow_vs,
                        unsigned slow_tcs) {
  return (struct pc){d, iter, d == slow_vs ? SPIN : 0,
                     d == slow_tcs ? SPIN : 0, prev};
}

/* The shaders' arithmetic on the CPU; returns the number of wrong draws. */
static unsigned check(const struct out *o, unsigned iter, bool prev,
                      unsigned slow_vs, unsigned slow_tcs) {
  uint32_t vs[DRAWS];
  unsigned bad = 0;
  for (unsigned d = 0; d < DRAWS; d++) {
    const struct pc p = params(d, iter, prev, slow_vs, slow_tcs);
    vs[d] = lcg(d * 0x9e3779b9u + iter, p.spin_vs) | 1u;
    uint32_t x = 0;
    if (prev) {
      x = vs[d ? d - 1 : 0];
    } else {
      for (unsigned j = 0; j <= d; j++)
        x = x * 31u + vs[j];
    }
    x ^= lcg(0x12345u ^ d, p.spin_tcs);
    const uint32_t level = 1 + (x & 3);
    if (o->vs[d] != vs[d] || o->tcs[d] != x || o->tes[d] != x ||
        o->level[d] != level) {
      if (bad++ < 4)
        fprintf(stderr,
                "pass %u (%s, slow vs %u tcs %u) draw %u: vs %08x/%08x "
                "tcs %08x/%08x tes %08x level %u/%u\n",
                iter, prev ? "prev" : "all", slow_vs, slow_tcs, d, o->vs[d],
                vs[d], o->tcs[d], x, o->tes[d], o->level[d], level);
    }
  }
  return bad;
}

int main(int argc, char **argv) {
  unsigned passes = 64;
  bool shim = false, indirect = false;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--shim"))
      shim = true;
    else if (!strcmp(argv[i], "--indirect"))
      indirect = true;
    else
      passes = atoi(argv[i]);
  }
  if (indirect && !shim) {
    fprintf(stderr, "--indirect requires --shim\n");
    return 2;
  }
  if (shim)
    passes = 1;

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
  float prio = 1;
  VkPhysicalDeviceVulkan13Features f13 = {
     .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
     .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE};
  CHECK(vkCreateDevice(
     pdev,
     &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
           .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
           .queueCount = 1, .pQueuePriorities = &prio},
        .pEnabledFeatures = &(VkPhysicalDeviceFeatures){
           .tessellationShader = VK_TRUE,
           .vertexPipelineStoresAndAtomics = VK_TRUE}},
     NULL, &dev));
  VkQueue q;
  vkGetDeviceQueue(dev, 0, 0, &q);

  /* The storage buffer, host visible. */
  VkBuffer buf;
  CHECK(vkCreateBuffer(
     dev,
     &(VkBufferCreateInfo){
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(struct out),
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT},
     NULL, &buf));
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(dev, buf, &mr);
  VkDeviceMemory bmem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)},
     NULL, &bmem));
  CHECK(vkBindBufferMemory(dev, buf, bmem, 0));
  struct out *out;
  CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&out));

  /* The indirect draw command, host visible. */
  VkBuffer ibuf;
  CHECK(vkCreateBuffer(
     dev,
     &(VkBufferCreateInfo){
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = sizeof(VkDrawIndirectCommand),
        .usage = VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT},
     NULL, &ibuf));
  vkGetBufferMemoryRequirements(dev, ibuf, &mr);
  VkDeviceMemory indmem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)},
     NULL, &indmem));
  CHECK(vkBindBufferMemory(dev, ibuf, indmem, 0));
  VkDrawIndirectCommand *icmd;
  CHECK(vkMapMemory(dev, indmem, 0, VK_WHOLE_SIZE, 0, (void **)&icmd));
  *icmd = (VkDrawIndirectCommand){.vertexCount = 3, .instanceCount = 1};

  /* A small colour target: the work under test is before the render. */
  const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
  VkImage img;
  CHECK(vkCreateImage(
     dev,
     &(VkImageCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D, .format = fmt, .extent = {64, 64, 1},
        .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT},
     NULL, &img));
  vkGetImageMemoryRequirements(dev, img, &mr);
  VkDeviceMemory imem;
  CHECK(vkAllocateMemory(
     dev,
     &(VkMemoryAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mr.size,
        .memoryTypeIndex =
           memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)},
     NULL, &imem));
  CHECK(vkBindImageMemory(dev, img, imem, 0));
  VkImageView view;
  CHECK(vkCreateImageView(
     dev,
     &(VkImageViewCreateInfo){
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
     NULL, &view));

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(
     dev,
     &(VkDescriptorSetLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &(VkDescriptorSetLayoutBinding){
           0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
           VK_SHADER_STAGE_ALL_GRAPHICS}},
     NULL, &dsl));
  VkDescriptorPool dpool;
  CHECK(vkCreateDescriptorPool(
     dev,
     &(VkDescriptorPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &(VkDescriptorPoolSize){
           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}},
     NULL, &dpool));
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(
     dev,
     &(VkDescriptorSetAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl},
     &set));
  vkUpdateDescriptorSets(
     dev, 1,
     &(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .pBufferInfo = &(VkDescriptorBufferInfo){buf, 0, VK_WHOLE_SIZE}},
     0, NULL);

  VkPipelineLayout pl;
  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
           VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(struct pc)}},
     NULL, &pl));
  const VkPipelineShaderStageCreateInfo stages[] = {
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_VERTEX_BIT,
      .module = module(tess_order_vert, sizeof(tess_order_vert)),
      .pName = "main"},
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
      .module = module(tess_order_tesc, sizeof(tess_order_tesc)),
      .pName = "main"},
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT,
      .module = module(tess_order_tese, sizeof(tess_order_tese)),
      .pName = "main"},
     {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
      .module = module(tess_order_frag, sizeof(tess_order_frag)),
      .pName = "main"},
  };
  VkPipeline pipe;
  CHECK(vkCreateGraphicsPipelines(
     dev, VK_NULL_HANDLE, 1,
     &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &(VkPipelineRenderingCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
           .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt},
        .stageCount = 4, .pStages = stages,
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
           .topology = VK_PRIMITIVE_TOPOLOGY_PATCH_LIST},
        .pTessellationState = &(VkPipelineTessellationStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO,
           .patchControlPoints = 3},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
           .viewportCount = 1,
           .pViewports = &(VkViewport){0, 0, 64, 64, 0, 1},
           .scissorCount = 1, .pScissors = &(VkRect2D){{0, 0}, {64, 64}}},
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
        .layout = pl},
     NULL, &pipe));

  VkCommandPool pool;
  CHECK(vkCreateCommandPool(
     dev,
     &(VkCommandPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT},
     NULL, &pool));
  VkCommandBuffer cb;
  CHECK(vkAllocateCommandBuffers(
     dev,
     &(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1},
     &cb));
  VkFence fence;
  CHECK(vkCreateFence(
     dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
     NULL, &fence));

  unsigned bad = 0;
  for (unsigned iter = 0; iter < passes; iter++) {
    /* Alternate all/prev; move the slow launches around the stage, the
     * vertex one mostly early so that many launches follow it. */
    const bool prev = iter & 1;
    const uint32_t h = (iter + 1) * 2654435761u;
    const unsigned slow_vs = (h >> 8) % (DRAWS / 4);
    const unsigned slow_tcs = (h >> 16) % DRAWS;

    CHECK(vkBeginCommandBuffer(
       cb, &(VkCommandBufferBeginInfo){
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
    vkCmdFillBuffer(cb, buf, 0, VK_WHOLE_SIZE, 0);
    vkCmdPipelineBarrier2(
       cb, &(VkDependencyInfo){
              .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
              .memoryBarrierCount = 1,
              .pMemoryBarriers = &(VkMemoryBarrier2){
                 .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                 .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT |
                                  VK_ACCESS_2_MEMORY_WRITE_BIT},
              .imageMemoryBarrierCount = 1,
              .pImageMemoryBarriers = &(VkImageMemoryBarrier2){
                 .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                 .dstStageMask =
                    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                 .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 .image = img,
                 .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}});
    vkCmdBeginRendering(
       cb, &(VkRenderingInfo){
              .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
              .renderArea = {{0, 0}, {64, 64}}, .layerCount = 1,
              .colorAttachmentCount = 1,
              .pColorAttachments = &(VkRenderingAttachmentInfo){
                 .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                 .imageView = view,
                 .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
                 .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE}});
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pl, 0, 1,
                            &set, 0, NULL);
    for (unsigned d = 0; d < (indirect ? 1 : DRAWS); d++) {
      const struct pc p = params(d, iter, prev, slow_vs, slow_tcs);
      vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_ALL_GRAPHICS, 0, sizeof(p),
                         &p);
      if (indirect)
        vkCmdDrawIndirect(cb, ibuf, 0, 1, 0);
      else
        vkCmdDraw(cb, 3, 1, 0, 0);
    }
    vkCmdEndRendering(cb);
    vkCmdPipelineBarrier2(
       cb, &(VkDependencyInfo){
              .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
              .memoryBarrierCount = 1,
              .pMemoryBarriers = &(VkMemoryBarrier2){
                 .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                 .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                 .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                 .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT}});
    CHECK(vkEndCommandBuffer(cb));
    CHECK(vkQueueSubmit(q, 1,
                        &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &cb},
                        fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    CHECK(vkResetFences(dev, 1, &fence));
    if (!shim)
      bad += check(out, iter, prev, slow_vs, slow_tcs);
  }

  if (shim) {
    printf("1 pass of %u draws recorded (no readback)\n",
           indirect ? 1 : DRAWS);
    return 0;
  }
  if (bad) {
    printf("FAIL: %u wrong draws in %u passes\n", bad, passes);
    return 1;
  }
  printf("PASS: %u passes x %u draws\n", passes, DRAWS);
  return 0;
}
