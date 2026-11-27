/* Copyright 2026 Jonathan Goyvaerts
 * SPDX-License-Identifier: MIT
 *
 * Exact readback of every way a game writes a storage image that is also a
 * colour attachment and a transfer source/destination (the usage set of
 * post-processing targets), for each format a compressed storage image may
 * take. Per format, in one command buffer and with the layouts a game uses:
 *
 *   1. vkCmdClearColorImage                       (TRANSFER_DST)
 *   2. a scissored LOAD/STORE render pass, fill   (COLOR_ATTACHMENT)
 *   3. compute imageLoad/imageStore of scattered texels in a region
 *                                                 (GENERAL)
 *   4. a scissored additive blend pass, +1        (COLOR_ATTACHMENT; float
 *                                                  and normalized formats)
 *   5. vkCmdCopyBufferToImage of an unaligned region (TRANSFER_DST)
 *   6. compute imageLoad/imageStore of scattered texels everywhere
 *
 * then reads the image back three ways: compute imageLoad (GENERAL), compute
 * texelFetch (SHADER_READ_ONLY) and vkCmdCopyImageToBuffer (TRANSFER_SRC, the
 * raw texel bits). Every channel of every texel holds a small integer k in
 * [0, 64], stored as k scaled to the format (k/64 for float formats, k/255 for
 * UNORM8, and so on), so every step is exact and a CPU model of the same steps
 * predicts every result bit.
 *
 * Run it with and without the driver compressing the images (M5:
 * ASAHI_M5_STORAGE_COMPRESSION; AGX_MESA_DEBUG=perf shows which are).
 *
 * The M4 Pro cannot compress storage images this way (2026-11-05): with the
 * M5 selection admitted on G17P, every compressed format failed all three
 * readbacks, also with no fast clear (SC_NOCLEAR=1), while all 22 formats are
 * exact uncompressed. The storage PBE descriptor was byte-identical to the
 * end-of-tile one that renders the same image correctly: a scattered shader
 * imageStore through the G17P (Gen9) PBE does not merge into a compressed
 * block. A block whose texels were stored took the last stored value in all
 * of them, and stores into other blocks were lost or wrote garbage. Metal
 * agrees (macOS 26.6.2, same machine): it compresses RenderTarget|ShaderRead
 * textures with this same PBE layout, but never one with ShaderWrite usage.
 *
 * Usage: vulkan-storage-compression [width=300] [height=200] [format-filter]
 * Exit status 0 = every format exact, 2 = no format tested. SC_NOCLEAR=1
 * replaces the clear by a fill of the whole image; SC_DUMP=1 prints the
 * top-left texels of a failure (got, * = wrong, then expected).
 */
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_storage_compression_shaders.h"

#define CHECK(call)                                                            \
  do {                                                                         \
    VkResult r = (call);                                                       \
    if (r != VK_SUCCESS) {                                                     \
      fprintf(stderr, "%s:%d %s: %d\n", __FILE__, __LINE__, #call, r);         \
      exit(2);                                                                 \
    }                                                                          \
  } while (0)

enum enc { F16, F32, UN8, SN8, U8, I8, UN16, SN16, U16, I16, U32, I32 };
struct fmt {
  VkFormat vk;
  const char *name;
  unsigned channels;
  enum enc enc;
  char type; /* f, u, i: the shader's component type */
  float scale;
  bool bgra; /* channel 0 is stored in byte 2 and channel 2 in byte 0 */
};

static const struct fmt formats[] = {
   {VK_FORMAT_R16G16B16A16_SFLOAT, "rgba16f", 4, F16, 'f', 1.0f / 64},
   {VK_FORMAT_R16G16_SFLOAT, "rg16f", 2, F16, 'f', 1.0f / 64},
   {VK_FORMAT_R16_SFLOAT, "r16f", 1, F16, 'f', 1.0f / 64},
   {VK_FORMAT_R32_SFLOAT, "r32f", 1, F32, 'f', 1.0f / 64},
   {VK_FORMAT_R32G32_SFLOAT, "rg32f", 2, F32, 'f', 1.0f / 64},
   {VK_FORMAT_R32G32B32A32_SFLOAT, "rgba32f", 4, F32, 'f', 1.0f / 64},
   {VK_FORMAT_R8G8B8A8_UNORM, "rgba8", 4, UN8, 'f', 1.0f / 255},
   {VK_FORMAT_R8G8B8A8_SNORM, "rgba8_snorm", 4, SN8, 'f', 1.0f / 127},
   {VK_FORMAT_R8G8B8A8_UINT, "rgba8ui", 4, U8, 'u', 1},
   {VK_FORMAT_R8G8B8A8_SINT, "rgba8i", 4, I8, 'i', 1},
   {VK_FORMAT_R16G16B16A16_UNORM, "rgba16", 4, UN16, 'f', 1.0f / 65535},
   {VK_FORMAT_R16G16B16A16_SNORM, "rgba16_snorm", 4, SN16, 'f', 1.0f / 32767},
   {VK_FORMAT_R16G16B16A16_UINT, "rgba16ui", 4, U16, 'u', 1},
   {VK_FORMAT_R16G16B16A16_SINT, "rgba16i", 4, I16, 'i', 1},
   {VK_FORMAT_R16G16_UNORM, "rg16", 2, UN16, 'f', 1.0f / 65535},
   {VK_FORMAT_R16G16_SNORM, "rg16_snorm", 2, SN16, 'f', 1.0f / 32767},
   {VK_FORMAT_R16G16_UINT, "rg16ui", 2, U16, 'u', 1},
   {VK_FORMAT_R16G16_SINT, "rg16i", 2, I16, 'i', 1},
   {VK_FORMAT_R16_UNORM, "r16", 1, UN16, 'f', 1.0f / 65535},
   {VK_FORMAT_R16_SNORM, "r16_snorm", 1, SN16, 'f', 1.0f / 32767},
   {VK_FORMAT_R16_UINT, "r16ui", 1, U16, 'u', 1},
   {VK_FORMAT_R16_SINT, "r16i", 1, I16, 'i', 1},
   {VK_FORMAT_B8G8R8A8_UNORM, "bgra8", 4, UN8, 'f', 1.0f / 255, true},
   {VK_FORMAT_R8G8_UNORM, "rg8", 2, UN8, 'f', 1.0f / 255},
   {VK_FORMAT_R8G8_SNORM, "rg8_snorm", 2, SN8, 'f', 1.0f / 127},
   {VK_FORMAT_R8G8_UINT, "rg8ui", 2, U8, 'u', 1},
   {VK_FORMAT_R8G8_SINT, "rg8i", 2, I8, 'i', 1},
   {VK_FORMAT_R8_UNORM, "r8", 1, UN8, 'f', 1.0f / 255},
   {VK_FORMAT_R8_SNORM, "r8_snorm", 1, SN8, 'f', 1.0f / 127},
   {VK_FORMAT_R8_UINT, "r8ui", 1, U8, 'u', 1},
   {VK_FORMAT_R8_SINT, "r8i", 1, I8, 'i', 1},
   {VK_FORMAT_R32G32_UINT, "rg32ui", 2, U32, 'u', 1},
   {VK_FORMAT_R32G32_SINT, "rg32i", 2, I32, 'i', 1},
   {VK_FORMAT_R32G32B32A32_UINT, "rgba32ui", 4, U32, 'u', 1},
   {VK_FORMAT_R32G32B32A32_SINT, "rgba32i", 4, I32, 'i', 1},
};

/* The byte offset of channel c within a texel. */
static unsigned channel_offset(const struct fmt *f, unsigned c, unsigned bytes) {
  return (f->bgra && (c == 0 || c == 2) ? 2 - c : c) * bytes;
}

static unsigned channel_bytes(enum enc e) {
  return (e == F32 || e == U32 || e == I32) ? 4
         : (e == UN8 || e == SN8 || e == U8 || e == I8) ? 1
                                                         : 2;
}

/* The format's bits for k (0 <= k <= 64). */
static uint32_t encode(enum enc e, unsigned k) {
  switch (e) {
  case F32: {
    float f = k / 64.0f;
    uint32_t u;
    memcpy(&u, &f, 4);
    return u;
  }
  case F16: {
    if (!k)
      return 0;
    /* k/64 has at most 7 significant bits: exact in a normal half. */
    float f = k / 64.0f;
    uint32_t u;
    memcpy(&u, &f, 4);
    return (((u >> 23) & 0xff) - 127 + 15) << 10 | ((u >> 13) & 0x3ff);
  }
  default:
    return k;
  }
}

/* Rectangles: x, y, w, h. */
struct rect {
  int x, y, w, h;
};

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue q;
static VkCommandBuffer cb;
static VkFence fence;
static unsigned W, H;

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
                            void **map, VkDeviceMemory *mem_out) {
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
  *mem_out = mem;
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

static void transition(VkImage img, VkImageLayout from, VkImageLayout to) {
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
               .oldLayout = from, .newLayout = to, .image = img,
               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}});
}

static VkPipeline graphics_pipeline(VkPipelineLayout layout, VkFormat format,
                                    VkShaderModule vs, VkShaderModule fs,
                                    bool blend) {
  const VkDynamicState dyn[] = {VK_DYNAMIC_STATE_VIEWPORT,
                                VK_DYNAMIC_STATE_SCISSOR};
  VkPipeline p;
  CHECK(vkCreateGraphicsPipelines(
     dev, VK_NULL_HANDLE, 1,
     &(VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = &(VkPipelineRenderingCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
           .colorAttachmentCount = 1, .pColorAttachmentFormats = &format},
        .stageCount = 2,
        .pStages = (VkPipelineShaderStageCreateInfo[]){
           {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs,
            .pName = "main"},
           {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs,
            .pName = "main"}},
        .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
        .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
           .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
        .pViewportState = &(VkPipelineViewportStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
           .viewportCount = 1, .scissorCount = 1},
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
              .blendEnable = blend,
              .srcColorBlendFactor = VK_BLEND_FACTOR_ONE,
              .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
              .colorBlendOp = VK_BLEND_OP_ADD,
              .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
              .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
              .alphaBlendOp = VK_BLEND_OP_ADD,
              .colorWriteMask = 0xf}},
        .pDynamicState = &(VkPipelineDynamicStateCreateInfo){
           .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
           .dynamicStateCount = 2, .pDynamicStates = dyn},
        .layout = layout},
     NULL, &p));
  return p;
}

static VkPipeline compute_pipeline(VkPipelineLayout layout, VkShaderModule cs) {
  VkPipeline p;
  CHECK(vkCreateComputePipelines(
     dev, VK_NULL_HANDLE, 1,
     &(VkComputePipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = cs,
                  .pName = "main"},
        .layout = layout},
     NULL, &p));
  return p;
}

struct rmw_push {
  float scale;
  uint32_t mul_x, mul_y, mod, add;
  uint32_t pad[3];
  int32_t r[4];
};

struct read_push {
  float scale;
  uint32_t fetch, w, h;
};

static void model_rmw(uint8_t *K, struct rect r, const struct rmw_push *p) {
  for (int y = r.y; y < r.y + r.h; y++)
    for (int x = r.x; x < r.x + r.w; x++)
      if (((unsigned)x * p->mul_x + (unsigned)y * p->mul_y) % p->mod == 0)
        for (unsigned c = 0; c < 4; c++)
          K[(y * W + x) * 4 + c] = (K[(y * W + x) * 4 + c] + p->add) % 64;
}

int main(int argc, char **argv) {
  W = argc > 1 ? strtoul(argv[1], NULL, 10) : 300;
  H = argc > 2 ? strtoul(argv[2], NULL, 10) : 200;
  const char *filter = argc > 3 ? argv[3] : NULL;
  if (W < 160 || H < 160) {
    fprintf(stderr, "the regions need at least 160x160\n");
    return 2;
  }

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
  if (!n)
    return 2;
  float prio = 1;
  CHECK(vkCreateDevice(
     pdev,
     &(VkDeviceCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &(VkPhysicalDeviceVulkan13Features){
           .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
           .dynamicRendering = VK_TRUE, .synchronization2 = VK_TRUE},
        .pEnabledFeatures = &(VkPhysicalDeviceFeatures){
           .shaderStorageImageReadWithoutFormat = VK_TRUE,
           .shaderStorageImageWriteWithoutFormat = VK_TRUE,
           .shaderStorageImageExtendedFormats = VK_TRUE},
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
           .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
           .queueCount = 1, .pQueuePriorities = &prio}},
     NULL, &dev));
  vkGetDeviceQueue(dev, 0, 0, &q);

  VkDescriptorSetLayout dsl;
  CHECK(vkCreateDescriptorSetLayout(
     dev,
     &(VkDescriptorSetLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = (VkDescriptorSetLayoutBinding[]){
           {0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1,
            VK_SHADER_STAGE_COMPUTE_BIT, NULL},
           {1, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1,
            VK_SHADER_STAGE_COMPUTE_BIT, NULL},
           {2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
            VK_SHADER_STAGE_COMPUTE_BIT, NULL}}},
     NULL, &dsl));
  VkDescriptorPool pool;
  CHECK(vkCreateDescriptorPool(
     dev,
     &(VkDescriptorPoolCreateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
        .poolSizeCount = 3,
        .pPoolSizes = (VkDescriptorPoolSize[]){
           {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1},
           {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1},
           {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}}},
     NULL, &pool));
  VkDescriptorSet set;
  CHECK(vkAllocateDescriptorSets(
     dev,
     &(VkDescriptorSetAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &dsl},
     &set));
  VkPipelineLayout layout;
  CHECK(vkCreatePipelineLayout(
     dev,
     &(VkPipelineLayoutCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &(VkPushConstantRange){
           VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT, 0,
           sizeof(struct rmw_push)}},
     NULL, &layout));

  const VkShaderModule vs = module(sc_draw_vert, sizeof(sc_draw_vert));
  const VkShaderModule fill[3] = {
     module(sc_fill_f_frag, sizeof(sc_fill_f_frag)),
     module(sc_fill_u_frag, sizeof(sc_fill_u_frag)),
     module(sc_fill_i_frag, sizeof(sc_fill_i_frag))};
  const VkPipeline rmw[3] = {
     compute_pipeline(layout, module(sc_rmw_f_comp, sizeof(sc_rmw_f_comp))),
     compute_pipeline(layout, module(sc_rmw_u_comp, sizeof(sc_rmw_u_comp))),
     compute_pipeline(layout, module(sc_rmw_i_comp, sizeof(sc_rmw_i_comp)))};
  const VkPipeline readk[3] = {
     compute_pipeline(layout, module(sc_read_f_comp, sizeof(sc_read_f_comp))),
     compute_pipeline(layout, module(sc_read_u_comp, sizeof(sc_read_u_comp))),
     compute_pipeline(layout, module(sc_read_i_comp, sizeof(sc_read_i_comp)))};

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

  /* Upload source, raw readback and k readback buffers. */
  const size_t texels = (size_t)W * H;
  VkDeviceMemory mems[3];
  void *upload_map, *raw_map, *k_map;
  VkBuffer upload = host_buffer(texels * 16, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                &upload_map, &mems[0]);
  VkBuffer raw = host_buffer(texels * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             &raw_map, &mems[1]);
  VkBuffer kbuf = host_buffer(texels * 2 * 16,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &k_map,
                              &mems[2]);
  uint8_t *K = malloc(texels * 4);

  /* SC_NOCLEAR: no fast clear; the fill pass covers the whole image, so
   * every block holds rendered data before the compute stores. */
  const bool noclear = getenv("SC_NOCLEAR") != NULL;
  const struct rect r1 = noclear ? (struct rect){0, 0, (int)W, (int)H}
                                 : (struct rect){13, 9, 150, 101},
                    r2 = {40, 30, 110, 120},
                    r3 = {100, 50, (int)W - 110, (int)H - 60},
                    r4 = {5, 117, 77, 33}, rall = {0, 0, (int)W, (int)H};
  const struct rmw_push p3 = {.mul_x = 7, .mul_y = 3, .mod = 5, .add = 3,
                              .r = {r2.x, r2.y, r2.w, r2.h}};
  const struct rmw_push p6 = {.mul_x = 1, .mul_y = 1, .mod = 3, .add = 5,
                              .r = {0, 0, (int)W, (int)H}};

  unsigned failed = 0, tested = 0;
  for (unsigned f = 0; f < sizeof(formats) / sizeof(formats[0]); f++) {
    const struct fmt *F = &formats[f];
    if (filter && strcmp(filter, F->name))
      continue;
    const unsigned t = F->type == 'f' ? 0 : F->type == 'u' ? 1 : 2;
    const unsigned cb_bytes = channel_bytes(F->enc);
    const unsigned texel_B = cb_bytes * F->channels;

    const VkImageUsageFlags usage =
       VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
       VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
       VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(pdev, F->vk, &fp);
    const VkFormatFeatureFlags need =
       VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
       VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
       VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
       (t == 0 ? VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT : 0);
    if ((fp.optimalTilingFeatures & need) != need) {
      printf("%-13s SKIP (format features %#x)\n", F->name,
             fp.optimalTilingFeatures);
      continue;
    }
    tested++;

    VkImage img;
    CHECK(vkCreateImage(
       dev,
       &(VkImageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .imageType = VK_IMAGE_TYPE_2D, .format = F->vk,
                            .extent = {W, H, 1}, .mipLevels = 1,
                            .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = usage},
       NULL, &img));
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(dev, img, &mr);
    VkDeviceMemory img_mem =
       bind_memory(mr, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    CHECK(vkBindImageMemory(dev, img, img_mem, 0));
    VkImageView view;
    CHECK(vkCreateImageView(
       dev,
       &(VkImageViewCreateInfo){
          .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img,
          .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = F->vk,
          .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
       NULL, &view));
    vkUpdateDescriptorSets(
       dev, 3,
       (VkWriteDescriptorSet[]){
          {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
           .dstBinding = 0, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
           .pImageInfo = &(VkDescriptorImageInfo){VK_NULL_HANDLE, view,
                                                  VK_IMAGE_LAYOUT_GENERAL}},
          {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
           .dstBinding = 1, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
           .pImageInfo = &(VkDescriptorImageInfo){
              VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
          {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
           .dstBinding = 2, .descriptorCount = 1,
           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
           .pBufferInfo = &(VkDescriptorBufferInfo){kbuf, 0, VK_WHOLE_SIZE}}},
       0, NULL);
    const VkPipeline fill_pipe =
       graphics_pipeline(layout, F->vk, vs, fill[t], false);
    const VkPipeline blend_pipe =
       t == 0 ? graphics_pipeline(layout, F->vk, vs, fill[t], true)
              : VK_NULL_HANDLE;

    /* The model, and the upload source in the format's bits. */
    for (size_t i = 0; i < texels * 4; i++)
      K[i] = 7;
    for (int y = r1.y; y < r1.y + r1.h; y++)
      for (int x = r1.x; x < r1.x + r1.w; x++)
        for (unsigned c = 0; c < 4; c++)
          K[(y * W + x) * 4 + c] = (x + 2 * y + 5 * c) % 64;
    model_rmw(K, r2, &p3);
    if (t == 0)
      for (int y = r3.y; y < r3.y + r3.h; y++)
        for (int x = r3.x; x < r3.x + r3.w; x++)
          for (unsigned c = 0; c < 4; c++)
            K[(y * W + x) * 4 + c] += 1;
    uint8_t *up = upload_map;
    for (int y = 0; y < r4.h; y++)
      for (int x = 0; x < r4.w; x++)
        for (unsigned c = 0; c < F->channels; c++) {
          const unsigned gx = r4.x + x, gy = r4.y + y;
          const unsigned k = (gx * 3 + gy + 11 * c) % 64;
          const uint32_t bits = encode(F->enc, k);
          memcpy(up + ((size_t)y * r4.w + x) * texel_B +
                    channel_offset(F, c, cb_bytes),
                 &bits, cb_bytes);
          K[(gy * W + gx) * 4 + c] = k;
        }
    model_rmw(K, rall, &p6);

    /* The GPU side. */
    CHECK(vkBeginCommandBuffer(
       cb, &(VkCommandBufferBeginInfo){
              .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1,
                            &set, 0, NULL);
    transition(img, VK_IMAGE_LAYOUT_UNDEFINED,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    VkClearColorValue clear;
    if (t == 0)
      for (unsigned c = 0; c < 4; c++)
        clear.float32[c] = 7 * F->scale;
    else
      for (unsigned c = 0; c < 4; c++)
        clear.uint32[c] = 7;
    if (!noclear)
      vkCmdClearColorImage(
         cb, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1,
         &(VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    for (unsigned pass = 0; pass < 2; pass++) {
      if (pass == 1) {
        /* 3: scattered compute read-modify-write */
        transition(img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   VK_IMAGE_LAYOUT_GENERAL);
        struct rmw_push p = p3;
        p.scale = F->scale;
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rmw[t]);
        vkCmdPushConstants(cb, layout,
                           VK_SHADER_STAGE_FRAGMENT_BIT |
                              VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(p), &p);
        vkCmdDispatch(cb, (r2.w + 7) / 8, (r2.h + 7) / 8, 1);
        if (t != 0)
          break;
        transition(img, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      }
      /* 2: fill, 4: additive blend */
      const struct rect r = pass ? r3 : r1;
      vkCmdBeginRendering(
         cb, &(VkRenderingInfo){
                .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                .renderArea = {{r.x, r.y}, {r.w, r.h}}, .layerCount = 1,
                .colorAttachmentCount = 1,
                .pColorAttachments = &(VkRenderingAttachmentInfo){
                   .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                   .imageView = view,
                   .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                   .storeOp = VK_ATTACHMENT_STORE_OP_STORE}});
      vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS,
                        pass ? blend_pipe : fill_pipe);
      const struct {
        float scale;
        uint32_t ones;
      } fp_push = {F->scale, pass};
      vkCmdPushConstants(cb, layout,
                         VK_SHADER_STAGE_FRAGMENT_BIT |
                            VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(fp_push), &fp_push);
      vkCmdSetViewport(cb, 0, 1, &(VkViewport){0, 0, W, H, 0, 1});
      vkCmdSetScissor(cb, 0, 1, &(VkRect2D){{r.x, r.y}, {r.w, r.h}});
      vkCmdDraw(cb, 3, 1, 0, 0);
      vkCmdEndRendering(cb);
    }
    /* 5: unaligned upload */
    transition(img,
               t == 0 ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                      : VK_IMAGE_LAYOUT_GENERAL,
               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    vkCmdCopyBufferToImage(
       cb, upload, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
       &(VkBufferImageCopy){
          .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .imageOffset = {r4.x, r4.y, 0},
          .imageExtent = {r4.w, r4.h, 1}});
    /* 6: scattered compute read-modify-write everywhere */
    transition(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
               VK_IMAGE_LAYOUT_GENERAL);
    struct rmw_push p = p6;
    p.scale = F->scale;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, rmw[t]);
    vkCmdPushConstants(cb, layout,
                       VK_SHADER_STAGE_FRAGMENT_BIT |
                          VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(p), &p);
    vkCmdDispatch(cb, (W + 7) / 8, (H + 7) / 8, 1);
    /* Readbacks: imageLoad, texelFetch, copy. */
    transition(img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, readk[t]);
    for (unsigned fetch = 0; fetch < 2; fetch++) {
      if (fetch)
        transition(img, VK_IMAGE_LAYOUT_GENERAL,
                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      const struct read_push rp = {F->scale, fetch, W, H};
      vkCmdPushConstants(cb, layout,
                         VK_SHADER_STAGE_FRAGMENT_BIT |
                            VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(rp), &rp);
      vkCmdDispatch(cb, (W + 7) / 8, (H + 7) / 8, 1);
    }
    transition(img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    vkCmdCopyImageToBuffer(
       cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, raw, 1,
       &(VkBufferImageCopy){
          .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
          .imageExtent = {W, H, 1}});
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
    memset(raw_map, 0xee, texels * texel_B);
    memset(k_map, 0xee, texels * 2 * 16);
    CHECK(vkQueueSubmit(q, 1,
                        &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                                        .commandBufferCount = 1,
                                        .pCommandBuffers = &cb},
                        fence));
    CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
    CHECK(vkResetFences(dev, 1, &fence));
    CHECK(vkResetCommandBuffer(cb, 0));

    /* Compare. */
    unsigned bad[3] = {0};
    const char *const how[3] = {"imageLoad", "texelFetch", "copy"};
    const uint32_t(*kr)[4] = k_map;
    const uint8_t *rb = raw_map;
    for (unsigned y = 0; y < H; y++) {
      for (unsigned x = 0; x < W; x++) {
        const size_t i = (size_t)y * W + x;
        for (unsigned c = 0; c < F->channels; c++) {
          const unsigned want = K[i * 4 + c];
          uint32_t got[3] = {kr[i][c], kr[texels + i][c], 0};
          memcpy(&got[2], rb + i * texel_B + channel_offset(F, c, cb_bytes),
                 cb_bytes);
          const uint32_t expect[3] = {want, want, encode(F->enc, want)};
          for (unsigned m = 0; m < 3; m++) {
            if (got[m] == expect[m])
              continue;
            if (bad[m]++ < 3)
              printf("%-13s %s: texel %u,%u channel %u = %#x, want %#x "
                     "(k=%u)\n",
                     F->name, how[m], x, y, c, got[m], expect[m], want);
          }
        }
      }
    }
    if (getenv("SC_DUMP") && (bad[0] || bad[1] || bad[2])) {
      for (unsigned y = 0; y < 8; y++) {
        for (unsigned x = 0; x < 24; x++)
          printf("%2u%c", kr[y * W + x][0],
                 kr[y * W + x][0] == K[(y * W + x) * 4] ? ' ' : '*');
        printf("   ");
        for (unsigned x = 0; x < 24; x++)
          printf("%2u ", K[(y * W + x) * 4]);
        printf("\n");
      }
    }
    if (bad[0] || bad[1] || bad[2]) {
      failed++;
      printf("%-13s FAIL: %u imageLoad, %u texelFetch, %u copy mismatches\n",
             F->name, bad[0], bad[1], bad[2]);
    } else {
      printf("%-13s exact\n", F->name);
    }

    vkDestroyPipeline(dev, fill_pipe, NULL);
    if (blend_pipe)
      vkDestroyPipeline(dev, blend_pipe, NULL);
    vkDestroyImageView(dev, view, NULL);
    vkDestroyImage(dev, img, NULL);
    vkFreeMemory(dev, img_mem, NULL);
  }

  printf("%u formats, %u failed, %ux%u\n", tested, failed, W, H);
  /* A filter that matches no format proves nothing. */
  return failed ? 1 : tested ? 0 : 2;
}
