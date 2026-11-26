/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A render pass that only clears its attachments, then a pass that loads the
 * depth, with no barrier between them, as Counter-Strike 2 clears and then
 * fills its depth buffer. Borealis runs a clear-only pass as compute clears
 * (hk_optimize_empty_vdm()); the depth pass must still see the cleared depth,
 * as it would after a clear render.
 *
 * Each frame: a long compute dispatch that nothing waits for, so the compute
 * clears queue behind it; the clear pass (colour and depth); a depth pass that
 * loads the depth and draws seeded triangles over a full-screen background;
 * a colour pass that loads both and draws the same triangles at depth EQUAL,
 * in white. Every pixel must end white. A depth pass that ran before the clear
 * tests against the previous frame's depth, and a clear that lands after it
 * resets its depth: either leaves pixels at the cleared black.
 *
 * Runs 2x D32_SFLOAT_S8_UINT (CS2's depth) and 1x D32_SFLOAT, 8 frames each.
 * Exit status 0 pass, 1 fail, 2 setup error.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vulkan/vulkan.h>

#include "g17p_vulkan_clear_pass_load_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define W         1024
#define H         1024
#define FRAMES    8
#define TRIANGLES 2000
#define BUSY      2000000u

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   }
   fprintf(stderr, "no memory type for %#x\n", flags);
   exit(2);
}

static VkDeviceMemory
allocate(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = memory_type(req.memoryTypeBits, flags),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   return memory;
}

static VkImage
image(VkFormat format, VkSampleCountFlagBits samples, VkImageUsageFlags usage,
      VkImageAspectFlags aspects, VkImageView *view)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &info, NULL, &img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   CHECK(vkBindImageMemory(
      dev, img, allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
   const VkImageViewCreateInfo vinfo = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = format,
      .subresourceRange = {aspects, 0, 1, 0, 1},
   };
   if (view)
      CHECK(vkCreateImageView(dev, &vinfo, NULL, view));
   return img;
}

static void
barrier(VkCommandBuffer cmd, VkImage img, VkImageAspectFlags aspects,
        VkImageLayout from, VkImageLayout to)
{
   const VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {aspects, 0, 1, 0, 1},
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   vkCmdPipelineBarrier2(cmd, &dep);
}

static VkShaderModule
shader(const uint32_t *code, size_t size)
{
   const VkShaderModuleCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size,
      .pCode = code,
   };
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &info, NULL, &m));
   return m;
}

/* depth_pass: depth test LESS and write, no colour; otherwise the white
 * colour pass at EQUAL. */
static VkPipeline
graphics(VkPipelineLayout layout, VkFormat zf, VkSampleCountFlagBits samples,
         bool depth_pass)
{
   const VkFormat cf = VK_FORMAT_R8G8B8A8_UNORM;
   const bool stencil = zf == VK_FORMAT_D32_SFLOAT_S8_UINT;
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = depth_pass ? 0 : 1,
      .pColorAttachmentFormats = &cf,
      .depthAttachmentFormat = zf,
      .stencilAttachmentFormat = stencil ? zf : VK_FORMAT_UNDEFINED,
   };
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = shader(cpl_vert, sizeof(cpl_vert)),
       .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = shader(cpl_frag, sizeof(cpl_frag)),
       .pName = "main"},
   };
   const VkViewport vp = {0, 0, W, H, 0, 1};
   const VkRect2D sc = {{0, 0}, {W, H}};
   const VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = 0xf};
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering,
      .stageCount = depth_pass ? 1 : 2,
      .pStages = stages,
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
         .viewportCount = 1, .pViewports = &vp,
         .scissorCount = 1, .pScissors = &sc},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .lineWidth = 1},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = samples},
      .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = VK_TRUE,
         .depthWriteEnable = depth_pass,
         .depthCompareOp =
            depth_pass ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_EQUAL},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
         .attachmentCount = depth_pass ? 0 : 1, .pAttachments = &blend},
      .layout = layout,
   };
   VkPipeline p;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &info, NULL, &p));
   return p;
}

/* Returns the number of frames with a pixel that is not white. */
static unsigned
run(VkFormat zf, VkSampleCountFlagBits samples, VkCommandPool pool,
    VkPipeline busy, VkPipelineLayout busy_layout, VkDescriptorSet busy_set)
{
   const bool stencil = zf == VK_FORMAT_D32_SFLOAT_S8_UINT;
   const VkImageAspectFlags zasp =
      VK_IMAGE_ASPECT_DEPTH_BIT | (stencil ? VK_IMAGE_ASPECT_STENCIL_BIT : 0);
   VkImageView cv, zv;
   VkImage ci = image(VK_FORMAT_R8G8B8A8_UNORM, samples,
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT, &cv);
   VkImage ri = image(VK_FORMAT_R8G8B8A8_UNORM, VK_SAMPLE_COUNT_1_BIT,
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_IMAGE_ASPECT_COLOR_BIT, NULL);
   /* CS2's depth usage: attachment, sampled and transfer. */
   VkImage zi = image(zf, samples,
                      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                         VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                         VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      zasp, &zv);

   const VkBufferCreateInfo binfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = W * H * 4,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer rb;
   CHECK(vkCreateBuffer(dev, &binfo, NULL, &rb));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, rb, &req);
   VkDeviceMemory rm = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkBindBufferMemory(dev, rb, rm, 0));
   const uint32_t *pixels;
   CHECK(vkMapMemory(dev, rm, 0, VK_WHOLE_SIZE, 0, (void **)&pixels));

   const VkPipelineLayoutCreateInfo linfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges =
         &(VkPushConstantRange){VK_SHADER_STAGE_VERTEX_BIT, 0, 4},
   };
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &linfo, NULL, &layout));
   VkPipeline depth_pipe = graphics(layout, zf, samples, true);
   VkPipeline colour_pipe = graphics(layout, zf, samples, false);

   VkCommandBuffer cmd;
   const VkCommandBufferAllocateInfo cinfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(dev, &cinfo, &cmd));
   VkFence fence;
   CHECK(vkCreateFence(
      dev, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO},
      NULL, &fence));

   const VkImageLayout ZL = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
   const VkImageLayout CL = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   unsigned bad_frames = 0;
   for (uint32_t f = 0; f < FRAMES; ++f) {
      CHECK(vkResetCommandBuffer(cmd, 0));
      CHECK(vkBeginCommandBuffer(
         cmd, &(VkCommandBufferBeginInfo){
                 .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
      barrier(cmd, zi, zasp, f ? ZL : VK_IMAGE_LAYOUT_UNDEFINED, ZL);
      barrier(cmd, ci, VK_IMAGE_ASPECT_COLOR_BIT,
              f ? VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                : VK_IMAGE_LAYOUT_UNDEFINED,
              CL);

      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, busy);
      vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, busy_layout,
                              0, 1, &busy_set, 0, NULL);
      const uint32_t iters = BUSY;
      vkCmdPushConstants(cmd, busy_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4,
                         &iters);
      vkCmdDispatch(cmd, 64, 1, 1);

      VkRenderingAttachmentInfo za = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = zv,
         .imageLayout = ZL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .clearValue = {.depthStencil = {1, 0}},
      };
      VkRenderingAttachmentInfo ca = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = cv,
         .imageLayout = CL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingInfo r = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {W, H}},
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &ca,
         .pDepthAttachment = &za,
         .pStencilAttachment = stencil ? &za : NULL,
      };
      /* The clear pass: nothing drawn, no barrier after it. */
      vkCmdBeginRendering(cmd, &r);
      vkCmdEndRendering(cmd);

      za.loadOp = ca.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      r.colorAttachmentCount = 0;
      vkCmdBeginRendering(cmd, &r);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, depth_pipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &f);
      vkCmdDraw(cmd, (TRIANGLES + 1) * 3, 1, 0, 0);
      vkCmdEndRendering(cmd);

      const VkMemoryBarrier2 zdep = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
         .srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                         VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
         .dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
      };
      vkCmdPipelineBarrier2(cmd, &(VkDependencyInfo){
                                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .memoryBarrierCount = 1,
                                    .pMemoryBarriers = &zdep});

      r.colorAttachmentCount = 1;
      za.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      vkCmdBeginRendering(cmd, &r);
      vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, colour_pipe);
      vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &f);
      vkCmdDraw(cmd, (TRIANGLES + 1) * 3, 1, 0, 0);
      vkCmdEndRendering(cmd);

      barrier(cmd, ci, VK_IMAGE_ASPECT_COLOR_BIT, CL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      barrier(cmd, ri, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
      const VkImageSubresourceLayers sl = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      if (samples > 1) {
         const VkImageResolve region = {sl, {0}, sl, {0}, {W, H, 1}};
         vkCmdResolveImage(cmd, ci, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ri,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
      } else {
         const VkImageCopy region = {sl, {0}, sl, {0}, {W, H, 1}};
         vkCmdCopyImage(cmd, ci, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, ri,
                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
      }
      barrier(cmd, ri, VK_IMAGE_ASPECT_COLOR_BIT,
              VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
              VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      const VkBufferImageCopy copy = {.imageSubresource = sl,
                                      .imageExtent = {W, H, 1}};
      vkCmdCopyImageToBuffer(cmd, ri, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, rb,
                             1, &copy);
      const VkMemoryBarrier2 host = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
         .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
         .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
      };
      vkCmdPipelineBarrier2(cmd, &(VkDependencyInfo){
                                    .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                                    .memoryBarrierCount = 1,
                                    .pMemoryBarriers = &host});
      CHECK(vkEndCommandBuffer(cmd));
      const VkSubmitInfo submit = {
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .commandBufferCount = 1,
         .pCommandBuffers = &cmd,
      };
      CHECK(vkQueueSubmit(queue, 1, &submit, fence));
      CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 20000000000ull));
      CHECK(vkResetFences(dev, 1, &fence));

      unsigned bad = 0;
      for (unsigned i = 0; i < W * H; ++i)
         bad += pixels[i] != 0xffffffffu;
      if (bad)
         printf("  frame %u: %u pixels not white\n", f, bad);
      bad_frames += bad != 0;
   }
   return bad_frames;
}

int
main(void)
{
   const VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .apiVersion = VK_API_VERSION_1_3,
   };
   const VkInstanceCreateInfo iinfo = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance instance;
   CHECK(vkCreateInstance(&iinfo, NULL, &instance));
   uint32_t n = 1;
   VkResult res = vkEnumeratePhysicalDevices(instance, &n, &pdev);
   if ((res != VK_SUCCESS && res != VK_INCOMPLETE) || !n) {
      fprintf(stderr, "no Vulkan device\n");
      return 2;
   }

   const float prio = 1.0f;
   const VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
   };
   const VkDeviceQueueCreateInfo qinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = 0,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   const VkDeviceCreateInfo dinfo = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &f13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qinfo,
   };
   CHECK(vkCreateDevice(pdev, &dinfo, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue);

   VkCommandPool pool;
   const VkCommandPoolCreateInfo pinfo = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   CHECK(vkCreateCommandPool(dev, &pinfo, NULL, &pool));

   /* The busy dispatch's output buffer and pipeline. */
   const VkBufferCreateInfo binfo = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 4096,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer bb;
   CHECK(vkCreateBuffer(dev, &binfo, NULL, &bb));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, bb, &req);
   CHECK(vkBindBufferMemory(
      dev, bb, allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
   const VkDescriptorSetLayoutBinding binding = {
      0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT};
   const VkDescriptorSetLayoutCreateInfo dslinfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1,
      .pBindings = &binding,
   };
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dslinfo, NULL, &dsl));
   const VkDescriptorPoolSize size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   const VkDescriptorPoolCreateInfo dpinfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &size,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpinfo, NULL, &dpool));
   const VkDescriptorSetAllocateInfo dsinfo = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &dsl,
   };
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dsinfo, &set));
   const VkWriteDescriptorSet write = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set,
      .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &(VkDescriptorBufferInfo){bb, 0, VK_WHOLE_SIZE},
   };
   vkUpdateDescriptorSets(dev, 1, &write, 0, NULL);
   const VkPipelineLayoutCreateInfo blinfo = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges =
         &(VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, 0, 4},
   };
   VkPipelineLayout busy_layout;
   CHECK(vkCreatePipelineLayout(dev, &blinfo, NULL, &busy_layout));
   const VkComputePipelineCreateInfo cinfo = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = shader(cpl_busy, sizeof(cpl_busy)),
                .pName = "main"},
      .layout = busy_layout,
   };
   VkPipeline busy;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cinfo, NULL,
                                  &busy));

   unsigned bad = 0, frames;
   frames = run(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_SAMPLE_COUNT_2_BIT, pool,
                busy, busy_layout, set);
   printf("2x D32S8: %u of %u frames wrong\n", frames, FRAMES);
   bad += frames;
   frames = run(VK_FORMAT_D32_SFLOAT, VK_SAMPLE_COUNT_1_BIT, pool, busy,
                busy_layout, set);
   printf("1x D32: %u of %u frames wrong\n", frames, FRAMES);
   bad += frames;

   printf("clear pass, then a pass that loads the depth: %s\n",
          bad ? "FAIL" : "PASS");
   return bad ? 1 : 0;
}
