/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * A render on one queue and a compute read of its result on another, ordered
 * only by timeline semaphores, as vkd3d-proton orders a D3D12 graphics queue
 * and an async compute queue (Cyberpunk 2077's haze buffer).
 *
 * Frame i: queue A waits until queue B has read frame i - 1, renders the
 * value i into a 960x600 R16_SFLOAT image with a deliberately slow fragment
 * shader and signals i. Queue B waits for i, reads a grid of texels in a
 * compute shader and signals i. The CPU submits every frame without waiting,
 * so the GPU sees the whole chain at once.
 *
 * Every texel B reads in frame i must be i. A texel that is i - 1 (or
 * anything else) means B ran before A's render had finished writing: the
 * semaphore did not order the work. Run once with a compressed image (ALIAS)
 * and once with an uncompressed one (MUTABLE_FORMAT, no format list): an
 * early read is visible in both, but in a compressed image it decodes whole
 * tiles wrongly.
 *
 * Variants: unrelated transfer work after the render, in the same submission
 * and on the other hardware pipe; and queue A's wait for B made the way
 * vkd3d-proton implements ID3D12CommandQueue::Wait, an empty batch that only
 * waits, followed by the render batch without waits. Vulkan includes every
 * later submission of the queue in that wait's second scope, so the render of
 * frame i must not start before B has read frame i - 1.
 *
 * usage: g17p-vulkan-cross-queue [--shim]
 *   --shim  no-op DRM shim: record and submit only, skip the checks
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "g17p_vulkan_cross_queue_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define W      960
#define H      600
#define FRAMES 300
#define ITERS  3000

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue[2];
static VkCommandPool pool;

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

static VkSemaphore
timeline(void)
{
   const VkSemaphoreTypeCreateInfo type = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
   };
   const VkSemaphoreCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
      .pNext = &type,
   };
   VkSemaphore s;
   CHECK(vkCreateSemaphore(dev, &info, NULL, &s));
   return s;
}

static void
barrier(VkCommandBuffer cmd, VkImage img, VkPipelineStageFlags2 src_stage,
        VkAccessFlags2 src_access, VkImageLayout from, VkImageLayout to)
{
   const VkImageMemoryBarrier2 b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = src_stage,
      .srcAccessMask = src_access,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .oldLayout = from,
      .newLayout = to,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   vkCmdPipelineBarrier2(cmd, &dep);
}

static void
submit(VkQueue q, VkCommandBuffer cmd, VkSemaphore wait, uint64_t wait_value,
       VkPipelineStageFlags2 wait_stage, VkSemaphore signal, uint64_t value)
{
   const VkSemaphoreSubmitInfo w = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = wait,
      .value = wait_value,
      .stageMask = wait_stage,
   };
   const VkSemaphoreSubmitInfo s = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .semaphore = signal,
      .value = value,
      .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
   };
   const VkCommandBufferSubmitInfo c = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd,
   };
   const VkSubmitInfo2 info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .waitSemaphoreInfoCount = wait_value ? 1 : 0,
      .pWaitSemaphoreInfos = &w,
      .commandBufferInfoCount = cmd ? 1 : 0,
      .pCommandBufferInfos = &c,
      .signalSemaphoreInfoCount = signal ? 1 : 0,
      .pSignalSemaphoreInfos = &s,
   };
   CHECK(vkQueueSubmit2(q, 1, &info, VK_NULL_HANDLE));
}

/* Returns the number of frames whose reads saw anything but the frame's value. */
static unsigned
run(bool compressed, bool tail, bool empty_wait, bool execute)
{
   const VkImageCreateInfo image_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = compressed ? VK_IMAGE_CREATE_ALIAS_BIT
                          : VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R16_SFLOAT,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &image_info, NULL, &img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   VkDeviceMemory img_mem = allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   CHECK(vkBindImageMemory(dev, img, img_mem, 0));
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R16_SFLOAT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &view_info, NULL, &view));

   const VkBufferCreateInfo buf_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = (FRAMES + 1) * 2 * sizeof(uint32_t),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &buf_info, NULL, &buf));
   vkGetBufferMemoryRequirements(dev, buf, &req);
   VkDeviceMemory buf_mem = allocate(req, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkBindBufferMemory(dev, buf, buf_mem, 0));
   uint32_t *result;
   CHECK(vkMapMemory(dev, buf_mem, 0, VK_WHOLE_SIZE, 0, (void **)&result));
   memset(result, 0, buf_info.size);

   /* tail: unrelated transfer work after the render, in the same submission
    * and without a barrier, so it runs beside the render's fragment work. */
   const VkBufferCreateInfo scratch_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 1 << 20,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer scratch;
   CHECK(vkCreateBuffer(dev, &scratch_info, NULL, &scratch));
   vkGetBufferMemoryRequirements(dev, scratch, &req);
   VkDeviceMemory scratch_mem = allocate(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
   CHECK(vkBindBufferMemory(dev, scratch, scratch_mem, 0));

   /* Pipelines */
   const VkPushConstantRange render_pc = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};
   const VkPipelineLayoutCreateInfo render_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &render_pc,
   };
   VkPipelineLayout render_layout, read_layout;
   CHECK(vkCreatePipelineLayout(dev, &render_layout_info, NULL, &render_layout));
   VkShaderModule vs = shader(xq_vert, sizeof(xq_vert));
   VkShaderModule fs = shader(xq_slow_frag, sizeof(xq_slow_frag));
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
       VK_SHADER_STAGE_VERTEX_BIT, vs, "main", NULL},
      {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
       VK_SHADER_STAGE_FRAGMENT_BIT, fs, "main", NULL},
   };
   const VkPipelineVertexInputStateCreateInfo vi = {
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkViewport viewport = {0, 0, W, H, 0, 1};
   const VkRect2D scissor = {{0, 0}, {W, H}};
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &viewport,
      .scissorCount = 1,
      .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState att = {.colorWriteMask = 0xf};
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &att,
   };
   const VkFormat target = VK_FORMAT_R16_SFLOAT;
   const VkPipelineRenderingCreateInfo rendering_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &target,
   };
   const VkGraphicsPipelineCreateInfo gp_info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rendering_info,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vp,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .layout = render_layout,
   };
   VkPipeline render_pipe;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp_info, NULL,
                                   &render_pipe));

   const VkDescriptorSetLayoutBinding bindings[2] = {
      {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
       VK_SHADER_STAGE_COMPUTE_BIT, NULL},
      {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT,
       NULL},
   };
   const VkDescriptorSetLayoutCreateInfo set_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2,
      .pBindings = bindings,
   };
   VkDescriptorSetLayout set_layout;
   CHECK(vkCreateDescriptorSetLayout(dev, &set_layout_info, NULL, &set_layout));
   const VkPushConstantRange read_pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   const VkPipelineLayoutCreateInfo read_layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &set_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &read_pc,
   };
   CHECK(vkCreatePipelineLayout(dev, &read_layout_info, NULL, &read_layout));
   VkShaderModule cs = shader(xq_read_comp, sizeof(xq_read_comp));
   const VkComputePipelineCreateInfo cp_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0,
                VK_SHADER_STAGE_COMPUTE_BIT, cs, "main", NULL},
      .layout = read_layout,
   };
   VkPipeline read_pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cp_info, NULL,
                                  &read_pipe));

   const VkSamplerCreateInfo sampler_info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .magFilter = VK_FILTER_NEAREST,
      .minFilter = VK_FILTER_NEAREST,
   };
   VkSampler sampler;
   CHECK(vkCreateSampler(dev, &sampler_info, NULL, &sampler));
   const VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
   };
   const VkDescriptorPoolCreateInfo dpool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 2,
      .pPoolSizes = sizes,
   };
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpool_info, NULL, &dpool));
   const VkDescriptorSetAllocateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool,
      .descriptorSetCount = 1,
      .pSetLayouts = &set_layout,
   };
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &set_info, &set));
   const VkDescriptorImageInfo ii = {sampler, view,
                                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
   const VkDescriptorBufferInfo bi = {buf, 0, VK_WHOLE_SIZE};
   const VkWriteDescriptorSet writes[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
       .dstBinding = 0, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .pImageInfo = &ii},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
       .dstBinding = 1, .descriptorCount = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &bi},
   };
   vkUpdateDescriptorSets(dev, 2, writes, 0, NULL);

   VkSemaphore produced = timeline(), consumed = timeline();
   static VkCommandBuffer cmds[2 * FRAMES];
   const VkCommandBufferAllocateInfo cmd_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 2 * FRAMES,
   };
   CHECK(vkAllocateCommandBuffers(dev, &cmd_info, cmds));
   const VkCommandBufferBeginInfo begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };

   for (uint32_t i = 1; i <= FRAMES; ++i) {
      /* Queue A: render the value i. */
      VkCommandBuffer g = cmds[2 * (i - 1)];
      CHECK(vkBeginCommandBuffer(g, &begin));
      barrier(g, img, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
              VK_ACCESS_2_MEMORY_READ_BIT,
              i == 1 ? VK_IMAGE_LAYOUT_UNDEFINED
                     : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
      const VkRenderingAttachmentInfo target_att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = view,
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      const VkRenderingInfo rendering = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = {{0, 0}, {W, H}},
         .layerCount = 1,
         .colorAttachmentCount = 1,
         .pColorAttachments = &target_att,
      };
      vkCmdBeginRendering(g, &rendering);
      vkCmdBindPipeline(g, VK_PIPELINE_BIND_POINT_GRAPHICS, render_pipe);
      const struct { float value; uint32_t iters; } rpc = {(float)i, ITERS};
      vkCmdPushConstants(g, render_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                         sizeof(rpc), &rpc);
      vkCmdDraw(g, 3, 1, 0, 0);
      vkCmdEndRendering(g);
      barrier(g, img, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
              VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
              VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
              VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
      if (tail)
         vkCmdFillBuffer(g, scratch, 0, VK_WHOLE_SIZE, i);
      CHECK(vkEndCommandBuffer(g));
      if (empty_wait) {
         /* vkd3d-proton's ID3D12CommandQueue::Wait: an empty batch that waits,
          * then the work, which relies on the wait's second scope including
          * every later submission of the queue. */
         if (i > 1)
            submit(queue[0], VK_NULL_HANDLE, consumed, i - 1,
                   VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_NULL_HANDLE, 0);
         submit(queue[0], g, VK_NULL_HANDLE, 0, 0, produced, i);
      } else {
         submit(queue[0], g, consumed, i - 1,
                VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, produced, i);
      }

      /* Queue B: read it. */
      VkCommandBuffer c = cmds[2 * (i - 1) + 1];
      CHECK(vkBeginCommandBuffer(c, &begin));
      vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_COMPUTE, read_pipe);
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_COMPUTE, read_layout, 0,
                              1, &set, 0, NULL);
      const struct { float expected; uint32_t slot; } cpc = {(float)i, i};
      vkCmdPushConstants(c, read_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(cpc), &cpc);
      vkCmdDispatch(c, (W / 4 + 15) / 16, (H / 4 + 15) / 16, 1);
      CHECK(vkEndCommandBuffer(c));
      submit(queue[1], c, produced, i, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
             consumed, i);
   }

   const uint64_t last = FRAMES;
   const VkSemaphoreWaitInfo wait = {
      .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
      .semaphoreCount = 1,
      .pSemaphores = &consumed,
      .pValues = &last,
   };
   CHECK(vkWaitSemaphores(dev, &wait, UINT64_MAX));
   CHECK(vkDeviceWaitIdle(dev));

   unsigned bad_frames = 0;
   if (execute) {
      for (uint32_t i = 1; i <= FRAMES; ++i) {
         if (!result[2 * i])
            continue;
         if (bad_frames++ < 8) {
            float seen;
            uint32_t f = result[2 * i + 1];
            memcpy(&seen, &f, sizeof(seen));
            printf("  frame %3u: %u of %u texels wrong, one read %g\n", i,
                   result[2 * i], (W / 4) * (H / 4), seen);
         }
      }
      printf("%s image%s%s: %u of %u frames read a value other than their own\n",
             compressed ? "compressed" : "uncompressed",
             tail ? ", transfer after the render" : "",
             empty_wait ? ", waits in an empty submission" : "", bad_frames,
             FRAMES);
   }

   vkDestroySemaphore(dev, produced, NULL);
   vkDestroySemaphore(dev, consumed, NULL);
   vkFreeCommandBuffers(dev, pool, 2 * FRAMES, cmds);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroySampler(dev, sampler, NULL);
   vkDestroyPipeline(dev, read_pipe, NULL);
   vkDestroyPipeline(dev, render_pipe, NULL);
   vkDestroyShaderModule(dev, cs, NULL);
   vkDestroyShaderModule(dev, fs, NULL);
   vkDestroyShaderModule(dev, vs, NULL);
   vkDestroyPipelineLayout(dev, read_layout, NULL);
   vkDestroyPipelineLayout(dev, render_layout, NULL);
   vkDestroyDescriptorSetLayout(dev, set_layout, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkFreeMemory(dev, buf_mem, NULL);
   vkDestroyBuffer(dev, scratch, NULL);
   vkFreeMemory(dev, scratch_mem, NULL);
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, img_mem, NULL);
   return bad_frames;
}

int
main(int argc, char **argv)
{
   const bool shim = argc > 1 && !strcmp(argv[1], "--shim");

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
   CHECK(vkEnumeratePhysicalDevices(instance, &count, &pdev));
   const float priorities[2] = {1.0f, 1.0f};
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 2,
      .pQueuePriorities = priorities,
   };
   VkPhysicalDeviceVulkan12Features features12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .timelineSemaphore = VK_TRUE,
   };
   VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .pNext = &features12,
      .synchronization2 = VK_TRUE,
      .dynamicRendering = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features13,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queue_info,
   };
   CHECK(vkCreateDevice(pdev, &device_info, NULL, &dev));
   vkGetDeviceQueue(dev, 0, 0, &queue[0]);
   vkGetDeviceQueue(dev, 0, 1, &queue[1]);
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
   };
   CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));

   unsigned bad = 0;
   for (unsigned empty_wait = 0; empty_wait < 2; ++empty_wait)
      for (unsigned tail = 0; tail < 2; ++tail)
         bad += run(false, tail, empty_wait, !shim) +
                run(true, tail, empty_wait, !shim);

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   printf("cross-queue render then read%s: %s\n", shim ? " (no checks)" : "",
          bad ? "FAIL" : "PASS");
   return bad ? 1 : 0;
}
