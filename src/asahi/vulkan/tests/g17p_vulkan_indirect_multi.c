/*
 * Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Exercise multi-draw indirect recording across:
 *   - vkCmdDrawIndirect (non-indexed, bind-tail elision + chunk rollover)
 *   - vkCmdDrawIndexedIndirect (both robustBufferAccess on and off)
 *   - indexed and non-indexed indirect count, including command buffer reuse
 *   - VS with neither base params nor gl_DrawID (fixed loader reuse)
 *   - VS with gl_InstanceIndex + gl_VertexIndex (dynamic DRAW_PARAMS reloc)
 *   - VS with gl_DrawID + gl_InstanceIndex (dynamic DRAW_PARAMS + DRAW_ID)
 *   - Non-standard strides, empty draws (empty_draw_jump), edge counts 0..3,
 *     and 2048-draw loops that cross multiple 64 KiB VDM chunks and trigger
 *     Ioto arena growth.
 *
 * Pass --shim under asahi_noop_drm_shim to validate command streams without
 * GPU readback; omit --shim on hardware to verify every rendered pixel.
 */

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "g17p_vulkan_indirect_multi_shaders.h"

#define IMG_W 64u
#define IMG_H 64u
#define DRAW_COUNT 2048u
#define INDIRECT_COUNT_ACTUAL 1050u

#define CHECK(expr)                                                            \
   do {                                                                        \
      VkResult _r = (expr);                                                    \
      if (_r != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s failed (%d) at line %d\n", #expr, (int)_r,        \
                 __LINE__);                                                    \
         exit(1);                                                              \
      }                                                                        \
   } while (0)

static uint32_t
find_mem_type(const VkPhysicalDeviceMemoryProperties *mp, uint32_t bits,
              VkMemoryPropertyFlags req)
{
   for (uint32_t i = 0; i < mp->memoryTypeCount; ++i) {
      if ((bits & (1u << i)) &&
          (mp->memoryTypes[i].propertyFlags & req) == req)
         return i;
   }
   fprintf(stderr, "No matching memory type (bits=0x%x req=0x%x)\n", bits, req);
   exit(1);
}

struct host_buf {
   VkBuffer buf;
   VkDeviceMemory mem;
   void *map;
   VkDeviceSize size;
};

static struct host_buf
create_host_buf(VkDevice dev, const VkPhysicalDeviceMemoryProperties *mp,
                VkDeviceSize size, VkBufferUsageFlags usage)
{
   struct host_buf b = {.size = size};
   VkBufferCreateInfo bci = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   };
   CHECK(vkCreateBuffer(dev, &bci, NULL, &b.buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, b.buf, &mr);
   VkMemoryAllocateInfo mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = mr.size,
      .memoryTypeIndex = find_mem_type(
         mp, mr.memoryTypeBits,
         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   CHECK(vkAllocateMemory(dev, &mai, NULL, &b.mem));
   CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
   CHECK(vkMapMemory(dev, b.mem, 0, size, 0, &b.map));
   memset(b.map, 0, size);
   return b;
}

static void
destroy_host_buf(VkDevice dev, struct host_buf *b)
{
   vkUnmapMemory(dev, b->mem);
   vkDestroyBuffer(dev, b->buf, NULL);
   vkFreeMemory(dev, b->mem, NULL);
}

static VkShaderModule
make_shader(VkDevice dev, const uint32_t *code, size_t size_B)
{
   VkShaderModuleCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size_B,
      .pCode = code,
   };
   VkShaderModule sm = VK_NULL_HANDLE;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &sm));
   return sm;
}

static VkPipeline
make_pipeline(VkDevice dev, VkRenderPass rp, VkPipelineLayout pl,
              VkShaderModule vsm, VkShaderModule fsm,
              VkPrimitiveTopology topology)
{
   VkPipelineShaderStageCreateInfo stages[2] = {
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_VERTEX_BIT,
         .module = vsm,
         .pName = "main",
      },
      {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
         .module = fsm,
         .pName = "main",
      },
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = topology,
   };
   VkViewport vp = {
      .x = 0.0f,
      .y = 0.0f,
      .width = (float)IMG_W,
      .height = (float)IMG_H,
      .minDepth = 0.0f,
      .maxDepth = 1.0f,
   };
   VkRect2D sc = {
      .offset = {0, 0},
      .extent = {IMG_W, IMG_H},
   };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1,
      .pViewports = &vp,
      .scissorCount = 1,
      .pScissors = &sc,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL,
      .cullMode = VK_CULL_MODE_NONE,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
      .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineColorBlendAttachmentState cba = {
      .colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT,
   };
   VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &cba,
   };
   VkGraphicsPipelineCreateInfo gpci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .stageCount = 2,
      .pStages = stages,
      .pVertexInputState = &vi,
      .pInputAssemblyState = &ia,
      .pViewportState = &vps,
      .pRasterizationState = &rs,
      .pMultisampleState = &ms,
      .pColorBlendState = &cb,
      .layout = pl,
      .renderPass = rp,
      .subpass = 0,
   };
   VkPipeline pipe = VK_NULL_HANDLE;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));
   return pipe;
}

static void
submit_and_wait(VkQueue queue, VkCommandBuffer cmd)
{
   VkSubmitInfo si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
}

static void
begin_pass_and_submit(VkQueue queue, VkCommandBuffer cmd,
                      VkRenderPass rp, VkFramebuffer fb, VkImage img,
                      VkBuffer readback,
                      void (*record)(VkCommandBuffer, void *), void *ctx)
{
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
   };
   CHECK(vkResetCommandBuffer(cmd, 0));
   CHECK(vkBeginCommandBuffer(cmd, &bi));

   VkClearValue clear = {.color = {.uint32 = {0, 0, 0, 0}}};
   VkRenderPassBeginInfo rpbi = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
      .renderPass = rp,
      .framebuffer = fb,
      .renderArea = {{0, 0}, {IMG_W, IMG_H}},
      .clearValueCount = 1,
      .pClearValues = &clear,
   };
   vkCmdBeginRenderPass(cmd, &rpbi, VK_SUBPASS_CONTENTS_INLINE);
   record(cmd, ctx);
   vkCmdEndRenderPass(cmd);

   VkImageMemoryBarrier to_src = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
      .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1,
                        &to_src);

   VkBufferImageCopy region = {
      .imageSubresource =
         {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .mipLevel = 0,
            .baseArrayLayer = 0,
            .layerCount = 1,
         },
      .imageExtent = {IMG_W, IMG_H, 1},
   };
   vkCmdCopyImageToBuffer(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          readback, 1, &region);

   VkMemoryBarrier to_host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &to_host, 0, NULL, 0,
                        NULL);

   CHECK(vkEndCommandBuffer(cmd));

   submit_and_wait(queue, cmd);
}

struct draw_ctx {
   VkPipeline pipe;
   VkBuffer index_buf;
   VkBuffer indirect_buf;
   VkBuffer count_buf;
   uint32_t draw_count;
   uint32_t stride;
   bool indexed;
   bool use_count;
   bool edge_counts;
};

static void
record_draws(VkCommandBuffer cmd, void *opaque)
{
   const struct draw_ctx *c = opaque;
   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, c->pipe);
   if (c->indexed)
      vkCmdBindIndexBuffer(cmd, c->index_buf, 0, VK_INDEX_TYPE_UINT32);

   if (c->edge_counts) {
      const uint32_t counts[] = {0, 1, 2, 3};
      for (unsigned i = 0; i < 4; ++i) {
         if (c->indexed) {
            vkCmdDrawIndexedIndirect(cmd, c->indirect_buf, 0, counts[i],
                                     c->stride);
         } else {
            vkCmdDrawIndirect(cmd, c->indirect_buf, 0, counts[i], c->stride);
         }
      }
      return;
   }

   if (c->use_count && c->indexed) {
      vkCmdDrawIndexedIndirectCount(cmd, c->indirect_buf, 0, c->count_buf, 0,
                                    c->draw_count, c->stride);
   } else if (c->use_count) {
      vkCmdDrawIndirectCount(cmd, c->indirect_buf, 0, c->count_buf, 0,
                             c->draw_count, c->stride);
   } else if (c->indexed) {
      vkCmdDrawIndexedIndirect(cmd, c->indirect_buf, 0, c->draw_count,
                               c->stride);
   } else {
      vkCmdDrawIndirect(cmd, c->indirect_buf, 0, c->draw_count, c->stride);
   }
}

static void
verify_pixels(const char *label, const uint32_t *px, uint32_t active_draws,
              uint32_t inst_base, uint32_t vtx_mod, uint32_t empty_mod,
              uint32_t tag)
{
   for (uint32_t d = 0; d < IMG_W * IMG_H; ++d) {
      const uint32_t *got = px + 4 * d;
      bool drawn = d < active_draws && (!empty_mod || (d % empty_mod) != 0);
      uint32_t exp[4] = {0, 0, 0, 0};
      if (drawn) {
         exp[0] = d;
         exp[1] = inst_base + d;
         exp[2] = d % vtx_mod;
         exp[3] = tag;
      }
      if (memcmp(got, exp, sizeof(exp)) != 0) {
         fprintf(stderr,
                 "FAIL %s pixel %u (%u,%u): got=(%u,%u,%u,%u) "
                 "exp=(%u,%u,%u,%u)\n",
                 label, d, d & 63u, d >> 6u, got[0], got[1], got[2], got[3],
                 exp[0], exp[1], exp[2], exp[3]);
         exit(1);
      }
   }
}

/* Reuse the same recording while the GPU-visible count moves across chunk
 * boundaries, drops to zero, and exceeds maxDrawCount. Each submit clears
 * the image, so skipped draws must leave their pixels clear.
 */
static void
verify_count_replay(VkQueue queue, VkCommandBuffer cmd, uint32_t *count,
                    const uint32_t *pixels, uint32_t vertex_mod,
                    const char *label)
{
   const uint32_t counts[] = {0, DRAW_COUNT - 3, 1, DRAW_COUNT + 1, 7};
   for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i) {
      *count = counts[i];
      submit_and_wait(queue, cmd);
      verify_pixels(label, pixels,
                    *count < DRAW_COUNT ? *count : DRAW_COUNT, 200u,
                    vertex_mod, 0u, 3u);
   }
   *count = INDIRECT_COUNT_ACTUAL;
}

static void
run_suite(VkPhysicalDevice pdev, uint32_t qf, bool robust_buffer_access,
          bool shim, bool non_indexed_only)
{
   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qf,
      .queueCount = 1,
      .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceVulkan12Features vk12 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
      .drawIndirectCount = VK_TRUE,
   };
   VkPhysicalDeviceVulkan11Features vk11 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES,
      .pNext = &vk12,
      .shaderDrawParameters = VK_TRUE,
   };
   VkPhysicalDeviceFeatures feats = {
      .multiDrawIndirect = VK_TRUE,
      .drawIndirectFirstInstance = VK_TRUE,
      .robustBufferAccess = robust_buffer_access ? VK_TRUE : VK_FALSE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &vk11,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &qci,
      .pEnabledFeatures = &feats,
   };
   VkDevice dev = VK_NULL_HANDLE;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));

   VkQueue queue = VK_NULL_HANDLE;
   vkGetDeviceQueue(dev, qf, 0, &queue);

   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);

   VkImageCreateInfo img_ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R32G32B32A32_UINT,
      .extent = {IMG_W, IMG_H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage =
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
   };
   VkImage img = VK_NULL_HANDLE;
   CHECK(vkCreateImage(dev, &img_ci, NULL, &img));
   VkMemoryRequirements img_mr;
   vkGetImageMemoryRequirements(dev, img, &img_mr);
   VkMemoryAllocateInfo img_mai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = img_mr.size,
      .memoryTypeIndex = find_mem_type(&mp, img_mr.memoryTypeBits,
                                       VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory img_mem = VK_NULL_HANDLE;
   CHECK(vkAllocateMemory(dev, &img_mai, NULL, &img_mem));
   CHECK(vkBindImageMemory(dev, img, img_mem, 0));

   VkImageViewCreateInfo ivci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = VK_FORMAT_R32G32B32A32_UINT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
   };
   VkImageView iv = VK_NULL_HANDLE;
   CHECK(vkCreateImageView(dev, &ivci, NULL, &iv));

   VkAttachmentDescription att = {
      .format = VK_FORMAT_R32G32B32A32_UINT,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
      .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkAttachmentReference ref = {
      .attachment = 0,
      .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
   };
   VkSubpassDescription sp = {
      .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
      .colorAttachmentCount = 1,
      .pColorAttachments = &ref,
   };
   VkRenderPassCreateInfo rpci = {
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
      .attachmentCount = 1,
      .pAttachments = &att,
      .subpassCount = 1,
      .pSubpasses = &sp,
   };
   VkRenderPass rp = VK_NULL_HANDLE;
   CHECK(vkCreateRenderPass(dev, &rpci, NULL, &rp));

   VkFramebufferCreateInfo fbci = {
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
      .renderPass = rp,
      .attachmentCount = 1,
      .pAttachments = &iv,
      .width = IMG_W,
      .height = IMG_H,
      .layers = 1,
   };
   VkFramebuffer fb = VK_NULL_HANDLE;
   CHECK(vkCreateFramebuffer(dev, &fbci, NULL, &fb));

   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
   };
   VkPipelineLayout pl = VK_NULL_HANDLE;
   CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));

   VkShaderModule sm_plain =
      make_shader(dev, vs_plain_spv, sizeof(vs_plain_spv));
   VkShaderModule sm_base =
      make_shader(dev, vs_base_param_spv, sizeof(vs_base_param_spv));
   VkShaderModule sm_draw_id =
      make_shader(dev, vs_draw_id_spv, sizeof(vs_draw_id_spv));
   VkShaderModule sm_fs =
      make_shader(dev, fs_uvec4_spv, sizeof(fs_uvec4_spv));

   VkPipeline pipe_plain = make_pipeline(dev, rp, pl, sm_plain, sm_fs,
                                         VK_PRIMITIVE_TOPOLOGY_POINT_LIST);
   VkPipeline pipe_base = make_pipeline(dev, rp, pl, sm_base, sm_fs,
                                        VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
   VkPipeline pipe_draw_id = make_pipeline(dev, rp, pl, sm_draw_id, sm_fs,
                                           VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);

   VkCommandPoolCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qf,
   };
   VkCommandPool pool = VK_NULL_HANDLE;
   CHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));

   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd = VK_NULL_HANDLE;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cmd));

   struct host_buf idx_buf =
      create_host_buf(dev, &mp, 3 * sizeof(uint32_t),
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
   uint32_t *indices = idx_buf.map;
   indices[0] = 0;
   indices[1] = 1;
   indices[2] = 2;

   struct host_buf readback =
      create_host_buf(dev, &mp, IMG_W * IMG_H * 4 * sizeof(uint32_t),
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT);

   struct host_buf count_buf =
      create_host_buf(dev, &mp, sizeof(uint32_t),
                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);
   *(uint32_t *)count_buf.map = INDIRECT_COUNT_ACTUAL;

   /* Allocate one indirect buffer large enough for stride=36 * DRAW_COUNT. */
   struct host_buf ind_buf =
      create_host_buf(dev, &mp, (VkDeviceSize)DRAW_COUNT * 40,
                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT);

   /* 1. Plain VS (no base params, no draw_id -> fixed loader reuse):
    *    non-indexed + indexed with non-standard strides. */
   {
      const uint32_t stride = 32;
      for (uint32_t d = 0; d < DRAW_COUNT; ++d) {
         VkDrawIndirectCommand *c =
            (VkDrawIndirectCommand *)((uint8_t *)ind_buf.map + d * stride);
         *c = (VkDrawIndirectCommand){
            .vertexCount = 1,
            .instanceCount = 1,
            .firstVertex = 0,
            .firstInstance = 0,
         };
      }
      struct draw_ctx ctx = {
         .pipe = pipe_plain,
         .index_buf = idx_buf.buf,
         .indirect_buf = ind_buf.buf,
         .count_buf = count_buf.buf,
         .draw_count = DRAW_COUNT,
         .stride = stride,
         .indexed = false,
      };
      for (unsigned counted = 0; counted <= 1; ++counted) {
         ctx.use_count = counted;
         begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                               record_draws, &ctx);
         if (!shim) {
            const uint32_t *px = readback.map;
            const uint32_t *center = px + 4 * ((IMG_H / 2) * IMG_W + IMG_W / 2);
            if (center[0] != 42 || center[1] != 7 || center[2] != 99 ||
                center[3] != 1) {
               fprintf(stderr, "FAIL plain non-indexed center=(%u,%u,%u,%u)\n",
                       center[0], center[1], center[2], center[3]);
               exit(1);
            }
         }
      }
   }

   /* 2. Base-param VS (uses_base_param=true, uses_draw_id=false),
    *    non-indexed DrawIndirect with empty draws and stride=32. */
   {
      const uint32_t stride = 32;
      memset(ind_buf.map, 0, ind_buf.size);
      for (uint32_t d = 0; d < DRAW_COUNT; ++d) {
         VkDrawIndirectCommand *c =
            (VkDrawIndirectCommand *)((uint8_t *)ind_buf.map + d * stride);
         const bool empty = (d % 37u) == 0;
         *c = (VkDrawIndirectCommand){
            .vertexCount = empty ? 0 : 3,
            .instanceCount = 1,
            .firstVertex = 3u * (d % 7u),
            .firstInstance = 100u + d,
         };
      }
      struct draw_ctx ctx = {
         .pipe = pipe_base,
         .index_buf = idx_buf.buf,
         .indirect_buf = ind_buf.buf,
         .draw_count = DRAW_COUNT,
         .stride = stride,
         .indexed = false,
      };
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim)
         verify_pixels("base_param non-indexed", readback.map, DRAW_COUNT, 100u,
                       7u, 37u, 2u);
   }

   /* 3. Base-param VS, indexed DrawIndexedIndirect with empty draws (tests
    *    empty_draw_jump on robust batch) and stride=36. */
   if (!non_indexed_only) {
      const uint32_t stride = 36;
      memset(ind_buf.map, 0, ind_buf.size);
      for (uint32_t d = 0; d < DRAW_COUNT; ++d) {
         VkDrawIndexedIndirectCommand *c =
            (VkDrawIndexedIndirectCommand *)((uint8_t *)ind_buf.map +
                                             d * stride);
         const bool empty = (d % 31u) == 0;
         *c = (VkDrawIndexedIndirectCommand){
            .indexCount = empty ? 0 : 3,
            .instanceCount = 1,
            .firstIndex = 0,
            .vertexOffset = (int32_t)(3u * (d % 11u)),
            .firstInstance = 100u + d,
         };
      }
      struct draw_ctx ctx = {
         .pipe = pipe_base,
         .index_buf = idx_buf.buf,
         .indirect_buf = ind_buf.buf,
         .draw_count = DRAW_COUNT,
         .stride = stride,
         .indexed = true,
      };
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim)
         verify_pixels("base_param indexed", readback.map, DRAW_COUNT, 100u,
                       11u, 31u, 2u);
   }

   /* 4. DrawID + base-param VS, non-indexed DrawIndirect, stride=24. */
   {
      const uint32_t stride = 24;
      memset(ind_buf.map, 0, ind_buf.size);
      for (uint32_t d = 0; d < DRAW_COUNT; ++d) {
         VkDrawIndirectCommand *c =
            (VkDrawIndirectCommand *)((uint8_t *)ind_buf.map + d * stride);
         *c = (VkDrawIndirectCommand){
            .vertexCount = 3,
            .instanceCount = 1,
            .firstVertex = 3u * (d % 5u),
            .firstInstance = 200u + d,
         };
      }
      struct draw_ctx ctx = {
         .pipe = pipe_draw_id,
         .index_buf = idx_buf.buf,
         .indirect_buf = ind_buf.buf,
         .draw_count = DRAW_COUNT,
         .stride = stride,
         .indexed = false,
      };
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim)
         verify_pixels("draw_id non-indexed", readback.map, DRAW_COUNT, 200u,
                       5u, 0u, 3u);

      ctx.use_count = true;
      ctx.count_buf = count_buf.buf;
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim) {
         verify_pixels("draw_id non-indexed count", readback.map,
                       INDIRECT_COUNT_ACTUAL, 200u, 5u, 0u, 3u);
         verify_count_replay(queue, cmd, count_buf.map, readback.map, 5u,
                             "non-indexed count replay");
      }
      ctx.use_count = false;

      /* Edge drawCounts 0, 1, 2, 3 (non-indexed). */
      ctx.edge_counts = true;
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim)
         verify_pixels("edge non-indexed 0..3", readback.map, 3u, 200u, 5u, 0u,
                       3u);
   }

   /* 5. DrawID + base-param VS, DrawIndexedIndirectCount (actual < max),
    *    stride=28. */
   if (!non_indexed_only) {
      const uint32_t stride = 28;
      memset(ind_buf.map, 0, ind_buf.size);
      for (uint32_t d = 0; d < DRAW_COUNT; ++d) {
         VkDrawIndexedIndirectCommand *c =
            (VkDrawIndexedIndirectCommand *)((uint8_t *)ind_buf.map +
                                             d * stride);
         *c = (VkDrawIndexedIndirectCommand){
            .indexCount = 3,
            .instanceCount = 1,
            .firstIndex = 0,
            .vertexOffset = (int32_t)(3u * (d % 13u)),
            .firstInstance = 200u + d,
         };
      }
      struct draw_ctx ctx = {
         .pipe = pipe_draw_id,
         .index_buf = idx_buf.buf,
         .indirect_buf = ind_buf.buf,
         .count_buf = count_buf.buf,
         .draw_count = DRAW_COUNT,
         .stride = stride,
         .indexed = true,
         .use_count = true,
      };
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim) {
         verify_pixels("draw_id indexed_count", readback.map,
                       INDIRECT_COUNT_ACTUAL, 200u, 13u, 0u, 3u);
         verify_count_replay(queue, cmd, count_buf.map, readback.map, 13u,
                             "indexed count replay");
      }

      /* 6. Edge drawCounts 0, 1, 2, 3 (indexed). */
      ctx.use_count = false;
      ctx.edge_counts = true;
      begin_pass_and_submit(queue, cmd, rp, fb, img, readback.buf,
                            record_draws, &ctx);
      if (!shim)
         verify_pixels("edge indexed 0..3", readback.map, 3u, 200u, 13u, 0u,
                       3u);
   }

   destroy_host_buf(dev, &ind_buf);
   destroy_host_buf(dev, &count_buf);
   destroy_host_buf(dev, &readback);
   destroy_host_buf(dev, &idx_buf);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe_draw_id, NULL);
   vkDestroyPipeline(dev, pipe_base, NULL);
   vkDestroyPipeline(dev, pipe_plain, NULL);
   vkDestroyShaderModule(dev, sm_fs, NULL);
   vkDestroyShaderModule(dev, sm_draw_id, NULL);
   vkDestroyShaderModule(dev, sm_base, NULL);
   vkDestroyShaderModule(dev, sm_plain, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyFramebuffer(dev, fb, NULL);
   vkDestroyRenderPass(dev, rp, NULL);
   vkDestroyImageView(dev, iv, NULL);
   vkDestroyImage(dev, img, NULL);
   vkFreeMemory(dev, img_mem, NULL);
   vkDestroyDevice(dev, NULL);
}

int
main(int argc, char **argv)
{
   bool shim = false;
   bool non_indexed_only = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--shim"))
         shim = true;
      else if (!strcmp(argv[i], "--non-indexed-only"))
         non_indexed_only = true;
   }

   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "g17p_vulkan_indirect_multi",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &app,
   };
   VkInstance inst = VK_NULL_HANDLE;
   CHECK(vkCreateInstance(&ici, NULL, &inst));

   uint32_t pdev_count = 0;
   CHECK(vkEnumeratePhysicalDevices(inst, &pdev_count, NULL));
   if (pdev_count == 0) {
      fprintf(stderr, "No Vulkan physical devices\n");
      return 1;
   }
   VkPhysicalDevice *pdevs = calloc(pdev_count, sizeof(*pdevs));
   CHECK(vkEnumeratePhysicalDevices(inst, &pdev_count, pdevs));
   VkPhysicalDevice pdev = pdevs[0];
   free(pdevs);

   uint32_t qf_count = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, NULL);
   VkQueueFamilyProperties *qfs = calloc(qf_count, sizeof(*qfs));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qfs);
   uint32_t qf = UINT32_MAX;
   for (uint32_t i = 0; i < qf_count; ++i) {
      if (qfs[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
         qf = i;
         break;
      }
   }
   free(qfs);
   if (qf == UINT32_MAX) {
      fprintf(stderr, "No graphics queue family\n");
      return 1;
   }

   run_suite(pdev, qf, true, shim, non_indexed_only);
   run_suite(pdev, qf, false, shim, non_indexed_only);

   vkDestroyInstance(inst, NULL);
   printf("PASS g17p_vulkan_indirect_multi (%s)\n", shim ? "shim" : "hw");
   return 0;
}
