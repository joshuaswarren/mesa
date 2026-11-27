/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Multisample tile reloads must be exact (mesa#1197). Draws N long, thin
 * triangles with additive blending into a 1920x1080 RGBA16F target with S
 * samples, resolved to 1x. Every fragment adds k/4096 (k = 1..4) to each
 * channel, so every sum is exact and independent of order: the same
 * triangles must give the same image however the work is split.
 *
 *   single     all N in one render pass (at large N its geometry overflows
 *              the parameter buffer, so the tile is stored and reloaded
 *              inside the pass)
 *   split      P render passes in one command buffer, each loading what the
 *              previous one stored
 *   splitsync  the same P passes, each submitted on its own with a wait
 *   loadonly   (S > 1) half in a clear+store pass, the other half in a pass
 *              that loads the samples without storing them
 *   loadempty  (S > 1) all N in a clear+store pass, then a pass that loads the
 *              samples and draws nothing (resolve only)
 *
 * Each runs twice. Every run's CRC32 of the resolved image (crc) must equal the
 * first single pass, and so must its CRC32 of every sample of the multisampled
 * image, [y][x][sample] RGBA16F read back with texelFetch (scrc), except after
 * loadonly, whose last pass does not store the samples.
 *
 *   aurora-msaa-reload [samples=4] [triangles=100000] [passes=50]
 *
 * Exit 0 = PASS, 1 = FAIL. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdbool.h>
#include <vulkan/vulkan.h>
#include "aurora_msaa_reload_shaders.h"
#define CK(x) do { VkResult r_ = (x); if (r_) { fprintf(stderr, "%s: %d\n", #x, r_); exit(2); } } while (0)
enum { W = 1920, H = 1080 };
static VkDevice dev; static VkPhysicalDeviceMemoryProperties mp;
static uint32_t mtype(uint32_t bits, VkMemoryPropertyFlags f)
{ for (uint32_t i = 0; i < mp.memoryTypeCount; i++) if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & f) == f) return i; exit(3); }
static VkShaderModule mod(const uint32_t *c, size_t n)
{ VkShaderModuleCreateInfo i = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = c};
  VkShaderModule m; CK(vkCreateShaderModule(dev, &i, NULL, &m)); return m; }
static VkImage image(VkSampleCountFlagBits s, VkImageUsageFlags u, VkImageView *view)
{
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R16G16B16A16_SFLOAT,
      .extent = {W, H, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = s, .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = u};
   VkImage img; CK(vkCreateImage(dev, &ci, NULL, &img)); VkMemoryRequirements r; vkGetImageMemoryRequirements(dev, img, &r);
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = r.size, .memoryTypeIndex = mtype(r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
   VkDeviceMemory m; CK(vkAllocateMemory(dev, &ai, NULL, &m)); CK(vkBindImageMemory(dev, img, m, 0));
   VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img, .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = ci.format, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
   CK(vkCreateImageView(dev, &vi, NULL, view)); return img;
}
static uint32_t crc32(const uint8_t *p, size_t n)
{
   static uint32_t t[256];
   if (!t[1]) for (uint32_t i = 0; i < 256; i++) { uint32_t c = i; for (int k = 0; k < 8; k++) c = c & 1 ? 0xedb88320u ^ (c >> 1) : c >> 1; t[i] = c; }
   uint32_t c = ~0u; for (size_t i = 0; i < n; i++) c = t[(c ^ p[i]) & 0xff] ^ (c >> 8); return ~c;
}
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }
int main(int argc, char **argv)
{
   const uint32_t S = argc > 1 ? strtoul(argv[1], NULL, 0) : 4;
   const uint32_t N = argc > 2 ? strtoul(argv[2], NULL, 0) : 100000;
   const uint32_t P = argc > 3 ? strtoul(argv[3], NULL, 0) : 50;
   if ((S != 1 && S != 4) || !N || P < 2) {
      fprintf(stderr, "usage: aurora-msaa-reload [samples=1|4] [triangles] [passes>=2]\n");
      return 2;
   }
   const VkSampleCountFlagBits SC = S == 1 ? VK_SAMPLE_COUNT_1_BIT : VK_SAMPLE_COUNT_4_BIT;
   const uint32_t natt = S == 1 ? 1 : 2;
   VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
   VkInstance inst; CK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd);
   VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd, &pp); vkGetPhysicalDeviceMemoryProperties(pd, &mp);
   fprintf(stderr, "device %s\n", pp.deviceName);
   float prio = 1; VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   CK(vkCreateDevice(pd, &dci, NULL, &dev)); VkQueue q; vkGetDeviceQueue(dev, 0, 0, &q);
   VkImageView msv, rsv = VK_NULL_HANDLE;
   VkImage msi = image(SC, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &msv);
   VkImage rs = S == 1 ? msi : image(VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, &rsv);
   const VkDeviceSize bytes = (VkDeviceSize)W * H * 8;
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buf; CK(vkCreateBuffer(dev, &bci, NULL, &buf)); VkMemoryRequirements r; vkGetBufferMemoryRequirements(dev, buf, &r);
   VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = r.size,
      .memoryTypeIndex = mtype(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
   VkDeviceMemory bm; CK(vkAllocateMemory(dev, &mai, NULL, &bm)); CK(vkBindBufferMemory(dev, buf, bm, 0));
   uint8_t *map; CK(vkMapMemory(dev, bm, 0, VK_WHOLE_SIZE, 0, (void **)&map));
   /* All samples, for scrc. */
   const VkDeviceSize sbytes = (VkDeviceSize)W * H * 4 * 8;
   VkBufferCreateInfo sbci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sbytes, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   VkBuffer sbuf; CK(vkCreateBuffer(dev, &sbci, NULL, &sbuf)); vkGetBufferMemoryRequirements(dev, sbuf, &r);
   mai.allocationSize = r.size; mai.memoryTypeIndex = mtype(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   VkDeviceMemory sbm; CK(vkAllocateMemory(dev, &mai, NULL, &sbm)); CK(vkBindBufferMemory(dev, sbuf, sbm, 0));
   uint8_t *smap; CK(vkMapMemory(dev, sbm, 0, VK_WHOLE_SIZE, 0, (void **)&smap));
   VkSamplerCreateInfo smci = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
   VkSampler smp; CK(vkCreateSampler(dev, &smci, NULL, &smp));
   VkDescriptorSetLayoutBinding dsb[2] = {{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
                                          {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT}};
   VkDescriptorSetLayoutCreateInfo dslci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = dsb};
   VkDescriptorSetLayout dsl; CK(vkCreateDescriptorSetLayout(dev, &dslci, NULL, &dsl));
   VkDescriptorPoolSize dps[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
   VkDescriptorPoolCreateInfo dpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 2, .pPoolSizes = dps};
   VkDescriptorPool dp; CK(vkCreateDescriptorPool(dev, &dpci, NULL, &dp));
   VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dp, .descriptorSetCount = 1, .pSetLayouts = &dsl};
   VkDescriptorSet ds; CK(vkAllocateDescriptorSets(dev, &dsai, &ds));
   VkDescriptorImageInfo dii = {smp, msv, VK_IMAGE_LAYOUT_GENERAL};
   VkDescriptorBufferInfo dbi = {sbuf, 0, VK_WHOLE_SIZE};
   VkWriteDescriptorSet wds[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &dii},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = ds, .dstBinding = 1, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi}};
   if (S > 1) vkUpdateDescriptorSets(dev, 2, wds, 0, NULL);
   VkPipelineLayoutCreateInfo cplci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl};
   VkPipelineLayout cpl; CK(vkCreatePipelineLayout(dev, &cplci, NULL, &cpl));
   VkComputePipelineCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = mod(msaa_reload_samples_comp, sizeof(msaa_reload_samples_comp)), .pName = "main"},
      .layout = cpl};
   VkPipeline cpipe = VK_NULL_HANDLE;
   if (S > 1) CK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &cpipe));
   /* Two compatible passes: clear and load; both store the samples and resolve. */
   VkRenderPass rp[3];
   for (int l = 0; l < 3; l++) {
      VkAttachmentDescription att[2] = {
         {.format = VK_FORMAT_R16G16B16A16_SFLOAT, .samples = SC, .loadOp = l ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = l == 2 ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = l ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_GENERAL},
         {.format = VK_FORMAT_R16G16B16A16_SFLOAT, .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}};
      VkAttachmentReference cref = {0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}, rref = {1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
      cref.layout = VK_IMAGE_LAYOUT_GENERAL;
      VkSubpassDescription sp = {.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1, .pColorAttachments = &cref,
         .pResolveAttachments = S == 1 ? NULL : &rref};
      VkSubpassDependency dep[2] = {
         {.srcSubpass = VK_SUBPASS_EXTERNAL, .dstSubpass = 0, .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT},
         {.srcSubpass = 0, .dstSubpass = VK_SUBPASS_EXTERNAL, .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
          .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
          .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT}};
      VkRenderPassCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = natt, .pAttachments = att, .subpassCount = 1,
         .pSubpasses = &sp, .dependencyCount = 2, .pDependencies = dep};
      CK(vkCreateRenderPass(dev, &ci, NULL, &rp[l]));
   }
   VkImageView views[2] = {msv, rsv};
   VkFramebufferCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = rp[0], .attachmentCount = natt, .pAttachments = views,
      .width = W, .height = H, .layers = 1};
   VkFramebuffer fb; CK(vkCreateFramebuffer(dev, &fci, NULL, &fb));
   VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT, 0, 4};
   VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout pl; CK(vkCreatePipelineLayout(dev, &plci, NULL, &pl));
   VkPipelineShaderStageCreateInfo st[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mod(msaa_reload_vert, sizeof(msaa_reload_vert)), .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = mod(msaa_reload_frag, sizeof(msaa_reload_frag)), .pName = "main"}};
   VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO, .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport vp = {0, 0, W, H, 0, 1}; VkRect2D sc = {{0, 0}, {W, H}};
   VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
   VkPipelineRasterizationStateCreateInfo rs_ = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1};
   VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO, .rasterizationSamples = SC};
   VkPipelineColorBlendAttachmentState ba = {.blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
      .colorBlendOp = VK_BLEND_OP_ADD, .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1, .pAttachments = &ba};
   VkGraphicsPipelineCreateInfo gci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2, .pStages = st, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vps, .pRasterizationState = &rs_, .pMultisampleState = &ms, .pColorBlendState = &cb,
      .layout = pl, .renderPass = rp[0]};
   VkPipeline pipe; CK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
   VkCommandPool pool; CK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool, .commandBufferCount = 1};
   VkCommandBuffer cmd; CK(vkAllocateCommandBuffers(dev, &cai, &cmd));
   uint32_t crc[5]; double ms_[5];
   const char *names[5] = {"single", "split", "splitsync", "loadonly", "loadempty"};
   const VkImageLayout src_layout = S == 1 ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
   uint32_t ref_crc = 0, ref_scrc = 0;
   bool pass = true;
   const int nmodes = S > 1 ? 5 : 3;
   for (int mode = 0; mode < nmodes; mode++) {
      const uint32_t passes = mode >= 3 ? 2 : mode ? P : 1;
      for (int rep = 0; rep < 2; rep++) {
         memset(map, 0xa5, bytes); memset(smap, 0xa5, sbytes);
         double total = 0;
         for (uint32_t p0 = 0; p0 < passes; p0 = mode >= 2 ? p0 + 1 : passes) {
            const uint32_t p1 = mode >= 2 ? p0 + 1 : passes;
            CK(vkResetCommandBuffer(cmd, 0));
            VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CK(vkBeginCommandBuffer(cmd, &bi));
            for (uint32_t p = p0; p < p1; p++) {
               uint32_t first = (uint64_t)N * p / passes, last = (uint64_t)N * (p + 1) / passes;
               if (mode == 4) { first = p ? N : 0; last = N; }
               VkClearValue cv = {0};
               VkRenderPassBeginInfo rbi = {.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = rp[p == 0 ? 0 : mode >= 3 ? 2 : 1], .framebuffer = fb,
                  .renderArea = {{0, 0}, {W, H}}, .clearValueCount = 1, .pClearValues = &cv};
               vkCmdBeginRenderPass(cmd, &rbi, VK_SUBPASS_CONTENTS_INLINE);
               vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
               vkCmdPushConstants(cmd, pl, VK_SHADER_STAGE_VERTEX_BIT, 0, 4, &first);
               if (last > first) vkCmdDraw(cmd, 3 * (last - first), 1, 0, 0);
               vkCmdEndRenderPass(cmd);
            }
            if (p1 == passes) {
               VkBufferImageCopy bic = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {W, H, 1}};
               vkCmdCopyImageToBuffer(cmd, rs, src_layout, buf, 1, &bic);
               if (S > 1) {
                  VkMemoryBarrier cb0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, .dstAccessMask = VK_ACCESS_SHADER_READ_BIT};
                  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &cb0, 0, NULL, 0, NULL);
                  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpipe);
                  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, cpl, 0, 1, &ds, 0, NULL);
                  vkCmdDispatch(cmd, (W * H + 63) / 64, 1, 1);
               }
               VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT, .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
               vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
            }
            CK(vkEndCommandBuffer(cmd));
            VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd};
            const double t0 = now_ms();
            CK(vkQueueSubmit(q, 1, &si, VK_NULL_HANDLE)); CK(vkQueueWaitIdle(q));
            total += now_ms() - t0;
         }
         ms_[mode] = total;
         crc[mode] = crc32(map, bytes);
         const uint32_t scrc = S > 1 ? crc32(smap, sbytes) : crc[mode];
         uint64_t lit = 0; const uint16_t *px = (const uint16_t *)map;
         for (size_t i = 0; i < (size_t)W * H; i++) lit += px[4 * i + 3] != 0;
         printf("%-9s N=%u passes=%u samples=%u rep=%d crc=%08x scrc=%08x ms=%.1f lit=%.3f\n", names[mode], N, passes, S, rep, crc[mode], scrc, ms_[mode],
                (double)lit / ((double)W * H));
         if (mode == 0 && rep == 0) {
            ref_crc = crc[mode];
            ref_scrc = scrc;
         } else if (crc[mode] != ref_crc || (mode != 3 && scrc != ref_scrc)) {
            pass = false;
         }
         fflush(stdout);
      }
   }
   printf("%s\n", pass ? "PASS" : "FAIL");
   return pass ? 0 : 1;
}
