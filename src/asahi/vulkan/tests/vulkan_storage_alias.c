/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Shader stores into a storage image whose memory, compression metadata
 * included, another alias overwrote.
 *
 * vkd3d-proton places D3D12 resources with VK_IMAGE_CREATE_ALIAS_BIT and
 * transitions one from UNDEFINED only at its activation (clear, discard, full
 * copy). A UAV-only texture that a dispatch writes after an aliasing barrier
 * therefore meets whatever compression metadata the previous alias left.
 *
 * Per format: image S (2D, one layer, STORAGE|SAMPLED|TRANSFER|COLOR_ATTACHMENT,
 * ALIAS, and with a same-layout format list when the format has more than one
 * view format) and a reference R (two layers, so never compressed). Each run:
 *   1. UNDEFINED -> GENERAL, store pattern A into every texel of S and R
 *   2. overwrite S's whole memory range through an aliasing buffer (random
 *      words, all ones, all zeros, 0x5a), with no layout transition
 *   3. store pattern B into S and R: about one texel in seven (every
 *      compression tile partly written), a checkerboard, or every texel
 *   4. read S and R back; every texel step 3 stored must be bit-identical
 * The "init" variant adds the UNDEFINED -> GENERAL transition after step 2,
 * the path valid Vulkan takes.
 *
 * Run it with and without the driver compressing the images
 * (ASAHI_MESA_DEBUG=nocompress); AGX_MESA_DEBUG=perf shows which are.
 *
 * usage: vulkan-storage-alias [format-filter]
 * Exit status 0 = every case exact, 1 = a mismatch, 2 = setup failure.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_storage_alias_shaders.h"

#define CHECK(x)                                                                         \
   do {                                                                                  \
      VkResult r_ = (x);                                                                 \
      if (r_ != VK_SUCCESS) {                                                            \
         fprintf(stderr, "FAIL %s:%d %s = %d\n", __FILE__, __LINE__, #x, r_);            \
         exit(2);                                                                        \
      }                                                                                  \
   } while (0)

static VkInstance inst;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkCommandPool pool;
static VkPhysicalDeviceMemoryProperties memprops;

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
      if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type for bits 0x%x flags 0x%x\n", bits, want);
   exit(2);
}

static VkCommandBuffer
begin(void)
{
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cb));
   CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT}));
   return cb;
}

static void
submit(VkCommandBuffer cb)
{
   CHECK(vkEndCommandBuffer(cb));
   CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1, .pCommandBuffers = &cb}, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static void
barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
   vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1, .pMemoryBarriers = &mb});
}

static void
transition(VkCommandBuffer cb, VkImage img, VkImageLayout old_layout,
           VkImageLayout new_layout, uint32_t layers)
{
   VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .oldLayout = old_layout, .newLayout = new_layout,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
   vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib});
}

static void
to_general(VkCommandBuffer cb, VkImage img, uint32_t layers)
{
   transition(cb, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, layers);
}

struct fmt {
   VkFormat f;
   const char *name;
   unsigned bpp;
   VkFormat list[5];
   unsigned nlist;
};

static const struct fmt fmts[] = {
   {VK_FORMAT_R16G16B16A16_SFLOAT, "rgba16f", 8,
    {VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_UINT,
     VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_SINT}, 5},
   {VK_FORMAT_R16G16_SFLOAT, "rg16f", 4,
    {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_UINT}, 3},
   {VK_FORMAT_R8G8B8A8_UNORM, "rgba8", 4, {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UINT}, 2},
   {VK_FORMAT_B8G8R8A8_UNORM, "bgra8", 4, {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB}, 2},
   {VK_FORMAT_R8G8_UNORM, "rg8", 2, {VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT}, 3},
   {VK_FORMAT_R8_UNORM, "r8", 1, {VK_FORMAT_R8_UNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT}, 3},
   {VK_FORMAT_B10G11R11_UFLOAT_PACK32, "r11g11b10f", 4, {VK_FORMAT_B10G11R11_UFLOAT_PACK32}, 1},
   {VK_FORMAT_R32G32B32A32_SFLOAT, "rgba32f", 16, {VK_FORMAT_R32G32B32A32_SFLOAT}, 1},
   {VK_FORMAT_R32G32_SFLOAT, "rg32f", 8, {VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT}, 3},
};

static VkPipeline pipe;
static VkPipelineLayout playout;
static VkDescriptorSetLayout dsl;
static VkDescriptorPool dpool;

struct pc { uint32_t seed, mode, w, h; };

static void
store(VkCommandBuffer cb, VkDescriptorSet set, uint32_t seed, uint32_t mode, uint32_t w, uint32_t h)
{
   struct pc pc = {seed, mode, w, h};
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, playout, 0, 1, &set, 0, NULL);
   vkCmdPushConstants(cb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
   vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
}

static bool
written(uint32_t x, uint32_t y, uint32_t seed, uint32_t mode)
{
   uint32_t h = (x * 73856093u) ^ (y * 19349663u) ^ (seed * 83492791u);
   if (mode == 1)
      return (h % 7u) == 0;
   if (mode == 2)
      return ((x + y) & 1u) == 0;
   return true;
}

static VkDescriptorSet
make_set(VkImageView view)
{
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = dpool,
      .descriptorSetCount = 1, .pSetLayouts = &dsl}, &set));
   VkDescriptorImageInfo ii = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
   vkUpdateDescriptorSets(dev, 1, &(VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set, .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &ii}, 0, NULL);
   return set;
}

static uint32_t rng = 0x12345678;
static uint32_t
xs(void)
{
   rng ^= rng << 13;
   rng ^= rng >> 17;
   rng ^= rng << 5;
   return rng;
}

/* returns mismatching written texels */
static unsigned
run_case(const struct fmt *F, bool mutable_list, bool init_after_garbage, unsigned garbage, uint32_t mode,
         uint32_t w, uint32_t h, uint32_t seed, VkDeviceSize *size_out)
{
   VkImageFormatListCreateInfo list = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .viewFormatCount = F->nlist, .pViewFormats = F->list};
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .pNext = mutable_list ? &list : NULL,
      .flags = VK_IMAGE_CREATE_ALIAS_BIT | (mutable_list ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0),
      .imageType = VK_IMAGE_TYPE_2D, .format = F->f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage S, R;
   CHECK(vkCreateImage(dev, &ci, NULL, &S));
   VkImageCreateInfo rci = ci;
   rci.pNext = NULL;
   rci.flags = 0;
   rci.arrayLayers = 2; /* never compressed as a storage image */
   CHECK(vkCreateImage(dev, &rci, NULL, &R));

   VkMemoryRequirements sreq, rreq;
   vkGetImageMemoryRequirements(dev, S, &sreq);
   vkGetImageMemoryRequirements(dev, R, &rreq);
   *size_out = sreq.size;

   VkBuffer B;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sreq.size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT},
      NULL, &B));
   VkMemoryRequirements breq;
   vkGetBufferMemoryRequirements(dev, B, &breq);
   VkDeviceMemory mS, mR;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = sreq.size > breq.size ? sreq.size : breq.size,
      .memoryTypeIndex = mem_type(sreq.memoryTypeBits & breq.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)}, NULL, &mS));
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = rreq.size,
      .memoryTypeIndex = mem_type(rreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)}, NULL, &mR));
   CHECK(vkBindImageMemory(dev, S, mS, 0));
   CHECK(vkBindBufferMemory(dev, B, mS, 0));
   CHECK(vkBindImageMemory(dev, R, mR, 0));

   VkImageView vS, vR;
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = S, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = F->f,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &vS));
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = R, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = F->f,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &vR));
   VkDescriptorSet sS = make_set(vS), sR = make_set(vR);

   /* host staging: garbage source and two readbacks */
   const VkDeviceSize plane = (VkDeviceSize)w * h * F->bpp;
   const VkDeviceSize hsz = sreq.size + 2 * plane;
   VkBuffer H;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = hsz, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT}, NULL, &H));
   VkMemoryRequirements hreq;
   vkGetBufferMemoryRequirements(dev, H, &hreq);
   VkDeviceMemory mH;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = hreq.size,
      .memoryTypeIndex = mem_type(hreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &mH));
   CHECK(vkBindBufferMemory(dev, H, mH, 0));
   uint8_t *hp;
   CHECK(vkMapMemory(dev, mH, 0, VK_WHOLE_SIZE, 0, (void **)&hp));
   uint32_t *g = (uint32_t *)hp;
   for (VkDeviceSize i = 0; i < sreq.size / 4; ++i)
      g[i] = garbage == 0 ? xs() : garbage == 1 ? 0xffffffffu : garbage == 2 ? 0 : 0x5a5a5a5au;

   VkCommandBuffer cb = begin();
   to_general(cb, S, 1);
   to_general(cb, R, 2);
   store(cb, sS, seed, 0, w, h);
   store(cb, sR, seed, 0, w, h);
   barrier(cb);
   vkCmdCopyBuffer(cb, H, B, 1, &(VkBufferCopy){0, 0, sreq.size});
   barrier(cb);
   if (init_after_garbage)
      to_general(cb, S, 1);
   store(cb, sS, seed + 1000, mode, w, h);
   store(cb, sR, seed + 1000, mode, w, h);
   barrier(cb);
   VkBufferImageCopy rc = {.bufferOffset = sreq.size, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           .imageExtent = {w, h, 1}};
   vkCmdCopyImageToBuffer(cb, S, VK_IMAGE_LAYOUT_GENERAL, H, 1, &rc);
   rc.bufferOffset = sreq.size + plane;
   vkCmdCopyImageToBuffer(cb, R, VK_IMAGE_LAYOUT_GENERAL, H, 1, &rc);
   submit(cb);

   unsigned bad = 0, total = 0, first = 1;
   for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x) {
         if (!written(x, y, seed + 1000, mode))
            continue;
         total++;
         const uint8_t *a = hp + sreq.size + ((size_t)y * w + x) * F->bpp;
         const uint8_t *b = hp + sreq.size + plane + ((size_t)y * w + x) * F->bpp;
         if (memcmp(a, b, F->bpp)) {
            if (first && getenv("SA_VERBOSE")) {
               printf("   first mismatch %u,%u:", x, y);
               for (unsigned k = 0; k < F->bpp; ++k) printf(" %02x/%02x", a[k], b[k]);
               printf("\n");
               first = 0;
            }
            bad++;
         }
      }
   (void)total;

   vkUnmapMemory(dev, mH);
   vkDestroyBuffer(dev, H, NULL);
   vkFreeMemory(dev, mH, NULL);
   vkDestroyImageView(dev, vS, NULL);
   vkDestroyImageView(dev, vR, NULL);
   vkDestroyBuffer(dev, B, NULL);
   vkDestroyImage(dev, S, NULL);
   vkDestroyImage(dev, R, NULL);
   vkFreeMemory(dev, mS, NULL);
   vkFreeMemory(dev, mR, NULL);
   CHECK(vkResetDescriptorPool(dev, dpool, 0));
   return bad;
}

/* Exercises all 256 metadata byte patterns (0x00..0xff) followed by:
 *   - two sequential partial compute imageStore dispatches (sparse + checker),
 *   - UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL partial-renderArea LOAD_OP_DONT_CARE
 *     render pass over unaligned coordinates, and
 *   - UNDEFINED -> GENERAL partial compute imageStore -> LOAD_OP_LOAD render pass
 *     updating an unaligned sub-rectangle while preserving compute-written texels.
 * Returns total mismatching texels across all three subcases. */
static unsigned
run_extended_cases(const struct fmt *F, bool renderable)
{
   const uint32_t w = 300, h = 180;
   VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = VK_IMAGE_CREATE_ALIAS_BIT,
      .imageType = VK_IMAGE_TYPE_2D, .format = F->f, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT |
               (renderable ? VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT : 0),
      .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
   VkImage S, R;
   CHECK(vkCreateImage(dev, &ci, NULL, &S));
   VkImageCreateInfo rci = ci;
   rci.flags = 0;
   rci.arrayLayers = 2;
   CHECK(vkCreateImage(dev, &rci, NULL, &R));

   VkMemoryRequirements sreq, rreq;
   vkGetImageMemoryRequirements(dev, S, &sreq);
   vkGetImageMemoryRequirements(dev, R, &rreq);

   VkBuffer B;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = sreq.size, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT},
      NULL, &B));
   VkMemoryRequirements breq;
   vkGetBufferMemoryRequirements(dev, B, &breq);
   VkDeviceMemory mS, mR;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = sreq.size > breq.size ? sreq.size : breq.size,
      .memoryTypeIndex = mem_type(sreq.memoryTypeBits & breq.memoryTypeBits,
                                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)}, NULL, &mS));
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = rreq.size,
      .memoryTypeIndex = mem_type(rreq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)}, NULL, &mR));
   CHECK(vkBindImageMemory(dev, S, mS, 0));
   CHECK(vkBindBufferMemory(dev, B, mS, 0));
   CHECK(vkBindImageMemory(dev, R, mR, 0));

   VkImageView vS, vR;
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = S, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = F->f,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &vS));
   CHECK(vkCreateImageView(dev, &(VkImageViewCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = R, .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = F->f,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &vR));
   VkDescriptorSet sS = make_set(vS), sR = make_set(vR);

   const VkDeviceSize plane = (VkDeviceSize)w * h * F->bpp;
   const VkDeviceSize hsz = sreq.size + 2 * plane;
   VkBuffer H;
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = hsz, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT}, NULL, &H));
   VkMemoryRequirements hreq;
   vkGetBufferMemoryRequirements(dev, H, &hreq);
   VkDeviceMemory mH;
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = hreq.size,
      .memoryTypeIndex = mem_type(hreq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)}, NULL, &mH));
   CHECK(vkBindBufferMemory(dev, H, mH, 0));
   uint8_t *hp;
   CHECK(vkMapMemory(dev, mH, 0, VK_WHOLE_SIZE, 0, (void **)&hp));

   /* Fill every 8-byte word with repeated byte 0x00..0xff in turn, so every
    * metadata tile mode encoding is hit across the surface. */
   uint64_t *g64 = (uint64_t *)hp;
   for (VkDeviceSize i = 0; i < sreq.size / 8; ++i)
      g64[i] = (uint64_t)(i & 0xffu) * UINT64_C(0x0101010101010101);

   unsigned bad = 0;
   VkBufferImageCopy rcS = {.bufferOffset = sreq.size,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};
   VkBufferImageCopy rcR = {.bufferOffset = sreq.size + plane,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {w, h, 1}};

   /* Subcase 1: all-256 byte sweep -> UNDEFINED -> GENERAL -> two sequential
    * partial compute stores (sparse + checker). */
   {
      VkCommandBuffer cb = begin();
      vkCmdCopyBuffer(cb, H, B, 1, &(VkBufferCopy){0, 0, sreq.size});
      barrier(cb);
      to_general(cb, S, 1);
      to_general(cb, R, 2);
      store(cb, sS, 101, 1, w, h);
      store(cb, sR, 101, 1, w, h);
      barrier(cb);
      store(cb, sS, 202, 2, w, h);
      store(cb, sR, 202, 2, w, h);
      barrier(cb);
      vkCmdCopyImageToBuffer(cb, S, VK_IMAGE_LAYOUT_GENERAL, H, 1, &rcS);
      vkCmdCopyImageToBuffer(cb, R, VK_IMAGE_LAYOUT_GENERAL, H, 1, &rcR);
      submit(cb);

      for (uint32_t y = 0; y < h; ++y)
         for (uint32_t x = 0; x < w; ++x) {
            if (!written(x, y, 101, 1) && !written(x, y, 202, 2))
               continue;
            const uint8_t *a = hp + sreq.size + ((size_t)y * w + x) * F->bpp;
            const uint8_t *b = hp + sreq.size + plane + ((size_t)y * w + x) * F->bpp;
            if (memcmp(a, b, F->bpp))
               bad++;
         }
   }

   if (renderable) {
      /* Subcase 2: all-256 byte sweep -> UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
       * -> partial unaligned renderArea LOAD_OP_DONT_CARE + clear draw. */
      const VkRect2D area = {.offset = {3, 5}, .extent = {131, 73}};
      const VkClearAttachment catt = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
         .clearValue = {.color = {.float32 = {0.25f, 0.5f, 0.75f, 1.0f}}},
      };
      const VkClearRect crect = {.rect = area, .baseArrayLayer = 0, .layerCount = 1};
      VkCommandBuffer cb = begin();
      vkCmdCopyBuffer(cb, H, B, 1, &(VkBufferCopy){0, 0, sreq.size});
      barrier(cb);
      transition(cb, S, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
      transition(cb, R, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 2);
      VkRenderingAttachmentInfo att = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = vS, .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
      VkRenderingInfo ri = {
         .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
         .renderArea = area, .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att,
      };
      vkCmdBeginRendering(cb, &ri);
      vkCmdClearAttachments(cb, 1, &catt, 1, &crect);
      vkCmdEndRendering(cb);
      att.imageView = vR;
      vkCmdBeginRendering(cb, &ri);
      vkCmdClearAttachments(cb, 1, &catt, 1, &crect);
      vkCmdEndRendering(cb);
      transition(cb, S, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);
      transition(cb, R, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 2);
      vkCmdCopyImageToBuffer(cb, S, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, H, 1, &rcS);
      vkCmdCopyImageToBuffer(cb, R, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, H, 1, &rcR);
      submit(cb);

      for (uint32_t y = (uint32_t)area.offset.y; y < (uint32_t)area.offset.y + area.extent.height; ++y)
         for (uint32_t x = (uint32_t)area.offset.x; x < (uint32_t)area.offset.x + area.extent.width; ++x) {
            const uint8_t *a = hp + sreq.size + ((size_t)y * w + x) * F->bpp;
            const uint8_t *b = hp + sreq.size + plane + ((size_t)y * w + x) * F->bpp;
            if (memcmp(a, b, F->bpp))
               bad++;
         }

      /* Subcase 3: all-256 byte sweep -> UNDEFINED -> GENERAL -> partial compute
       * imageStore -> GENERAL -> COLOR_ATTACHMENT_OPTIMAL -> LOAD_OP_LOAD render
       * pass updating an unaligned sub-rectangle. */
      const VkRect2D sub = {.offset = {19, 13}, .extent = {97, 61}};
      const VkClearAttachment catt2 = {
         .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = 0,
         .clearValue = {.color = {.float32 = {0.75f, 0.25f, 0.5f, 1.0f}}},
      };
      const VkClearRect crect2 = {.rect = sub, .baseArrayLayer = 0, .layerCount = 1};
      cb = begin();
      vkCmdCopyBuffer(cb, H, B, 1, &(VkBufferCopy){0, 0, sreq.size});
      barrier(cb);
      to_general(cb, S, 1);
      to_general(cb, R, 2);
      store(cb, sS, 303, 2, w, h);
      store(cb, sR, 303, 2, w, h);
      transition(cb, S, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 1);
      transition(cb, R, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, 2);
      att.imageView = vS;
      att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      ri.renderArea = (VkRect2D){{0, 0}, {w, h}};
      vkCmdBeginRendering(cb, &ri);
      vkCmdClearAttachments(cb, 1, &catt2, 1, &crect2);
      vkCmdEndRendering(cb);
      att.imageView = vR;
      vkCmdBeginRendering(cb, &ri);
      vkCmdClearAttachments(cb, 1, &catt2, 1, &crect2);
      vkCmdEndRendering(cb);
      transition(cb, S, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 1);
      transition(cb, R, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, 2);
      vkCmdCopyImageToBuffer(cb, S, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, H, 1, &rcS);
      vkCmdCopyImageToBuffer(cb, R, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, H, 1, &rcR);
      submit(cb);

      for (uint32_t y = 0; y < h; ++y)
         for (uint32_t x = 0; x < w; ++x) {
            const bool in_sub =
               x >= (uint32_t)sub.offset.x && x < (uint32_t)sub.offset.x + sub.extent.width &&
               y >= (uint32_t)sub.offset.y && y < (uint32_t)sub.offset.y + sub.extent.height;
            if (!in_sub && !written(x, y, 303, 2))
               continue;
            const uint8_t *a = hp + sreq.size + ((size_t)y * w + x) * F->bpp;
            const uint8_t *b = hp + sreq.size + plane + ((size_t)y * w + x) * F->bpp;
            if (memcmp(a, b, F->bpp))
               bad++;
         }
   }

   vkUnmapMemory(dev, mH);
   vkDestroyBuffer(dev, H, NULL);
   vkFreeMemory(dev, mH, NULL);
   vkDestroyImageView(dev, vS, NULL);
   vkDestroyImageView(dev, vR, NULL);
   vkDestroyBuffer(dev, B, NULL);
   vkDestroyImage(dev, S, NULL);
   vkDestroyImage(dev, R, NULL);
   vkFreeMemory(dev, mS, NULL);
   vkFreeMemory(dev, mR, NULL);
   CHECK(vkResetDescriptorPool(dev, dpool, 0));
   return bad;
}

int
main(int argc, char **argv)
{
   const char *only = argc > 1 ? argv[1] : NULL;
   CHECK(vkCreateInstance(&(VkInstanceCreateInfo){.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                               .apiVersion = VK_API_VERSION_1_3}}, NULL, &inst));
   uint32_t n = 1;
   vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if (!n) {
      fprintf(stderr, "no device\n");
      return 2;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);
   uint32_t nq = 8;
   VkQueueFamilyProperties qf[8];
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qf);
   for (qfam = 0; qfam < nq && (qf[qfam].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) !=
                                  (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
        ++qfam)
      ;
   float prio = 1.0f;
   VkPhysicalDeviceVulkan13Features f13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                           .synchronization2 = VK_TRUE,
                                           .dynamicRendering = VK_TRUE};
   VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f13,
      .features = {.shaderStorageImageWriteWithoutFormat = VK_TRUE}};
   CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f2,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = qfam, .queueCount = 1,
         .pQueuePriorities = &prio}}, NULL, &dev));
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = qfam}, NULL, &pool));

   CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){.binding = 0,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}}, NULL, &dsl));
   CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 4, .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4}}, NULL, &dpool));
   CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct pc)}},
      NULL, &playout));
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(store_comp), .pCode = store_comp}, NULL, &sm));
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
      .layout = playout}, NULL, &pipe));

   printf("DEVICE %s\n", props.deviceName);
   static const char *gname[] = {"random", "ones", "zeros", "5a"};
   static const char *mname[] = {"full", "sparse", "checker"};
   unsigned fails = 0, cases = 0;
   for (unsigned fi = 0; fi < sizeof(fmts) / sizeof(fmts[0]); ++fi) {
      const struct fmt *F = &fmts[fi];
      if (only && !strstr(F->name, only))
         continue;
      VkFormatProperties fp;
      vkGetPhysicalDeviceFormatProperties(pdev, F->f, &fp);
      if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) {
         printf("SKIP %s: no storage\n", F->name);
         continue;
      }
      for (int ml = 0; ml < 2; ++ml) {
         if (ml && F->nlist < 2)
            continue;
         for (int init = 0; init < 2; ++init) {
            unsigned bad_total = 0, runs = 0;
            VkDeviceSize sz = 0;
            for (unsigned gi = 0; gi < 4; ++gi)
               for (uint32_t mode = 0; mode < 3; ++mode)
                  for (uint32_t seed = 1; seed <= 3; ++seed) {
                     const uint32_t w = seed == 2 ? 300 : 512, h = seed == 3 ? 77 : 256;
                     unsigned bad = run_case(F, ml, init, gi, mode, w, h, seed * 17 + gi, &sz);
                     runs++;
                     if (bad) {
                        printf("  MISMATCH %s %s %s garbage=%s mode=%s %ux%u seed=%u: %u texels\n", F->name,
                               ml ? "mutable+list" : "plain", init ? "init" : "no-init", gname[gi],
                               mname[mode], w, h, seed, bad);
                        bad_total += bad;
                     }
                  }
            cases++;
            if (bad_total)
               fails++;
            printf("RESULT %-10s %-12s %-7s size=%llu runs=%u mismatching_texels=%u %s\n", F->name,
                   ml ? "mutable+list" : "plain", init ? "init" : "no-init", (unsigned long long)sz, runs,
                   bad_total, bad_total ? "FAIL" : "PASS");
         }
      }
      const bool renderable =
         (fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT) != 0;
      unsigned ext_bad = run_extended_cases(F, renderable);
      cases++;
      if (ext_bad)
         fails++;
      printf("RESULT %-10s %-12s %-7s mismatching_texels=%u %s\n", F->name,
             "byte-sweep", renderable ? "cs+gfx" : "cs-only", ext_bad,
             ext_bad ? "FAIL" : "PASS");
   }
   printf("SUMMARY %u/%u case groups pass\n", cases - fails, cases);
   return fails ? 1 : 0;
}
