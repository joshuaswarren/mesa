/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Visibility test for non-coherent compute device stores (AGX3 stdev f75=1 /
 * AGX_CAP_COMPUTE_COHERENT_STORE on G17P).
 *
 * Exercises:
 *   1. Post-barrier inter-dispatch visibility (pre-warm stale L1 lines ->
 *      WAR barrier -> non-coherent producer SSBO stores in u32/uvec2/uvec4 ->
 *      Vulkan memory barrier -> shifted-workgroup consumer dispatch), across
 *      L1-sized (16 KiB), L2-sized (1 MiB), and SLC-spilling (16 MiB) buffers,
 *      with VkMemoryBarrier, VkBufferMemoryBarrier, vkCmdPipelineBarrier2, and
 *      separate-queue-submit boundaries plus host readback (SHADER_WRITE ->
 *      HOST_READ).
 *   2. Same-thread RAW (runtime-zero offset reload immediately after stdev)
 *      and intra-workgroup cross-thread RAW after memoryBarrierBuffer() +
 *      barrier().
 *   3. Cross-workgroup visibility within a single dispatch: producer
 *      workgroups write non-coherent payload SSBOs, execute
 *      memoryBarrierBuffer() + barrier(), and publish an atomic flag;
 *      consumer workgroups observe the flag, execute memoryBarrierBuffer(),
 *      and verify every word across 64 ping-pong rounds.
 *
 * Exit status: 0 = all checks pass, 1 = mismatch/visibility failure, 2 = setup.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_compute_store_visibility_shaders.h"

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

struct buf {
   VkBuffer b;
   VkDeviceMemory m;
   VkDeviceSize size;
   void *map;
};

static uint32_t
mem_type(uint32_t bits, VkMemoryPropertyFlags want)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; ++i)
      if ((bits & (1u << i)) &&
          (memprops.memoryTypes[i].propertyFlags & want) == want)
         return i;
   fprintf(stderr, "no memory type for bits 0x%x flags 0x%x\n", bits, want);
   exit(2);
}

static struct buf
make_buf(VkDeviceSize size, VkBufferUsageFlags usage,
         VkMemoryPropertyFlags props, bool map_it)
{
   struct buf out = { .size = size };
   CHECK(vkCreateBuffer(dev, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
      .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
   }, NULL, &out.b));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, out.b, &req);
   CHECK(vkAllocateMemory(dev, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex = mem_type(req.memoryTypeBits, props),
   }, NULL, &out.m));
   CHECK(vkBindBufferMemory(dev, out.b, out.m, 0));
   if (map_it)
      CHECK(vkMapMemory(dev, out.m, 0, size, 0, &out.map));
   return out;
}

static void
destroy_buf(struct buf *b)
{
   if (b->map)
      vkUnmapMemory(dev, b->m);
   vkDestroyBuffer(dev, b->b, NULL);
   vkFreeMemory(dev, b->m, NULL);
   memset(b, 0, sizeof(*b));
}

static VkCommandBuffer
begin_cb(void)
{
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   }, &cb));
   CHECK(vkBeginCommandBuffer(cb, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   }));
   return cb;
}

static void
submit_cb(VkCommandBuffer cb)
{
   CHECK(vkEndCommandBuffer(cb));
   CHECK(vkQueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cb,
   }, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

static uint32_t
host_pat32(uint32_t idx, uint32_t r)
{
   uint32_t x = (idx * 2654435761u) ^ (r * 2246822519u) ^ 0x9e3779b9u;
   x ^= x >> 16;
   x *= 0x7feb352du;
   x ^= x >> 15;
   return x | 1u;
}

enum barrier_kind {
   BARRIER_GLOBAL_V1 = 0,
   BARRIER_BUFFER_V1 = 1,
   BARRIER_GLOBAL_V2 = 2,
   BARRIER_SEPARATE_SUBMIT = 3,
};

struct barrier_pc {
   uint32_t mode;
   uint32_t count;
   uint32_t round_id;
   uint32_t shift;
};

static void
emit_cs_barrier(VkCommandBuffer cb, enum barrier_kind kind,
                VkAccessFlags src_acc, VkAccessFlags dst_acc,
                VkBuffer b0, VkBuffer b1, VkBuffer b2)
{
   if (kind == BARRIER_GLOBAL_V1) {
      VkMemoryBarrier mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
         .srcAccessMask = src_acc,
         .dstAccessMask = dst_acc,
      };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           1, &mb, 0, NULL, 0, NULL);
   } else if (kind == BARRIER_BUFFER_V1) {
      VkBufferMemoryBarrier bb[3] = {
         { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
           .srcAccessMask = src_acc, .dstAccessMask = dst_acc,
           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
           .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
           .buffer = b0, .offset = 0, .size = VK_WHOLE_SIZE },
         { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
           .srcAccessMask = src_acc, .dstAccessMask = dst_acc,
           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
           .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
           .buffer = b1, .offset = 0, .size = VK_WHOLE_SIZE },
         { .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
           .srcAccessMask = src_acc, .dstAccessMask = dst_acc,
           .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
           .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
           .buffer = b2, .offset = 0, .size = VK_WHOLE_SIZE },
      };
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           0, NULL, 3, bb, 0, NULL);
   } else if (kind == BARRIER_GLOBAL_V2) {
      VkMemoryBarrier2 mb = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .srcAccessMask = src_acc,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstAccessMask = dst_acc,
      };
      vkCmdPipelineBarrier2(cb, &(VkDependencyInfo){
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1,
         .pMemoryBarriers = &mb,
      });
   }
}

static unsigned
run_post_barrier_suite(void)
{
   VkDescriptorSetLayoutBinding binds[4];
   for (uint32_t i = 0; i < 4; ++i) {
      binds[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 4, .pBindings = binds,
   }, NULL, &dsl));
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 },
   }, NULL, &dpool));
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   }, &ds));
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(struct barrier_pc) },
   }, NULL, &pl));
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(barrier_comp), .pCode = barrier_comp,
   }, NULL, &sm));
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1,
      &(VkComputePipelineCreateInfo){
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm, .pName = "main",
         },
         .layout = pl,
      }, NULL, &pipe));

   const VkMemoryPropertyFlags host_vis =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   struct buf res_buf = make_buf(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 host_vis, true);

   const struct {
      const char *name;
      uint32_t count; /* elements of u32/uvec2/uvec4 */
      bool check_host;
   } sizes[] = {
      { "l1_16k",   4096u,   true  },
      { "l2_1m",    65536u,  true  },
      { "slc_16m",  524288u, false }, /* 524288 * (4+8+16) = 14.7 MiB total */
   };
   static const char *bnames[] = {
      "global_v1", "buffer_v1", "global_v2", "separate_submit"
   };

   unsigned fails = 0;
   uint32_t round_id = 1;

   for (unsigned si = 0; si < sizeof(sizes) / sizeof(sizes[0]); ++si) {
      const uint32_t n = sizes[si].count;
      const uint32_t num_wgs = n / 64u;
      const VkMemoryPropertyFlags buf_props =
         sizes[si].check_host ? host_vis : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
      struct buf b32 = make_buf((VkDeviceSize)n * 4u,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                buf_props, sizes[si].check_host);
      struct buf b64 = make_buf((VkDeviceSize)n * 8u,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                buf_props, false);
      struct buf b128 = make_buf((VkDeviceSize)n * 16u,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                 buf_props, false);

      VkDescriptorBufferInfo bi[4] = {
         { b32.b, 0, VK_WHOLE_SIZE },
         { b64.b, 0, VK_WHOLE_SIZE },
         { b128.b, 0, VK_WHOLE_SIZE },
         { res_buf.b, 0, VK_WHOLE_SIZE },
      };
      VkWriteDescriptorSet w[4];
      for (uint32_t i = 0; i < 4; ++i) {
         w[i] = (VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &bi[i],
         };
      }
      vkUpdateDescriptorSets(dev, 4, w, 0, NULL);

      /* First run mode 3: same-thread RAW + intra-workgroup RAW. */
      memset(res_buf.map, 0, 16);
      {
         VkCommandBuffer cb = begin_cb();
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
         vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                 0, 1, &ds, 0, NULL);
         struct barrier_pc pc = { 3u, n, round_id++, 0u };
         vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
         vkCmdDispatch(cb, num_wgs, 1, 1);
         submit_cb(cb);
      }
      uint32_t *rm = (uint32_t *)res_buf.map;
      if (rm[0] != 0) {
         fprintf(stderr, "FAIL same_core_and_wg_raw %s errs=%u idx=0x%x got=0x%x want=0x%x\n",
                 sizes[si].name, rm[0], rm[1], rm[2], rm[3]);
         fails++;
      } else {
         printf("PASS same_core_and_wg_raw %-8s count=%u\n", sizes[si].name, n);
      }

      /* Now run all 4 barrier kinds x 3 workgroup shifts. */
      const uint32_t shifts[3] = { 1u, 7u, (num_wgs / 2u) | 1u };
      for (unsigned bk = 0; bk < 4; ++bk) {
         unsigned bk_errs = 0;
         for (unsigned sh = 0; sh < 3; ++sh) {
            memset(res_buf.map, 0, 16);
            const uint32_t r_cur = round_id++;
            if (bk == BARRIER_SEPARATE_SUBMIT) {
               VkCommandBuffer cb0 = begin_cb();
               vkCmdBindPipeline(cb0, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
               vkCmdBindDescriptorSets(cb0, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                       0, 1, &ds, 0, NULL);
               struct barrier_pc pc0 = { 0u, n, r_cur - 1u, 0u };
               vkCmdPushConstants(cb0, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc0), &pc0);
               vkCmdDispatch(cb0, num_wgs, 1, 1);
               submit_cb(cb0);

               VkCommandBuffer cb1 = begin_cb();
               vkCmdBindPipeline(cb1, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
               vkCmdBindDescriptorSets(cb1, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                       0, 1, &ds, 0, NULL);
               struct barrier_pc pc1 = { 1u, n, r_cur, 0u };
               vkCmdPushConstants(cb1, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
               vkCmdDispatch(cb1, num_wgs, 1, 1);
               submit_cb(cb1);

               VkCommandBuffer cb2 = begin_cb();
               vkCmdBindPipeline(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
               vkCmdBindDescriptorSets(cb2, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                       0, 1, &ds, 0, NULL);
               struct barrier_pc pc2 = { 2u, n, r_cur, shifts[sh] };
               vkCmdPushConstants(cb2, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
               vkCmdDispatch(cb2, num_wgs, 1, 1);
               submit_cb(cb2);
            } else {
               VkCommandBuffer cb = begin_cb();
               vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
               vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                                       0, 1, &ds, 0, NULL);

               /* 1. Pre-warm consumer L1 lines with stale data. */
               struct barrier_pc pc0 = { 0u, n, r_cur - 1u, 0u };
               vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc0), &pc0);
               vkCmdDispatch(cb, num_wgs, 1, 1);

               emit_cs_barrier(cb, (enum barrier_kind)bk,
                               VK_ACCESS_SHADER_READ_BIT,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               b32.b, b64.b, b128.b);

               /* 2. Producer non-coherent stores. */
               struct barrier_pc pc1 = { 1u, n, r_cur, 0u };
               vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
               vkCmdDispatch(cb, num_wgs, 1, 1);

               emit_cs_barrier(cb, (enum barrier_kind)bk,
                               VK_ACCESS_SHADER_WRITE_BIT,
                               VK_ACCESS_SHADER_READ_BIT,
                               b32.b, b64.b, b128.b);

               /* 3. Shifted-workgroup consumer reads. */
               struct barrier_pc pc2 = { 2u, n, r_cur, shifts[sh] };
               vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
               vkCmdDispatch(cb, num_wgs, 1, 1);

               if (sizes[si].check_host && sh == 2) {
                  VkMemoryBarrier hmb = {
                     .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                     .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                     .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
                  };
                  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                       VK_PIPELINE_STAGE_HOST_BIT, 0,
                                       1, &hmb, 0, NULL, 0, NULL);
               }
               submit_cb(cb);
            }

            if (rm[0] != 0) {
               fprintf(stderr, "FAIL post_barrier %s %s shift=%u errs=%u idx=0x%x got=0x%x want=0x%x\n",
                       sizes[si].name, bnames[bk], shifts[sh], rm[0], rm[1], rm[2], rm[3]);
               bk_errs += rm[0];
            }

            if (sizes[si].check_host && sh == 2) {
               const uint32_t *h32 = (const uint32_t *)b32.map;
               for (uint32_t i = 0; i < n; ++i) {
                  uint32_t w32 = host_pat32(i, r_cur);
                  if (h32[i] != w32) {
                     fprintf(stderr, "FAIL host_readback %s %s idx=%u got=0x%x want=0x%x\n",
                             sizes[si].name, bnames[bk], i, h32[i], w32);
                     bk_errs++;
                     break;
                  }
               }
            }
         }
         if (bk_errs)
            fails++;
         else
            printf("PASS post_barrier %-8s %-15s shifts=3 host=%s\n",
                   sizes[si].name, bnames[bk], sizes[si].check_host ? "checked" : "device-local");
      }

      destroy_buf(&b32);
      destroy_buf(&b64);
      destroy_buf(&b128);
   }

   destroy_buf(&res_buf);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   return fails;
}

static unsigned
run_cross_wg_suite(void)
{
   VkDescriptorSetLayoutBinding binds[5];
   for (uint32_t i = 0; i < 5; ++i) {
      binds[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      };
   }
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 5, .pBindings = binds,
   }, NULL, &dsl));
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1,
      .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){
         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5 },
   }, NULL, &dpool));
   VkDescriptorSet ds;
   CHECK(vkAllocateDescriptorSets(dev, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
   }, &ds));
   VkPipelineLayout pl;
   CHECK(vkCreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &dsl,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &(VkPushConstantRange){
         VK_SHADER_STAGE_COMPUTE_BIT, 0, 8u },
   }, NULL, &pl));
   VkShaderModule sm;
   CHECK(vkCreateShaderModule(dev, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(xwg_comp), .pCode = xwg_comp,
   }, NULL, &sm));
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1,
      &(VkComputePipelineCreateInfo){
         .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
         .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sm, .pName = "main",
         },
         .layout = pl,
      }, NULL, &pipe));

   const VkMemoryPropertyFlags host_vis =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   const uint32_t pairs = 2u; /* 4 workgroups: guaranteed concurrent on multi-core GPUs */
   const uint32_t num_rounds = 64u;
   struct buf p32 = make_buf(pairs * 64u * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_vis, true);
   struct buf p128 = make_buf(pairs * 64u * 16u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_vis, true);
   struct buf flag = make_buf(pairs * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_vis, true);
   struct buf ack = make_buf(pairs * 4u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_vis, true);
   struct buf stats = make_buf(64u, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host_vis, true);

   memset(p32.map, 0, p32.size);
   memset(p128.map, 0, p128.size);
   memset(flag.map, 0, flag.size);
   memset(ack.map, 0, ack.size);
   memset(stats.map, 0, stats.size);

   VkDescriptorBufferInfo bi[5] = {
      { p32.b, 0, VK_WHOLE_SIZE },
      { p128.b, 0, VK_WHOLE_SIZE },
      { flag.b, 0, VK_WHOLE_SIZE },
      { ack.b, 0, VK_WHOLE_SIZE },
      { stats.b, 0, VK_WHOLE_SIZE },
   };
   VkWriteDescriptorSet w[5];
   for (uint32_t i = 0; i < 5; ++i) {
      w[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = ds, .dstBinding = i, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &bi[i],
      };
   }
   vkUpdateDescriptorSets(dev, 5, w, 0, NULL);

   VkCommandBuffer cb = begin_cb();
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl,
                           0, 1, &ds, 0, NULL);
   const uint32_t pc[2] = { num_rounds, 1u << 22 };
   vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), pc);
   vkCmdDispatch(cb, pairs * 2u, 1, 1);
   submit_cb(cb);

   const uint32_t *st = (const uint32_t *)stats.map;
   unsigned fails = 0;
   if (st[0] != 0 || st[1] != pairs * num_rounds || st[2] != 0) {
      fprintf(stderr, "FAIL cross_wg_device_barrier errs=%u rounds_done=%u/%u timeouts=%u got=0x%x want=0x%x\n",
              st[0], st[1], pairs * num_rounds, st[2], st[3], st[4]);
      fails = 1;
   } else {
      printf("PASS cross_wg_device_barrier pairs=%u rounds=%u (%u handoffs, 0 stale)\n",
             pairs, num_rounds, st[1]);
   }

   destroy_buf(&p32);
   destroy_buf(&p128);
   destroy_buf(&flag);
   destroy_buf(&ack);
   destroy_buf(&stats);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyPipelineLayout(dev, pl, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   return fails;
}

int
main(void)
{
   CHECK(vkCreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
         .pApplicationName = "vulkan-compute-store-visibility",
         .apiVersion = VK_API_VERSION_1_3,
      },
   }, NULL, &inst));

   uint32_t ndev = 1;
   CHECK(vkEnumeratePhysicalDevices(inst, &ndev, &pdev));
   if (ndev == 0) {
      fprintf(stderr, "no Vulkan physical device\n");
      return 2;
   }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);

   uint32_t nqf = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nqf, NULL);
   VkQueueFamilyProperties *qf = calloc(nqf, sizeof(*qf));
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nqf, qf);
   bool found_q = false;
   for (uint32_t i = 0; i < nqf; ++i) {
      if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
         qfam = i;
         found_q = true;
         break;
      }
   }
   free(qf);
   if (!found_q) {
      fprintf(stderr, "no compute queue family\n");
      return 2;
   }

   const float prio = 1.0f;
   VkPhysicalDeviceSynchronization2Features sync2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES,
      .synchronization2 = VK_TRUE,
   };
   CHECK(vkCreateDevice(pdev, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &sync2,
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueFamilyIndex = qfam,
         .queueCount = 1,
         .pQueuePriorities = &prio,
      },
   }, NULL, &dev));
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   CHECK(vkCreateCommandPool(dev, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qfam,
   }, NULL, &pool));

   printf("DEVICE %s\n", props.deviceName);
   unsigned fails = 0;
   fails += run_post_barrier_suite();
   fails += run_cross_wg_suite();

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);

   printf("SUMMARY %s (fails=%u)\n", fails ? "FAIL" : "PASS", fails);
   return fails ? 1 : 0;
}
