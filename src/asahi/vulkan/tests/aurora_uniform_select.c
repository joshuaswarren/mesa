// SPDX-License-Identifier: MIT
/* aurora-uniform-select: float compare-selects that read a uniform in place.
 *
 * One compute dispatch per (u, y) pair of edge values (signed zeros,
 * denormals, NaNs, infinities, normals) passes u as a push constant, so
 * the compiler sees a uniform, and stores u's bits in memory, so the same
 * comparison also runs on a register. Each dispatch writes six pairs of
 * results (aurora_uniform_select.comp): u == 0 and u != 0, u as the second
 * and the first comparand, u as a selected value, and min(y, u), each beside
 * its register reference. The two halves of every pair must agree.
 *
 * It prints one line per mismatch and a hash of every result word, which
 * must also be the same for a driver that copies the uniform to a register
 * first.
 *
 * usage: aurora-uniform-select
 */
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "aurora_uniform_select_shaders.h"

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

static void
die(const char *fmt, ...)
{
   va_list ap;
   va_start(ap, fmt);
   fputs("FATAL ", stdout);
   vprintf(fmt, ap);
   fputc('\n', stdout);
   va_end(ap);
   exit(2);
}

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS)                                                    \
         die("%s:%d %s = %d", __FILE__, __LINE__, #call, (int)r_);            \
   } while (0)

static const uint32_t values[] = {
   0x00000000, /* +0 */
   0x80000000, /* -0 */
   0x00000001, /* smallest denormal */
   0x80000001,
   0x007fffff, /* largest denormal */
   0x807fffff,
   0x00800000, /* smallest normal */
   0x3f800000, /* 1 */
   0xbf800000, /* -1 */
   0x3f800001,
   0x7f800000, /* +inf */
   0xff800000, /* -inf */
   0x7fc00000, /* quiet NaN */
   0xffc00000, /* negative quiet NaN */
   0x7f800001, /* signalling NaN */
};

#define WORDS_PER_RECORD 16u

static const char *const pair_names[] = {
   "u == 0", "u != 0", "y < u", "u < y", "y == 0 ? u : f", "min(y, u)",
};

int
main(void)
{
   const VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                                  .apiVersion = VK_API_VERSION_1_3};
   const VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                                     .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pdev;
   VkResult er = vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if ((er != VK_SUCCESS && er != VK_INCOMPLETE) || n == 0)
      die("no Vulkan device");
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   printf("device %s, driver 0x%x\n", props.deviceName, props.driverVersion);

   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, NULL);
   VkQueueFamilyProperties qf[16];
   if (nq > ARRAY_LEN(qf))
      nq = ARRAY_LEN(qf);
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qf);
   uint32_t qfam = UINT32_MAX;
   for (uint32_t i = 0; i < nq && qfam == UINT32_MAX; ++i)
      if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
         qfam = i;
   if (qfam == UINT32_MAX)
      die("no compute queue");

   const float prio = 1.0f;
   const VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .queueFamilyIndex = qfam, .queueCount = 1,
                                        .pQueuePriorities = &prio};
   const VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                   .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
   VkDevice dev;
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, qfam, 0, &queue);

   const unsigned nv = ARRAY_LEN(values);
   const unsigned records = nv * nv;
   const VkDeviceSize size = (VkDeviceSize)records * WORDS_PER_RECORD * 4;
   const VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                                  .size = size,
                                  .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bi, NULL, &buf));
   VkMemoryRequirements mr;
   vkGetBufferMemoryRequirements(dev, buf, &mr);
   VkPhysicalDeviceMemoryProperties mp;
   vkGetPhysicalDeviceMemoryProperties(pdev, &mp);
   const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
   uint32_t type = UINT32_MAX;
   for (uint32_t i = 0; i < mp.memoryTypeCount && type == UINT32_MAX; ++i)
      if ((mr.memoryTypeBits & (1u << i)) &&
          (mp.memoryTypes[i].propertyFlags & want) == want)
         type = i;
   if (type == UINT32_MAX)
      die("no host-visible memory");
   const VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .allocationSize = mr.size, .memoryTypeIndex = type};
   VkDeviceMemory mem;
   CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
   CHECK(vkBindBufferMemory(dev, buf, mem, 0));
   uint32_t *w;
   CHECK(vkMapMemory(dev, mem, 0, VK_WHOLE_SIZE, 0, (void **)&w));

   for (unsigned r = 0; r < records; ++r) {
      uint32_t *rec = w + r * WORDS_PER_RECORD;
      memset(rec, 0xee, WORDS_PER_RECORD * 4);
      rec[0] = values[r / nv];
      rec[1] = values[r % nv];
      rec[2] = 0x7a000000u | r; /* the "true" value */
      rec[3] = 0x5c000000u | r; /* the "false" value */
   }

   const VkDescriptorSetLayoutBinding b0 = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   const VkDescriptorSetLayoutCreateInfo dsli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 1, .pBindings = &b0};
   VkDescriptorSetLayout dsl;
   CHECK(vkCreateDescriptorSetLayout(dev, &dsli, NULL, &dsl));
   const VkPushConstantRange pcr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                    .offset = 0, .size = 8};
   const VkPipelineLayoutCreateInfo pli = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
   VkPipelineLayout layout;
   CHECK(vkCreatePipelineLayout(dev, &pli, NULL, &layout));

   const VkDescriptorPoolSize ps = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                    .descriptorCount = 1};
   const VkDescriptorPoolCreateInfo dpi = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
      .poolSizeCount = 1, .pPoolSizes = &ps};
   VkDescriptorPool dpool;
   CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &dpool));
   const VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = dpool, .descriptorSetCount = 1, .pSetLayouts = &dsl};
   VkDescriptorSet set;
   CHECK(vkAllocateDescriptorSets(dev, &dsai, &set));
   const VkDescriptorBufferInfo dbi = {.buffer = buf, .offset = 0, .range = size};
   const VkWriteDescriptorSet wds = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set,
      .dstBinding = 0, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi};
   vkUpdateDescriptorSets(dev, 1, &wds, 0, NULL);

   VkShaderModule sm;
   const VkShaderModuleCreateInfo smi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(uniform_select), .pCode = uniform_select};
   CHECK(vkCreateShaderModule(dev, &smi, NULL, &sm));
   const VkComputePipelineCreateInfo cpci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = "main"},
      .layout = layout};
   VkPipeline pipe;
   CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, NULL, &pipe));

   const VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                        .queueFamilyIndex = qfam};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &cpi, NULL, &pool));
   const VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
   const VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                          .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   CHECK(vkBeginCommandBuffer(cb, &cbbi));
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
   for (uint32_t r = 0; r < records; ++r) {
      const uint32_t push[2] = {values[r / nv], r};
      vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
      vkCmdDispatch(cb, 1, 1, 1);
   }
   const VkMemoryBarrier host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                 .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                                 .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
   vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, NULL, 0, NULL);
   CHECK(vkEndCommandBuffer(cb));
   const VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));
   const VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                            .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));

   unsigned mismatches = 0, written = 0;
   uint64_t hash = 0xcbf29ce484222325ull;
   for (unsigned r = 0; r < records; ++r) {
      const uint32_t *rec = w + r * WORDS_PER_RECORD;
      for (unsigned k = 4; k < WORDS_PER_RECORD; ++k) {
         hash = (hash ^ rec[k]) * 0x100000001b3ull;
         written += rec[k] != 0xeeeeeeeeu;
      }
      for (unsigned p = 0; p < ARRAY_LEN(pair_names); ++p) {
         /* pairs: (4,6) (5,7) (8,10) (9,11) (12,13) (14,15) */
         static const unsigned at[][2] = {{4, 6}, {5, 7}, {8, 10}, {9, 11},
                                          {12, 13}, {14, 15}};
         const uint32_t native = rec[at[p][0]], ref = rec[at[p][1]];
         if (native != ref) {
            printf("MISMATCH %-16s u=%08x y=%08x: uniform %08x, register %08x\n",
                   pair_names[p], rec[0], rec[1], native, ref);
            mismatches++;
         }
      }
   }
   printf("records %u, result words written %u/%u, mismatches %u, hash %016llx\n",
          records, written, records * (WORDS_PER_RECORD - 4), mismatches,
          (unsigned long long)hash);
   const bool ok = !mismatches && written == records * (WORDS_PER_RECORD - 4);
   printf("%s\n", ok ? "PASS" : "FAIL");

   vkDestroyFence(dev, fence, NULL);
   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyPipeline(dev, pipe, NULL);
   vkDestroyShaderModule(dev, sm, NULL);
   vkDestroyDescriptorPool(dev, dpool, NULL);
   vkDestroyPipelineLayout(dev, layout, NULL);
   vkDestroyDescriptorSetLayout(dev, dsl, NULL);
   vkUnmapMemory(dev, mem);
   vkFreeMemory(dev, mem, NULL);
   vkDestroyBuffer(dev, buf, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(inst, NULL);
   return ok ? 0 : 1;
}
