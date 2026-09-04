/*
 * Copyright 2026 Joshua Warren
 * SPDX-License-Identifier: MIT
 *
 * Vulkan plumbing for the Honeykrisp miscompile reproductions. See vk.h.
 */

#include "vk.h"

#include <alloca.h>

static VkResult
find_memory_type(struct hk_ctx *k, uint32_t want, uint32_t flags,
                 uint32_t *out)
{
   VkPhysicalDeviceMemoryProperties p;
   vkGetPhysicalDeviceMemoryProperties(k->pdev, &p);
   for (uint32_t i = 0; i < p.memoryTypeCount; i++) {
      if (!(want & (1u << i)))
         continue;
      if ((p.memoryTypes[i].propertyFlags & flags) == flags) {
         *out = i;
         return VK_SUCCESS;
      }
   }
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

bool
hk_init(struct hk_ctx *k)
{
   memset(k, 0, sizeof(*k));

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "hk-repro",
                            .apiVersion = VK_API_VERSION_1_1};
   VkInstanceCreateInfo ici = {.sType =
                                  VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   if (vkCreateInstance(&ici, NULL, &k->instance) != VK_SUCCESS) {
      fprintf(stderr, "vkCreateInstance failed\n");
      return false;
   }

   uint32_t ndev = 8;
   VkPhysicalDevice devs[8];
   if (vkEnumeratePhysicalDevices(k->instance, &ndev, devs) != VK_SUCCESS ||
       ndev == 0) {
      fprintf(stderr, "no Vulkan device\n");
      return false;
   }

   for (uint32_t i = 0; i < ndev && !k->pdev; i++) {
      uint32_t nq = 0;
      vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, NULL);
      if (nq == 0)
         continue;
      VkQueueFamilyProperties *qps = alloca(sizeof(*qps) * nq);
      vkGetPhysicalDeviceQueueFamilyProperties(devs[i], &nq, qps);
      for (uint32_t q = 0; q < nq; q++) {
         if (qps[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            k->pdev = devs[i];
            k->queue_family = q;
            break;
         }
      }
   }
   if (!k->pdev) {
      fprintf(stderr, "no device with a compute queue\n");
      return false;
   }

   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(k->pdev, &props);
   snprintf(k->name, sizeof(k->name), "%s", props.deviceName);
   k->api_version = props.apiVersion;
   printf("device: %s (Vulkan %u.%u.%u)\n", k->name,
          VK_API_VERSION_MAJOR(props.apiVersion),
          VK_API_VERSION_MINOR(props.apiVersion),
          VK_API_VERSION_PATCH(props.apiVersion));

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = k->queue_family,
      .queueCount = 1,
      .pQueuePriorities = &prio};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   if (vkCreateDevice(k->pdev, &dci, NULL, &k->dev) != VK_SUCCESS) {
      fprintf(stderr, "vkCreateDevice failed\n");
      return false;
   }
   vkGetDeviceQueue(k->dev, k->queue_family, 0, &k->queue);

   VkCommandPoolCreateInfo cpi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .queueFamilyIndex = k->queue_family};
   if (vkCreateCommandPool(k->dev, &cpi, NULL, &k->cmdpool) != VK_SUCCESS)
      return false;

   VkDescriptorSetLayoutBinding b[HK_MAX_SSBOS];
   for (unsigned i = 0; i < HK_MAX_SSBOS; i++) {
      b[i] = (VkDescriptorSetLayoutBinding){
         .binding = i,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   }
   VkDescriptorSetLayoutCreateInfo slci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = HK_MAX_SSBOS,
      .pBindings = b};
   if (vkCreateDescriptorSetLayout(k->dev, &slci, NULL, &k->set_layout) !=
       VK_SUCCESS)
      return false;

   VkDescriptorPoolSize ps = {
      .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 64 * HK_MAX_SSBOS};
   VkDescriptorPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 64,
      .poolSizeCount = 1,
      .pPoolSizes = &ps};
   if (vkCreateDescriptorPool(k->dev, &pci, NULL, &k->pool) != VK_SUCCESS)
      return false;

   VkPushConstantRange pcr = {
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
      .offset = 0,
      .size = HK_PUSH_SIZE};
   VkPipelineLayoutCreateInfo plci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1,
      .pSetLayouts = &k->set_layout,
      .pushConstantRangeCount = 1,
      .pPushConstantRanges = &pcr};
   if (vkCreatePipelineLayout(k->dev, &plci, NULL, &k->pipe_layout) !=
       VK_SUCCESS)
      return false;

   VkDescriptorSetAllocateInfo dsai = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = k->pool,
      .descriptorSetCount = 1,
      .pSetLayouts = &k->set_layout};
   return vkAllocateDescriptorSets(k->dev, &dsai, &k->set) == VK_SUCCESS;
}

void
hk_finish(struct hk_ctx *k)
{
   if (k->pipe_layout)
      vkDestroyPipelineLayout(k->dev, k->pipe_layout, NULL);
   if (k->pool)
      vkDestroyDescriptorPool(k->dev, k->pool, NULL);
   if (k->set_layout)
      vkDestroyDescriptorSetLayout(k->dev, k->set_layout, NULL);
   if (k->cmdpool)
      vkDestroyCommandPool(k->dev, k->cmdpool, NULL);
   if (k->dev)
      vkDestroyDevice(k->dev, NULL);
   if (k->instance)
      vkDestroyInstance(k->instance, NULL);
}

bool
hk_alloc_buffer(struct hk_ctx *k, struct hk_buffer *b, uint64_t bytes)
{
   memset(b, 0, sizeof(*b));
   b->size = bytes;
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                            .size = bytes,
                            .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
   if (vkCreateBuffer(k->dev, &bi, NULL, &b->buf) != VK_SUCCESS)
      return false;

   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(k->dev, b->buf, &req);
   uint32_t type;
   if (find_memory_type(k, req.memoryTypeBits,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &type) != VK_SUCCESS)
      return false;

   VkMemoryAllocateInfo ai = {.sType =
                                 VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                              .allocationSize = req.size,
                              .memoryTypeIndex = type};
   if (vkAllocateMemory(k->dev, &ai, NULL, &b->mem) != VK_SUCCESS)
      return false;
   if (vkBindBufferMemory(k->dev, b->buf, b->mem, 0) != VK_SUCCESS)
      return false;
   return vkMapMemory(k->dev, b->mem, 0, bytes, 0, &b->map) == VK_SUCCESS;
}

void
hk_free_buffer(struct hk_ctx *k, struct hk_buffer *b)
{
   if (b->map)
      vkUnmapMemory(k->dev, b->mem);
   if (b->mem)
      vkFreeMemory(k->dev, b->mem, NULL);
   if (b->buf)
      vkDestroyBuffer(k->dev, b->buf, NULL);
}

bool
hk_dispatch(struct hk_ctx *k, const char *spv_path, uint32_t groups,
            const uint32_t *push, struct hk_buffer *ssbos, unsigned n_ssbos)
{
   FILE *f = fopen(spv_path, "rb");
   if (!f) {
      fprintf(stderr, "cannot open %s (run make first)\n", spv_path);
      return false;
   }
   fseek(f, 0, SEEK_END);
   long len = ftell(f);
   fseek(f, 0, SEEK_SET);
   uint32_t *code = malloc(len);
   if (fread(code, 1, len, f) != (size_t)len) {
      fclose(f);
      free(code);
      return false;
   }
   fclose(f);

   VkShaderModuleCreateInfo smci = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = len,
      .pCode = code};
   VkShaderModule sm;
   bool ok = vkCreateShaderModule(k->dev, &smci, NULL, &sm) == VK_SUCCESS;
   free(code);
   if (!ok) {
      fprintf(stderr, "vkCreateShaderModule failed for %s\n", spv_path);
      return false;
   }

   VkPipeline p;
   VkComputePipelineCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType =
                   VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = sm,
                .pName = "main"},
      .layout = k->pipe_layout};
   ok = vkCreateComputePipelines(k->dev, VK_NULL_HANDLE, 1, &pci, NULL, &p) ==
        VK_SUCCESS;
   vkDestroyShaderModule(k->dev, sm, NULL);
   if (!ok) {
      fprintf(stderr, "pipeline creation failed for %s\n", spv_path);
      return false;
   }

   /* Update descriptors. */
   VkWriteDescriptorSet w[HK_MAX_SSBOS];
   VkDescriptorBufferInfo bi[HK_MAX_SSBOS];
   for (unsigned i = 0; i < n_ssbos; i++) {
      bi[i] = (VkDescriptorBufferInfo){.buffer = ssbos[i].buf,
                                       .offset = 0,
                                       .range = ssbos[i].size};
      w[i] = (VkWriteDescriptorSet){
         .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = k->set,
         .dstBinding = i,
         .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .pBufferInfo = &bi[i]};
   }
   vkUpdateDescriptorSets(k->dev, n_ssbos, w, 0, NULL);

   VkCommandBuffer cb;
   VkCommandBufferAllocateInfo cbai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = k->cmdpool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1};
   if (vkAllocateCommandBuffers(k->dev, &cbai, &cb) != VK_SUCCESS)
      return false;

   VkCommandBufferBeginInfo bbi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
   vkBeginCommandBuffer(cb, &bbi);
   vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, p);
   vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
                           k->pipe_layout, 0, 1, &k->set, 0, NULL);
   if (push)
      vkCmdPushConstants(cb, k->pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         HK_PUSH_SIZE, push);
   vkCmdDispatch(cb, groups, 1, 1);
   vkEndCommandBuffer(cb);

   VkFence fence;
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   vkCreateFence(k->dev, &fci, NULL, &fence);

   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1,
                      .pCommandBuffers = &cb};
   ok = vkQueueSubmit(k->queue, 1, &si, fence) == VK_SUCCESS &&
        vkWaitForFences(k->dev, 1, &fence, VK_TRUE, 5ull * 1000 * 1000 *
                                                          1000) == VK_SUCCESS;

   vkDestroyFence(k->dev, fence, NULL);
   vkFreeCommandBuffers(k->dev, k->cmdpool, 1, &cb);
   vkDestroyPipeline(k->dev, p, NULL);
   return ok;
}

void
hk_print_case_header(const struct hk_case *c)
{
   printf("\n== case %d: %s ==\n", c->number, c->name);
}

void
hk_print_expect(const char *what)
{
   printf("  expect: %s\n", what);
}

bool
hk_check_words(const char *label, const uint32_t *got, const uint32_t *want,
               unsigned n)
{
   unsigned bad = 0;
   for (unsigned i = 0; i < n; i++) {
      if (got[i] != want[i]) {
         if (bad < 8)
            printf("  %s[%u]: got %u (0x%x), want %u (0x%x)\n", label, i,
                   got[i], got[i], want[i], want[i]);
         bad++;
      }
   }
   if (bad)
      printf("  FAIL: %u/%u wrong\n", bad, n);
   else
      printf("  PASS: %s matches (%u words)\n", label, n);
   return bad == 0;
}

int
hk_run(int only_case)
{
   struct hk_ctx k;
   if (!hk_init(&k))
      return 2;

   int fails = 0;
   for (int i = 0; i < hk_case_count; i++) {
      const struct hk_case *c = &hk_cases[i];
      if (only_case >= 0 && c->number != only_case)
         continue;

      hk_print_case_header(c);

      struct hk_buffer b[HK_MAX_SSBOS];
      memset(b, 0, sizeof(b));
      bool ok = true;
      for (unsigned s = 0; s < HK_MAX_SSBOS; s++) {
         if (c->size[s] == 0)
            continue;
         uint64_t bytes = (uint64_t)c->size[s] * 4;
         if (!hk_alloc_buffer(&k, &b[s], bytes)) {
            fprintf(stderr, "alloc failed\n");
            ok = false;
            break;
         }
         memset(b[s].map, 0, bytes);
         if (c->data[s])
            memcpy(b[s].map, c->data[s], bytes);
      }

      if (ok) {
         unsigned n_ssbos = 0;
         while (n_ssbos < HK_MAX_SSBOS && b[n_ssbos].size)
            n_ssbos++;
         ok = hk_dispatch(&k, c->spv, c->groups, c->push, b, n_ssbos);
         if (ok) {
            const uint32_t *out[HK_MAX_SSBOS];
            for (unsigned s = 0; s < HK_MAX_SSBOS; s++)
               out[s] = b[s].size ? (const uint32_t *)b[s].map : NULL;
            if (!c->check(c, out))
               ok = false;
         }
      }

      if (!ok)
         fails++;
      for (unsigned s = 0; s < HK_MAX_SSBOS; s++)
         if (b[s].size)
            hk_free_buffer(&k, &b[s]);
   }

   hk_finish(&k);
   return fails ? 1 : 0;
}
