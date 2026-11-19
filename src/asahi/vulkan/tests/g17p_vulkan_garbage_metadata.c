/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Compressed G17P images over memory that held anything: aliased
 * (VK_IMAGE_CREATE_ALIAS_BIT) colour and depth/stencil attachments are bound
 * to memory the host filled first, metadata included, and then used the way
 * a D3D12 placed resource is after an aliasing barrier.
 *
 * Every image gets a transition from UNDEFINED and a render pass that loads
 * and stores it with a small clear inside, so the background program
 * (colour) or ZLS (depth/stencil) decodes every tile with whatever metadata
 * the memory held. A copy to a buffer then reads it through the texture
 * unit, and a second identical pass and copy must give the same bytes: once
 * stored, the image must decode as it was written. The cleared rectangle
 * must hold the clear value whatever the memory held.
 *
 * The memory is filled with each of the 256 byte values in turn (so every
 * subtile mode appears in every tile) and then with pseudo-random bytes.
 * A GPU fault or hang shows as VK_ERROR_DEVICE_LOST (exit 2) and as a GPU
 * recovery in the kernel log.
 *
 * usage: g17p-vulkan-garbage-metadata [--shim] [--expect-compressed]
 *   --shim               no-op DRM shim: nothing executes, skip readback
 *   --expect-compressed  require the driver's "compression enabled" message
 *                        for every format (needs ASAHI_MESA_DEBUG=perf)
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define W 256
#define H 256
#define RANDOM_ROUNDS 16

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static unsigned failures;

static FILE *driver_log;
static int saved_stderr = -1;

static void
restore_stderr(void)
{
   if (saved_stderr < 0)
      return;
   fflush(stderr);
   dup2(saved_stderr, STDERR_FILENO);
   close(saved_stderr);
   saved_stderr = -1;
}

static void
replay_driver_log(void)
{
   restore_stderr();
   if (!driver_log)
      return;
   rewind(driver_log);
   char line[1024];
   while (fgets(line, sizeof(line), driver_log))
      fputs(line, stderr);
   fclose(driver_log);
   driver_log = NULL;
}

struct test_format {
   VkFormat format;
   VkImageAspectFlags aspects;
   /* bytes per texel of the depth (or colour) and stencil copies */
   unsigned texel_B, stencil_B;
   /* float formats: garbage may decode to NaNs a load need not keep */
   bool exact;
   const char *names[2];
};

static const struct test_format formats[] = {
   {VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, true, {"r8_unorm"}},
   {VK_FORMAT_R8G8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 2, 0, true,
    {"r8g8_unorm"}},
   {VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, 4, 0, true,
    {"r8g8b8a8_unorm"}},
   {VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_IMAGE_ASPECT_COLOR_BIT, 4, 0, true,
    {"r10g10b10a2_unorm"}},
   {VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, 8, 0, false,
    {"r16g16b16a16_float"}},
   {VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT, 2, 0, true, {"z16_unorm"}},
   {VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, 4, 0, false,
    {"z32_float"}},
   {VK_FORMAT_D32_SFLOAT_S8_UINT,
    VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 4, 1, false,
    {"z32_float", "s8_uint"}},
};

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
host_memory(VkMemoryRequirements req, void **map)
{
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, map));
   return memory;
}

static VkCommandBuffer
begin(void)
{
   const VkCommandBufferAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cmd;
   CHECK(vkAllocateCommandBuffers(dev, &info, &cmd));
   const VkCommandBufferBeginInfo begin_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
   };
   CHECK(vkBeginCommandBuffer(cmd, &begin_info));
   return cmd;
}

static void
submit(VkCommandBuffer cmd)
{
   CHECK(vkEndCommandBuffer(cmd));
   const VkSubmitInfo info = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
      .commandBufferCount = 1,
      .pCommandBuffers = &cmd,
   };
   CHECK(vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));
   vkFreeCommandBuffers(dev, pool, 1, &cmd);
}

static void
image_barrier(VkCommandBuffer cmd, VkImage image, VkImageAspectFlags aspects,
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
      .image = image,
      .subresourceRange = {aspects, 0, 1, 0, 1},
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &b,
   };
   vkCmdPipelineBarrier2(cmd, &dep);
}

/* The cleared rectangle, in whole 16x16 compression tiles and across one. */
static const VkRect2D clear_rect = {{40, 24}, {64, 48}};

/* One load/store pass with a clear inside it, then a copy of the image to
 * buf (depth or colour first, stencil after it). */
static void
pass_and_copy(VkCommandBuffer cmd, const struct test_format *f, VkImage image,
              VkImageView view, VkBuffer buf, VkImageLayout from)
{
   const bool zs = f->aspects != VK_IMAGE_ASPECT_COLOR_BIT;
   const VkImageLayout layout =
      zs ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
         : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
   image_barrier(cmd, image, f->aspects, from, layout);

   const VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = layout,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {W, H}},
      .layerCount = 1,
      .colorAttachmentCount = zs ? 0 : 1,
      .pColorAttachments = zs ? NULL : &att,
      .pDepthAttachment =
         (f->aspects & VK_IMAGE_ASPECT_DEPTH_BIT) ? &att : NULL,
      .pStencilAttachment =
         (f->aspects & VK_IMAGE_ASPECT_STENCIL_BIT) ? &att : NULL,
   };
   vkCmdBeginRendering(cmd, &rendering);
   const VkClearAttachment clear = {
      .aspectMask = f->aspects,
      .colorAttachment = 0,
      .clearValue = zs ? (VkClearValue){.depthStencil = {0.5f, 0x5a}}
                       : (VkClearValue){.color = {{0.25f, 0.5f, 0.75f, 1.0f}}},
   };
   const VkClearRect rect = {clear_rect, 0, 1};
   vkCmdClearAttachments(cmd, 1, &clear, 1, &rect);
   vkCmdEndRendering(cmd);

   image_barrier(cmd, image, f->aspects, layout,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
   VkBufferImageCopy regions[2] = {
      {
         .imageSubresource = {f->aspects & ~VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0,
                              1},
         .imageExtent = {W, H, 1},
      },
      {
         .bufferOffset = (VkDeviceSize)W * H * f->texel_B,
         .imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1},
         .imageExtent = {W, H, 1},
      },
   };
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                          buf, f->stencil_B && f->texel_B ? 2 : 1, regions);
   const VkMemoryBarrier2 host = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
   };
   const VkDependencyInfo dep = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1,
      .pMemoryBarriers = &host,
   };
   vkCmdPipelineBarrier2(cmd, &dep);
}

static bool
in_clear_rect(unsigned x, unsigned y)
{
   return x >= (unsigned)clear_rect.offset.x &&
          x < clear_rect.offset.x + clear_rect.extent.width &&
          y >= (unsigned)clear_rect.offset.y &&
          y < clear_rect.offset.y + clear_rect.extent.height;
}

static void
fill(uint8_t *map, VkDeviceSize size, unsigned round)
{
   if (round < 256) {
      memset(map, (int)round, size);
      return;
   }
   uint64_t x = 0x9e3779b97f4a7c15ull * (round + 1);
   for (VkDeviceSize i = 0; i < size; ++i) {
      x ^= x << 13;
      x ^= x >> 7;
      x ^= x << 17;
      map[i] = (uint8_t)(x >> 24);
   }
}

static void
run_format(const struct test_format *f, bool execute)
{
   const bool zs = f->aspects != VK_IMAGE_ASPECT_COLOR_BIT;
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      /* As vkd3d-proton creates a placed render target or depth buffer. */
      .flags = VK_IMAGE_CREATE_ALIAS_BIT,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = f->format,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               (zs ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                   : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &info, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   uint8_t *memory_map;
   VkDeviceMemory memory = host_memory(req, (void **)&memory_map);
   CHECK(vkBindImageMemory(dev, image, memory, 0));

   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = f->format,
      .subresourceRange = {f->aspects, 0, 1, 0, 1},
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &view_info, NULL, &view));

   const VkDeviceSize copy_B = (VkDeviceSize)W * H * (f->texel_B + f->stencil_B);
   VkBuffer bufs[2];
   uint8_t *maps[2];
   VkDeviceMemory buf_memory[2];
   for (unsigned i = 0; i < 2; ++i) {
      const VkBufferCreateInfo buf_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = copy_B,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      };
      CHECK(vkCreateBuffer(dev, &buf_info, NULL, &bufs[i]));
      VkMemoryRequirements breq;
      vkGetBufferMemoryRequirements(dev, bufs[i], &breq);
      buf_memory[i] = host_memory(breq, (void **)&maps[i]);
      CHECK(vkBindBufferMemory(dev, bufs[i], buf_memory[i], 0));
   }

   uint8_t *reference = calloc(1, copy_B);
   bool have_reference = false;
   unsigned unstable_rounds = 0, rect_rounds = 0;
   for (unsigned round = 0; round < 256 + RANDOM_ROUNDS; ++round) {
      fill(memory_map, req.size, round);
      memset(maps[0], 0xcc, copy_B);
      memset(maps[1], 0x33, copy_B);

      VkCommandBuffer cmd = begin();
      pass_and_copy(cmd, f, image, view, bufs[0], VK_IMAGE_LAYOUT_UNDEFINED);
      pass_and_copy(cmd, f, image, view, bufs[1],
                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
      submit(cmd);
      if (!execute)
         continue;

      /* What the first copy read must be what the second one read. */
      if (memcmp(maps[0], maps[1], copy_B)) {
         unsigned n = 0;
         for (VkDeviceSize i = 0; i < copy_B; ++i)
            n += maps[0][i] != maps[1][i];
         if (f->exact || unstable_rounds < 4)
            fprintf(stderr, "%s round %u: %u bytes differ between reads\n",
                    f->names[0], round, n);
         unstable_rounds++;
      }

      /* The cleared rectangle holds the clear value in every round. */
      bool rect_ok = true;
      for (unsigned part = 0; part < 2; ++part) {
         const unsigned B = part ? f->stencil_B : f->texel_B;
         const VkDeviceSize base = part ? (VkDeviceSize)W * H * f->texel_B : 0;
         for (unsigned y = 0; B && y < H; ++y) {
            for (unsigned x = 0; x < W; ++x) {
               if (!in_clear_rect(x, y))
                  continue;
               const VkDeviceSize o = base + ((VkDeviceSize)y * W + x) * B;
               if (!have_reference)
                  memcpy(reference + o, maps[1] + o, B);
               else if (memcmp(reference + o, maps[1] + o, B))
                  rect_ok = false;
            }
         }
      }
      have_reference = true;
      if (!rect_ok) {
         fprintf(stderr, "%s round %u: cleared rectangle wrong\n",
                 f->names[0], round);
         rect_rounds++;
      }
   }

   if (execute) {
      const bool fail = rect_rounds || (f->exact && unstable_rounds);
      printf("%-20s %u rounds: %u unstable, %u wrong clear%s\n", f->names[0],
             256 + RANDOM_ROUNDS, unstable_rounds, rect_rounds,
             fail ? " FAIL" : "");
      failures += fail;
   }

   free(reference);
   for (unsigned i = 0; i < 2; ++i) {
      vkDestroyBuffer(dev, bufs[i], NULL);
      vkFreeMemory(dev, buf_memory[i], NULL);
   }
   vkDestroyImageView(dev, view, NULL);
   vkDestroyImage(dev, image, NULL);
   vkFreeMemory(dev, memory, NULL);
}

int
main(int argc, char **argv)
{
   bool shim = false, expect_compressed = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--shim"))
         shim = true;
      else if (!strcmp(argv[i], "--expect-compressed"))
         expect_compressed = true;
   }

   fflush(stderr);
   saved_stderr = dup(STDERR_FILENO);
   driver_log = tmpfile();
   if (saved_stderr < 0 || !driver_log ||
       atexit(replay_driver_log) != 0 ||
       dup2(fileno(driver_log), STDERR_FILENO) < 0) {
      replay_driver_log();
      fprintf(stderr, "could not capture driver log\n");
      return 2;
   }

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
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceVulkan13Features features13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
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
   vkGetDeviceQueue(dev, 0, 0, &queue);
   const VkCommandPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
   };
   CHECK(vkCreateCommandPool(dev, &pool_info, NULL, &pool));

   const unsigned nr = sizeof(formats) / sizeof(formats[0]);
   for (unsigned i = 0; i < nr; ++i) {
      /* Results go to the real stderr, the driver log stays captured. */
      run_format(&formats[i], !shim);
   }

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   restore_stderr();
   rewind(driver_log);
   char line[1024], want[96];
   unsigned missing = 0;
   char *log = NULL;
   size_t log_len = 0;
   FILE *mem = open_memstream(&log, &log_len);
   while (fgets(line, sizeof(line), driver_log)) {
      fputs(line, stderr);
      fputs(line, mem);
   }
   fclose(mem);
   fclose(driver_log);
   driver_log = NULL;
   for (unsigned i = 0; expect_compressed && i < nr; ++i) {
      for (unsigned p = 0; p < 2 && formats[i].names[p]; ++p) {
         snprintf(want, sizeof(want), "compression enabled: %s %ux%u",
                  formats[i].names[p], W, H);
         if (!strstr(log, want)) {
            fprintf(stderr, "FAIL: %s not compressed\n", formats[i].names[p]);
            missing++;
         }
      }
   }
   free(log);
   failures += missing;

   printf("garbage metadata, %u formats%s: %s\n", nr,
          shim ? " (no readback)" : "", failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
