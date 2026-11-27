/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Linear vkCmdBlitImage mip chains on images G17P compresses (RGBA8/BGRA8
 * render targets), the commands zink records for glGenerateMipmap: upload
 * level 0, blit each level from the one above, read every level back and
 * compare it bit for bit with the G17P linear filter computed here.
 *
 * Then, for each image: a render pass that loads and stores level 1 without
 * drawing must leave it unchanged, and for RGBA8 a blit from the compressed
 * level 0 into a plain (storage) image must match level 1.
 *
 * usage: g17p-vulkan-compressed-mip-blit [--shim] [--expect-compute]
 *   --shim            no-op DRM shim: nothing executes, skip readback
 *   --expect-compute  require that recording the blit chains logs a render
 *                     pass only for the levels hk stores through the PBE
 *                     (HK_G17P_MIP_PBE_MIN_TEXELS) and that the images are
 *                     compressed (needs ASAHI_MESA_DEBUG=perf)
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

static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static VkCommandPool pool;
static unsigned failures;

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

static VkImage
image_create(VkFormat format, uint32_t w, uint32_t h, uint32_t levels,
             VkImageUsageFlags usage)
{
   const VkImageCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = format,
      .extent = {w, h, 1},
      .mipLevels = levels,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
   };
   VkImage image;
   CHECK(vkCreateImage(dev, &info, NULL, &image));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, image, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindImageMemory(dev, image, memory, 0));
   return image;
}

static VkBuffer
buffer_create(VkDeviceSize size, VkBufferUsageFlags usage, uint8_t **map)
{
   const VkBufferCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = size,
      .usage = usage,
   };
   VkBuffer buffer;
   CHECK(vkCreateBuffer(dev, &info, NULL, &buffer));
   VkMemoryRequirements req;
   vkGetBufferMemoryRequirements(dev, buffer, &req);
   const VkMemoryAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
   };
   VkDeviceMemory memory;
   CHECK(vkAllocateMemory(dev, &alloc, NULL, &memory));
   CHECK(vkBindBufferMemory(dev, buffer, memory, 0));
   CHECK(vkMapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)map));
   memset(*map, 0, size);
   return buffer;
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
barrier(VkCommandBuffer cmd)
{
   const VkMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
      .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &b, 0, NULL,
                        0, NULL);
}

static void
to_general(VkCommandBuffer cmd, VkImage image)
{
   const VkImageMemoryBarrier b = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = image,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0,
                           VK_REMAINING_MIP_LEVELS, 0,
                           VK_REMAINING_ARRAY_LAYERS},
   };
   vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, NULL, 0, NULL,
                        1, &b);
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

static uint32_t
minify(uint32_t size, uint32_t level)
{
   return size >> level ? size >> level : 1;
}

static uint32_t
level_count(uint32_t w, uint32_t h)
{
   uint32_t levels = 1;
   while ((w >> levels) || (h >> levels))
      levels++;
   return levels;
}

/* One destination coordinate of the G17P linear filter: the lower source
 * coordinate and the Q8 weight of the upper one. Exact 2x and 1x axes are
 * (2i, 128) and (i, 0); 2x+1 axes follow vk_meta's float scale. */
static void
axis_entry(uint32_t src, uint32_t dst, uint32_t i, uint32_t *lower,
           uint32_t *weight)
{
   if (src == dst) {
      *lower = i;
      *weight = 0;
      return;
   }
   if (src == 2 * dst) {
      *lower = 2 * i;
      *weight = 128;
      return;
   }

   const float scale = (float)(1.0 / (double)dst);
   const float uv = ((float)i + 0.5f) * scale;
   const float source = uv * (float)src - 0.5f;
   uint32_t l = (uint32_t)source;
   uint32_t w = (uint32_t)((source - (float)l) * 256.0f + 0.5f);
   if (w == 256) {
      ++l;
      w = 0;
   }
   if (l >= src)
      l = src - 1;
   *lower = l;
   *weight = w;
}

/* dst (dw x dh) from src (sw x sh), four bytes per texel */
static void
reference_level(const uint8_t *src, uint32_t sw, uint32_t sh, uint8_t *dst,
                uint32_t dw, uint32_t dh)
{
   for (uint32_t y = 0; y < dh; ++y) {
      uint32_t y0, wy;
      axis_entry(sh, dh, y, &y0, &wy);
      const uint32_t y1 = y0 + (wy != 0), iy = 256 - wy;
      for (uint32_t x = 0; x < dw; ++x) {
         uint32_t x0, wx;
         axis_entry(sw, dw, x, &x0, &wx);
         const uint32_t x1 = x0 + (wx != 0), ix = 256 - wx;
         for (uint32_t c = 0; c < 4; ++c) {
            const uint32_t value = src[4 * (y0 * sw + x0) + c] * ix * iy +
                                   src[4 * (y0 * sw + x1) + c] * wx * iy +
                                   src[4 * (y1 * sw + x0) + c] * ix * wy +
                                   src[4 * (y1 * sw + x1) + c] * wx * wy;
            const uint32_t fixed4 = (value + 2048) >> 12;
            uint32_t out = fixed4 >> 4;
            const uint32_t rem = fixed4 & 15;
            out += rem > 8 || (rem == 8 && out < 128);
            dst[4 * (y * dw + x) + c] = (uint8_t)out;
         }
      }
   }
}

static void
compare(const char *what, const uint8_t *got, const uint8_t *want,
        uint32_t w, uint32_t h)
{
   unsigned bad = 0, max_diff = 0;
   for (uint32_t i = 0; i < 4 * w * h; ++i) {
      const unsigned diff = abs((int)got[i] - (int)want[i]);
      if (diff) {
         if (!bad)
            fprintf(stderr, "%s: texel (%u,%u) channel %u is %u, want %u\n",
                    what, (i / 4) % w, (i / 4) / w, i % 4, got[i], want[i]);
         bad++;
         max_diff = diff > max_diff ? diff : max_diff;
      }
   }
   if (bad) {
      fprintf(stderr, "%s: %u of %u bytes differ, max difference %u\n", what,
              bad, 4 * w * h, max_diff);
      failures++;
   }
}

static FILE *capture;
static int saved_stderr = -1;

static void
capture_begin(void)
{
   fflush(stderr);
   saved_stderr = dup(2);
   capture = tmpfile();
   if (saved_stderr < 0 || !capture || dup2(fileno(capture), 2) < 0)
      exit(2);
}

/* Echo the captured log; returns how many lines contain needle. */
static unsigned
capture_end(const char *needle)
{
   fflush(stderr);
   dup2(saved_stderr, 2);
   close(saved_stderr);
   rewind(capture);
   unsigned count = 0;
   char line[1024];
   while (fgets(line, sizeof(line), capture)) {
      fputs(line, stderr);
      count += strstr(line, needle) != NULL;
   }
   fclose(capture);
   return count;
}

struct test_case {
   VkFormat format;
   uint32_t width, height;
};

static void
run_case(const struct test_case *t, bool readback, bool expect_compute)
{
   const uint32_t w = t->width, h = t->height;
   const uint32_t levels = level_count(w, h);
   char name[64];
   snprintf(name, sizeof(name), "%s %ux%u",
            t->format == VK_FORMAT_B8G8R8A8_UNORM ? "bgra8" : "rgba8", w, h);

   /* A render target with transfer usage: G17P compresses it. */
   const VkImageUsageFlags usage =
      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   if (expect_compute)
      capture_begin();
   VkImage image = image_create(t->format, w, h, levels, usage);
   /* Storage usage keeps this one uncompressed. */
   const bool with_plain = t->format == VK_FORMAT_R8G8B8A8_UNORM;
   VkImage plain = with_plain ? image_create(t->format, minify(w, 1),
                                             minify(h, 1), 1,
                                             usage | VK_IMAGE_USAGE_STORAGE_BIT)
                              : VK_NULL_HANDLE;
   if (expect_compute) {
      char needle[96];
      snprintf(needle, sizeof(needle), "compression enabled: %s_unorm %ux%u",
               t->format == VK_FORMAT_B8G8R8A8_UNORM ? "b8g8r8a8" : "r8g8b8a8",
               w, h);
      if (capture_end(needle) != 1) {
         fprintf(stderr, "%s: FAIL: the image is not compressed\n", name);
         failures++;
      }
   }

   /* Level sizes and readback offsets */
   uint64_t offsets[16], total = 0;
   for (uint32_t l = 0; l < levels; ++l) {
      offsets[l] = total;
      total += 4ull * minify(w, l) * minify(h, l);
   }
   const uint64_t level1_size = 4ull * minify(w, 1) * minify(h, 1);

   uint8_t *upload, *out, *again, *from_plain;
   VkBuffer upload_buf =
      buffer_create(4ull * w * h, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, &upload);
   VkBuffer out_buf =
      buffer_create(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &out);
   VkBuffer again_buf =
      buffer_create(level1_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT, &again);
   VkBuffer plain_buf =
      buffer_create(level1_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    &from_plain);

   uint32_t seed = w * 2654435761u ^ h;
   for (uint64_t i = 0; i < 4ull * w * h; ++i) {
      seed = seed * 1664525u + 1013904223u;
      upload[i] = seed >> 24;
   }

   VkCommandBuffer cmd = begin();
   to_general(cmd, image);
   if (with_plain)
      to_general(cmd, plain);
   const VkBufferImageCopy up = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {w, h, 1},
   };
   vkCmdCopyBufferToImage(cmd, upload_buf, image, VK_IMAGE_LAYOUT_GENERAL, 1,
                          &up);
   barrier(cmd);

   if (expect_compute)
      capture_begin();
   for (uint32_t l = 1; l < levels; ++l) {
      const VkImageBlit blit = {
         .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0, 1},
         .srcOffsets = {{0, 0, 0},
                        {minify(w, l - 1), minify(h, l - 1), 1}},
         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1},
         .dstOffsets = {{0, 0, 0}, {minify(w, l), minify(h, l), 1}},
      };
      vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, image,
                     VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
      barrier(cmd);
   }
   /* From the compressed level 0 into the plain image */
   const VkImageBlit to_plain = {
      .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .srcOffsets = {{0, 0, 0}, {w, h, 1}},
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .dstOffsets = {{0, 0, 0}, {minify(w, 1), minify(h, 1), 1}},
   };
   if (with_plain) {
      vkCmdBlitImage(cmd, image, VK_IMAGE_LAYOUT_GENERAL, plain,
                     VK_IMAGE_LAYOUT_GENERAL, 1, &to_plain, VK_FILTER_LINEAR);
      barrier(cmd);
   }
   /* Exact 2x2 compressed levels use the compressed compute block-mip path
    * on HAL200/AGX3B; only non-exact-2x2 levels of at least
    * HK_G17P_MIP_PBE_MIN_TEXELS (1 << 17) texels take a render pass. */
   unsigned passes = 0;
   for (uint32_t l = 1; l < levels; ++l) {
      const uint32_t sw = minify(w, l - 1), sh = minify(h, l - 1);
      const uint32_t dw = minify(w, l), dh = minify(h, l);
      const bool exact_2x2 = sw == 2 * dw && sh == 2 * dh;
      passes += !exact_2x2 && (uint64_t)dw * dh >= (1u << 17);
   }
   if (expect_compute && capture_end("Rendering ") != passes) {
      fprintf(stderr, "%s: FAIL: the blits did not record %u render passes\n",
              name, passes);
      failures++;
   }

   for (uint32_t l = 0; l < levels; ++l) {
      const VkBufferImageCopy back = {
         .bufferOffset = offsets[l],
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1},
         .imageExtent = {minify(w, l), minify(h, l), 1},
      };
      vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, out_buf, 1,
                             &back);
   }
   const VkBufferImageCopy back_plain = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
      .imageExtent = {minify(w, 1), minify(h, 1), 1},
   };
   if (with_plain)
      vkCmdCopyImageToBuffer(cmd, plain, VK_IMAGE_LAYOUT_GENERAL, plain_buf, 1,
                             &back_plain);
   barrier(cmd);

   /* Load and store level 1 in a render pass that draws nothing */
   const VkImageViewCreateInfo view_info = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = t->format,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 0, 1},
   };
   VkImageView view;
   CHECK(vkCreateImageView(dev, &view_info, NULL, &view));
   const VkRenderingAttachmentInfo att = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
   };
   const VkRenderingInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {minify(w, 1), minify(h, 1)}},
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &att,
   };
   vkCmdBeginRendering(cmd, &rendering);
   vkCmdEndRendering(cmd);
   barrier(cmd);
   const VkBufferImageCopy back1 = {
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 0, 1},
      .imageExtent = {minify(w, 1), minify(h, 1), 1},
   };
   vkCmdCopyImageToBuffer(cmd, image, VK_IMAGE_LAYOUT_GENERAL, again_buf, 1,
                          &back1);
   barrier(cmd);
   submit(cmd);

   if (readback) {
      char what[96];
      uint8_t *want = malloc(total);
      memcpy(want, upload, 4ull * w * h);
      for (uint32_t l = 1; l < levels; ++l)
         reference_level(want + offsets[l - 1], minify(w, l - 1),
                         minify(h, l - 1), want + offsets[l], minify(w, l),
                         minify(h, l));
      for (uint32_t l = 0; l < levels; ++l) {
         snprintf(what, sizeof(what), "%s level %u", name, l);
         compare(what, out + offsets[l], want + offsets[l], minify(w, l),
                 minify(h, l));
      }
      snprintf(what, sizeof(what), "%s level 1 after load/store", name);
      compare(what, again, want + offsets[1], minify(w, 1), minify(h, 1));
      if (with_plain) {
         snprintf(what, sizeof(what), "%s blit into a plain image", name);
         compare(what, from_plain, want + offsets[1], minify(w, 1),
                 minify(h, 1));
      }
      free(want);
   }

   vkDestroyImageView(dev, view, NULL);
   vkDestroyBuffer(dev, upload_buf, NULL);
   vkDestroyBuffer(dev, out_buf, NULL);
   vkDestroyBuffer(dev, again_buf, NULL);
   vkDestroyBuffer(dev, plain_buf, NULL);
   vkDestroyImage(dev, image, NULL);
   if (with_plain)
      vkDestroyImage(dev, plain, NULL);
}

int
main(int argc, char **argv)
{
   bool shim = false, expect_compute = false;
   for (int i = 1; i < argc; ++i) {
      if (!strcmp(argv[i], "--shim"))
         shim = true;
      else if (!strcmp(argv[i], "--expect-compute"))
         expect_compute = true;
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

   /* Even and odd chains, wide and tall metadata grids, a level 0 whose
    * levels soon fall below the 16-pixel compression limit, and BGRA. */
   static const struct test_case cases[] = {
      {VK_FORMAT_R8G8B8A8_UNORM, 1920, 1080},
      {VK_FORMAT_R8G8B8A8_UNORM, 256, 256},
      {VK_FORMAT_R8G8B8A8_UNORM, 129, 127},
      {VK_FORMAT_R8G8B8A8_UNORM, 1000, 40},
      {VK_FORMAT_R8G8B8A8_UNORM, 40, 1000},
      {VK_FORMAT_B8G8R8A8_UNORM, 300, 17},
      {VK_FORMAT_B8G8R8A8_UNORM, 3840, 2160},
   };
   for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i)
      run_case(&cases[i], !shim, expect_compute);

   vkDestroyCommandPool(dev, pool, NULL);
   vkDestroyDevice(dev, NULL);
   vkDestroyInstance(instance, NULL);

   printf("%u mip chains%s: %s\n",
          (unsigned)(sizeof(cases) / sizeof(cases[0])),
          shim ? " (no readback)" : "", failures ? "FAIL" : "PASS");
   return failures ? 1 : 0;
}
