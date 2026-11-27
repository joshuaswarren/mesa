/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Multisample colour attachments of mixed formats in one pass, as Transport
 * Fever 3 records at 4x: with an R11G11B10F target among them G17P refused
 * the tilebuffer layout ("G17P tilebuffer layout refused: attachments=3
 * samples=4 sample_size=8 tile_size=512") and vkEndCommandBuffer returned
 * VK_ERROR_FEATURE_NOT_PRESENT. The default set reproduces that line.
 *
 * With a reloaded attachment the 4x tile keeps eight bytes a sample and the
 * rest spill to memory. A pipeline draw with alpha-to-coverage then failed to
 * compile its fragment epilog ("Unhandled intrinsic load_sample_id"), which
 * also ended in VK_ERROR_FEATURE_NOT_PRESENT: `... load a2c` reproduces that.
 *
 * Pass 1 clears each attachment (loadOp CLEAR, storeOp STORE). Pass 2 reloads
 * them (loadOp LOAD), writes a band with vkCmdClearAttachments or a
 * pipeline draw (opaque colours, optionally with alpha-to-coverage) and
 * resolves them. Each resolved image must equal a single-sample reference
 * given the same colours. `none` writes nothing in pass 2, so every tile is
 * one the application did not draw to.
 *
 * On M4 Pro the single-layer 2x attachments of formats up to eight bytes are
 * compressed. G17P cannot spill a compressed target: in
 * `rgba16f,rgba32f,rgba32f,rgba32f,rgba32f,rgba16f 1 2` the compressed
 * RGBA16F targets must stay in the tilebuffer and an RGBA32F target spill
 * instead (the plain order spilled both RGBA16F and failed the pass);
 * `r8,rgba32f,rgba32f,rgba32f,rgba32f,rg8 1 2` pads between the groups.
 *
 * With `layers` above one the images are arrays and both passes are layered;
 * vkCmdClearAttachments writes every layer, or with `layer0` only layer 0,
 * which leaves the other layers without any geometry.
 *
 * With rgba8, 4 samples and write=rounding, compare the hardware resolve and
 * the shader fallback to a CPU byte average (discard-mask 1 and 0).
 *
 *   vulkan-msaa-mrt [formats=rgba8,r11g11b10f,none] [attachments=3] [samples=4]
 *                   [pass2=load|clear] [write=clear|layer0|draw|a2c|rounding|none] [layers=1] [discard-mask=0]
 *   formats: one name repeated `attachments` times, or a comma-separated list
 *   with one name per attachment ("none" = VK_NULL_HANDLE view):
 *   rgba16f rgba8 rg16f r32f rg32f rgba32f rgb10a2 bgr10a2 r11g11b10f r8 rg8
 *   none
 * Exit 0 = PASS, 1 = FAIL. */
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "vulkan_msaa_mrt_shaders.h"

#define CHECK(x)                                                              \
   do {                                                                       \
      VkResult r_ = (x);                                                      \
      if (r_ != VK_SUCCESS) {                                                 \
         printf("FAIL: %s = %d (line %d)\n", #x, r_, __LINE__);               \
         exit(1);                                                             \
      }                                                                       \
   } while (0)

#define W 64
#define H 64
#define MAX_ATT 8
/* Pass 2 writes columns 5..36: tiles partly written lie on both edges. */
#define WRITE_RECT ((VkRect2D){{5, 0}, {W / 2, H}})

static const struct { const char *name; VkFormat fmt; unsigned bpp; } formats[] = {
   {"rgba16f", VK_FORMAT_R16G16B16A16_SFLOAT, 8},
   {"rgba8", VK_FORMAT_R8G8B8A8_UNORM, 4},
   {"rg16f", VK_FORMAT_R16G16_SFLOAT, 4},
   {"r32f", VK_FORMAT_R32_SFLOAT, 4},
   {"rg32f", VK_FORMAT_R32G32_SFLOAT, 8},
   {"rgba32f", VK_FORMAT_R32G32B32A32_SFLOAT, 16},
   {"rgb10a2", VK_FORMAT_A2B10G10R10_UNORM_PACK32, 4},
   {"bgr10a2", VK_FORMAT_A2R10G10B10_UNORM_PACK32, 4},
   {"r11g11b10f", VK_FORMAT_B10G11R11_UFLOAT_PACK32, 4},
   {"r8", VK_FORMAT_R8_UNORM, 1},
   {"rg8", VK_FORMAT_R8G8_UNORM, 2},
   {"none", VK_FORMAT_UNDEFINED, 0},
};

static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memprops;
static unsigned layers = 1;

static VkDeviceMemory
bind_memory(VkMemoryRequirements req, VkMemoryPropertyFlags flags)
{
   for (uint32_t i = 0; i < memprops.memoryTypeCount; i++) {
      if ((req.memoryTypeBits & (1u << i)) &&
          (memprops.memoryTypes[i].propertyFlags & flags) == flags) {
         VkMemoryAllocateInfo ai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL,
                                    req.size, i};
         VkDeviceMemory mem;
         CHECK(vkAllocateMemory(dev, &ai, NULL, &mem));
         return mem;
      }
   }
   printf("FAIL: no memory type\n");
   exit(1);
}

static VkImage
make_image(VkFormat fmt, VkSampleCountFlagBits samples, VkImageView *view)
{
   VkImageCreateInfo ci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = layers,
      .samples = samples,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   VkImage img;
   CHECK(vkCreateImage(dev, &ci, NULL, &img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, img, &req);
   CHECK(vkBindImageMemory(dev, img,
                           bind_memory(req, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT), 0));
   if (view) {
      VkImageViewCreateInfo vi = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = img,
         .viewType = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
         .format = fmt,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers},
      };
      CHECK(vkCreateImageView(dev, &vi, NULL, view));
   }
   return img;
}

static void
barrier(VkCommandBuffer cb)
{
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
   };
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
   vkCmdPipelineBarrier2(cb, &di);
}

static void
to_general(VkCommandBuffer cb, VkImage img)
{
   VkImageMemoryBarrier2 ib = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
      .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
      .newLayout = VK_IMAGE_LAYOUT_GENERAL,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = img,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers},
   };
   VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
                          .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
   vkCmdPipelineBarrier2(cb, &di);
}

static VkClearColorValue
colour(unsigned att, unsigned pass)
{
   /* Exactly representable in every listed format; distinct per attachment
    * and per pass. */
   VkClearColorValue c = {{0, 0, 0, 1}};
   c.float32[att % 3] = 1.0f;
   c.float32[(att + pass + 1) % 3] = pass ? 0.5f : 0.0f;
   /* Pass 2 is opaque, so alpha-to-coverage keeps every sample. Pass 1 alpha
    * is non-zero too: the upper eight bytes of a wide sample must not be all
    * zero, or losing them goes unseen. */
   c.float32[3] = 1.0f;
   return c;
}

/* Three samples keep the clear value and the fourth gets the draw value.
 * Exercise both sides of a tie and ties with odd and even lower bytes. */
static VkClearColorValue
rounding_colour(unsigned phase)
{
   static const uint8_t clear[4] = {165, 164, 46, 20};
   static const uint8_t draw[4] = {255, 254, 47, 23};
   VkClearColorValue c;
   for (unsigned i = 0; i < 4; i++) {
      unsigned resolved = (3u * clear[i] + draw[i] + 2) / 4;
      const unsigned values[] = {clear[i], draw[i], resolved};
      c.float32[i] = values[phase] / 255.0f;
   }
   return c;
}

static VkShaderModule
make_module(const uint32_t *code, size_t size)
{
   VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = size, .pCode = code};
   VkShaderModule m;
   CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
   return m;
}

static VkPipeline
make_pipeline(const VkFormat *fmt, unsigned count, unsigned samples, bool a2c,
              const VkSampleMask *mask, VkPipelineLayout *layout)
{
   VkPushConstantRange pcr = {VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8 * 16};
   VkPipelineLayoutCreateInfo lci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .pushConstantRangeCount = 1,
                                     .pPushConstantRanges = &pcr};
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, layout));

   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT,
       .module = make_module(msaa_mrt_vert, sizeof(msaa_mrt_vert)), .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
       .module = make_module(msaa_mrt_frag, sizeof(msaa_mrt_frag)), .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
   VkViewport vp = {0, 0, W, H, 0, 1};
   VkRect2D scissor = WRITE_RECT;
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &scissor};
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
      .lineWidth = 1.0f};
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = samples, .alphaToCoverageEnable = a2c, .pSampleMask = mask};
   VkPipelineColorBlendAttachmentState blend[MAX_ATT];
   for (unsigned i = 0; i < count; i++)
      blend[i] = (VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf};
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = count, .pAttachments = blend};
   VkPipelineRenderingCreateInfo rci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = count, .pColorAttachmentFormats = fmt};
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &rci, .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia,
      .pViewportState = &vps, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pColorBlendState = &cbs, .layout = *layout};
   VkPipeline pipeline;
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipeline));
   return pipeline;
}

int
main(int argc, char **argv)
{
   const char *list = argc > 1 ? argv[1] : "rgba8,r11g11b10f,none";
   unsigned count = argc > 2 ? atoi(argv[2]) : 3;
   unsigned samples = argc > 3 ? atoi(argv[3]) : 4;
   bool pass2_load = !(argc > 4 && !strcmp(argv[4], "clear"));
   const char *write = argc > 5 ? argv[5] : "clear";
   const bool rounding = !strcmp(write, "rounding");
   const bool draw = !strcmp(write, "draw") || !strcmp(write, "a2c") || rounding;
   const bool none = !strcmp(write, "none");
   const bool layer0 = !strcmp(write, "layer0");
   layers = argc > 6 ? atoi(argv[6]) : 1;
   unsigned discard_mask = argc > 7 ? strtoul(argv[7], NULL, 0) : 0;
   /* The draw writes layer 0 only. */
   if ((!draw && !none && !layer0 && strcmp(write, "clear")) || !layers ||
       (draw && layers > 1))
      goto usage;
   VkFormat fmt[MAX_ATT];
   unsigned bpp[MAX_ATT];
   char names[256];
   snprintf(names, sizeof(names), "%s", list);
   unsigned listed = 0;
   for (char *tok = strtok(names, ","); tok; tok = strtok(NULL, ",")) {
      unsigned f = 0;
      while (f < sizeof(formats) / sizeof(formats[0]) && strcmp(formats[f].name, tok))
         f++;
      if (f == sizeof(formats) / sizeof(formats[0]) || listed == MAX_ATT)
         goto usage;
      fmt[listed] = formats[f].fmt;
      bpp[listed++] = formats[f].bpp;
   }
   if (listed > 1)
      count = listed;
   if (!listed || !count || count > MAX_ATT ||
       (rounding && (count != 1 || fmt[0] != VK_FORMAT_R8G8B8A8_UNORM || samples != 4))) {
   usage:
      fprintf(stderr, "usage: %s [format[,format...]] [attachments<=8] [samples] [load|clear] "
                      "[clear|layer0|draw|a2c|rounding|none] [layers (not draw|a2c)] [discard-mask]\n",
              argv[0]);
      return 2;
   }
   for (unsigned i = listed; i < count; i++) {
      fmt[i] = fmt[0];
      bpp[i] = bpp[0];
   }
   printf("%u attachments (%s), %ux, %u layers, pass 2 loadOp %s, %s\n", count, list,
          samples, layers, pass2_load ? "LOAD" : "CLEAR",
          draw ? (strcmp(write, "a2c") ? "draw" : "draw with alpha-to-coverage")
          : none ? "no writes" : "vkCmdClearAttachments");

   VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .apiVersion = VK_API_VERSION_1_3};
   VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                               .pApplicationInfo = &app};
   VkInstance inst;
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkPhysicalDevice pdev;
   VkResult r = vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !n) {
      printf("FAIL: no physical device\n");
      return 1;
   }
   VkPhysicalDeviceProperties props;
   vkGetPhysicalDeviceProperties(pdev, &props);
   vkGetPhysicalDeviceMemoryProperties(pdev, &memprops);
   printf("device: %s\n", props.deviceName);

   float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                  .queueFamilyIndex = 0, .queueCount = 1,
                                  .pQueuePriorities = &prio};
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE};
   VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                             .pNext = &f13, .queueCreateInfoCount = 1,
                             .pQueueCreateInfos = &qci};
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   VkQueue queue;
   vkGetDeviceQueue(dev, 0, 0, &queue);

   VkPipelineLayout layout = VK_NULL_HANDLE;
   VkPipeline pipeline = draw ? make_pipeline(fmt, count, samples,
                                              !strcmp(write, "a2c"), rounding ? &(VkSampleMask){0x8} : NULL, &layout)
                              : VK_NULL_HANDLE;

   VkImage ms[MAX_ATT], res[MAX_ATT], ref[MAX_ATT];
   VkImageView ms_view[MAX_ATT] = {0}, res_view[MAX_ATT] = {0}, ref_view[MAX_ATT] = {0};
   for (unsigned i = 0; i < count; i++) {
      if (!fmt[i])
         continue;
      ms[i] = make_image(fmt[i], samples, &ms_view[i]);
      res[i] = make_image(fmt[i], VK_SAMPLE_COUNT_1_BIT, &res_view[i]);
      ref[i] = make_image(fmt[i], VK_SAMPLE_COUNT_1_BIT, &ref_view[i]);
   }

   const VkDeviceSize img_size = (VkDeviceSize)W * H * 16 * layers;
   VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                             .size = img_size * count * 2,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   VkBuffer buf;
   CHECK(vkCreateBuffer(dev, &bci, NULL, &buf));
   VkMemoryRequirements breq;
   vkGetBufferMemoryRequirements(dev, buf, &breq);
   VkDeviceMemory bmem = bind_memory(breq, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
   CHECK(vkBindBufferMemory(dev, buf, bmem, 0));

   VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool pool;
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                      .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                      .commandBufferCount = 1};
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &cai, &cb));
   VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(vkBeginCommandBuffer(cb, &cbi));

   for (unsigned i = 0; i < count; i++) {
      if (!fmt[i])
         continue;
      to_general(cb, ms[i]);
      to_general(cb, res[i]);
      to_general(cb, ref[i]);
   }

   /* Reference: pass-1 colour everywhere; the band gets the pass-2
    * colour below. */
   for (unsigned i = 0; i < count; i++) {
      if (!fmt[i])
         continue;
      VkClearColorValue c = rounding ? rounding_colour(0) : colour(i, 0);
      VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
      vkCmdClearColorImage(cb, ref[i], VK_IMAGE_LAYOUT_GENERAL, &c, 1, &range);
   }

   VkRenderingAttachmentInfo att[MAX_ATT];
   VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
                         .renderArea = {{0, 0}, {W, H}},
                         .layerCount = layers,
                         .colorAttachmentCount = count,
                         .pColorAttachments = att};
   for (unsigned pass = 0; pass < 2; pass++) {
      for (unsigned i = 0; i < count; i++) {
         att[i] = (VkRenderingAttachmentInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = ms_view[i],
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp = pass && pass2_load ? VK_ATTACHMENT_LOAD_OP_LOAD
                                         : VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue.color = rounding ? rounding_colour(0) : colour(i, 0),
         };
         /* A single-sample pass has nothing to resolve: read it back. */
         if (pass && fmt[i] && samples > 1) {
            if (discard_mask & (1u << i))
               att[i].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            att[i].resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT;
            att[i].resolveImageView = res_view[i];
            att[i].resolveImageLayout = VK_IMAGE_LAYOUT_GENERAL;
         }
      }
      barrier(cb);
      vkCmdBeginRendering(cb, &ri);
      if (pass && draw) {
         VkClearColorValue pc[8] = {0};
         for (unsigned i = 0; i < count; i++)
            pc[i] = rounding ? rounding_colour(1) : colour(i, 1);
         vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
         vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(pc), pc);
         vkCmdDraw(cb, 3, 1, 0, 0);
      } else if (pass && !none) {
         VkClearAttachment ca[MAX_ATT];
         for (unsigned i = 0; i < count; i++)
            ca[i] = (VkClearAttachment){VK_IMAGE_ASPECT_COLOR_BIT, i,
                                        {.color = colour(i, 1)}};
         VkClearRect rect = {WRITE_RECT, 0, layer0 ? 1 : layers};
         vkCmdClearAttachments(cb, count, ca, 1, &rect);
      }
      vkCmdEndRendering(cb);
   }
   barrier(cb);

   /* The band of the reference: render a single-sample pass with the same
    * vkCmdClearAttachments call, which takes no tilebuffer path under test. */
   if (!none) {
      VkRenderingAttachmentInfo ra[MAX_ATT];
      for (unsigned i = 0; i < count; i++) {
         ra[i] = (VkRenderingAttachmentInfo){
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = ref_view[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
      }
      VkRenderingInfo rri = ri;
      rri.pColorAttachments = ra;
      vkCmdBeginRendering(cb, &rri);
      VkClearAttachment ca[MAX_ATT];
      for (unsigned i = 0; i < count; i++)
         ca[i] = (VkClearAttachment){VK_IMAGE_ASPECT_COLOR_BIT, i,
                                     {.color = rounding ? rounding_colour(2) : colour(i, 1)}};
      VkClearRect rect = {WRITE_RECT, 0, layer0 ? 1 : layers};
      vkCmdClearAttachments(cb, count, ca, 1, &rect);
      vkCmdEndRendering(cb);
   }
   barrier(cb);

   for (unsigned i = 0; i < count; i++) {
      if (!fmt[i])
         continue;
      VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                                .imageExtent = {W, H, 1}};
      copy.bufferOffset = img_size * (2 * i);
      vkCmdCopyImageToBuffer(cb, samples > 1 ? res[i] : ms[i], VK_IMAGE_LAYOUT_GENERAL,
                             buf, 1, &copy);
      copy.bufferOffset = img_size * (2 * i + 1);
      vkCmdCopyImageToBuffer(cb, ref[i], VK_IMAGE_LAYOUT_GENERAL, buf, 1, &copy);
   }
   barrier(cb);

   r = vkEndCommandBuffer(cb);
   printf("vkEndCommandBuffer = %d%s\n", r,
          r == VK_ERROR_FEATURE_NOT_PRESENT ? " (VK_ERROR_FEATURE_NOT_PRESENT)" : "");
   if (r != VK_SUCCESS) {
      printf("FAIL\n");
      return 1;
   }

   VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                      .commandBufferCount = 1, .pCommandBuffers = &cb};
   CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
   CHECK(vkQueueWaitIdle(queue));

   uint8_t *map;
   CHECK(vkMapMemory(dev, bmem, 0, VK_WHOLE_SIZE, 0, (void **)&map));
   unsigned bad = 0;
   for (unsigned i = 0; i < count; i++) {
      if (!fmt[i])
         continue;
      const unsigned bpp_i = bpp[i];
      const uint8_t *got = map + img_size * (2 * i);
      const uint8_t *want = map + img_size * (2 * i + 1);
      unsigned bad_px = 0;
      for (unsigned px = 0; px < W * H * layers; px++) {
         if (memcmp(got + px * bpp_i, want + px * bpp_i, bpp_i)) {
            if (!bad_px)
               printf("attachment %u: first mismatch at (%u,%u) layer %u\n", i, px % W,
                      px / W % H, px / (W * H));
            bad_px++;
         }
      }
      if (bad_px)
         printf("attachment %u: %u/%u pixels wrong\n", i, bad_px, W * H * layers);
      bad += bad_px;
   }
   printf("%s\n", bad ? "FAIL (contents)" : "PASS");
   return bad ? 1 : 0;
}
