/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Submit-time render packing (hk_render_pack.h) on reusable command buffers.
 *
 * Each command buffer records one render pass instance that loads and stores
 * the same two colour targets and depth. Submitted together, consecutive
 * ones run as one tile pass, chained through stream links written over
 * their terminate records for that submission. The command buffers are
 * recorded once, without ONE_TIME_SUBMIT, and submitted again and again in
 * other groupings: the same run twice, each one alone (a member that kept a
 * stale link would also run its old neighbours' draws), in another order,
 * around a pass on other targets, after an ordered (UAV) dependency, with a
 * simultaneous-use member, as two submissions of one vkQueueSubmit2, and
 * with the layout decay vkd3d-proton records around every render pass (the
 * attachments go to GENERAL after each pass and back before the next).
 * After every submission both colour targets of both target sets are read
 * back and compared with a CPU painter's-algorithm reference.
 *
 * Usage: hk-vulkan-render-pack-replay [rounds]   Exit status 0 = correct.
 */
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vulkan/vulkan.h>

#include "hk_vulkan_render_pack_replay_shaders.h"

#define CHECK(call)                                                            \
   do {                                                                        \
      VkResult r_ = (call);                                                    \
      if (r_ != VK_SUCCESS) {                                                  \
         fprintf(stderr, "%s: %d\n", #call, r_);                               \
         exit(2);                                                              \
      }                                                                        \
   } while (0)

#define W 96
#define H 96
#define TARGETS 2

static VkInstance inst;
static VkPhysicalDevice pdev;
static VkDevice dev;
static VkQueue queue;
static uint32_t qfam;
static VkCommandPool pool;
static VkFence fence;
static VkPipelineLayout layout;
static VkPipeline pipe;

/* A set of attachments: two colour targets and depth, with readback. */
struct set {
   VkImage img[TARGETS + 1];
   VkDeviceMemory mem[TARGETS + 1];
   VkImageView view[TARGETS + 1];
   VkBuffer rb;
   VkDeviceMemory rb_mem;
   uint32_t *rb_map;
   /* Reference contents. */
   uint32_t color[TARGETS][W * H];
   float depth[W * H];
};
static struct set sets[2];

struct draw {
   int32_t rect[4];
   uint32_t color;
   float z;
};

struct push {
   int32_t rect[4];
   float color[4];
   float scale[2];
   float z;
   float pad;
};

/* A recorded command buffer: one instance on `set`, optionally after a UAV
 * barrier (an ordered dependency on the previous instance). */
struct rec {
   const char *name;
   unsigned set;
   bool uav, simultaneous;
   /* The attachments come in GENERAL and go back to it (layout decay). */
   bool decay;
   unsigned ndraws;
   struct draw draws[3];
   /* The depth clear value passed with the LOAD (0: 1.0). */
   float clear_z;
   VkCommandBuffer cb;
};

static struct rec recs[] = {
   {"A", 0, false, false, false, 2, {{{4, 4, 60, 60}, 0xff2040e0u, 0.50f},
                              {{20, 10, 80, 40}, 0xff10c030u, 0.45f}}},
   {"B", 0, false, false, false, 2, {{{30, 30, 90, 90}, 0xffe0a010u, 0.40f},
                              {{0, 50, 40, 96}, 0xff808080u, 0.70f}}},
   {"C", 0, false, false, false, 2, {{{10, 60, 70, 95}, 0xff00ffffu, 0.42f},
                              {{50, 0, 96, 30}, 0xffff00ffu, 0.90f}}},
   {"D", 0, true, false, false, 1, {{{40, 40, 70, 70}, 0xff4080c0u, 0.30f}}},
   {"S", 0, false, true, false, 1, {{{0, 0, 30, 30}, 0xffc04080u, 0.35f}}},
   {"X", 1, false, false, false, 1, {{{8, 8, 88, 88}, 0xff123456u, 0.60f}}},
   {"E", 0, false, false, true, 2, {{{4, 4, 60, 60}, 0xff2040e0u, 0.50f},
                                    {{20, 10, 80, 40}, 0xff10c030u, 0.45f}}},
   {"F", 0, false, false, true, 2, {{{30, 30, 90, 90}, 0xffe0a010u, 0.40f},
                                    {{0, 50, 40, 96}, 0xff808080u, 0.70f}}},
   {"G", 0, false, false, true, 1, {{{10, 60, 70, 95}, 0xff00ffffu, 0.42f}}},
   /* Aliasable images (set 1): Y packs after X; Z's UAV barrier is an
    * ordered dependency, which keeps an aliasable image's instances apart. */
   {"Y", 1, false, false, false, 2, {{{30, 30, 90, 90}, 0xffe0a010u, 0.40f},
                                    {{0, 50, 40, 96}, 0xff808080u, 0.70f}},
    0.25f},
   {"Z", 1, true, false, false, 1, {{{10, 60, 70, 95}, 0xff00ffffu, 0.42f}}},
};
enum { A, B, C, D, S, X, E, F, G, Y, Z, NREC };

static uint32_t
memory_type(uint32_t bits, VkMemoryPropertyFlags flags)
{
   VkPhysicalDeviceMemoryProperties m;
   vkGetPhysicalDeviceMemoryProperties(pdev, &m);
   for (uint32_t i = 0; i < m.memoryTypeCount; i++)
      if ((bits & (1u << i)) && (m.memoryTypes[i].propertyFlags & flags) == flags)
         return i;
   fprintf(stderr, "no memory type\n");
   exit(2);
}

static void
make_image(VkImageCreateFlags flags, VkFormat fmt, VkImageUsageFlags usage,
           VkImageAspectFlags aspect, VkImage *img, VkDeviceMemory *mem,
           VkImageView *view)
{
   VkImageCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .flags = flags,
      .imageType = VK_IMAGE_TYPE_2D,
      .format = fmt,
      .extent = {W, H, 1},
      .mipLevels = 1,
      .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = usage,
   };
   CHECK(vkCreateImage(dev, &ici, NULL, img));
   VkMemoryRequirements req;
   vkGetImageMemoryRequirements(dev, *img, &req);
   VkMemoryAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = req.size,
      .memoryTypeIndex =
         memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
   };
   CHECK(vkAllocateMemory(dev, &ai, NULL, mem));
   CHECK(vkBindImageMemory(dev, *img, *mem, 0));
   VkImageViewCreateInfo vci = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
      .image = *img,
      .viewType = VK_IMAGE_VIEW_TYPE_2D,
      .format = fmt,
      .subresourceRange = {aspect, 0, 1, 0, 1},
   };
   CHECK(vkCreateImageView(dev, &vci, NULL, view));
}

static void
begin(VkCommandBuffer cb, VkCommandBufferUsageFlags flags)
{
   VkCommandBufferBeginInfo bi = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
      .flags = flags,
   };
   CHECK(vkBeginCommandBuffer(cb, &bi));
}

static void
barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
        VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access)
{
   VkMemoryBarrier2 mb = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src, .srcAccessMask = src_access,
      .dstStageMask = dst, .dstAccessMask = dst_access,
   };
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

/* Every attachment of set s between its attachment layout and GENERAL, as
 * vkd3d-proton's render pass transitions record it: attachment writes to
 * shader reads after a pass, and back before the next. */
static void
transition(VkCommandBuffer cb, const struct set *s, bool to_general)
{
   VkImageMemoryBarrier2 b[TARGETS + 1];
   for (unsigned t = 0; t <= TARGETS; t++) {
      const bool depth = t == TARGETS;
      const VkImageLayout att = depth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                                      : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      const VkPipelineStageFlags2 att_stages =
         depth ? VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT
               : VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      const VkAccessFlags2 att_access =
         depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
               : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                    VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
      const VkPipelineStageFlags2 shader_stages =
         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
      b[t] = (VkImageMemoryBarrier2){
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = to_general ? att_stages : shader_stages,
         .srcAccessMask = to_general ? att_access : 0,
         .dstStageMask = to_general ? shader_stages : att_stages,
         .dstAccessMask = to_general ? VK_ACCESS_2_SHADER_SAMPLED_READ_BIT : att_access,
         .oldLayout = to_general ? att : VK_IMAGE_LAYOUT_GENERAL,
         .newLayout = to_general ? VK_IMAGE_LAYOUT_GENERAL : att,
         .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
         .image = s->img[t],
         .subresourceRange = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                              0, 1, 0, 1},
      };
   }
   VkDependencyInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .imageMemoryBarrierCount = TARGETS + 1, .pImageMemoryBarriers = b,
   };
   vkCmdPipelineBarrier2(cb, &di);
}

static void
render(VkCommandBuffer cb, const struct set *s, VkAttachmentLoadOp load,
       float clear_z)
{
   VkRenderingAttachmentInfo ca[TARGETS];
   for (unsigned t = 0; t < TARGETS; t++)
      ca[t] = (VkRenderingAttachmentInfo){
         .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
         .imageView = s->view[t],
         .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
         .loadOp = load,
         .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      };
   VkRenderingAttachmentInfo da = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
      .imageView = s->view[TARGETS],
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      .loadOp = load,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
      .clearValue.depthStencil = {clear_z, 0},
   };
   VkRenderingInfo ri = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {W, H}},
      .layerCount = 1,
      .colorAttachmentCount = TARGETS,
      .pColorAttachments = ca,
      .pDepthAttachment = &da,
   };
   vkCmdBeginRendering(cb, &ri);
}

static void
record(struct rec *r)
{
   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   CHECK(vkAllocateCommandBuffers(dev, &ai, &r->cb));
   begin(r->cb, r->simultaneous ? VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT : 0);
   if (r->uav) {
      const VkPipelineStageFlags2 stages = VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT |
                                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
      const VkAccessFlags2 access = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      barrier(r->cb, stages, access, stages, access);
   }
   if (r->decay)
      transition(r->cb, &sets[r->set], false);
   /* A loaded attachment's clear value is unused; vkd3d-proton passes one
    * anyway, and Y's differs from X's. */
   render(r->cb, &sets[r->set], VK_ATTACHMENT_LOAD_OP_LOAD,
          r->clear_z ? r->clear_z : 1.0f);
   VkViewport v = {0, 0, W, H, 0, 1};
   VkRect2D sc = {{0, 0}, {W, H}};
   vkCmdSetViewport(r->cb, 0, 1, &v);
   vkCmdSetScissor(r->cb, 0, 1, &sc);
   vkCmdBindPipeline(r->cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
   for (unsigned d = 0; d < r->ndraws; d++) {
      const struct draw *dr = &r->draws[d];
      struct push pc = {
         .rect = {dr->rect[0], dr->rect[1], dr->rect[2], dr->rect[3]},
         .color = {(float)(dr->color & 0xff) / 255.0f,
                   (float)((dr->color >> 8) & 0xff) / 255.0f,
                   (float)((dr->color >> 16) & 0xff) / 255.0f,
                   (float)(dr->color >> 24) / 255.0f},
         .scale = {2.0f / W, 2.0f / H},
         .z = dr->z,
      };
      vkCmdPushConstants(r->cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(pc), &pc);
      vkCmdDraw(r->cb, 6, 1, 0, 0);
   }
   vkCmdEndRendering(r->cb);
   if (r->decay)
      transition(r->cb, &sets[r->set], true);
   CHECK(vkEndCommandBuffer(r->cb));
}

static uint32_t
swizzle_bgra(uint32_t c)
{
   return (c & 0xff00ff00u) | ((c & 0xffu) << 16) | ((c >> 16) & 0xffu);
}

/* What the instance of `r` leaves in its set's reference. */
static void
apply(const struct rec *r)
{
   struct set *s = &sets[r->set];
   for (unsigned d = 0; d < r->ndraws; d++) {
      const struct draw *dr = &r->draws[d];
      for (int y = dr->rect[1]; y < dr->rect[3]; y++) {
         for (int x = dr->rect[0]; x < dr->rect[2]; x++) {
            const unsigned i = y * W + x;
            if (dr->z <= s->depth[i]) {
               s->depth[i] = dr->z;
               s->color[0][i] = dr->color;
               s->color[1][i] = swizzle_bgra(dr->color);
            }
         }
      }
   }
}

static VkCommandBuffer
oneshot(void)
{
   VkCommandBufferAllocateInfo ai = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
      .commandBufferCount = 1,
   };
   VkCommandBuffer cb;
   CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
   begin(cb, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
   return cb;
}

static void
run(VkCommandBuffer cb)
{
   CHECK(vkEndCommandBuffer(cb));
   VkCommandBufferSubmitInfo ci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb,
   };
   VkSubmitInfo2 si = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1, .pCommandBufferInfos = &ci,
   };
   CHECK(vkQueueSubmit2(queue, 1, &si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
   CHECK(vkResetFences(dev, 1, &fence));
   vkFreeCommandBuffers(dev, pool, 1, &cb);
}

/* Both sets back to colour 0, depth 1. */
static void
clear(void)
{
   VkCommandBuffer cb = oneshot();
   for (unsigned s = 0; s < 2; s++) {
      render(cb, &sets[s], VK_ATTACHMENT_LOAD_OP_CLEAR, 1.0f);
      vkCmdEndRendering(cb);
      memset(sets[s].color, 0, sizeof(sets[s].color));
      for (unsigned i = 0; i < W * H; i++)
         sets[s].depth[i] = 1.0f;
   }
   run(cb);
}

/* One vkQueueSubmit2 of `count` submissions, submission i holding the
 * command buffers lists[i][0..lens[i]). */
static void
submit(unsigned count, const unsigned *const *lists, const unsigned *lens)
{
   VkCommandBufferSubmitInfo ci[2][NREC];
   VkSubmitInfo2 si[2];
   for (unsigned s = 0; s < count; s++) {
      for (unsigned i = 0; i < lens[s]; i++)
         ci[s][i] = (VkCommandBufferSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
            .commandBuffer = recs[lists[s][i]].cb,
         };
      si[s] = (VkSubmitInfo2){
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
         .commandBufferInfoCount = lens[s], .pCommandBufferInfos = ci[s],
      };
   }
   CHECK(vkQueueSubmit2(queue, count, si, fence));
   CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
   CHECK(vkResetFences(dev, 1, &fence));
   for (unsigned s = 0; s < count; s++)
      for (unsigned i = 0; i < lens[s]; i++)
         apply(&recs[lists[s][i]]);
}

static bool
verify(const char *what)
{
   VkCommandBuffer cb = oneshot();
   for (unsigned s = 0; s < 2; s++) {
      for (unsigned t = 0; t < TARGETS; t++) {
         VkImageMemoryBarrier2 b = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sets[s].img[t],
            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
         };
         VkDependencyInfo di = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b,
         };
         vkCmdPipelineBarrier2(cb, &di);
         VkBufferImageCopy r = {
            .bufferOffset = (VkDeviceSize)t * W * H * 4,
            .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {W, H, 1},
         };
         vkCmdCopyImageToBuffer(cb, sets[s].img[t], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                sets[s].rb, 1, &r);
         b.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
         b.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
         b.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
         b.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
         b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
         b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
         vkCmdPipelineBarrier2(cb, &di);
      }
   }
   barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
           VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
   run(cb);

   for (unsigned s = 0; s < 2; s++) {
      for (unsigned t = 0; t < TARGETS; t++) {
         const uint32_t *got = sets[s].rb_map + (size_t)t * W * H;
         for (unsigned i = 0; i < W * H; i++) {
            if (got[i] != sets[s].color[t][i]) {
               fprintf(stderr, "FAIL %s: set %u target %u pixel (%u,%u): got 0x%08x want 0x%08x\n",
                       what, s, t, i % W, i / W, got[i], sets[s].color[t][i]);
               return false;
            }
         }
      }
   }
   return true;
}

static void
setup(void)
{
   VkApplicationInfo app = {
      .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
      .pApplicationName = "hk-vulkan-render-pack-replay",
      .apiVersion = VK_API_VERSION_1_3,
   };
   VkInstanceCreateInfo ici = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app,
   };
   CHECK(vkCreateInstance(&ici, NULL, &inst));
   uint32_t n = 1;
   VkResult r = vkEnumeratePhysicalDevices(inst, &n, &pdev);
   if ((r != VK_SUCCESS && r != VK_INCOMPLETE) || !n) {
      fprintf(stderr, "no Vulkan device\n");
      exit(2);
   }
   uint32_t nq = 0;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, NULL);
   VkQueueFamilyProperties qp[16];
   nq = nq > 16 ? 16 : nq;
   vkGetPhysicalDeviceQueueFamilyProperties(pdev, &nq, qp);
   for (qfam = 0; qfam < nq && !(qp[qfam].queueFlags & VK_QUEUE_GRAPHICS_BIT); qfam++)
      ;
   const float prio = 1.0f;
   VkDeviceQueueCreateInfo qci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = qfam, .queueCount = 1, .pQueuePriorities = &prio,
   };
   VkPhysicalDeviceVulkan13Features f13 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
      .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE,
   };
   VkDeviceCreateInfo dci = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &f13,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci,
   };
   CHECK(vkCreateDevice(pdev, &dci, NULL, &dev));
   vkGetDeviceQueue(dev, qfam, 0, &queue);
   VkCommandPoolCreateInfo pci = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
      .queueFamilyIndex = qfam,
   };
   CHECK(vkCreateCommandPool(dev, &pci, NULL, &pool));
   VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   CHECK(vkCreateFence(dev, &fci, NULL, &fence));

   for (unsigned s = 0; s < 2; s++) {
      /* Set 1 is created as vkd3d-proton creates placed resources. */
      const VkImageCreateFlags flags = s ? VK_IMAGE_CREATE_ALIAS_BIT : 0;
      for (unsigned t = 0; t < TARGETS; t++)
         make_image(flags, VK_FORMAT_R8G8B8A8_UNORM,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, &sets[s].img[t], &sets[s].mem[t],
                    &sets[s].view[t]);
      make_image(flags, VK_FORMAT_D32_SFLOAT, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                 VK_IMAGE_ASPECT_DEPTH_BIT, &sets[s].img[TARGETS], &sets[s].mem[TARGETS],
                 &sets[s].view[TARGETS]);
      VkBufferCreateInfo bci = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
         .size = (VkDeviceSize)TARGETS * W * H * 4,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      };
      CHECK(vkCreateBuffer(dev, &bci, NULL, &sets[s].rb));
      VkMemoryRequirements req;
      vkGetBufferMemoryRequirements(dev, sets[s].rb, &req);
      VkMemoryAllocateInfo ai = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = req.size,
         .memoryTypeIndex = memory_type(req.memoryTypeBits,
                                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                           VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
      };
      CHECK(vkAllocateMemory(dev, &ai, NULL, &sets[s].rb_mem));
      CHECK(vkBindBufferMemory(dev, sets[s].rb, sets[s].rb_mem, 0));
      CHECK(vkMapMemory(dev, sets[s].rb_mem, 0, VK_WHOLE_SIZE, 0, (void **)&sets[s].rb_map));
   }

   /* Attachment layouts once; every pass after keeps them. */
   VkCommandBuffer cb = oneshot();
   for (unsigned s = 0; s < 2; s++) {
      for (unsigned t = 0; t <= TARGETS; t++) {
         const bool depth = t == TARGETS;
         VkImageMemoryBarrier2 b = {
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_GRAPHICS_BIT,
            .dstAccessMask = depth ? VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                                   : VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = depth ? VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL
                               : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = sets[s].img[t],
            .subresourceRange = {depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                                 0, 1, 0, 1},
         };
         VkDependencyInfo di = {
            .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
            .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b,
         };
         vkCmdPipelineBarrier2(cb, &di);
      }
   }
   run(cb);

   VkPushConstantRange pr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                             sizeof(struct push)};
   VkPipelineLayoutCreateInfo lci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .pushConstantRangeCount = 1, .pPushConstantRanges = &pr,
   };
   CHECK(vkCreatePipelineLayout(dev, &lci, NULL, &layout));
   VkShaderModuleCreateInfo vs = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(rp_vert), .pCode = rp_vert,
   };
   VkShaderModuleCreateInfo fs = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = sizeof(rp_frag), .pCode = rp_frag,
   };
   VkShaderModule mv, mf;
   CHECK(vkCreateShaderModule(dev, &vs, NULL, &mv));
   CHECK(vkCreateShaderModule(dev, &fs, NULL, &mf));
   VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = mv, .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = mf, .pName = "main"},
   };
   VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
   };
   VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   VkPipelineViewportStateCreateInfo vps = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1,
   };
   VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f,
   };
   VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE,
      .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
   };
   VkPipelineColorBlendAttachmentState ba[TARGETS] = {{.colorWriteMask = 0xf},
                                                      {.colorWriteMask = 0xf}};
   VkPipelineColorBlendStateCreateInfo cbs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = TARGETS, .pAttachments = ba,
   };
   VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dyn,
   };
   const VkFormat fmts[TARGETS] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
   VkPipelineRenderingCreateInfo rci = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = TARGETS, .pColorAttachmentFormats = fmts,
      .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT,
   };
   VkGraphicsPipelineCreateInfo gci = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rci,
      .stageCount = 2, .pStages = stages,
      .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vps,
      .pRasterizationState = &rs, .pMultisampleState = &ms, .pDepthStencilState = &dss,
      .pColorBlendState = &cbs, .pDynamicState = &ds, .layout = layout,
   };
   CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gci, NULL, &pipe));
   vkDestroyShaderModule(dev, mv, NULL);
   vkDestroyShaderModule(dev, mf, NULL);
}

struct scenario {
   const char *name;
   /* Set 0 waits in GENERAL around the submission (layout decay). */
   bool decay;
   unsigned count;
   unsigned lens[2];
   unsigned lists[2][NREC];
};

/* Run, replay, then the members alone and in other groupings: each must
 * show exactly its own command buffers' draws. */
static const struct scenario scenarios[] = {
   {"A B C", false, 1, {3}, {{A, B, C}}},
   {"A B C replayed", false, 1, {3}, {{A, B, C}}},
   {"C alone", false, 1, {1}, {{C}}},
   {"A alone (no stale link to B)", false, 1, {1}, {{A}}},
   {"B alone (no stale link to C)", false, 1, {1}, {{B}}},
   {"B A (other order)", false, 1, {2}, {{B, A}}},
   {"A B C after B A", false, 1, {3}, {{A, B, C}}},
   {"A D (ordered UAV dependency)", false, 1, {2}, {{A, D}}},
   {"A X B (other targets between)", false, 1, {3}, {{A, X, B}}},
   {"A S (simultaneous-use last)", false, 1, {2}, {{A, S}}},
   {"S A (simultaneous-use first)", false, 1, {2}, {{S, A}}},
   {"A B + C (two submissions)", false, 2, {2, 1}, {{A, B}, {C}}},
   {"C B A D", false, 1, {4}, {{C, B, A, D}}},
   {"A alone again", false, 1, {1}, {{A}}},
   {"E F G (vkd3d layout decay)", true, 1, {3}, {{E, F, G}}},
   {"E F G replayed", true, 1, {3}, {{E, F, G}}},
   {"F alone (no stale link to G)", true, 1, {1}, {{F}}},
   {"G E", true, 1, {2}, {{G, E}}},
   {"X Y (aliasable images)", false, 1, {2}, {{X, Y}}},
   {"X Y replayed", false, 1, {2}, {{X, Y}}},
   {"Y alone (no stale link)", false, 1, {1}, {{Y}}},
   {"X Y Z (ordered after an alias)", false, 1, {3}, {{X, Y, Z}}},
};

int
main(int argc, char **argv)
{
   const unsigned rounds = argc > 1 ? (unsigned)atoi(argv[1]) : 20;
   setup();
   for (unsigned i = 0; i < NREC; i++)
      record(&recs[i]);

   unsigned checks = 0;
   for (unsigned round = 0; round < rounds; round++) {
      for (unsigned s = 0; s < sizeof(scenarios) / sizeof(scenarios[0]); s++) {
         const struct scenario *sc = &scenarios[s];
         const unsigned *lists[2] = {sc->lists[0], sc->lists[1]};
         clear();
         if (sc->decay) {
            VkCommandBuffer cb = oneshot();
            transition(cb, &sets[0], true);
            run(cb);
         }
         submit(sc->count, lists, sc->lens);
         if (sc->decay) {
            VkCommandBuffer cb = oneshot();
            transition(cb, &sets[0], false);
            run(cb);
         }
         char what[128];
         snprintf(what, sizeof(what), "round %u, %s", round, sc->name);
         if (!verify(what))
            return 1;
         checks++;
      }
   }
   printf("hk-vulkan-render-pack-replay: %u scenario submissions exact\n", checks);
   return 0;
}
