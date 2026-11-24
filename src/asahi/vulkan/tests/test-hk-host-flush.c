/* Copyright 2026 Aurora Silicon
 * SPDX-License-Identifier: MIT
 *
 * Which dependencies ask for the memory-scoped host-availability flush, and
 * after which device work. No Vulkan device, DRM or GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_queue.h"

static unsigned tests, failures;

#define T(name, cond)                                                          \
   do {                                                                        \
      tests++;                                                                 \
      if (!(cond)) {                                                           \
         failures++;                                                           \
         printf("FAIL %s line%d\n", name, __LINE__);                           \
      } else                                                                   \
         printf("PASS %s\n", name);                                            \
   } while (0)

static VkMemoryBarrier2
mem(VkPipelineStageFlags2 src, VkAccessFlags2 src_access,
    VkPipelineStageFlags2 dst, VkAccessFlags2 dst_access)
{
   return (VkMemoryBarrier2){
      .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = src,
      .srcAccessMask = src_access,
      .dstStageMask = dst,
      .dstAccessMask = dst_access,
   };
}

static VkDependencyInfo
dep(const VkMemoryBarrier2 *b, uint32_t count)
{
   return (VkDependencyInfo){
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = count,
      .pMemoryBarriers = b,
   };
}

#define ALL_CMDS  VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT
#define TOP       VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT
#define BOTTOM    VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT
#define HOST      VK_PIPELINE_STAGE_2_HOST_BIT
#define CS        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
#define FS        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
#define COLOR     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#define XFER      VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT
#define H_R       VK_ACCESS_2_HOST_READ_BIT
#define H_W       VK_ACCESS_2_HOST_WRITE_BIT
#define M_R       VK_ACCESS_2_MEMORY_READ_BIT
#define M_W       VK_ACCESS_2_MEMORY_WRITE_BIT
#define S_R       VK_ACCESS_2_SHADER_READ_BIT
#define S_W       VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
#define C_W       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
#define T_W       VK_ACCESS_2_TRANSFER_WRITE_BIT

/* Which dependencies perform a device-to-host domain operation, and after
 * which device work. */
static void
test_kinds(void)
{
   const struct {
      const char *name;
      VkPipelineStageFlags2 src, dst;
      VkAccessFlags2 src_access, dst_access;
      bool host;
      uint8_t kinds;
   } cases[] = {
      /* An empty or host-only first scope follows no device write. */
      {"NONE -> HOST_READ", VK_PIPELINE_STAGE_2_NONE, HOST, 0, H_R, true, 0},
      {"TOP -> HOST_READ", TOP, HOST, 0, H_R, true, 0},
      {"HOST_WRITE -> HOST_READ", HOST, HOST, H_W, H_R, true, 0},
      /* The domain operation also takes what earlier availability
       * operations of a chain made available: read-only or empty source
       * access still flushes once device work is in the first scope. */
      {"SHADER_READ -> HOST_READ", CS, HOST, S_R, H_R, true, HK_SYNC_COMPUTE},
      {"compute, no access -> HOST_READ", CS, HOST, 0, H_R, true,
       HK_SYNC_COMPUTE},
      {"SHADER_WRITE -> HOST_READ", CS, HOST, S_W, H_R, true,
       HK_SYNC_COMPUTE},
      {"COLOR_WRITE -> HOST_WRITE", COLOR, HOST, C_W, H_W, true, HK_SYNC_GFX},
      {"TRANSFER_WRITE -> HOST_READ", XFER, HOST, T_W, H_R, true,
       HK_SYNC_XFER},
      {"MEMORY_WRITE -> MEMORY_READ at HOST", ALL_CMDS, HOST, M_W, M_R, true,
       HK_SYNC_ALL},
      {"MEMORY_WRITE -> MEMORY_WRITE at HOST", FS, HOST, M_W, M_W, true,
       HK_SYNC_GFX},
      /* BOTTOM_OF_PIPE in the first scope is ALL_COMMANDS. */
      {"BOTTOM|HOST -> TOP|HOST, HOST_WRITE|MEMORY_READ", BOTTOM | HOST,
       TOP | HOST, H_W | M_R, H_W | M_R, true, HK_SYNC_ALL},
      {"BOTTOM -> HOST, MEMORY_READ -> HOST_WRITE", BOTTOM, HOST, M_R, H_W,
       true, HK_SYNC_ALL},
      /* No host access: an execution dependency into the host stage, the
       * host waits through a fence or event instead. */
      {"execution only into HOST", CS, HOST, 0, 0, false, 0},
      /* ALL_COMMANDS does not include the host stage. */
      {"MEMORY_READ at ALL_COMMANDS", CS, ALL_CMDS, S_W, M_R, false, 0},
      /* The other direction is not a device-to-host operation. */
      {"HOST_WRITE -> SHADER_READ", HOST, CS, H_W, S_R, false, 0},
      {"SHADER_WRITE -> SHADER_READ", CS, CS, S_W, S_R, false, 0},
   };

   for (unsigned i = 0; i < ARRAY_SIZE(cases); ++i) {
      const VkMemoryBarrier2 b = mem(cases[i].src, cases[i].src_access,
                                     cases[i].dst, cases[i].dst_access);
      const VkDependencyInfo info = dep(&b, 1);
      T(cases[i].name, hk_sync_dependency_has_host_access(&info) ==
                          cases[i].host &&
                          hk_sync_host_flush_kinds(&info) == cases[i].kinds);
   }

   /* Host write to device read: no device work in the first scope, so no
    * device dependency either; the submission makes host writes before it
    * visible. Nothing changes for it. */
   {
      const VkMemoryBarrier2 b = mem(HOST, H_W, CS, S_R);
      const VkDependencyInfo info = dep(&b, 1);
      uint8_t src, dst;
      hk_sync_dependency_kinds(&info, false, &src, &dst);
      T("HOST_WRITE -> SHADER_READ: no device source", src == 0);
   }

   /* Only the barriers with a host access contribute their first scope. */
   {
      const VkMemoryBarrier2 b[] = {
         mem(COLOR, C_W, CS, S_R),
         mem(CS, S_W, HOST, H_R),
      };
      const VkDependencyInfo info = dep(b, 2);
      T("mixed dependency: host barrier's scope only",
        hk_sync_host_flush_kinds(&info) == HK_SYNC_COMPUTE);
   }

   /* Buffer and image barriers. */
   {
      const VkBufferMemoryBarrier2 bb = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
         .srcStageMask = XFER,
         .srcAccessMask = T_W,
         .dstStageMask = HOST,
         .dstAccessMask = H_R,
      };
      const VkImageMemoryBarrier2 ib = {
         .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
         .srcStageMask = COLOR,
         .srcAccessMask = C_W,
         .dstStageMask = HOST,
         .dstAccessMask = H_R,
      };
      VkDependencyInfo info = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .bufferMemoryBarrierCount = 1,
         .pBufferMemoryBarriers = &bb,
      };
      T("buffer barrier TRANSFER_WRITE -> HOST_READ",
        hk_sync_dependency_has_host_access(&info) &&
           hk_sync_host_flush_kinds(&info) == HK_SYNC_XFER);
      info = (VkDependencyInfo){
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .imageMemoryBarrierCount = 1,
         .pImageMemoryBarriers = &ib,
      };
      T("image barrier COLOR_WRITE -> HOST_READ",
        hk_sync_dependency_has_host_access(&info) &&
           hk_sync_host_flush_kinds(&info) == HK_SYNC_GFX);
   }
}

int
main(void)
{
   test_kinds();
   printf("%u tests, %u failures\n", tests, failures);
   return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
