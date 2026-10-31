/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_event.h"
#include "vulkan/vulkan_core.h"

#include "agx_bo.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"

#define HK_EVENT_MEM_SIZE sizeof(VkResult)

VKAPI_ATTR VkResult VKAPI_CALL
hk_CreateEvent(VkDevice device, const VkEventCreateInfo *pCreateInfo,
               const VkAllocationCallbacks *pAllocator, VkEvent *pEvent)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   struct hk_event *event;

   event = vk_object_zalloc(&dev->vk, pAllocator, sizeof(*event),
                            VK_OBJECT_TYPE_EVENT);
   if (!event)
      return vk_error(dev, VK_ERROR_OUT_OF_HOST_MEMORY);

   /* TODO: this is really wasteful, bring back the NVK heap!
    *
    * XXX
    */
   event->bo =
      agx_bo_create(&dev->dev, HK_EVENT_MEM_SIZE, 0, AGX_BO_WRITEBACK, "Event");
   event->status = event->bo ? agx_bo_map(event->bo) : NULL;
   if (!event->status) {
      agx_bo_unreference(&dev->dev, event->bo);
      vk_object_free(&dev->vk, pAllocator, event);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }
   event->addr = event->bo->va->addr;

   *event->status = VK_EVENT_RESET;

   *pEvent = hk_event_to_handle(event);

   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
hk_DestroyEvent(VkDevice device, VkEvent _event,
                const VkAllocationCallbacks *pAllocator)
{
   VK_FROM_HANDLE(hk_device, dev, device);
   VK_FROM_HANDLE(hk_event, event, _event);

   if (!event)
      return;

   agx_bo_unreference(&dev->dev, event->bo);
   vk_object_free(&dev->vk, pAllocator, event);
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_GetEventStatus(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   return *event->status;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_SetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   *event->status = VK_EVENT_SET;

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   *event->status = VK_EVENT_RESET;

   return VK_SUCCESS;
}

/*
 * Write an event's status from a compute stream ordered after every earlier
 * stream of the given source kinds (Vulkan 1.3, 7.5 "Events": the first
 * synchronization scope of vkCmdSetEvent2 and vkCmdResetEvent2 includes all
 * commands earlier in submission order, restricted to the stage mask). The
 * stream is the driver's own, so every later source scope covers it.
 */
static void
hk_event_write(struct hk_cmd_buffer *cmd, struct hk_event *event,
               uint8_t src_kinds, uint32_t value)
{
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, src_kinds, cmd->current_cs.cs, dep);

   cmd->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   if (cs)
      hk_queue_write(cmd, event->bo->va->addr, value, false);
   cmd->sync.kind_override = 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_event, event, _event);

   perf_debug(cmd, "Set event");

   assert(pDependencyInfo->sType == VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
   uint8_t src_kinds, dst_kinds;
   hk_sync_dependency_kinds(pDependencyInfo, hk_sync_image_class(cmd),
                            &src_kinds, &dst_kinds);

   /* The wait orders later work behind this stream's source work, but on
    * G17P a read through another address of the same memory also needs
    * the source writes flushed with memory scope (hk_cdm_alias_scope()). */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_cs *cs = cmd->current_cs.cs;
   if (cs && (src_kinds & (HK_SYNC_COMPUTE | HK_SYNC_XFER)) &&
       hk_cdm_alias_scope(dev) && hk_cdm_visibility_flush_needed(dev, cs, true)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, true);
   }

   hk_event_write(cmd, event, src_kinds, VK_EVENT_SET);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_event, event, _event);

   perf_debug(cmd, "Reset event");
   hk_event_write(cmd, event, hk_sync_kinds_src(stageMask), VK_EVENT_RESET);
}

/* A wait polls the event word on the GPU, so a later host vkSetEvent is
 * honoured. A poll whose set never comes hangs the queue, which only G17P
 * recovers from; elsewhere the wait is the plain barrier below. */
static bool
hk_event_wait_polls(const struct hk_device *dev)
{
   return dev->dev.chip == AGX_CHIP_G17P;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdWaitEvents2(VkCommandBuffer commandBuffer, uint32_t eventCount,
                  const VkEvent *pEvents,
                  const VkDependencyInfo *pDependencyInfos)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_flush_deferred_clears(cmd);
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   perf_debug(cmd, "Wait events");

   for (uint32_t i = 0; i < eventCount; ++i) {
      VK_FROM_HANDLE(hk_event, event, pEvents[i]);
      assert(pDependencyInfos[i].sType == VK_STRUCTURE_TYPE_DEPENDENCY_INFO);

      uint8_t src_kinds, dst_kinds;
      hk_sync_dependency_kinds(&pDependencyInfos[i], hk_sync_image_class(cmd),
                               &src_kinds, &dst_kinds);

      /* Vulkan 1.3, 7.5: the first synchronization scope of vkCmdWaitEvents2
       * is the union of the set operations' scopes, which are earlier in
       * submission order; the second scope is every later command in the
       * destination stages. The wait itself is a compute stream that polls
       * the status word on G17P, so a host-side vkSetEvent is honoured;
       * ordering behind the set's source scope then carries through it. */
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, src_kinds, cmd->current_cs.cs, dep);

      if (hk_event_wait_polls(dev)) {
         cmd->sync.kind_override = HK_SYNC_ALL;
         struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
         if (cs)
            hk_g17p_meta_wait_addr(cmd, event->addr, VK_EVENT_SET);
         cmd->sync.kind_override = 0;
         if (vk_command_buffer_has_error(&cmd->vk))
            return;

         /* Later work of the destination kinds runs after the poll. The
          * poll stream is ended so it cannot absorb that work. */
         cs = cmd->current_cs.cs ? cmd->current_cs.cs : cs;
         hk_sync_after_stream(cmd, cs, dst_kinds);
         hk_cmd_buffer_end_compute(cmd);
      }

      hk_sync_barrier(cmd, src_kinds, dst_kinds);
   }
}
