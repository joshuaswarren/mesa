/*
 * Copyright 2024 Valve Corporation
 * Copyright 2024 Alyssa Rosenzweig
 * Copyright 2022-2023 Collabora Ltd. and Red Hat Inc.
 * SPDX-License-Identifier: MIT
 */
#include "hk_event.h"
#include "vulkan/vulkan_core.h"

#include "agx_bo.h"
#include "agx_device.h"
#include "hk_cmd_buffer.h"
#include "hk_device.h"
#include "hk_entrypoints.h"
#include "hk_frame_anatomy.h"
#include "hk_query_oq.h"
#include "hk_query_pool.h"
#include "util/bitscan.h"
#include "util/u_atomic.h"
#include "vk_common_entrypoints.h"

#define HK_EVENT_SLAB_SIZE  (16u * 1024u)
#define HK_EVENT_SLOT_SIZE  sizeof(uint64_t)
#define HK_EVENT_SLAB_SLOTS (HK_EVENT_SLAB_SIZE / HK_EVENT_SLOT_SIZE)
#define HK_EVENT_SLAB_WORDS (HK_EVENT_SLAB_SLOTS / 64u)

struct hk_event_slab {
   struct list_head link;
   struct agx_bo *bo;
   uint64_t *map;
   uint32_t ts_handle;
   uint32_t used_count;
   uint64_t used[HK_EVENT_SLAB_WORDS];
};

void
hk_events_init(struct hk_device *dev)
{
   simple_mtx_init(&dev->events.lock, mtx_plain);
   list_inithead(&dev->events.slabs);
   dev->events.ready = true;
}

void
hk_events_finish(struct hk_device *dev)
{
   if (!dev->events.ready)
      return;

   list_for_each_entry_safe(struct hk_event_slab, slab, &dev->events.slabs,
                            link) {
      list_del(&slab->link);
      if (slab->ts_handle) {
         hk_query_stamps_unbind(dev, slab->ts_handle);
         hk_frame_anatomy_unbind_pool(dev, slab->ts_handle);
         dev->dev.ops.bo_unbind_object(&dev->dev, slab->ts_handle);
      }
      agx_bo_unreference(&dev->dev, slab->bo);
      free(slab);
   }

   simple_mtx_destroy(&dev->events.lock);
   dev->events.ready = false;
}

static struct hk_event_slab *
hk_event_slab_create(struct hk_device *dev)
{
   struct hk_event_slab *slab = calloc(1, sizeof(*slab));
   if (!slab)
      return NULL;

   /* Status words ride on retirement timestamps only where the kernel lets a
    * compute command skip host timestamp publication (hk_sync_stamp_event());
    * elsewhere a timestamp binding would only add HK_SYNC_STAMP ordering and
    * query access to every event write. */
   const bool want_stamp =
      hk_oq_render_stamps_supported(dev->dev.chip) &&
      (dev->dev.params.features & DRM_ASAHI_FEATURE_COMPUTE_NO_QUERY_ACCESS);
   enum agx_bo_flags flags = AGX_BO_WRITEBACK;
   if (want_stamp)
      flags |= AGX_BO_SHARED;

   slab->bo =
      agx_bo_create(&dev->dev, HK_EVENT_SLAB_SIZE, 0, flags, "Event slab");
   slab->map = slab->bo ? agx_bo_map(slab->bo) : NULL;
   if (!slab->map) {
      agx_bo_unreference(&dev->dev, slab->bo);
      free(slab);
      return NULL;
   }

   memset(slab->map, 0, HK_EVENT_SLAB_SIZE);

   if (want_stamp) {
      if (agx_bind_timestamps(&dev->dev, slab->bo, &slab->ts_handle) == 0) {
         hk_frame_anatomy_bind_pool(dev, slab->ts_handle, slab->map,
                                    slab->bo->size);
      } else {
         slab->ts_handle = 0;
      }
   }

   list_addtail(&slab->link, &dev->events.slabs);
   return slab;
}

static bool
hk_event_alloc_slot(struct hk_device *dev, struct hk_event_slab **slab_out,
                    uint32_t *slot_out)
{
   simple_mtx_lock(&dev->events.lock);
   struct hk_event_slab *chosen = NULL;
   list_for_each_entry(struct hk_event_slab, slab, &dev->events.slabs, link) {
      if (slab->used_count < HK_EVENT_SLAB_SLOTS) {
         chosen = slab;
         break;
      }
   }
   if (!chosen) {
      chosen = hk_event_slab_create(dev);
      if (!chosen) {
         simple_mtx_unlock(&dev->events.lock);
         return false;
      }
   }

   for (unsigned w = 0; w < HK_EVENT_SLAB_WORDS; ++w) {
      const uint64_t avail = ~chosen->used[w];
      if (!avail)
         continue;
      const unsigned bit = ffsll(avail) - 1;
      chosen->used[w] |= BITFIELD64_BIT(bit);
      chosen->used_count++;
      *slab_out = chosen;
      *slot_out = w * 64u + bit;
      simple_mtx_unlock(&dev->events.lock);
      return true;
   }

   UNREACHABLE("slab reported free slot but bitmap was full");
}

static void
hk_event_free_slot(struct hk_device *dev, struct hk_event_slab *slab,
                   uint32_t slot)
{
   simple_mtx_lock(&dev->events.lock);
   const unsigned w = slot / 64u;
   const unsigned bit = slot % 64u;
   assert(slab->used[w] & BITFIELD64_BIT(bit));
   slab->used[w] &= ~BITFIELD64_BIT(bit);
   slab->used_count--;
   simple_mtx_unlock(&dev->events.lock);
}

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

   if (!hk_event_alloc_slot(dev, &event->slab, &event->slot)) {
      vk_object_free(&dev->vk, pAllocator, event);
      return vk_error(dev, VK_ERROR_OUT_OF_DEVICE_MEMORY);
   }

   event->bo = event->slab->bo;
   event->ts_handle = event->slab->ts_handle;
   event->ts_offset_B = event->slot * HK_EVENT_SLOT_SIZE;
   event->addr = event->bo->va->addr + event->ts_offset_B;
   event->status = &event->slab->map[event->slot];

   p_atomic_set(event->status, 0);

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

   if (event->slab)
      hk_event_free_slot(dev, event->slab, event->slot);
   vk_object_free(&dev->vk, pAllocator, event);
}

/* Status encoding: 0 is VK_EVENT_RESET; any nonzero 64-bit value (1 from
 * host/GPU set, or a strictly positive kernel retirement timestamp) is
 * VK_EVENT_SET. */
VKAPI_ATTR VkResult VKAPI_CALL
hk_GetEventStatus(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   return p_atomic_read(event->status) ? VK_EVENT_SET : VK_EVENT_RESET;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_SetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   p_atomic_set(event->status, 1);

   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
hk_ResetEvent(VkDevice device, VkEvent _event)
{
   VK_FROM_HANDLE(hk_event, event, _event);

   p_atomic_set(event->status, 0);

   return VK_SUCCESS;
}

/*
 * Write an event's status from a compute stream ordered after every earlier
 * stream of the given source kinds (Vulkan 1.3, 7.5 "Events": the first
 * synchronization scope of vkCmdSetEvent2 and vkCmdResetEvent2 includes all
 * commands earlier in submission order, restricted to the stage mask). The
 * stream is the driver's own, so every later source scope covers it.
 *
 * The host can read the status (vkGetEventStatus) before the submission
 * completes, so the host availability deferred to the end of the submission
 * (hk_sync_defer_host_flush()) is made here instead: the stream also follows
 * those dependencies' first scopes and flushes with memory scope before the
 * write.
 */
static void
hk_event_write(struct hk_cmd_buffer *cmd, struct hk_event *event,
               uint8_t src_kinds, bool set)
{
   /* An earlier timestamp write is in a non-empty first scope: the host may
    * read its query once it sees the event. */
   if (src_kinds)
      hk_sync_flush_stamp_copies(cmd);
   const uint8_t host_kinds = hk_sync_take_host_flush(cmd);
   uint8_t scope = hk_sync_stamp_scope(src_kinds) | host_kinds;
   /* A stamped event may have been set by a retirement timestamp on an
    * earlier stream; include HK_SYNC_STAMP even for an empty source scope so
    * a subsequent reset orders after that retirement publication. */
   if (event->ts_handle)
      scope |= HK_SYNC_STAMP;
   struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
   hk_sync_src_dep(cmd, scope, cmd->current_cs.cs, dep);

   cmd->sync.kind_override = HK_SYNC_ALL;
   struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
   if (cs) {
      struct hk_device *dev = hk_cmd_buffer_device(cmd);
      if (src_kinds) {
         /* A host observer polling vkGetEventStatus() may read query results
          * once this event is set; withhold NO_QUERY_ACCESS so pending
          * host-published stamps in the first scope land before the write. */
         cs->query_access_untracked = true;
      }
      if (event->ts_handle) {
         hk_cmd_note_query_range(cmd, cs, event->ts_handle, event->ts_offset_B,
                                 HK_EVENT_SLOT_SIZE);
      }
      if (host_kinds) {
         hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
         hk_cdm_visibility_flush(dev, cs, true);
      }
      if (event->ts_handle) {
         if (agx_chip_native_agx3(dev->dev.chip)) {
            /* Clear/write all 64 bits in a single dispatch so a reset wipes any
             * upper bits left by an earlier 64-bit retirement timestamp. */
            hk_cdm_cache_flush(dev, cs);
            hk_g17p_meta_fill_addr(cmd, &cmd->current_cs.cs, event->addr,
                                   HK_EVENT_SLOT_SIZE, set ? 1 : 0);
         } else {
            /* Write the upper 32-bit half first on reset so a host poll never
             * observes a cleared lower word while upper timestamp bits remain. */
            if (!set)
               hk_queue_write(cmd, event->addr + sizeof(uint32_t), 0, false);
            hk_queue_write(cmd, event->addr, set ? 1 : 0, false);
         }
      } else {
         hk_queue_write(cmd, event->addr, set ? 1 : 0, false);
      }
   } else {
      cmd->sync.host_flush_kinds |= host_kinds;
   }
   cmd->sync.kind_override = 0;
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdSetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                const VkDependencyInfo *pDependencyInfo)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_event, event, _event);

   cmd->state.uses_events = true;
   perf_debug(cmd, "Set event");

   assert(pDependencyInfo->sType == VK_STRUCTURE_TYPE_DEPENDENCY_INFO);
   uint8_t src_kinds, dst_kinds;
   hk_sync_dependency_kinds(pDependencyInfo, hk_sync_image_class(cmd),
                            &src_kinds, &dst_kinds);

   /* The wait orders later work behind this stream's source work, but on
    * G17P and M5 a read through another address of the same memory also
    * needs the source writes flushed with memory scope
    * (hk_cdm_alias_scope()). A command buffer that uses events records no
    * patch site for a later host import: it has the scope whenever an
    * import could add such an address (hk_cmd_alias_scope()). */
   struct hk_device *dev = hk_cmd_buffer_device(cmd);
   struct hk_cs *cs = cmd->current_cs.cs;
   if (cs && (src_kinds & (HK_SYNC_COMPUTE | HK_SYNC_XFER)) &&
       hk_cmd_alias_scope(cmd) && hk_cdm_visibility_flush_needed(dev, cs, true)) {
      hk_ensure_cs_has_space(cmd, cs, AGX3_CDM_BARRIER_SIZE);
      hk_cdm_visibility_flush(dev, cs, true);
   }

   /* A host access in this dependency is made available before the status
    * the host polls is written (hk_event_write()). */
   if (hk_cdm_memory_scope_form(dev))
      hk_sync_defer_host_flush(cmd, pDependencyInfo);

   if (event->ts_handle) {
      const struct agx_timestamp_req req = {
         .handle = event->ts_handle,
         .offset_B = event->ts_offset_B,
         .event = true,
      };
      if (hk_sync_stamp_event(cmd, req, src_kinds))
         return;
   }

   hk_event_write(cmd, event, src_kinds, true);
}

VKAPI_ATTR void VKAPI_CALL
hk_CmdResetEvent2(VkCommandBuffer commandBuffer, VkEvent _event,
                  VkPipelineStageFlags2 stageMask)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);
   hk_flush_deferred_clears(cmd);
   VK_FROM_HANDLE(hk_event, event, _event);

   cmd->state.uses_events = true;
   perf_debug(cmd, "Reset event");
   hk_event_write(cmd, event, hk_sync_kinds_src(stageMask), false);
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
   cmd->state.uses_events = true;
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
      uint8_t wait_scope = src_kinds;
      if (event->ts_handle)
         wait_scope |= HK_SYNC_STAMP;
      struct hk_sync_dep dep[HK_SYNC_NUM_SQ];
      hk_sync_src_dep(cmd, wait_scope, cmd->current_cs.cs, dep);

      if (hk_event_wait_polls(dev)) {
         cmd->sync.kind_override = HK_SYNC_ALL;
         struct hk_cs *cs = hk_sync_get_cs_after(cmd, dep);
         if (cs) {
            if (event->ts_handle) {
               hk_cmd_note_query_range(cmd, cs, event->ts_handle,
                                       event->ts_offset_B, HK_EVENT_SLOT_SIZE);
            }
            hk_g17p_meta_wait_addr(cmd, event->addr, 1);
         }
         cmd->sync.kind_override = 0;
         if (vk_command_buffer_has_error(&cmd->vk))
            return;

         /* Later work of the destination kinds runs after the poll. The
          * poll stream is ended so it cannot absorb that work. */
         cs = cmd->current_cs.cs ? cmd->current_cs.cs : cs;
         hk_sync_after_stream(cmd, cs, dst_kinds);
         hk_cmd_buffer_end_compute(cmd);
         brl_sync_event_poll(cmd, cs);
      }

      /* A layout transition of this wait (brl_discard_init_prepare()) follows
       * the polls recorded so far, so the legacy route below, whose image
       * barriers come in a pipeline barrier after the last wait, stays
       * ordered after every event it waits for. Without a poll it follows
       * the wait's first scope as the stage history has it now. A host
       * operation there needs no GPU wait: the host sets the event before it
       * submits the wait (VUID-vkSetEvent-event-09543). */
      uint8_t init_dst;
      const bool init =
         brl_discard_init_prepare(cmd, &pDependencyInfos[i], &init_dst);

      /* AGX2 has no polling wait; this boundary is an ordering dependency
       * like a pipeline barrier. Carry it to the next CDM launch (or to the
       * trailing edge of the open compute stream when it closes). */
      if (agx_chip_command_stream_generation(dev->dev.chip) ==
              AGX_CS_GENERATION_AGX2 &&
          !HK_PERF(dev, FULLCDMBARRIER)) {
         cmd->state.cs.cdm_barrier_pending = true;
         cmd->state.cs.cdm_barrier_full_needed = true;
      }
      hk_sync_barrier(cmd, src_kinds, dst_kinds);
      if (init)
         brl_discard_init_record(cmd, &pDependencyInfos[i], init_dst);
   }

   if (!cmd->state.brl_legacy_event_wait)
      brl_sync_event_done(cmd);
}

/* The common lowering of vkCmdWaitEvents waits on the events with an
 * execution-only dependency and passes the barriers in a following
 * vkCmdPipelineBarrier, which knows nothing of the events. Keep the polls
 * recorded (brl_sync_event_poll()) until that barrier has been recorded, so a
 * layout transition it performs is still ordered after the events. The
 * barrier's first scope is the wait's source stages, so a wait on host and
 * device events keeps the device stages. */
VKAPI_ATTR void VKAPI_CALL
hk_CmdWaitEvents(VkCommandBuffer commandBuffer, uint32_t eventCount,
                 const VkEvent *pEvents, VkPipelineStageFlags srcStageMask,
                 VkPipelineStageFlags dstStageMask, uint32_t memoryBarrierCount,
                 const VkMemoryBarrier *pMemoryBarriers,
                 uint32_t bufferMemoryBarrierCount,
                 const VkBufferMemoryBarrier *pBufferMemoryBarriers,
                 uint32_t imageMemoryBarrierCount,
                 const VkImageMemoryBarrier *pImageMemoryBarriers)
{
   VK_FROM_HANDLE(hk_cmd_buffer, cmd, commandBuffer);

   cmd->state.brl_legacy_event_wait = true;
   vk_common_CmdWaitEvents(commandBuffer, eventCount, pEvents, srcStageMask,
                           dstStageMask, memoryBarrierCount, pMemoryBarriers,
                           bufferMemoryBarrierCount, pBufferMemoryBarriers,
                           imageMemoryBarrierCount, pImageMemoryBarriers);
   cmd->state.brl_legacy_event_wait = false;
   brl_sync_event_done(cmd);
}
