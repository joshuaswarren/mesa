/*
 * Copyright 2026 Aurora Silicon contributors
 * SPDX-License-Identifier: MIT
 */

#include "hk_gpu_waits.h"

#include "util/macros.h"
#include "vk_drm_syncobj.h"

#include "hk_device.h"
#include "hk_physical_device.h"
#include "hk_queue.h"

static inline struct hk_device *
gw_dev(struct vk_device *vk)
{
   return container_of(vk, struct hk_device, vk);
}

static inline const struct vk_sync_type *
gw_base(struct vk_device *vk)
{
   return &hk_device_physical(gw_dev(vk))->syncobj_base_type;
}

/* Lock held. */
static struct hk_gpu_wait_tag *
gw_find(struct hk_gpu_waits *gw, const struct vk_sync *sync)
{
   for (unsigned i = 0; i < HK_GPU_WAIT_TAGS; ++i) {
      if (gw->tags[i].sync == sync)
         return &gw->tags[i];
   }
   return NULL;
}

static void
gw_forget(struct vk_device *vk, const struct vk_sync *sync)
{
   struct hk_gpu_waits *gw = &gw_dev(vk)->gpu_waits;
   if (!gw->enabled)
      return;

   simple_mtx_lock(&gw->lock);
   struct hk_gpu_wait_tag *tag = gw_find(gw, sync);
   if (tag)
      tag->sync = NULL;
   simple_mtx_unlock(&gw->lock);
}

/* Re-initializing a vk_sync (including at an allocator-reused address) clears
 * any stale record before creating the underlying syncobj. */
static VkResult
gw_init(struct vk_device *vk, struct vk_sync *sync, uint64_t initial_value)
{
   gw_forget(vk, sync);
   return gw_base(vk)->init(vk, sync, initial_value);
}

static VkResult
gw_signal(struct vk_device *vk, struct vk_sync *sync, uint64_t value)
{
   gw_forget(vk, sync);
   return gw_base(vk)->signal(vk, sync, value);
}

static VkResult
gw_signal_many(struct vk_device *vk, uint32_t count,
               const struct vk_sync_signal *signals)
{
   for (uint32_t i = 0; i < count; ++i)
      gw_forget(vk, signals[i].sync);
   return gw_base(vk)->signal_many(vk, count, signals);
}

static VkResult
gw_reset(struct vk_device *vk, struct vk_sync *sync)
{
   gw_forget(vk, sync);
   return gw_base(vk)->reset(vk, sync);
}

static VkResult
gw_reset_many(struct vk_device *vk, uint32_t count,
              struct vk_sync *const *syncs)
{
   for (uint32_t i = 0; i < count; ++i)
      gw_forget(vk, syncs[i]);
   return gw_base(vk)->reset_many(vk, count, syncs);
}

/* The payload moves from src to dst, and its record with it unless shared. */
static VkResult
gw_move(struct vk_device *vk, struct vk_sync *dst, struct vk_sync *src)
{
   struct hk_gpu_waits *gw = &gw_dev(vk)->gpu_waits;
   if (!gw->enabled)
      return gw_base(vk)->move(vk, dst, src);

   simple_mtx_lock(&gw->lock);
   struct hk_gpu_wait_tag *d = gw_find(gw, dst);
   if (d)
      d->sync = NULL;
   VkResult result = gw_base(vk)->move(vk, dst, src);
   /* A shared payload can change elsewhere: its record never carries. */
   struct hk_gpu_wait_tag *s = gw_find(gw, src);
   if (s) {
      s->sync = result == VK_SUCCESS &&
                      !((src->flags | dst->flags) & VK_SYNC_IS_SHARED)
                   ? dst
                   : NULL;
   }
   simple_mtx_unlock(&gw->lock);
   return result;
}

static VkResult
gw_import_opaque_fd(struct vk_device *vk, struct vk_sync *sync, int fd)
{
   gw_forget(vk, sync);
   return gw_base(vk)->import_opaque_fd(vk, sync, fd);
}

static VkResult
gw_export_opaque_fd(struct vk_device *vk, struct vk_sync *sync, int *fd)
{
   gw_forget(vk, sync);
   return gw_base(vk)->export_opaque_fd(vk, sync, fd);
}

static VkResult
gw_import_sync_file(struct vk_device *vk, struct vk_sync *sync, int fd)
{
   gw_forget(vk, sync);
   return gw_base(vk)->import_sync_file(vk, sync, fd);
}

static VkResult
gw_export_sync_file(struct vk_device *vk, struct vk_sync *sync, int *fd)
{
   gw_forget(vk, sync);
   return gw_base(vk)->export_sync_file(vk, sync, fd);
}

void
hk_gpu_waits_wrap_type(struct vk_sync_type *type,
                       const struct vk_sync_type *base)
{
   /* vk_sync_type_is_drm_syncobj() identifies the type by its finish
    * operation, so finish stays the original: destruction is covered by
    * hk_gpu_waits_forget_destroyed(), waits consuming the record, and
    * gw_init() clearing any reused address. */
   assert(type->finish == base->finish);
   if (base->init)
      type->init = gw_init;
   if (base->signal)
      type->signal = gw_signal;
   if (base->signal_many)
      type->signal_many = gw_signal_many;
   if (base->reset)
      type->reset = gw_reset;
   if (base->reset_many)
      type->reset_many = gw_reset_many;
   if (base->move)
      type->move = gw_move;
   if (base->import_opaque_fd)
      type->import_opaque_fd = gw_import_opaque_fd;
   if (base->export_opaque_fd)
      type->export_opaque_fd = gw_export_opaque_fd;
   if (base->import_sync_file)
      type->import_sync_file = gw_import_sync_file;
   if (base->export_sync_file)
      type->export_sync_file = gw_export_sync_file;
}

void
hk_gpu_waits_init(struct hk_device *dev)
{
   struct hk_gpu_waits *gw = &dev->gpu_waits;
   simple_mtx_init(&gw->lock, mtx_plain);
   gw->next = 0;
   for (unsigned i = 0; i < HK_GPU_WAIT_TAGS; ++i)
      gw->tags[i].sync = NULL;

   /* Needs the wrapped sync type and exact prior ordinals, which each queue
    * checks again at submission (serialized G13/G14 queues keep exact_prior
    * off). */
   gw->enabled = !dev->dev.is_virtio &&
                 (dev->dev.params.features &
                  DRM_ASAHI_FEATURE_EXACT_PRIOR_BARRIERS) &&
                 hk_device_physical(dev)->syncobj_sync_type.signal ==
                    gw_signal;
}

void
hk_gpu_waits_finish(struct hk_device *dev)
{
   simple_mtx_destroy(&dev->gpu_waits.lock);
}

VkResult
hk_gpu_waits_copy_sync_payloads(struct vk_device *vk, uint32_t wait_count,
                                const struct vk_sync_wait *waits,
                                uint32_t signal_count,
                                const struct vk_sync_signal *signals)
{
   for (uint32_t i = 0; i < signal_count; ++i)
      gw_forget(vk, signals[i].sync);
   hk_gpu_waits_forget_waits(gw_dev(vk), wait_count, waits);
   return vk_drm_syncobj_copy_payloads(vk, wait_count, waits, signal_count,
                                       signals);
}

/* Only a sync of the wrapped type: any other type's operations would change
 * its payload unseen. */
static bool
gw_eligible_sync(const struct vk_sync *sync)
{
   return sync->type->signal == gw_signal &&
          !(sync->flags & (VK_SYNC_IS_TIMELINE | VK_SYNC_IS_SHARED));
}

void
hk_gpu_waits_forget_signals(struct hk_device *dev, uint32_t signal_count,
                            const struct vk_sync_signal *signals)
{
   for (uint32_t i = 0; i < signal_count; ++i) {
      if (!(signals[i].sync->flags & VK_SYNC_IS_TIMELINE))
         gw_forget(&dev->vk, signals[i].sync);
   }
}

void
hk_gpu_waits_forget_waits(struct hk_device *dev, uint32_t wait_count,
                          const struct vk_sync_wait *waits)
{
   uint64_t unused[HK_SYNC_NUM_SQ] = {0};
   for (uint32_t i = 0; i < wait_count; ++i)
      hk_gpu_waits_take(dev, NULL, waits[i].sync, unused);
}

void
hk_gpu_waits_forget_destroyed(struct hk_device *dev, struct vk_sync *permanent,
                              struct vk_sync *temporary)
{
   if (permanent)
      gw_forget(&dev->vk, permanent);
   if (temporary)
      gw_forget(&dev->vk, temporary);
}

void
hk_gpu_waits_note_signals(struct hk_device *dev, const struct hk_queue *queue,
                          bool eligible, uint32_t signal_count,
                          const struct vk_sync_signal *signals)
{
   struct hk_gpu_waits *gw = &dev->gpu_waits;
   if (!gw->enabled || !signal_count)
      return;

   simple_mtx_lock(&gw->lock);
   for (uint32_t i = 0; i < signal_count; ++i) {
      const struct vk_sync *sync = signals[i].sync;
      if (sync->flags & VK_SYNC_IS_TIMELINE)
         continue;

      struct hk_gpu_wait_tag *tag = gw_find(gw, sync);
      if (!eligible || !gw_eligible_sync(sync)) {
         if (tag)
            tag->sync = NULL;
         continue;
      }

      if (!tag) {
         tag = &gw->tags[gw->next];
         gw->next = (gw->next + 1) % HK_GPU_WAIT_TAGS;
      }
      *tag = (struct hk_gpu_wait_tag){
         .sync = sync,
         .queue = queue,
         .generation = queue->gpu_wait_generation,
      };
      for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
         tag->submitted[sq] = queue->sync.submitted[sq];
   }
   simple_mtx_unlock(&gw->lock);
}

bool
hk_gpu_waits_take(struct hk_device *dev, const struct hk_queue *queue,
                  const struct vk_sync *sync, uint64_t floor[HK_SYNC_NUM_SQ])
{
   struct hk_gpu_waits *gw = &dev->gpu_waits;
   if (!gw->enabled || (sync->flags & VK_SYNC_IS_TIMELINE))
      return false;

   bool taken = false;
   simple_mtx_lock(&gw->lock);
   struct hk_gpu_wait_tag *tag = gw_find(gw, sync);
   if (tag) {
      if (gw_eligible_sync(sync) && queue && tag->queue == queue &&
          tag->generation == queue->gpu_wait_generation) {
         for (unsigned sq = 0; sq < HK_SYNC_NUM_SQ; ++sq)
            floor[sq] = MAX2(floor[sq], tag->submitted[sq]);
         taken = true;
      }
      /* A binary wait consumes the payload, on whichever queue it is: the
       * record never outlives it, also for a temporary payload the runtime
       * destroys after the submission. */
      tag->sync = NULL;
   }
   simple_mtx_unlock(&gw->lock);
   return taken;
}
